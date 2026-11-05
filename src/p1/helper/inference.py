"""每张生成卡共享一个异步 vLLM 引擎，交错推进多个独立题目。

由 pilot_parallel 或 Pilot.model 调用，不单独启动。输入固定模型配置和请求，
输出与原 generate 相同的 CompletionOutput；不改变提示词、采样参数和 token 预算。
模型引擎只在自己的 asyncio 线程内操作；各题控制器使用独立工作区和消息历史。
"""

import asyncio
from concurrent.futures import TimeoutError as FutureTimeout
import json
from pathlib import Path
import threading
import time
import uuid

from .baseline_state import beijing_now


def concurrency(cfg, name, gpu):
    """输入配置、并发字段和 GPU 编号；返回正整数，支持统一整数或逐卡字典。"""
    value = cfg.get(name, 1)
    if isinstance(value, dict):
        value = value.get(str(gpu), value.get(gpu))
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} 必须为正整数，或包含 GPU {gpu} 的正整数字典")
    return value


class InferenceService:
    """同一卡上的题目共享权重及动态批处理，Python 控制器可同步等待各自请求。"""

    def __init__(self, cfg, root, gpu, run, stop=None):
        """输入配置、项目根目录、卡号、运行目录和停止事件；加载唯一模型副本。"""
        from transformers import AutoTokenizer

        self.cfg = cfg
        self.root = Path(root)
        self.gpu = gpu
        self.stop = stop
        self.started = time.monotonic()
        self.active = 0
        self.peak_active = 0
        self.requests = 0
        self.log = Path(run) / "scheduler" / f"generation-gpu-{gpu}.jsonl"
        self.log.parent.mkdir(parents=True, exist_ok=True)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.root / cfg["model"], local_files_only=True
        )
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(
            target=self.loop.run_forever, name=f"vllm-async-{gpu}", daemon=True
        )
        self.thread.start()
        try:
            asyncio.run_coroutine_threadsafe(self._start(), self.loop).result()
        except BaseException:
            self.loop.call_soon_threadsafe(self.loop.stop)
            self.thread.join()
            raise
        self.startup_seconds = time.monotonic() - self.started

    async def _start(self):
        """在事件循环中创建 AsyncLLM；整个进程只有这一个模型实例，无返回值。"""
        from vllm.engine.arg_utils import AsyncEngineArgs
        from vllm.v1.engine.async_llm import AsyncLLM

        args = AsyncEngineArgs(
            model=str(self.root / self.cfg["model"]),
            dtype="bfloat16",
            max_model_len=self.cfg["context_tokens"],
            max_num_seqs=concurrency(self.cfg, "generation_concurrency", self.gpu),
            gpu_memory_utilization=0.85,
            enforce_eager=self.cfg.get("enforce_eager", True),
            enable_prefix_caching=self.cfg.get("enable_prefix_caching", False),
            limit_mm_per_prompt={"image": 0, "video": 0},
            seed=self.cfg["seed"],
            disable_log_stats=True,
        )
        self.engine = AsyncLLM.from_engine_args(args)
        self._record(
            "engine_ready",
            generation_concurrency=args.max_num_seqs,
            enable_prefix_caching=args.enable_prefix_caching,
            enforce_eager=args.enforce_eager,
        )

    def _record(self, event, **fields):
        """输入事件及字段；只在引擎线程内追加北京时间日志，不保存提示词正文。"""
        row = {"at": beijing_now().isoformat(timespec="milliseconds"),
               "event": event, "generation_gpu": self.gpu, **fields}
        with self.log.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")

    async def _generate(self, ids, params, request_id, label):
        """输入 token、采样参数、请求 ID 和轨迹标签；返回本请求最后一次完整输出。"""
        self.active += 1
        self.requests += 1
        self.peak_active = max(self.peak_active, self.active)
        self._record("request_started", request_id=request_id, trajectory=label,
                     active=self.active, prompt_tokens=len(ids), seed=params.seed)
        output = None
        try:
            async for item in self.engine.generate(
                {"prompt_token_ids": ids}, params, request_id=request_id
            ):
                output = item
            if output is None:
                raise RuntimeError("异步引擎没有返回请求结果")
            return output.outputs[0]
        finally:
            self.active -= 1
            self._record("request_finished", request_id=request_id,
                         trajectory=label, active=self.active)

    def generate(self, ids, params, label=""):
        """输入单条请求；提交给共享引擎并等待自己的结果，其他题目可同时生成。"""
        request_id = uuid.uuid4().hex
        future = asyncio.run_coroutine_threadsafe(
            self._generate(ids, params, request_id, label), self.loop
        )
        while True:
            try:
                return future.result(timeout=0.5)
            except FutureTimeout:
                if future.done():
                    raise
                if self.stop is not None and self.stop.is_set():
                    asyncio.run_coroutine_threadsafe(
                        self.engine.abort(request_id), self.loop
                    ).result(timeout=15)
                    future.cancel()
                    raise InterruptedError("停止未完成模型请求")

    def close(self):
        """无输入；在所属事件循环中关闭引擎，再退出线程，不影响其他卡的引擎。"""
        async def shutdown():
            self.engine.shutdown()
            await asyncio.sleep(0)

        try:
            asyncio.run_coroutine_threadsafe(shutdown(), self.loop).result()
        finally:
            self.loop.call_soon_threadsafe(self.loop.stop)
            self.thread.join()

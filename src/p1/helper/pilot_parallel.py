"""P1 多卡调度：每张生成卡一个模型副本，共享完整评测作业的 GPU 池。

由 p1.run 调用，不单独启动。配置的 GPU 字段接受整数或整数列表。
父进程独占写根汇总；子进程只写各自题目的轨迹和原有 evaluations 数字目录。
文件锁置于 .runtime/locks，所有卡只读同一份 worker-config.json。
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
import fcntl
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
from queue import Empty
import signal
import threading
import time
import traceback
import uuid

from .baseline_state import (
    archive_incomplete, atomic_json, beijing_now, cleanup_reference_artifacts,
)
from ..evaluation import Evaluator
from .inference import InferenceService, concurrency

LOCK_ROOT = Path(__file__).resolve().parents[3] / ".runtime/locks"
MODES = ("single", "agent")


def gpu_numbers(value, name):
    """输入单个编号或列表和字段名；返回非空、无重复的非负编号列表，非法输入抛错。"""
    devices = [value] if type(value) is int else value
    if (not isinstance(devices, list) or not devices
            or any(type(device) is not int or device < 0 for device in devices)
            or len(devices) != len(set(devices))):
        raise ValueError(f"{name} 必须为非负 GPU 编号或无重复的编号列表")
    return list(devices)


def gpu_roles(cfg):
    """输入配置；返回生成/评测编号列表，拒绝两种用途共享同一卡以免干扰测时。"""
    generation = gpu_numbers(cfg.get("generation_gpu", 2), "generation_gpu")
    evaluation = gpu_numbers(cfg.get("evaluation_gpu", 3), "evaluation_gpu")
    if set(generation) & set(evaluation):
        raise ValueError("generation_gpu 与 evaluation_gpu 不能包含相同 GPU")
    for gpu in generation:
        concurrency(cfg, "generation_concurrency", gpu)
    for gpu in evaluation:
        concurrency(cfg, "evaluation_concurrency", gpu)
    return generation, evaluation


def scheduling_directory(run):
    """输入 run；返回该运行的宿主调度锁目录，保持 run 内原有输出布局。"""
    identity = hashlib.sha256(str(Path(run).resolve()).encode()).hexdigest()
    return LOCK_ROOT / f"p1-pilot-{identity}"


@contextmanager
def blocking_lock(path):
    """输入锁路径；跨进程等待并持锁，异常或进程退出自动释放，不删除锁文件。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


class EvaluationPool:
    """每卡提供可配置验算槽位；性能测量使用写锁，等待同卡全部读作业结束。"""

    def __init__(self, devices, directory, stop=None, cfg=None, log=None):
        """输入卡号、锁目录、停止事件和配置；初始化读写锁及逐卡并发上限。"""
        self.devices = gpu_numbers(devices, "evaluation_gpu")
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.stop = stop
        self.cfg = cfg or {}
        self.log = Path(log) if log is not None else self.directory / "evaluation-events.jsonl"
        self.log.parent.mkdir(parents=True, exist_ok=True)
        self.offset = 0

    def record(self, event, gpu, exclusive, slot=None):
        """输入租约状态；追加北京时间日志，用于核对实际并发和独占测时。"""
        with self.log.open("a", encoding="utf-8") as stream:
            fcntl.flock(stream, fcntl.LOCK_EX)
            stream.write(json.dumps({
                "at": beijing_now().isoformat(timespec="milliseconds"),
                "event": event, "evaluation_gpu": gpu,
                "exclusive": exclusive, "slot": slot, "pid": os.getpid(),
                "thread": threading.current_thread().name,
            }) + "\n")
            stream.flush()

    @contextmanager
    def lease(self, exclusive=True):
        """输入是否正式计时；返回物理卡号，验算读锁可并发、计时写锁完全独占。"""
        while True:
            if self.stop is not None and self.stop.is_set():
                raise InterruptedError("P1 已请求停止，取消等待评测卡")
            order = self.devices[self.offset:] + self.devices[:self.offset]
            for gpu in order:
                gate = (self.directory / f"evaluation-gpu-{gpu}.gate").open("a+")
                main = (self.directory / f"evaluation-gpu-{gpu}.lock").open("a+")
                slot = None
                slot_index = None
                acquired = False
                try:
                    # 写者先占门闩，阻止新验算继续进入，避免测速长期饥饿。
                    fcntl.flock(
                        gate, (fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)
                        | fcntl.LOCK_NB,
                    )
                    if exclusive:
                        while True:
                            try:
                                fcntl.flock(main, fcntl.LOCK_EX | fcntl.LOCK_NB)
                                break
                            except BlockingIOError:
                                if self.stop is not None and self.stop.is_set():
                                    raise InterruptedError("停止等待独占测时")
                                time.sleep(0.05)
                    else:
                        fcntl.flock(main, fcntl.LOCK_SH | fcntl.LOCK_NB)
                        for index in range(concurrency(self.cfg, "evaluation_concurrency", gpu)):
                            candidate = (self.directory / f"evaluation-gpu-{gpu}-slot-{index}.lock").open("a+")
                            try:
                                fcntl.flock(candidate, fcntl.LOCK_EX | fcntl.LOCK_NB)
                            except BlockingIOError:
                                candidate.close()
                                continue
                            slot, slot_index = candidate, index
                            break
                        if slot is None:
                            continue
                        # 验算只短暂经过门闩；写者可以占门闩等待已有读者自然退出。
                        fcntl.flock(gate, fcntl.LOCK_UN)
                    acquired = True
                except BlockingIOError:
                    pass
                finally:
                    if not acquired:
                        if slot is not None:
                            slot.close()
                        main.close()
                        gate.close()
                if not acquired:
                    continue
                self.offset = (self.devices.index(gpu) + 1) % len(self.devices)
                self.record("lease_started", gpu, exclusive, slot_index)
                try:
                    yield gpu
                finally:
                    self.record("lease_finished", gpu, exclusive, slot_index)
                    if slot is not None:
                        slot.close()
                    main.close()
                    gate.close()
                return
            time.sleep(0.05)


class ParallelEvaluator(Evaluator):
    """给原评测器增加卡池和快照编号锁；数值检查及测时仍使用原 Evaluator。"""

    def __init__(self, cfg, run, pool):
        """输入公共配置、run 和卡池；只读复用父进程写好的公共 worker 配置。"""
        super().__init__(cfg, run, gpu=pool.devices[0], write_config=False)
        self.pool = pool
        self._operation_lock = threading.RLock()
        self._leased = False

    @contextmanager
    def operation(self, exclusive=True):
        """无输入；完整作业持有同一张卡，嵌套 prepare/重试复用已有租约。"""
        with self._operation_lock:
            if self._leased:
                yield
                return
            with self.pool.lease(exclusive=exclusive) as gpu:
                self.gpu = gpu
                self._leased = True
                try:
                    yield
                finally:
                    self._leased = False

    def prepare(self, task):
        """输入任务；参数预检及初始基线作为一个作业使用同一卡，返回原预检索引。"""
        with self.operation():
            return super().prepare(task)

    def profile_reference(self, task):
        """输入任务；参考性能分析使用一个租约，返回原性能结果。"""
        with self.operation():
            return super().profile_reference(task)

    def gpu_operation(self, exclusive=True):
        """输入阶段类型；编译不取卡，验算共享槽位，正式计时独占。"""
        return self.operation(exclusive=exclusive)

    def snapshot(self, workspace):
        """输入工作区；跨进程串行分配 evaluations 编号，返回原快照与审查结果。"""
        with blocking_lock(self.pool.directory / "evaluation-id.lock"):
            self.counter = max(
                (int(path.name) for path in (self.run / "evaluations").iterdir()
                 if path.is_dir() and path.name.isdigit()),
                default=0,
            ) if (self.run / "evaluations").exists() else 0
            return super().snapshot(workspace)


def saved_trajectory(run, task_id, mode):
    """输入 run、题号、模式；返回完整结果的标量摘要，缺文件返回 None，损坏时拒绝覆盖。"""
    path = Path(run) / "trajectories" / task_id / mode / "result.json"
    if not path.exists():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("task_id") != task_id or value.get("mode") != mode or "correct" not in value:
        raise ValueError(f"轨迹结果身份不匹配：{path}")
    return compact_result(value)


def compact_result(value):
    """输入完整轨迹结果；返回根汇总所需标量，避免跨进程复制完整候选和工具日志。"""
    return {key: item for key, item in value.items()
            if item is None or isinstance(item, (str, int, float, bool))}


def archive_partial_trajectory(run, task_id, mode):
    """输入待重做轨迹；归档没有最终结果的旧工作区，返回 None，避免遗留源码影响新生成。"""
    old = Path(run) / "trajectories" / task_id / mode
    if not old.exists():
        return
    if (old / "result.json").exists():
        raise ValueError("不能归档已有最终结果的轨迹")
    stamp = beijing_now().strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:8]
    target = Path(run) / "interrupted" / stamp / task_id / mode
    target.parent.mkdir(parents=True, exist_ok=True)
    old.rename(target)


def generation_worker(cfg, run, gpu, directory, jobs, events, stop, pilot_factory=None):
    """输入卡号和任务队列；一份模型服务多个独立题目，逐题回传两种模式结果。"""
    real_model = pilot_factory is None
    if real_model:
        os.setsid()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    os.environ["VLLM_NO_USAGE_STATS"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    if threading.current_thread() is threading.main_thread():
        signal.signal(signal.SIGTERM, lambda signum, frame: stop.set())
    if real_model:
        from ..run import Pilot
        from ..data import ROOT
        pilot_factory = Pilot
    service = None
    pilots = []
    finished = 0
    state_lock = threading.Lock()
    started = time.monotonic()

    def consume():
        """每个控制线程有自己的 Pilot/评测器；只共享模型服务和跨进程资源锁。"""
        nonlocal finished
        pool = EvaluationPool(
            gpu_roles(cfg)[1], directory, stop, cfg,
            log=Path(run) / "scheduler/evaluation.jsonl",
        )
        evaluator = ParallelEvaluator(cfg, run, pool)
        evaluator.stop_event = stop
        pilot = pilot_factory(cfg, Path(run), generation_gpu=gpu, evaluator=evaluator)
        pilot.stop_event = stop
        if service is not None:
            pilot.inference = service
            pilot.tokenizer = service.tokenizer
            pilot.model_started = service.started
        with state_lock:
            pilots.append(pilot)
        try:
            while not stop.is_set():
                job = jobs.get()
                if job is None:
                    break
                index, task = job
                events.put(("task_started", {
                    "task_id": task["task_id"], "generation_gpu": gpu,
                    "thread": threading.current_thread().name,
                    "at": beijing_now().isoformat(timespec="milliseconds"),
                }))
                for mode in MODES:
                    if stop.is_set():
                        raise InterruptedError("P1 已请求停止")
                    row = saved_trajectory(run, task["task_id"], mode)
                    if row is None:
                        archive_partial_trajectory(run, task["task_id"], mode)
                        pilot.model()
                        row = compact_result(pilot.trajectory(task, mode, index))
                    events.put(("result", row))
                with state_lock:
                    finished += 1
                events.put(("task_complete", task["task_id"]))
        finally:
            pilot.close_model()

    try:
        if real_model:
            service = InferenceService(cfg, ROOT, gpu, run, stop)
        count = concurrency(cfg, "generation_concurrency", gpu)
        with ThreadPoolExecutor(max_workers=count, thread_name_prefix=f"trajectory-gpu-{gpu}") as pool:
            futures = [pool.submit(consume) for _ in range(count)]
            for future in as_completed(futures):
                try:
                    future.result()
                except InterruptedError:
                    stop.set()
                except BaseException:
                    stop.set()
                    events.put(("error", {"generation_gpu": gpu, "traceback": traceback.format_exc()}))
    except BaseException:
        stop.set()
        events.put(("error", {"generation_gpu": gpu, "traceback": traceback.format_exc()}))
    finally:
        if service is not None:
            try:
                service.close()
            except BaseException:
                stop.set()
                events.put(("error", {"generation_gpu": gpu, "traceback": traceback.format_exc()}))
        events.put(("worker_done", {
            "generation_gpu": gpu, "tasks_completed": finished,
            "generation_gpu_reserved_seconds": time.monotonic() - started,
            "container_seconds": sum(p.ev.seconds for p in pilots),
            "compiles": sum(p.ev.compiles for p in pilots),
            "compile_cache_hits": sum(getattr(p.ev, "cache_hits", 0) for p in pilots),
            "max_active_sequences": service.peak_active if service is not None else None,
            "model_startup_seconds": service.startup_seconds if service is not None else None,
            "generation_requests": service.requests if service is not None else None,
        }))


def run_generation_workers(cfg, run, pending, directory, stop, on_event,
                           worker_target=generation_worker):
    """输入待处理整题、停止事件和父进程回调；spawn 模型进程并收集结果，返回成本列表。

    worker_target 可注入 CPU 假进程。显式检查退出码，进程异常退出不能无限等待队列。
    """
    context = multiprocessing.get_context("spawn")
    jobs = context.Queue()
    events = context.Queue()
    devices = gpu_roles(cfg)[0][:len(pending)]
    for job in pending:
        jobs.put(job)
    for gpu in devices:
        for _ in range(concurrency(cfg, "generation_concurrency", gpu)):
            jobs.put(None)
    workers = [context.Process(target=worker_target,
                               args=(cfg, str(run), gpu, str(directory), jobs, events, stop),
                               name=f"p1-generation-gpu-{gpu}") for gpu in devices]
    statistics = []
    errors = []
    dead_reported = set()
    callback_failed = False
    try:
        for worker in workers:
            try:
                worker.start()
            except BaseException:
                stop.set()
                raise
        while len(statistics) < len(workers):
            try:
                kind, payload = events.get(timeout=0.5)
            except Empty:
                dead = [worker for worker in workers if worker.exitcode not in (None, 0)]
                if dead:
                    stop.set()
                    for worker in dead:
                        if worker.name not in dead_reported:
                            errors.append(f"{worker.name} 异常退出：{worker.exitcode}")
                            dead_reported.add(worker.name)
                if all(worker.exitcode is not None for worker in workers):
                    # 退出前 put 的事件会由 multiprocessing feeder 刷入；再等一轮读取尾部。
                    try:
                        kind, payload = events.get(timeout=1)
                    except Empty:
                        if len(statistics) != len(workers):
                            errors.append("生成进程退出但缺少最终成本事件")
                        break
                    else:
                        pass
                else:
                    continue
            if kind == "worker_done":
                statistics.append(payload)
            elif kind == "error":
                errors.append(payload["traceback"])
                stop.set()
            elif not callback_failed:
                try:
                    on_event(kind, payload)
                except Exception:
                    # 汇总写入失败后不再处理整题清理事件，保留参数供下次恢复。
                    callback_failed = True
                    errors.append(traceback.format_exc())
                    stop.set()
    finally:
        for worker in workers:
            if worker.pid is not None:
                worker.join()
        # 未消费完的任务无需刷入管道，否则中断退出时 feeder 可能等待已离开的消费者。
        jobs.cancel_join_thread()
        jobs.close()
        events.close()
    for worker in workers:
        if worker.exitcode not in (None, 0) and worker.name not in dead_reported:
            errors.append(f"{worker.name} 异常退出：{worker.exitcode}")
    if errors:
        raise RuntimeError("P1 多卡运行中断，已保存完成结果：\n" + "\n".join(errors))
    return statistics


def execute_parallel(pilot, prepare_only=False, stop=None):
    """输入父 Pilot 和预检选项；并行预检、派发整题、统一落盘并在完成题后清理。

    父进程不加载 Qwen；完整轨迹保存在原路径，可从最终结果恢复而无需重新生成。
    """
    context = multiprocessing.get_context("spawn")
    stop = stop if stop is not None else context.Event()
    generation, evaluation = gpu_roles(pilot.cfg)
    directory = scheduling_directory(pilot.run)
    task_file = pilot.run / "tasks.json"
    from ..data import ROOT
    tasks = json.loads((task_file if task_file.exists() else ROOT / "data/processed/pilot.json").read_text())
    ids = [task["task_id"] for task in tasks]
    if len(ids) != len(set(ids)):
        raise ValueError("pilot 题单含重复 task_id，拒绝并行覆盖结果")
    atomic_json(task_file, tasks)
    manifest = pilot.make_manifest(tasks)
    manifest["execution"] = {"generation_gpus": generation, "evaluation_gpus": evaluation,
                             "generation_strategy": "shared_async_model_per_gpu",
                             "generation_concurrency": pilot.cfg.get("generation_concurrency", 1),
                             "evaluation_concurrency": pilot.cfg.get("evaluation_concurrency", 1)}
    atomic_json(pilot.run / "manifest.json", manifest)
    records = {}
    indices = {task["task_id"]: index for index, task in enumerate(tasks)}
    eligible = []
    progress = getattr(pilot, "progress", None)
    if progress is not None:
        progress.start_references(tasks)
    evaluators = []
    local = threading.local()

    def persist():
        """按固定题单顺序由父进程唯一写入摘要；子进程不会竞争 results.json。"""
        keys = [(task["task_id"], mode) for task in tasks for mode in ("reference", *MODES)]
        pilot.results = [records[key] for key in keys if key in records]
        pilot.save_result_summaries()

    def prepare(task):
        """读取已有预检或租卡准备参考；缺少最终索引的旧参考目录先归档。"""
        index = pilot.run / "references" / task["task_id"] / "preflight.json"
        if index.exists():
            cached = json.loads(index.read_text())
            # 缺依赖等基础设施故障可在环境修复后重试，不能当作永久拒绝跳过。
            if cached.get("status") in {"ok", "task_invalid"}:
                return cached
        if stop.is_set():
            raise InterruptedError("停止参考预检")
        if not hasattr(local, "evaluator"):
            pool = EvaluationPool(
                evaluation, directory, stop, pilot.cfg,
                log=pilot.run / "scheduler/evaluation.jsonl",
            )
            local.evaluator = ParallelEvaluator(pilot.cfg, pilot.run, pool)
            evaluators.append(local.evaluator)
        archive_incomplete(pilot.run, task["task_id"])
        return local.evaluator.prepare(task)

    workers = []
    try:
        with ThreadPoolExecutor(max_workers=len(evaluation), thread_name_prefix="p1-preflight") as pool:
            futures = {pool.submit(prepare, task): task for task in tasks}
            for future in as_completed(futures):
                task = futures[future]
                try:
                    preflight = future.result()
                except BaseException:
                    # 必须在等待线程池退出前通知其他预检停止，取消尚未开始的任务。
                    stop.set()
                    for queued in futures:
                        queued.cancel()
                    raise
                if preflight["status"] == "ok":
                    eligible.append(task)
                    for mode in MODES:
                        row = saved_trajectory(pilot.run, task["task_id"], mode)
                        if row is not None:
                            records[(task["task_id"], mode)] = row
                else:
                    records[(task["task_id"], "reference")] = {
                        "task_id": task["task_id"], "mode": "reference", "status": "task_invalid",
                    }
                persist()
                if progress is not None:
                    progress.reference_done(task["task_id"], preflight["status"])
                if preflight["status"] != "ok":
                    cleanup_reference_artifacts(pilot.run, task["task_id"])
        eligible.sort(key=lambda task: indices[task["task_id"]])
        if prepare_only:
            manifest.update(status="prepared", updated_at=beijing_now().isoformat(timespec="seconds"))
            atomic_json(pilot.run / "manifest.json", manifest)
            return
        if progress is not None:
            progress.start_modes(eligible, pilot.results)
        pending = []
        for index, task in enumerate(eligible):
            if all((task["task_id"], mode) in records for mode in MODES):
                cleanup_reference_artifacts(pilot.run, task["task_id"])
                continue
            inputs = pilot.run / "references" / task["task_id"] / "cases/input-spec"
            required = [inputs / "model.py"] + [inputs / f"state-{seed}.pt"
                                                for seed in [71001, *pilot.cfg.get("hidden_init_seeds", [])]]
            if not all(path.is_file() for path in required):
                raise RuntimeError(f"待完成题缺少参考参数，请重新预检或新建运行：{task['task_id']}")
            pending.append((index, task))

        def on_event(kind, payload):
            """接收轨迹或整题完成事件；先保存结果，再清理该题参考参数。"""
            if kind == "task_started":
                log = pilot.run / "scheduler/tasks.jsonl"
                log.parent.mkdir(exist_ok=True)
                with log.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps({"event": kind, **payload}) + "\n")
                if progress is not None:
                    progress.write("TASK_START", payload["task_id"], "GPU", payload["generation_gpu"])
            elif kind == "result":
                records[(payload["task_id"], payload["mode"])] = payload
                persist()
            elif kind == "task_complete":
                log = pilot.run / "scheduler/tasks.jsonl"
                with log.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps({
                        "event": "task_finished", "task_id": payload,
                        "at": beijing_now().isoformat(timespec="milliseconds"),
                    }) + "\n")
                if not all((payload, mode) in records for mode in MODES):
                    raise RuntimeError("整题完成事件早于两条轨迹结果")
                cleanup_reference_artifacts(pilot.run, payload)

        if pending:
            workers = run_generation_workers(pilot.cfg, pilot.run, pending, directory, stop, on_event)
        costs = {
            "container_seconds_this_invocation": sum(item.seconds for item in evaluators)
                                                 + sum(item["container_seconds"] for item in workers),
            "compiles_this_invocation": sum(item.compiles for item in evaluators)
                                        + sum(item["compiles"] for item in workers),
            "compile_cache_hits": sum(getattr(item, "cache_hits", 0) for item in evaluators)
                                  + sum(item.get("compile_cache_hits", 0) for item in workers),
            "generation_gpu_reserved_seconds": sum(item["generation_gpu_reserved_seconds"] for item in workers),
            "generation_workers": workers,
        }
        pilot.finish_summary(tasks, eligible, manifest, costs=costs, complete=not stop.is_set())
    except BaseException as error:
        stop.set()
        persist()
        manifest.update(status="interrupted" if isinstance(error, (InterruptedError, KeyboardInterrupt)) else "failed",
                        updated_at=beijing_now().isoformat(timespec="seconds"), error=str(error))
        atomic_json(pilot.run / "manifest.json", manifest)
        raise

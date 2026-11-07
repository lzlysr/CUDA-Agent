"""可信 CUDA 奖励适配器：单次最终输出 → 五组反馈验算 → 后测基线奖励。

由 verl RewardLoopWorker 导入，不直接运行。输入原始 response token 的解码文本，
输出终局 -1/1/2/3；环境错误抛异常并留下证据，不伪装成模型负分。
test 验证额外运行隐藏验收；不修复 JSON、不选择历史最优源码。
"""

import json
from pathlib import Path
import shutil
import uuid

from p2.common import ROOT, now, write_json


def evaluate_response(config, response, task, hidden=False):
    """输入项目配置、原始输出和训练/验证任务；返回 verl 标量奖励及审查索引。

    每次请求独立保存生成源码和结果；同题请求持锁保护参考输入，测速独占评测卡。
    完成后清理 input-spec、compile-cache；代码/日志和后测基线记录继续保留。
    """
    from p1.evaluation import Evaluator, InfrastructureError
    from p1.helper.baseline_state import cleanup_reference_artifacts
    from p1.helper.pilot_parallel import EvaluationPool, ParallelEvaluator, blocking_lock
    from p1.helper.rewards import training_reward
    from p1.helper.tool_calls import parse_single
    from p1.run import write_files

    run = Path(config["run"])
    request = run / "samples" / uuid.uuid4().hex
    request.mkdir(parents=True)
    (request / "response.txt").write_text(response, encoding="utf-8")
    cfg = config["kernel"]
    evaluator_run = run / "kernel"
    # 初始化只写独立工作区；公共配置由训练入口创建，避免多 reward worker 相互覆盖。
    pool = EvaluationPool(cfg["evaluation_gpu"], run / "scheduler", cfg=cfg)
    evaluator = ParallelEvaluator(cfg, evaluator_run, pool)
    try:
        try:
            files = parse_single(response)
            workspace = request / "workspace"
            workspace.mkdir()
            write_files(workspace, files)
        except (ValueError, TypeError, KeyError, json.JSONDecodeError) as error:
            result = {"reward": -1, "reward_valid": True, "failure_kind": "candidate", "reason": str(error)}
        else:
            official = ROOT / "upstream/CUDA-Agent/agent_workdir"
            for name in ("binding.cpp", "binding_registry.h"):
                shutil.copyfile(official / name, workspace / name)
            # 同题参考目录不可被并行重建或清理；不同题仍使用共享卡池并发验算。
            with blocking_lock(run / "scheduler" / f"task-{task['task_id']}.lock"):
                try:
                    preflight = evaluator.prepare(task)
                    if preflight["status"] != "ok":
                        raise InfrastructureError("已通过准备阶段的参考任务本次无法评分")
                    evaluation = evaluator.evaluate(task, workspace)
                    acceptance = evaluator.evaluate(task, workspace, final=True) if hidden and evaluation.get("correct") else None
                    result = dict(training_reward(evaluation, acceptance), correct=bool(evaluation.get("correct")))
                    result["speedup_eager"] = evaluation.get("speedup_eager")
                    result["speedup_compile"] = evaluation.get("speedup_compile")
                    if acceptance is not None and not acceptance.get("correct"):
                        result["correct"] = False
                    if not result["reward_valid"]:
                        raise InfrastructureError(f"评测返回无效奖励：{result}")
                finally:
                    cleanup_reference_artifacts(evaluator_run, task["task_id"])
        result.update(task_id=task["task_id"], hidden=hidden, created_at=now().isoformat())
        write_json(request / "result.json", result)
        return {"score": result["reward"], "acc": int(result.get("correct", False)),
                "reward_valid": True, "hidden": hidden, "detail": str(request / "result.json")}
    except Exception as error:
        write_json(request / "infrastructure-error.json", {
            "at": now().isoformat(), "error_type": type(error).__name__, "error": str(error), "task_id": task["task_id"],
        })
        raise


from verl.experimental.reward_loop.reward_manager.base import RewardManagerBase


class CudaRewardManager(RewardManagerBase):
    """verl 公开奖励管理器扩展；保留 think 分隔符，精确解码实际生成 token。"""

    def __init__(self, config, tokenizer, compute_score=None,
                 reward_router_address=None, reward_model_tokenizer=None):
        """接收官方 loader 的配置/分词器/可选 RM 参数；初始化规则奖励管理器。"""
        if reward_router_address is not None or reward_model_tokenizer is not None:
            raise ValueError("CUDA 规则奖励不使用额外的神经奖励模型")
        super().__init__(config, tokenizer, compute_score)

    async def run_single(self, data):
        """输入一条 verl DataProto；返回终局 reward_score 和日志字段。"""
        item = data[-1:][0]
        response_ids = item.batch["responses"]
        length = int(item.batch["attention_mask"][-response_ids.shape[-1]:].sum())
        if length <= 0:
            raise ValueError("空 response 无法放置终局奖励，停止本批次")
        tokens = response_ids[:length].tolist()
        # 仅移除实际 EOS；保留 <think>/</think>，严格解析器用它排除推理区 JSON。
        while tokens and tokens[-1] == self.tokenizer.eos_token_id:
            tokens.pop()
        response = self.tokenizer.decode(tokens, skip_special_tokens=False)
        extra = item.non_tensor_batch["extra_info"]
        from omegaconf import OmegaConf
        config = OmegaConf.to_container(self.config.p2, resolve=True)
        task = json.loads(extra["task_json"])
        result = await self.loop.run_in_executor(
            None, lambda: evaluate_response(config, response, task, bool(extra.get("hidden", False)))
        )
        return {"reward_score": result.pop("score"), "reward_extra_info": result}

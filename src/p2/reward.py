"""可信 CUDA 奖励适配器：单次最终输出 → 五组反馈验算 → 后测基线奖励。

由 verl RewardLoopWorker 导入，不直接运行。输入原始 response token 的解码文本，
输出终局 -1/1/2/3；环境错误抛异常并留下证据，不伪装成模型负分。
test 验证额外运行隐藏验收；共享解析器容错结束标签，不补源码、不选择历史最优源码。
"""

import json
from pathlib import Path
import shutil
import uuid

from p2.common import ROOT, now, write_json


def compact_evaluation_result(run, evaluation):
    """输入 P2 run 与完整内存评测结果；精简磁盘总结果，返回相对 run 的结果路径。

    奖励仍使用未修改的 evaluation；阶段详情保存在原文件中，总结果只存路径引用。
    此函数仅由 P2 调用，不改变 P1 的返回值或默认输出格式。
    """
    run = Path(run).resolve()
    directory = run / "kernel/evaluations" / str(evaluation["evaluation_id"])
    if directory.resolve().parent != (run / "kernel/evaluations").resolve():
        raise ValueError("评测结果目录超出本次 run")
    path = directory / "result.json"
    summary = json.loads(path.read_text(encoding="utf-8"))
    details = {}
    for field, relative in (
        ("compilation", "compilation/result.json"),
        ("execution", "candidate-stream/result.json"),
        ("reference_execution", "reference-stream/result.json"),
        ("timing", "timing/result.json"),
        ("baseline_after", "baseline-after/bench_result.json"),
    ):
        detail = directory / relative
        if detail.is_file():
            # 控制器可能补充传输方式等字段；先合入阶段文件，再移除总结果副本。
            if isinstance(summary.get(field), dict):
                record = json.loads(detail.read_text(encoding="utf-8"))
                record.update(summary.pop(field))
                write_json(detail, record)
            details[field] = detail.relative_to(run).as_posix()
    summary["detail_paths"] = details
    summary["candidate_path"] = (directory / "candidate").relative_to(run).as_posix()
    write_json(path, summary)
    return path.relative_to(run).as_posix()


def cleanup_candidate_workspace(request, candidate):
    """输入本次样本目录和可信候选快照；仅在全部源码已有相同副本时删除工作区。

    返回是否删除。未形成完整快照、含符号链接或内容不一致时保留现场，便于排查。
    调用方必须先保存样本结果，并确认反馈评测及可选隐藏验收均已结束。
    """
    request = Path(request).resolve()
    workspace = request / "workspace"
    candidate = Path(candidate)
    if not workspace.is_dir() or workspace.is_symlink() or not candidate.is_dir():
        return False
    if workspace.resolve().parent != request:
        raise ValueError("清理工作区超出本次样本目录")
    paths = list(workspace.rglob("*"))
    if any(path.is_symlink() for path in paths):
        return False
    sources = [path for path in paths if path.is_file()]
    for source in sources:
        snapshot = candidate / source.relative_to(workspace)
        if not snapshot.is_file() or snapshot.read_bytes() != source.read_bytes():
            return False
    if not sources:
        return False
    shutil.rmtree(workspace)
    return True


def evaluate_response(config, response, task, hidden=False):
    """输入项目配置、原始输出和训练/验证任务；返回 verl 标量奖励及审查索引。

    每次请求保存原始回答和结果索引；源码只长期保留评测快照，完整快照写好后清理工作区。
    同题请求持锁保护参考输入，测速独占评测卡；继续清理 input-spec、compile-cache。
    """
    from p1.evaluation import InfrastructureError
    from p1.helper.baseline_state import cleanup_reference_artifacts
    from p1.helper.pilot_parallel import EvaluationPool, ParallelEvaluator, blocking_lock
    from p1.helper.rewards import training_reward
    from p1.helper.tool_calls import parse_single
    from p1.run import write_files

    run = Path(config["run"]).resolve()
    request = run / "samples" / uuid.uuid4().hex
    request.mkdir(parents=True)
    (request / "response.txt").write_text(response, encoding="utf-8")
    cfg = config["kernel"]
    evaluator_run = run / "kernel"
    # 初始化只写独立工作区；公共配置由训练入口创建，避免多 reward worker 相互覆盖。
    pool = EvaluationPool(cfg["evaluation_gpu"], run / "scheduler", cfg=cfg)
    evaluator = ParallelEvaluator(cfg, evaluator_run, pool)
    evaluation = acceptance = None
    reference_failure = None
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
                        # 准备阶段只有轻量检查；GPU 参考失败时保留实际失败阶段及原始原因。
                        detail_relative = preflight.get("bench_result_path") or preflight["prepare_result_path"]
                        detail_path = evaluator_run / detail_relative
                        detail = json.loads(detail_path.read_text(encoding="utf-8"))
                        reference_failure = {
                            "preflight_status": preflight["status"],
                            "failure_kind": detail.get("failure_kind", "reference_invalid"),
                            "error_type": detail.get("error_type"),
                            "error": detail.get("error") or (
                                "reference output rejected as near-constant"
                                if detail.get("near_constant_sampled_output")
                                else f"reference stage returned {detail.get('status')}"
                            ),
                            "detail_path": detail_path.relative_to(run).as_posix(),
                        }
                        raise InfrastructureError(
                            f"参考预检失败：task_id={task['task_id']}; "
                            f"status={preflight['status']}; "
                            f"failure_kind={reference_failure['failure_kind']}; "
                            f"{reference_failure['error_type']}: {reference_failure['error']}; "
                            f"detail_path={detail_path}"
                        )
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
        # 先由完整内存结果计算奖励，再精简持久化详情；反馈与隐藏验收分别索引。
        feedback_path = compact_evaluation_result(run, evaluation) if evaluation is not None else None
        acceptance_path = compact_evaluation_result(run, acceptance) if acceptance is not None else None
        result.update(
            sample_id=request.name, task_id=task["task_id"], hidden=hidden,
            correct=bool(result.get("correct", False)), created_at=now().isoformat(),
            response_path=(request / "response.txt").relative_to(run).as_posix(),
            feedback_result_path=feedback_path, acceptance_result_path=acceptance_path,
        )
        write_json(request / "result.json", result)
        if feedback_path is not None:
            # 结果及源码快照落盘之后才清理；失败时保留工作区，不改变已经算出的奖励。
            try:
                cleanup_candidate_workspace(request, (run / feedback_path).parent / "candidate")
            except OSError as error:
                result["workspace_cleanup_error"] = str(error)
                write_json(request / "result.json", result)
        return {"score": result["reward"], "acc": int(result.get("correct", False)),
                "reward_valid": True, "hidden": hidden, "detail": str(request / "result.json")}
    except Exception as error:
        failure = {
            "at": now().isoformat(), "error_type": type(error).__name__, "error": str(error), "task_id": task["task_id"],
        }
        # 原阶段文件保留完整 traceback；样本索引保存底层分类、原因及相对 run 的详情路径。
        if reference_failure is not None:
            failure["reference_failure"] = reference_failure
        write_json(request / "infrastructure-error.json", failure)
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

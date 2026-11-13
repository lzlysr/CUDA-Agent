"""统一评测奖励的语义，避免把编译日志或隐藏验收占位分直接送入 PPO。

由 p1.evaluation 调用 set_reward；未来训练入口可调用 training_reward。
输入是同一冻结候选的评测字典，输出含 reward、reward_valid 和失败归属。
本模块不启动 GPU，不采样，不保存 PPO batch，也不根据历史最优候选重新归奖。
"""


def set_reward(result, value, *, stage, failure_kind=None, reason=None, valid=True):
    """输入待更新结果及奖励来源；原地写入明确语义并返回同一个结果字典。

    valid=True 才表示可作训练目标；None、仅编译和仅正确性成功均不能冒充性能奖励。
    failure_kind 为 candidate、environment 或 None；负奖励并不等于无效奖励。
    """
    if valid and value not in {-1, 1, 2, 3}:
        raise ValueError("有效训练奖励只能为 -1、1、2、3")
    result.update(
        reward=value,
        reward_valid=valid,
        scored=valid,
        reward_stage=stage,
        failure_kind=failure_kind,
        reward_reason=reason,
    )
    return result


def training_reward(evaluation, acceptance=None):
    """输入一次正式评测和可选的同候选隐藏验收；返回训练奖励摘要，不修改输入。

    不接受轨迹根摘要、旧版未声明有效性的结果或 compile_only 的占位分。
    隐藏成功保留原性能奖励；隐藏失败才可将同一候选的奖励改为 -1。
    环境失败返回无效奖励，由训练调度器决定有限重试/补采，不在这里伪造分数。
    """
    source = evaluation
    if acceptance is not None:
        candidate = evaluation.get("candidate_hash")
        if not candidate or candidate != acceptance.get("candidate_hash"):
            raise ValueError("性能奖励与隐藏验收必须属于同一冻结候选")
        if not acceptance.get("correct") or acceptance.get("failure_kind") == "environment":
            source = acceptance

    valid = source.get("reward_valid") is True and source.get("scored") is True
    value = source.get("reward")
    if value not in {-1, 1, 2, 3}:
        valid = False
    return {
        "reward": value if valid else None,
        "reward_valid": valid,
        "failure_kind": source.get("failure_kind"),
        "reason": source.get("reward_reason") or (None if valid else "not_a_scored_evaluation"),
        "evaluation_id": source.get("evaluation_id"),
        "candidate_hash": source.get("candidate_hash"),
    }


def environment_failure(worker_result, *, candidate=False):
    """输入容器结果；识别无法归责给候选的启动、依赖和异常退出，返回布尔值。

    正常预算内的候选 TimeoutExpired、数值错误和输出容量超限仍是候选失败。
    candidate=True 仅用于候选执行/测时：SIGILL、SIGABRT、SIGFPE、SIGSEGV
    属于候选进程失败；参考、编译器、Docker 启动失败和 OOM 仍按环境故障停止。
    """
    code = worker_result.get("container_returncode")
    # 候选原生扩展可以直接使进程崩溃，不能因此终止整个训练。
    # 137/SIGKILL、143/SIGTERM 不在此表中，避免把资源争抢或人工终止当作模型错误。
    if (candidate and worker_result.get("error_type") == "WorkerProcessError"
            and code in {132, 134, 136, 139, -4, -6, -8, -11}):
        return False
    missing = (
        worker_result.get("error_type") == "ModuleNotFoundError"
        and worker_result.get("missing_module") in {"torch", "numpy", "ninja", "packaging", "setuptools"}
    )
    return (
        worker_result.get("error_type") == "WorkerProcessError"
        # 候选导入不存在的 torch API 或无法加载自己生成的扩展，不自动归咎于环境。
        or missing
        # 当前没有每作业独立显存配额，无法仅凭 OOM 区分候选过大和外部显存干扰。
        or worker_result.get("error_type") == "OutOfMemoryError"
        or worker_result.get("failure_kind") == "storage_unavailable"
        or (isinstance(code, int) and (code < 0 or code in {125, 126, 127, 137, 143}))
    )

"""P2 单题丢弃与致命错误的分类、证据及隔离记录。

由 rollout/reward/sampler 调用，不单独运行，不启动 GPU。
输入本次运行配置、verl 请求 UID 和原始异常；输出按 UID 索引的失败证据。
仅明确的参考无效和真正零 token 可以丢弃，其余异常默认停止。
"""

import json
from pathlib import Path
import re

from p2.common import now, write_json


class RecoverableSampleError(RuntimeError):
    """输入可丢弃原因及详情路径；调用方不为此轨迹计算或伪造奖励。"""

    def __init__(self, reason, detail_path=None, *, quarantine=False):
        super().__init__(reason)
        self.detail_path = str(detail_path) if detail_path else None
        self.quarantine = quarantine


class NoValidValidationSamples(RuntimeError):
    """当前验证批次全部不可评分；只报告覆盖率，不输出虚假的正确率。"""


def quarantine_path(run, task_id):
    """输入 run/题号；返回本次 run 的隔离记录路径，拒绝任意目录访问。"""
    if not re.fullmatch(r"[A-Za-z0-9-]+", str(task_id)):
        raise ValueError("隔离题号不是合法标识")
    return Path(run) / "scheduler/quarantine" / f"{task_id}.json"


def failure_path(run, uid):
    """输入 run/框架 UID；返回唯一失败索引，不通过历史日志猜测失败归属。"""
    if not re.fullmatch(r"[A-Za-z0-9-]+", str(uid)):
        raise ValueError("采样 UID 不是合法标识")
    return Path(run) / "scheduler/failures" / f"{uid}.json"


def record_failure(config, uid, task_id, error, *, hidden=False):
    """输入配置、UID、题号、异常及验证标记；保存分类和证据并返回记录。

    引用可恢复异常才能允许补采；磁盘写入失败会继续抛错，不能伪装成已记录。
    """
    recoverable = isinstance(error, RecoverableSampleError)
    record = {
        "at": now().isoformat(), "uid": str(uid), "task_id": task_id,
        "partition": "val" if hidden else "train", "recoverable": recoverable,
        "error_type": type(error).__name__, "error": str(error),
        "detail_path": getattr(error, "detail_path", None),
    }
    write_json(failure_path(config["run"], uid), record)
    if recoverable and error.quarantine:
        write_json(quarantine_path(config["run"], task_id), record)
    return record


def check_quarantine(config, task_id):
    """输入配置和题号；已隔离则在生成/评测前丢弃该请求，否则返回 None。"""
    path = quarantine_path(config["run"], task_id)
    if path.is_file():
        previous = json.loads(path.read_text(encoding="utf-8"))
        raise RecoverableSampleError(
            f"参考题已在本次运行隔离：{task_id}; {previous['error']}", path,
        )

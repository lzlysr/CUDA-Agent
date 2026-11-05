"""验算输出的短生命周期传输目录；参考与候选分别挂载自己的通道。

优先使用宿主 /dev/shm，空间不足时回退到本次评测目录。两端容器不能读取
对方目录；日志和判定 JSON 仍留在 evaluations，逐例张量消费后立即删除。
"""

from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import shutil
import stat
import tempfile
import uuid


class OutputFileError(ValueError):
    """发布的输出不是普通文件、是符号链接，或超过单文件容量上限。"""


@contextmanager
def checked_output(path, limit):
    """输入输出路径和容量上限；返回固定文件描述符路径，供 torch.load 安全读取。

    在反序列化前用 O_NOFOLLOW/fstat 校验，拒绝符号链接、非普通文件及超大文件。
    使用 /proc/self/fd 固定已检查的 inode，避免校验后文件路径又被替换。
    """
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except OSError as error:
        raise OutputFileError(f"cannot open published output: {error}") from error
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode):
            raise OutputFileError("published output must be a regular file")
        if info.st_size > limit:
            raise OutputFileError(f"published output exceeds {limit} bytes: {info.st_size}")
        yield f"/proc/self/fd/{descriptor}"
    finally:
        os.close(descriptor)


@contextmanager
def storage_reservation(cfg, evaluation):
    """输入配置/评测目录；协作预留双通道容量，返回选定的临时目录父路径。

    全项目通过同一把文件锁登记预留，避免多个任务同时把同一份空闲空间算作可用。
    这是本项目的并发预算，不是宿主磁盘配额；外部进程耗尽磁盘仍按环境失败处理。
    """
    limit = cfg.get("case_storage_limit_bytes", 8 * 1024**3)
    if type(limit) is not int or limit <= 0:
        raise ValueError("case_storage_limit_bytes 必须为正整数")
    required = 2 * limit
    locks = Path(__file__).resolve().parents[3] / ".runtime/locks/p1-storage"
    locks.mkdir(parents=True, exist_ok=True)
    reservation = locks / (uuid.uuid4().hex + ".json")
    owner = None
    parent = None
    try:
        with (locks / "pool.lock").open("a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            # 创建、加锁和写登记都在池锁内，其他任务不会读到尚未初始化的空记录。
            owner = reservation.open("w+")
            fcntl.flock(owner, fcntl.LOCK_EX)
            pending = {}
            for path in locks.glob("*.json"):
                if path == reservation:
                    continue
                with path.open("r+") as other:
                    try:
                        fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except BlockingIOError:
                        entry = json.load(other)
                        device = entry["device"]
                        pending[device] = pending.get(device, 0) + entry["bytes"]
                    else:
                        # 持有者已退出；其遗留文件占用仍体现在实际空闲空间里。
                        path.unlink(missing_ok=True)
            for option in [Path("/dev/shm"), Path(evaluation)]:
                if not option.is_dir():
                    continue
                device = str(option.stat().st_dev)
                if shutil.disk_usage(option).free - pending.get(device, 0) >= required:
                    parent = option
                    json.dump({"device": device, "bytes": required}, owner)
                    owner.flush()
                    break
            if parent is None:
                raise OSError(28, f"no storage available for two bounded outputs ({required} bytes)")
        yield parent
    finally:
        # 删除登记也在池锁下完成，避免另一个分配者读到半更新记录。
        with (locks / "pool.lock").open("a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            reservation.unlink(missing_ok=True)
        if owner is not None:
            owner.close()


@contextmanager
def stream_directories(cfg, evaluation):
    """输入配置和评测目录；返回参考/候选通道及传输名称，结束自动清理临时目录。"""
    with storage_reservation(cfg, evaluation) as parent:
        with tempfile.TemporaryDirectory(prefix="cuda-agent-p1-", dir=parent) as name:
            root = Path(name)
            expected, actual = root / "reference", root / "candidate"
            expected.mkdir()
            actual.mkdir()
            transport = "shared_memory" if parent == Path("/dev/shm") else "filesystem"
            yield expected, actual, transport

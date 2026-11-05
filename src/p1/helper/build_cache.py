"""可信编译缓存与 CPU 编译配额；不接受 Agent 工作区中的二进制。

由 Evaluator 调用。缓存键包含冻结源码、worker 编译实现和实际依赖安装状态；
缓存元数据放在容器不可写的 build 目录之外，并校验扩展哈希。
"""

from contextlib import contextmanager
import fcntl
import json
from pathlib import Path
import time

from ..data import sha


def build_identity(root, cfg, candidate_hash):
    """输入根目录、配置和候选摘要；返回编译缓存键及用于审计的环境指纹。"""
    root = Path(root)
    environment = {
        "candidate_hash": candidate_hash,
        "worker": sha((root / "src/p1/worker.py").read_bytes()),
        "image": cfg["docker_image"],
        "architecture": "sm_80",
        "cuda": "/usr/local/cuda-12.9",
    }
    # 包元数据及关键编译入口的状态变化会使旧缓存失效，无需运行 GPU 查询。
    paths = list((root / ".runtime/conda/kernel/lib").glob(
        "python*/site-packages/*.dist-info/METADATA"
    ))
    paths += [root / ".runtime/conda/kernel/bin/python",
              Path("/usr/local/cuda-12.9/bin/nvcc"), Path("/usr/bin/c++")]
    environment["installed_files"] = [
        [str(p.resolve()), p.stat().st_size, p.stat().st_mtime_ns]
        for p in sorted(paths) if p.exists()
    ]
    return sha(json.dumps(environment, sort_keys=True)), environment


@contextmanager
def cache_lock(path):
    """输入单个锁文件；跨线程/进程独占该缓存项，退出自动释放。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


@contextmanager
def compilation_slot(root, count, stop=None):
    """输入全局编译并发数与停止事件；获取 CPU 编译槽位，不占用评测 GPU。"""
    if type(count) is not int or count < 1:
        raise ValueError("compile_workers 必须为正整数")
    directory = Path(root) / ".runtime/locks"
    directory.mkdir(parents=True, exist_ok=True)
    while True:
        if stop is not None and stop.is_set():
            raise InterruptedError("停止等待 CPU 编译")
        for index in range(count):
            stream = (directory / f"p1-compile-{index}.lock").open("a+")
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                stream.close()
                continue
            try:
                yield
            finally:
                fcntl.flock(stream, fcntl.LOCK_UN)
                stream.close()
            return
        time.sleep(0.05)

"""验算输出的短生命周期传输目录；参考与候选分别挂载自己的通道。

优先使用宿主 /dev/shm，空间不足时回退到本次评测目录。两端容器不能读取
对方目录；日志和判定 JSON 仍留在 evaluations，逐例张量消费后立即删除。
"""

from contextlib import contextmanager
from pathlib import Path
import shutil
import tempfile


@contextmanager
def stream_directories(cfg, evaluation):
    """输入配置和评测目录；返回参考/候选通道及传输名称，结束自动清理临时目录。"""
    memory = Path("/dev/shm")
    required = 2 * cfg.get("case_storage_limit_bytes", 8 * 1024**3)
    use_memory = memory.is_dir() and shutil.disk_usage(memory).free > required
    parent = memory if use_memory else Path(evaluation)
    with tempfile.TemporaryDirectory(prefix="cuda-agent-p1-", dir=parent) as name:
        root = Path(name)
        expected, actual = root / "reference", root / "candidate"
        expected.mkdir()
        actual.mkdir()
        yield expected, actual, "shared_memory" if use_memory else "filesystem"

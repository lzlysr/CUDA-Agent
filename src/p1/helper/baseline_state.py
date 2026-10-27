"""P1 基线与 pilot 的持久化、运行锁及已完成阶段识别；只使用标准库，不启动 GPU。

调用：由 p1.baseline 和评测控制器导入，不单独执行。
输入：运行目录、固定题目和阶段 JSON；输出：原子写盘、锁和可恢复的阶段结果。
保留阶段文件作为唯一详情，不增加一份逐题 checkpoint.json 副本。
"""

from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import shutil
import tempfile
import uuid


def beijing_now():
    """无输入；返回带 UTC+8 时区的当前时间，供日志、状态和归档命名使用。"""
    return datetime.now(timezone(timedelta(hours=8)))


def atomic_text(path, content):
    """输入目标路径和文本；同目录写临时文件、刷盘后原子替换，返回 None。

    单个文件更新要么保留旧版本，要么呈现完整新版本；三个汇总文件不是跨文件事务。
    临时文件位于目标目录，保证 os.replace 不跨文件系统；异常时清理临时文件。
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="", dir=path.parent,
            prefix=f".{path.name}.", suffix=".tmp", delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def atomic_json(path, value):
    """输入路径和可序列化对象；以 UTF-8 原子写入 JSON，返回 None。"""
    atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


@contextmanager
def exclusive_lock(path):
    """输入 Linux 锁文件路径；持锁期间禁止另一控制器进入，退出或进程死亡后释放。

    锁文件保留供查看持有者；判断占用依赖 flock，不依赖文件是否存在或旧 PID。
    文件不能在释放时删除，否则后来的进程可能对不同 inode 同时持锁。
    """
    import fcntl

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"运行或 GPU 已由另一基线控制器持锁：{path}") from error
        try:
            stream.seek(0)
            stream.truncate()
            json.dump(
                {"pid": os.getpid(), "host": socket.gethostname(),
                 "started_at": beijing_now().isoformat(timespec="seconds")},
                stream, ensure_ascii=False,
            )
            stream.flush()
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def task_fingerprints(tasks):
    """输入固定任务列表；返回各题源码及 split 的指纹，阻止续跑时悄悄更换数据。"""
    return {
        task["task_id"]: hashlib.sha256(
            json.dumps({"code": task["code"], "split": task["split"]},
                       ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()
        for task in tasks
    }


def protocol_fingerprints(source):
    """输入 p1 源码目录；返回评测实现指纹，调度改动不改变计时协议的身份。"""
    return {
        name: hashlib.sha256((Path(source) / name).read_bytes()).hexdigest()
        for name in ("worker.py", "evaluation.py", "contract.py")
    }


def protocol_matches(recorded, current):
    """输入原/当前协议指纹；仅兼容已审查的配置共享和辅助模块路径调整。

    兼容改动仅涉及配置传递和辅助模块导入路径，不修改数值、容差或测时算法。
    旧版多卡 run 可继续使用；worker/contract 指纹仍须完全一致，未知修改仍拒绝。
    """
    compatible = {
        # 历史配置共享版本和本次移动前版本，均只放行到当前精确源码指纹。
        "2afbf8ca1dd5c643fbddadcd21be7f9db89c9731556e63e3162a57874e29ff3b":
        "52ba32e10c4f7b74bf2c918837960a700ee2eb175c12475a20f84697a33d874e",
        "10d655b9bfa9ef16844c4dd4aaaea0ad24e65207244d6104b644cd2c8802081d":
        "52ba32e10c4f7b74bf2c918837960a700ee2eb175c12475a20f84697a33d874e",
    }
    if recorded == current:
        return True
    upgraded = dict(recorded)
    old = recorded.get("evaluation.py")
    upgraded["evaluation.py"] = compatible.get(old, old)
    return upgraded == current


def run_lock_path(output, lock_root):
    """输入 run 和公共锁目录；返回由 run 绝对路径确定的锁路径，不在 run 内新增文件。"""
    identity = hashlib.sha256(str(Path(output).resolve()).encode("utf-8")).hexdigest()
    return Path(lock_root) / f"p1-baseline-run-{identity}.lock"


def guard_legacy_controller(output, project):
    """输入旧 run 和项目路径；识别没有运行锁的旧版控制器，防止同时续跑。

    旧版自动命名进程没有保存 run 参数，无法可靠区分目录；存在这类同项目进程时
    保守拒绝旧目录续跑。此函数只读 /proc，不发送信号、不停止现有实验。
    """
    for process in Path("/proc").glob("[0-9]*"):
        if int(process.name) == os.getpid():
            continue
        try:
            arguments = (process / "cmdline").read_bytes().decode().split("\0")
            if "p1.baseline" not in arguments or (process / "cwd").resolve() != Path(project).resolve():
                continue
            if "--run" in arguments:
                run = arguments[arguments.index("--run") + 1]
                if (Path(project) / run).resolve() != Path(output).resolve():
                    continue
            elif any(value.startswith("--run=") for value in arguments):
                run = next(value.split("=", 1)[1] for value in arguments if value.startswith("--run="))
                if (Path(project) / run).resolve() != Path(output).resolve():
                    continue
            raise RuntimeError(
                f"检测到同项目旧基线进程 PID {process.name}，尚不能安全恢复此目录；"
                "请先让原运行结束或自行停止，当前进程不会停止它。"
            )
        except (OSError, UnicodeDecodeError, IndexError):
            # 进程退出或属于其他用户时可读性会变化，跳过无法读取的条目。
            continue


def guard_active_containers(output):
    """输入待恢复目录；只读检查遗留 Docker 容器，仍挂载此 run 时拒绝恢复，不主动清理。

    SIGKILL 可能让 Docker 容器继续运行，即使宿主文件锁已释放也不能立即重做同题。
    兼容旧容器的随机名称，通过挂载路径识别所属 run；不启动容器或 GPU。
    """
    identifiers = subprocess.check_output(
        ["docker", "ps", "--filter", "name=cuda-agent-p1-", "--format", "{{.ID}}"],
        text=True, timeout=30,
    ).split()
    if not identifiers:
        return
    records = json.loads(subprocess.check_output(
        ["docker", "inspect", *identifiers], text=True, timeout=30,
    ))
    root = Path(output).resolve()
    for record in records:
        if not record.get("State", {}).get("Running"):
            continue
        if any(Path(mount.get("Source", "/")).resolve().is_relative_to(root)
               for mount in record.get("Mounts", [])):
            raise RuntimeError(
                f"原 run 仍有运行中的容器 {record.get('Name', record['Id'])}；"
                "请等待其结束或自行检查清理，当前恢复不会停止它。"
            )


def _record(path):
    """输入阶段路径；返回完整且经控制器收尾的记录，损坏或未收尾时返回 None。"""
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(value, dict):
        return None
    seconds = value.get("controller_seconds")
    if (isinstance(seconds, bool) or not isinstance(seconds, (int, float))
            or not math.isfinite(seconds) or seconds < 0):
        return None
    # worker 的 JSON 先写出，控制器之后才补 controller_seconds；缺失时不能当作完成。
    return value if value.get("status") in {"ok", "error", "timeout"} else None


def completed_reference(output, task):
    """输入 run 和题目；根据正式阶段详情重建结果，缺文件/索引或未完成则返回 None。

    成功和明确拒绝均为终态；不因存在 input-spec 就跳过原预检。根汇总损坏不影响
    此处识别，恢复时会重新生成三种汇总。只接受本题标准相对路径，避免跨题串用。
    """
    output = Path(output)
    relative = Path("references") / task["task_id"]
    ref = output / relative
    try:
        index = json.loads((ref / "preflight.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    prepare_path = (relative / "cases/prepare_result.json").as_posix()
    bench_path = (relative / "initial-bench/bench_result.json").as_posix()
    if not isinstance(index, dict) or index.get("prepare_result_path") != prepare_path:
        return None
    prepared = _record(output / prepare_path)
    if prepared is None:
        return None
    rejected = prepared["status"] != "ok" or prepared.get("near_constant_sampled_output")
    detail = prepared
    if not rejected:
        if index.get("bench_result_path") != bench_path:
            return None
        detail = _record(output / bench_path)
        if detail is None:
            return None
        rejected = detail["status"] != "ok"
    if index.get("status") != ("task_invalid" if rejected else "ok"):
        return None
    if rejected:
        return {"status": "reference_rejected",
                "result_path": (relative / "preflight.json").as_posix(),
                "evaluation_gpu": detail.get("evaluation_gpu")}
    # 完整性检查只要求原有数值字段存在，不修改既有有效性判定或容差。
    if not all(isinstance(detail.get(mode), dict)
               and all(isinstance(detail[mode].get(key), (int, float)) for key in ("cuda_ms", "wall_ms"))
               for mode in ("eager", "compile")):
        return None
    return dict(detail, result_path=bench_path)


def archive_incomplete(output, task_id):
    """输入 run 和待重做题号；将残缺参考目录移至 interrupted，返回归档相对路径或 None。"""
    output = Path(output)
    ref = output / "references" / task_id
    if not ref.exists():
        return None
    name = beijing_now().strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:8]
    target = output / "interrupted" / name / task_id
    target.parent.mkdir(parents=True, exist_ok=True)
    ref.rename(target)
    return target.relative_to(output).as_posix()


def cleanup_reference_artifacts(output, task_id):
    """输入 run 和已完成题号；仅删除本题输入/参数快照及编译缓存，返回删除的相对路径。

    调用方必须先持久保存本题最终结果，且确保本题没有正在运行的评测。
    保留源码、预检索引、阶段 JSON 和日志，续跑仍可识别完成状态；重复调用无副作用。
    所有目标先验证再删除；拒绝路径逃逸和目录符号链接，避免误删其他题或 run 外内容。
    """
    if not isinstance(task_id, str) or task_id in {"", ".", ".."} or Path(task_id).name != task_id:
        raise ValueError("清理题号必须是单个目录名")
    root = Path(output).resolve(strict=True)
    ref = root / "references" / task_id
    targets = [ref / "cases/input-spec", ref / "compile-cache"]
    for target in targets:
        # 包括父目录在内逐级拒绝符号链接，不跟随链接进入其他题或外部目录。
        components = [root / "references", ref]
        if target.name == "input-spec":
            components.append(ref / "cases")
        components.append(target)
        if any(path.is_symlink() for path in components):
            raise ValueError(f"清理目标或父目录不允许是符号链接：{target}")
        resolved = target.resolve()
        if not resolved.is_relative_to(ref) or resolved == ref:
            raise ValueError(f"清理目标越出本题目录：{target}")
        if target.exists() and not target.is_dir():
            raise ValueError(f"清理目标不是目录：{target}")
    deleted = []
    for target in targets:
        if target.exists():
            shutil.rmtree(target)
            deleted.append(target.relative_to(root).as_posix())
    return deleted

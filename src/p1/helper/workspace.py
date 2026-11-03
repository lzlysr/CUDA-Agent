"""统一 Agent 的相对路径、/workspace 路径与宿主工作区路径。

由 p1.run、p1.tools 和 p1.evaluation 调用，不独立运行。
输入：工作区、模型给出的路径及写入范围；输出：经过边界检查的宿主 Path。
辅助脚本和 notebook 仅供开发使用，不作为正式候选源码参与计分。
"""

from pathlib import Path, PurePosixPath


def workspace_path(root, name, *, writing=False, auxiliary=False):
    """把三种工作区路径映射到同一文件；拒绝越界和对固定设施的写入。"""
    root = Path(root).resolve()
    if not isinstance(name, str) or not name or "\x00" in name:
        raise ValueError("invalid workspace path")
    # 容器中的 /workspace 与本轨迹宿主目录是一一对应的，不是宿主绝对路径。
    sandbox = PurePosixPath(name)
    if sandbox.is_relative_to("/workspace"):
        destination = root / str(sandbox.relative_to("/workspace"))
    else:
        destination = root / name
    destination = destination.resolve()
    if not destination.is_relative_to(root):
        raise ValueError("path escapes workspace")
    relative = destination.relative_to(root)
    if writing:
        candidate = relative.as_posix() == "model_new.py" or (
            len(relative.parts) == 2
            and relative.parts[0] == "kernels"
            and relative.suffix in {".cu", ".cpp", ".h", ".cuh"}
        )
        # 开发脚本可单独运行；不放宽 single 正式交付或固定 model/utils 的保护。
        development = (
            auxiliary and bool(relative.parts)
            and relative.parts[0] not in {"utils", "kernels"}
            and relative.name not in {"model.py", "SKILL.md"}
            and relative.suffix in {".py", ".ipynb"}
        )
        if not candidate and not development:
            raise ValueError("write only candidate sources or auxiliary Python/notebook files")
    return destination


def workspace_glob(root, pattern):
    """输入相对或 /workspace glob；返回规范化的相对模式，拒绝 .. 穿越。"""
    if ".." in PurePosixPath(pattern).parts:
        raise ValueError("glob must stay inside workspace")
    return workspace_path(root, pattern).relative_to(Path(root).resolve()).as_posix()

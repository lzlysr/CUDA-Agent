"""CUDA 开发工具：对齐论文中的 Bash、Read/Write、Edit/MultiEdit、Glob/Grep、TodoWrite。

输入：模型生成的工具名和 JSON 参数；输出：受限工作区内的操作反馈。
文件修改采用先读后写；evaluate/profile_reference 是本地隔离评测扩展。
没有 NotebookEdit：当前任务不含 notebook，不冒充完整 OpenHands 实现。
CPU 回归：PYTHONPATH=src python -m unittest discover -s tests/p1 -v。"""

import json
from pathlib import Path
import re


def schema(name, description, properties, required=()):
    """输入工具名、说明和参数字段；返回语言模型 function-call schema。"""
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": properties, "required": list(required)},
        },
    }


STRING = {"type": "string"}
EDIT = {
    "type": "object",
    "properties": {"old_string": STRING, "new_string": STRING},
    "required": ["old_string", "new_string"],
}
TOOLS = [
    schema(
        "Bash",
        "Run a command in the persistent CPU shell. cwd, environment, /tmp and functions survive calls. Use bash utils/compile.sh; python3 -m utils.verification; python3 -m utils.profiling --reference-only.",
        {"command": STRING},
        ["command"],
    ),
    schema(
        "Read",
        "Read a workspace UTF-8 file before changing an existing file.",
        {"path": STRING},
        ["path"],
    ),
    schema(
        "Write",
        "Create/replace model_new.py or kernels/* source. Read existing files before overwriting.",
        {"path": STRING, "content": STRING},
        ["path", "content"],
    ),
    schema(
        "Edit",
        "Replace one exact string in a file already read.",
        {"path": STRING, "old_string": STRING, "new_string": STRING},
        ["path", "old_string", "new_string"],
    ),
    schema(
        "MultiEdit",
        "Apply exact replacements atomically to one previously read file.",
        {"path": STRING, "edits": {"type": "array", "items": EDIT}},
        ["path", "edits"],
    ),
    schema("Glob", "Find workspace files by a relative glob.", {"pattern": STRING}, ["pattern"]),
    schema(
        "Grep",
        "Search workspace UTF-8 files using a regex; return paths, line numbers and matching text.",
        {"pattern": STRING, "glob": STRING},
        ["pattern"],
    ),
    schema(
        "TodoWrite",
        "Record the current development plan.",
        {
            "todos": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {"content": STRING, "status": STRING},
                    "required": ["content", "status"],
                },
            }
        },
        ["todos"],
    ),
    schema(
        "profile_reference",
        "Measure/profile original eager and torch.compile before writing a candidate; no candidate is required.",
        {},
    ),
    schema(
        "evaluate",
        "Compile, check five feedback inputs, and if correct measure the current immutable CUDA candidate.",
        {},
    ),
    schema(
        "submit",
        "Stop development; the controller selects the fastest correct trajectory candidate for hidden validation.",
        {},
    ),
]


class WorkspaceTools:
    """只持有单条轨迹的工作区访问状态；不授予宿主其他路径或 GPU 权限。"""

    def __init__(self, workspace, shell, action):
        """输入工作区、Bash 回调和固定评测回调；初始化已读文件集合，无返回值。"""
        self.root = Path(workspace).resolve()
        self.shell = shell
        self.action = action
        self.read_paths = set()

    def path(self, name, writing=False):
        """输入相对路径及写标记；返回受限绝对路径，逃逸或修改固定设施则抛错。"""
        destination = (self.root / name).resolve()
        if not destination.is_relative_to(self.root):
            raise ValueError("path escapes workspace")
        relative = destination.relative_to(self.root)
        if writing and not (
            str(relative) == "model_new.py"
            or (
                len(relative.parts) == 2
                and relative.parts[0] == "kernels"
                and destination.suffix in {".cu", ".cpp", ".h", ".cuh"}
            )
        ):
            raise ValueError("write only model_new.py and kernels/* source files")
        return destination

    def write(self, name, content):
        """输入路径及完整内容；先读后写检查通过才写文件，返回写入路径。"""
        path = self.path(name, True)
        if path.exists() and path not in self.read_paths:
            raise ValueError("Read existing file before writing: " + name)
        if not isinstance(content, str) or len(content.encode()) > 1000000:
            raise ValueError("invalid source file or file too large")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        self.read_paths.add(path)
        return {"written": [name]}

    def dispatch(self, name, arguments):
        """输入工具名及参数字典；返回 JSON 可序列化反馈，未知工具或参数会抛错。"""
        aliases = {"shell": "Bash", "read_file": "Read", "list_files": "Glob"}
        original = name
        name = aliases.get(name, name)
        if name == "Bash":
            return self.shell(arguments["command"])
        if name == "Read":
            path = self.path(arguments["path"])
            text = path.read_text(encoding="utf-8")
            self.read_paths.add(path)
            return {"content": text}
        if name == "Write":
            return self.write(arguments["path"], arguments["content"])
        if original == "write_files":
            return {
                "written": [
                    self.write(path, content)["written"][0]
                    for path, content in arguments["files"].items()
                ]
            }
        if name in {"Edit", "MultiEdit"}:
            path = self.path(arguments["path"], True)
            if path not in self.read_paths:
                raise ValueError("Read existing file before editing")
            text = path.read_text(encoding="utf-8")
            edits = [arguments] if name == "Edit" else arguments["edits"]
            for edit in edits:
                old = edit["old_string"]
                if not old or text.count(old) != 1:
                    raise ValueError("replacement must match exactly once")
                text = text.replace(old, edit["new_string"], 1)
            return self.write(arguments["path"], text)
        if name in {"Glob", "Grep"}:
            pattern = (
                arguments.get("glob", "**/*") if name == "Grep" else arguments.get("pattern", "**/*")
            )
            if Path(pattern).is_absolute() or ".." in Path(pattern).parts:
                raise ValueError("glob must stay inside workspace")
            files = [
                path for path in self.root.glob(pattern) if path.is_file() and not path.is_symlink()
            ]
            if name == "Glob":
                return {"paths": [str(path.relative_to(self.root)) for path in sorted(files)][:500]}
            expression = re.compile(arguments["pattern"])
            matches = []
            for path in sorted(files):
                if path.stat().st_size > 1000000:
                    continue
                for line, text in enumerate(
                    path.read_text(encoding="utf-8", errors="replace").splitlines(), 1
                ):
                    if expression.search(text):
                        matches.append(
                            {"path": str(path.relative_to(self.root)), "line": line, "text": text}
                        )
                    if len(matches) >= 200:
                        return {"matches": matches, "truncated": True}
            return {"matches": matches, "truncated": False}
        if name == "TodoWrite":
            (self.root / ".agent-todo.json").write_text(
                json.dumps(arguments["todos"], ensure_ascii=False, indent=2), encoding="utf-8"
            )
            return {"todos": arguments["todos"]}
        if name in {"evaluate", "profile_reference"}:
            return self.action(name)
        raise ValueError("unknown tool: " + name)

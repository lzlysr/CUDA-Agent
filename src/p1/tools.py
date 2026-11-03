"""CUDA 开发工具：持久/后台 Bash、文件编辑、ripgrep 检索和 notebook 编辑。

输入：模型生成的工具名和 JSON 参数；输出：受限工作区内的操作反馈。
文件修改采用先读后写；evaluate/profile_reference 是本地隔离评测扩展。
由 p1.run 调用，不独立运行；辅助脚本/notebook 不进入正式候选计分。"""

import json
from pathlib import Path
import subprocess
import uuid

from .data import ROOT
from .helper.workspace import workspace_glob, workspace_path


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
    "properties": {"old_string": STRING, "new_string": STRING, "replace_all": {"type": "boolean"}},
    "required": ["old_string", "new_string"],
}
TOOLS = [
    schema(
        "Bash",
        "Run a command in the persistent CPU shell. cwd, variables and functions survive calls. run_in_background=true returns a bash_id for BashOutput/KillBash. For GPU tuning use python3 -m utils.run tune_kernel.py; normal CPU Bash has no GPU.",
        {"command": STRING, "run_in_background": {"type": "boolean"}, "timeout": {"type": "integer"}},
        ["command"],
    ),
    schema("BashOutput", "Read only new output from a background job.", {"bash_id": STRING}, ["bash_id"]),
    schema("KillBash", "Terminate a background job and its process group.", {"bash_id": STRING}, ["bash_id"]),
    schema(
        "Read",
        "Read a workspace UTF-8 file before changing an existing file.",
        {"path": STRING, "offset": {"type": "integer"}, "limit": {"type": "integer"}},
        ["path"],
    ),
    schema(
        "Write",
        "Create/replace candidate sources or auxiliary .py/.ipynb development files. Read before overwriting. Fixed infrastructure stays read-only.",
        {"path": STRING, "content": STRING},
        ["path", "content"],
    ),
    schema(
        "Edit",
        "Replace one exact string in a file already read.",
        {"path": STRING, "old_string": STRING, "new_string": STRING, "replace_all": {"type": "boolean"}},
        ["path", "old_string", "new_string"],
    ),
    schema(
        "MultiEdit",
        "Apply exact replacements atomically to one previously read file.",
        {"path": STRING, "edits": {"type": "array", "items": EDIT}},
        ["path", "edits"],
    ),
    schema("Glob", "Find files using a relative or /workspace glob.", {"pattern": STRING}, ["pattern"]),
    schema(
        "Grep",
        "Search with ripgrep regex, glob/file-type filters and contextual lines. Paths may be relative or under /workspace.",
        {
            "pattern": STRING, "glob": STRING, "path": STRING, "type": STRING,
            "context": {"type": "integer"}, "ignore_case": {"type": "boolean"},
            "multiline": {"type": "boolean"},
            "output_mode": {"type": "string", "enum": ["content", "files_with_matches", "count"]},
        },
        ["pattern"],
    ),
    schema(
        "NotebookEdit", "Replace, insert or delete a notebook cell. Read an existing notebook before editing. Cells are not executed.",
        {
            "path": STRING, "cell_index": {"type": "integer"}, "new_source": STRING,
            "edit_mode": {"type": "string", "enum": ["replace", "insert", "delete"]},
            "cell_type": {"type": "string", "enum": ["code", "markdown"]},
        },
        ["path", "cell_index", "edit_mode"],
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

    def __init__(self, workspace, shell, action, *, background=None, bash_output=None, kill_bash=None):
        """输入工作区、Bash 回调和固定评测回调；初始化已读文件集合，无返回值。"""
        self.root = Path(workspace).resolve()
        self.shell = shell
        self.action = action
        self.read_paths = set()
        self.background = background
        self.bash_output = bash_output
        self.kill_bash = kill_bash

    def path(self, name, writing=False):
        """输入相对路径及写标记；返回受限绝对路径，逃逸或修改固定设施则抛错。"""
        return workspace_path(self.root, name, writing=writing, auxiliary=True)

    def grep(self, arguments):
        """输入搜索参数；调用真正的 rg，返回结构化匹配/上下文，不经过宿主 shell。"""
        # kernel Conda 环境同时提供给宿主工具和 Docker，不依赖当前激活的是哪个环境。
        executable = ROOT / ".runtime/conda/kernel/bin/rg"
        if not executable.is_file():
            raise FileNotFoundError(
                "kernel Conda 环境中缺少 ripgrep；请按 reports/p0/requirements.txt "
                "说明使用 kernel Python 的 pip 安装 ripgrep==14.1.0"
            )
        target = self.path(arguments.get("path", "."))
        context = arguments.get("context", 0)
        if type(context) is not int or not 0 <= context <= 50:
            raise ValueError("context must be an integer between 0 and 50")
        mode = arguments.get("output_mode", "content")
        if mode not in {"content", "files_with_matches", "count"}:
            raise ValueError("invalid output_mode")
        command = [str(executable), "--json", "--no-config", "--color=never", "--max-filesize=1M"]
        if arguments.get("glob"):
            command += ["--glob", workspace_glob(self.root, arguments["glob"])]
        if arguments.get("type"):
            command += ["--type", arguments["type"]]
        if arguments.get("ignore_case"):
            command.append("--ignore-case")
        if arguments.get("multiline"):
            command.append("--multiline")
        command += ["--context", str(context), "--regexp", arguments["pattern"], "--", str(target)]
        result = subprocess.run(command, cwd=self.root, capture_output=True, text=True, timeout=30)
        if result.returncode not in {0, 1}:
            raise ValueError("ripgrep: " + result.stderr[-3000:])
        matches, counts, surrounding = [], {}, []
        for line in result.stdout.splitlines():
            event = json.loads(line)
            if event["type"] not in {"match", "context"}:
                continue
            data = event["data"]
            if "text" not in data["path"]:
                continue
            path = self.path(data["path"]["text"])
            row = {"path": path.relative_to(self.root).as_posix(), "line": data["line_number"],
                   "text": data["lines"].get("text", "").rstrip("\n")}
            if event["type"] == "match":
                counts[row["path"]] = counts.get(row["path"], 0) + 1
                matches.append(row)
            else:
                surrounding.append(row)
        if mode == "files_with_matches":
            return {"paths": sorted(counts)[:500], "truncated": len(counts) > 500}
        if mode == "count":
            return {"counts": counts}
        return {"matches": matches[:200], "context": surrounding[:400],
                "truncated": len(matches) > 200 or len(surrounding) > 400}

    def notebook_edit(self, arguments):
        """输入 notebook 路径、单元序号和编辑内容；原子修改单元，不运行代码。"""
        name = arguments["path"]
        path = self.path(name, True)
        if path.suffix != ".ipynb":
            raise ValueError("NotebookEdit requires an .ipynb file")
        if path.exists():
            if path not in self.read_paths:
                raise ValueError("Read existing notebook before editing")
            value = json.loads(path.read_text(encoding="utf-8"))
        else:
            value = {"cells": [], "metadata": {}, "nbformat": 4, "nbformat_minor": 5}
        if value.get("nbformat") != 4 or not isinstance(value.get("cells"), list):
            raise ValueError("only nbformat 4 notebooks are supported")
        cells, index, mode = value["cells"], arguments["cell_index"], arguments["edit_mode"]
        if type(index) is not int or index < 0 or index > len(cells):
            raise ValueError("cell_index out of range")
        if mode in {"replace", "delete"} and index == len(cells):
            raise ValueError("cell_index out of range")
        if mode == "delete":
            del cells[index]
        elif mode in {"replace", "insert"}:
            kind = arguments.get("cell_type", cells[index]["cell_type"] if mode == "replace" else "code")
            if kind not in {"code", "markdown"}:
                raise ValueError("cell_type must be code or markdown")
            source = arguments["new_source"]
            if not isinstance(source, str):
                raise ValueError("new_source must be text")
            # 编辑后清掉旧执行输出，避免展示与新源码无关的结果。
            cell = dict(cells[index]) if mode == "replace" else {"metadata": {}}
            cell.update(cell_type=kind, source=source.splitlines(keepends=True))
            if kind == "code":
                cell.update(execution_count=None, outputs=[])
                cell.pop("attachments", None)
            else:
                cell.pop("execution_count", None)
                cell.pop("outputs", None)
            if mode == "insert":
                cell["id"] = uuid.uuid4().hex[:8]
                cells.insert(index, cell)
            else:
                if "id" in cells[index]:
                    cell["id"] = cells[index]["id"]
                cells[index] = cell
        else:
            raise ValueError("invalid edit_mode")
        return self.write(name, json.dumps(value, ensure_ascii=False, indent=2) + "\n")

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
            if arguments.get("run_in_background"):
                if self.background is None:
                    raise RuntimeError("background Bash is unavailable")
                return self.background(arguments["command"], arguments.get("timeout"))
            if arguments.get("timeout") is not None:
                return self.shell(arguments["command"], timeout=arguments["timeout"])
            return self.shell(arguments["command"])
        if name in {"BashOutput", "KillBash"}:
            callback = self.bash_output if name == "BashOutput" else self.kill_bash
            if callback is None:
                raise RuntimeError(name + " is unavailable")
            return callback(arguments["bash_id"])
        if name == "Read":
            path = self.path(arguments["path"])
            text = path.read_text(encoding="utf-8")
            if "offset" in arguments or "limit" in arguments:
                offset, limit = arguments.get("offset", 1), arguments.get("limit", 2000)
                if type(offset) is not int or type(limit) is not int or offset < 1 or limit < 1:
                    raise ValueError("offset/limit must be positive integers")
                text = "".join(text.splitlines(keepends=True)[offset - 1:offset - 1 + limit])
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
                replace_all = edit.get("replace_all", False)
                if not old or not text.count(old) or (not replace_all and text.count(old) != 1):
                    raise ValueError("replacement must match exactly once")
                text = text.replace(old, edit["new_string"], -1 if replace_all else 1)
            return self.write(arguments["path"], text)
        if name == "Glob":
            pattern = workspace_glob(self.root, arguments.get("pattern", "**/*"))
            files = [
                path for path in self.root.glob(pattern) if path.is_file() and not path.is_symlink()
            ]
            return {"paths": [str(path.relative_to(self.root)) for path in sorted(files)][:500]}
        if name == "Grep":
            return self.grep(arguments)
        if name == "NotebookEdit":
            return self.notebook_edit(arguments)
        if name == "TodoWrite":
            path = self.path(".agent-todo.json")
            if (self.root / ".agent-todo.json").is_symlink():
                raise ValueError("todo file must not be a symlink")
            path.write_text(
                json.dumps(arguments["todos"], ensure_ascii=False, indent=2), encoding="utf-8"
            )
            return {"todos": arguments["todos"]}
        if name in {"evaluate", "profile_reference"}:
            return self.action(name)
        raise ValueError("unknown tool: " + name)

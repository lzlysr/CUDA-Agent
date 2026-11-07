"""解析模型的文件交付和工具调用；容错文件结束标签，不执行 thinking 中的文本。

由 p1.run 和 p2.reward 调用，不独立运行。single 按文件开始标签划分原始源码，
容忍独占行的结束标签缺失或多余；Agent 的原生 tool_call 仍严格解析并保持顺序。
"""

import json
from pathlib import PurePosixPath
import re


# P1 single 和 P2 共用一份交付说明，避免提示词与解析器各自采用不同格式。
SINGLE_FILE_DELIVERY = '''Generate one complete solution with no execution feedback.
After your reasoning, return only complete raw file blocks in this exact format:
<file path="model_new.py">
COMPLETE_PYTHON_SOURCE
</file>
<file path="kernels/kernel.cu">
COMPLETE_CUDA_SOURCE
</file>
<file path="kernels/kernel_binding.cpp">
COMPLETE_BINDING_SOURCE
</file>
Replace the illustrative source placeholders with complete actual source code.
Kernel file names above are examples; include only the source files your implementation needs.
Each opening tag and closing </file> tag must occupy its own line, without indentation.
Write source code directly: preserve normal newlines, quotes, backslashes, <, > and &.
Do not JSON-escape or XML-escape source code. Do not add Markdown code fences.
Do not put explanatory text outside the file blocks. Never repeat a file path or nest blocks.
Use canonical relative paths, without /workspace prefixes, backslashes, . or .. path components.
All file contents must be complete. model_new.py must define class ModelNew and import
cuda_extension directly. Only model_new.py and source files directly under kernels/ are
writable; omit binding.cpp, binding_registry.h, model.py, utils/, setup.py and other files.
Deliver only the allowed candidate files; do not include auxiliary scripts/notebooks.
Missing file opening tags, invalid paths or incomplete source fail this single attempt;
there is no automatic repair or extra generation.
'''


# 在原题末尾重申文件边界和停止位置；P1 single 与 P2 使用相同的简短提醒。
SINGLE_FILE_DELIVERY_REMINDER = '''
Final delivery rules:
- Return only independent file blocks for the final implementation, with no outer wrapper.
- Each <file path="..."> must have exactly one matching </file>.
- End your response immediately after the last file's </file>; do not add another </file>.
'''


def answer_body(text):
    """输入原始生成文本；返回 thinking 之后的正式回答，未闭合 thinking 返回空串。"""
    body = text.rsplit("</think>", 1)[-1]
    return "" if "<think>" in body else body.strip()


def json_answer(body):
    """输入 Agent 的 JSON 调用正文；只去除完整围栏，严格解析且不补括号。"""
    if body.startswith("```"):
        match = re.fullmatch(r"```(?:json)?\s*\n?(.*?)\n?```", body, re.S)
        if not match:
            raise ValueError("incomplete JSON fence")
        body = match.group(1)
    return json.loads(body)


def parse_single(text):
    """输入单次生成文本，返回路径→原始源码字典；仅容错结束标签的数量。

    遇到独占一行的结束标签、下一个开始标签或回答结束时结束当前文件。
    多余的独占行结束标签作为分隔符跳过；开始标签、路径及源码仍须合法。
    不猜测缺失的文件名，不剥除 Markdown 围栏，不补源码，不兼容旧 files JSON。
    路径写入范围与文件大小继续由 write_files 在整批落盘前统一检查。
    """
    remaining = answer_body(text)
    files = {}
    while remaining:
        # 容忍文件前后及文件之间多余的结束标签，绝不把它们写入源码。
        separator = re.match(r"</file>(?:\r?\n|$)", remaining)
        if separator is not None:
            remaining = remaining[separator.end():].lstrip()
            continue

        # 文件名只能来自规范的开始标签，不从代码或说明文字中猜测。
        opening = re.match(r'<file path="([^"\r\n<>]+)">\r?\n', remaining)
        if opening is None:
            raise ValueError('single output must contain only <file path="..."> blocks')
        path = opening.group(1)
        relative = PurePosixPath(path)
        if (
            relative.is_absolute()
            or relative.as_posix() != path
            or ".." in relative.parts
            or "\\" in path
        ):
            raise ValueError(f"file block requires a canonical relative path: {path}")
        if path in files:
            raise ValueError(f"duplicate file path: {path}")

        contents = remaining[opening.end():]
        # 只认独占行的标签，普通字符串中的 "</file>" 不构成边界。
        # 下一个开始标签也构成边界，因此缺少结束标签不会吞并下一份文件。
        # 包含畸形开始标签时也先停下，留给下一轮严格检查，避免当作源码放过。
        boundary = re.search(r"(?m)^</file>(?:\r?\n|$)|^<file(?:[ \t>])", contents)
        source = contents[:boundary.start()] if boundary is not None else contents
        if not source.strip():
            raise ValueError(f"empty source file: {path}")
        if source.lstrip().startswith("```"):
            raise ValueError(f"Markdown code fences are not allowed inside {path}")
        files[path] = source

        # 回到边界本身：结束标签由下一轮跳过，开始标签由下一轮解析。
        # 无边界时直接以回答结束收尾；源码完整性仍由编译和正确性评测判断。
        remaining = contents[boundary.start():] if boundary is not None else ""

    if "model_new.py" not in files:
        raise ValueError("single output must include model_new.py")
    return files


def parse_calls(text):
    """输入 Agent 输出；返回有序 (name, arguments) 列表，不静默截掉后续调用。"""
    body = answer_body(text)
    blocks = re.findall(r"<tool_call>\s*(.*?)\s*</tool_call>", body, re.S)
    if blocks:
        # 部分闭合也属于格式错误，不执行一个没有完整声明的调用批次。
        if body.count("<tool_call>") != len(blocks) or body.count("</tool_call>") != len(blocks):
            raise ValueError("incomplete tool_call block")
        actions = []
        for index, raw in enumerate(blocks):
            native = re.fullmatch(r"<function=([^>]+)>\s*(.*?)\s*</function>", raw.strip(), re.S)
            if native:
                arguments = {}
                params = native.group(2)
                matches = list(re.finditer(r"<parameter=([^>]+)>\s*(.*?)\s*</parameter>", params, re.S))
                remainder = re.sub(r"<parameter=([^>]+)>\s*(.*?)\s*</parameter>", "", params, flags=re.S)
                if remainder.strip():
                    raise ValueError(f"tool_call {index}: incomplete parameter")
                for match in matches:
                    key, value = match.groups()
                    if key in arguments:
                        raise ValueError(f"tool_call {index}: duplicate parameter {key}")
                    try:
                        arguments[key] = json.loads(value)
                    except json.JSONDecodeError:
                        arguments[key] = value
                name = native.group(1)
            else:
                value = json.loads(raw)
                name, arguments = value["name"], value.get("arguments", {})
                if isinstance(arguments, str):
                    arguments = json.loads(arguments)
            if not isinstance(name, str) or not isinstance(arguments, dict):
                raise ValueError(f"tool_call {index}: invalid name or arguments")
            actions.append((name, arguments))
        return actions
    # 保留已有 JSON 调用兼容入口，仍使用严格 JSON 解析。
    if not body:
        return []
    value = json_answer(body)
    if isinstance(value, dict) and "files" in value:
        return [("write_files", {"files": value["files"]})]
    if isinstance(value, dict) and "name" in value:
        arguments = value.get("arguments", {})
        if isinstance(arguments, str):
            arguments = json.loads(arguments)
        if isinstance(value["name"], str) and isinstance(arguments, dict):
            return [(value["name"], arguments)]
    raise ValueError("no complete tool call found")

"""严格解析模型的正式输出，不修补 JSON、不执行 thinking 中的文本。

由 p1.run 调用，不独立运行。single 只接受完整 files JSON；Agent 接受
Qwen 原生的多个 tool_call，并保持输出顺序。任何格式错误均显式返回给控制器。
"""

import json
import re


def answer_body(text):
    """输入原始生成文本；返回 thinking 之后的正式回答，未闭合 thinking 返回空串。"""
    body = text.rsplit("</think>", 1)[-1]
    return "" if "<think>" in body else body.strip()


def json_answer(body):
    """输入正式回答；只去除完整的 Markdown JSON 围栏，严格解析且不补括号。"""
    if body.startswith("```"):
        match = re.fullmatch(r"```(?:json)?\s*\n?(.*?)\n?```", body, re.S)
        if not match:
            raise ValueError("incomplete JSON fence")
        body = match.group(1)
    return json.loads(body)


def parse_single(text):
    """输入单次生成；返回 files 字典，格式不完整或不是 files 对象则抛错。"""
    value = json_answer(answer_body(text))
    if not isinstance(value, dict) or not isinstance(value.get("files"), dict):
        raise ValueError("single output must be a complete JSON object containing files")
    return value["files"]


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

"""保存一轮模型输出、工具调用和工具返回，并兼容旧日志的状态计数。

调用：由 p1.run 写入 turn-000.json，由 p1.status 读取；不独立运行。
输入：模型原文、生成统计和工具反馈；输出：逐轮 JSON，不执行模型或工具。
每轮文件在生成、调用工具、收尾时原子更新，运行中可通过 status 判断是否完整。
"""

import json
from pathlib import Path

from .baseline_state import atomic_json, beijing_now


class TurnLog:
    """持有当前一轮记录；原始模型文本与结构化工具信息分别保存。"""

    def __init__(self, directory, turn, mode):
        """输入轨迹目录、零起算轮号和模式；创建尚未完成的日志，无返回值。"""
        self.path = Path(directory) / f"turn-{turn:03d}.json"
        self.record = {
            "turn": turn,
            "mode": mode,
            "status": "generating",
            "started_at": beijing_now().isoformat(timespec="seconds"),
            "finished_at": None,
            "content": None,
            "generation": None,
            "tool_call": None,
            "tool_response": None,
        }
        self.save()

    def save(self):
        """无输入；原子保存当前记录，返回 None；磁盘错误向调用方传播。"""
        atomic_json(self.path, self.record)

    def generation(self, result):
        """输入生成结果；分开保存完整文本与 token/耗时统计，返回 None。"""
        self.record["content"] = result.get("text", "")
        self.record["generation"] = {key: value for key, value in result.items() if key != "text"}
        self.record["status"] = "generated"
        self.save()

    def call(self, ident, name, arguments):
        """输入调用 ID、名称、参数；在执行前保存调用，返回 None。"""
        self.record["tool_call"] = {"id": ident, "name": name, "arguments": arguments}
        self.record["status"] = "tool_running"
        self.save()

    def response(self, feedback, returned_text, tokens, truncated):
        """输入完整反馈、实际回传文本、token 数及截断标记；保存工具结果，返回 None。"""
        self.record["tool_response"] = feedback
        self.record["tool_response_meta"] = {"feedback_tokens": tokens, "truncated": truncated}
        # 默认完整反馈只存一份；发生截断时，另存模型实际看到的前缀，便于复核。
        if truncated:
            self.record["tool_response_meta"]["returned_text"] = returned_text
        self.record["status"] = "completed"

    def finish(self):
        """无输入；标记本轮收尾时间并保存，返回 None；不把失败改写为成功。"""
        self.record["finished_at"] = beijing_now().isoformat(timespec="seconds")
        self.save()


def turn_counts(directory):
    """输入轨迹目录；返回已完成生成/工具返回数，兼容旧格式且按轮号去重。"""
    directory = Path(directory)
    generations = {
        path.stem.removeprefix("generation-") for path in directory.glob("generation-*.json")
    }
    tools = {path.stem.removeprefix("tool-") for path in directory.glob("tool-*.json")}
    for path in directory.glob("turn-*.json"):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            # 原子更新期间文件可能被恢复/归档；状态查询不因单条记录不可读而失败。
            continue
        ident = path.stem.removeprefix("turn-")
        generation = record.get("generation") or {}
        # 上下文耗尽时没有真正调用模型，不算一次已完成生成。
        if "seconds" in generation:
            generations.add(ident)
        if record.get("tool_call") and record.get("tool_response") is not None:
            tools.add(ident)
    return {"completed_generations": len(generations), "completed_tools": len(tools)}

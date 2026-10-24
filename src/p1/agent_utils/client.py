"""评测请求客户端。由 utils.compile/verification/profiling 调用，不单独运行。

输入：固定动作名；输出：JSON 反馈并返回退出码。仅在当前轨迹 /rpc 下通信。"""

import json
import os
from pathlib import Path
import time
import uuid


def request(action):
    """输入允许的动作；写入原子请求并等待宿主回复，打印 JSON，返回 0/1。"""
    directory = Path(os.environ["P1_TOOL_RPC"])
    ident = uuid.uuid4().hex
    target = directory / f"request-{ident}.json"
    temporary = target.with_suffix(".tmp")
    temporary.write_text(json.dumps({"action": action}), encoding="utf-8")
    temporary.replace(target)
    reply = directory / f"reply-{ident}.json"
    deadline = time.monotonic() + 1800
    while not reply.exists():
        if time.monotonic() > deadline:
            raise TimeoutError("宿主评测没有在 1800 秒内回复")
        time.sleep(0.05)
    value = json.loads(reply.read_text(encoding="utf-8"))
    reply.unlink()
    print(json.dumps(value, ensure_ascii=False, indent=2))
    success = value.get("status") == "compiled" or (
        value.get("status") == "ok" and value.get("correct") is not False
    )
    return 0 if success else 1

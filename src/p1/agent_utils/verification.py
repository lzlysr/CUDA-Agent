"""验证当前候选：在 Agent 工作区执行 python3 -m utils.verification。

输入：当前候选；输出：编译及五例正确性反馈；不运行最终隐藏验收。"""

from .client import request

if __name__ == "__main__":
    raise SystemExit(request("verify"))

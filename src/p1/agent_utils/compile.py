"""只编译当前候选：在 Agent 工作区执行 python3 -m utils.compile。

输入：当前工作区源码；输出：宿主编译日志，不进行数值验收或性能评分。"""

from .client import request

if __name__ == "__main__":
    raise SystemExit(request("compile"))

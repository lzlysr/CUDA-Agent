#!/usr/bin/env bash
# 作用：兼容原 SKILL 的 bash utils/compile.sh，委托宿主隔离编译。
# 运行：Agent 工作区内 bash utils/compile.sh；输入当前源码，输出编译反馈。
set -euo pipefail
python3 -m utils.compile "$@"

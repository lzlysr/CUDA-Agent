"""分析基线或候选：python3 -m utils.profiling [--reference-only]。

没有 model_new.py 时自动分析原始 eager/compile；有候选时先验收再测时。
输入：命令选项和当前工作区；输出：耗时及 profiler 事件，不加载 Qwen。"""

import argparse
from pathlib import Path
from .client import request


def main():
    """读取 --reference-only；选择参考或候选分析请求，返回进程退出码。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-only", action="store_true")
    args = parser.parse_args()
    action = (
        "profile_reference"
        if args.reference_only or not Path("model_new.py").exists()
        else "profile"
    )
    return request(action)


if __name__ == "__main__":
    raise SystemExit(main())

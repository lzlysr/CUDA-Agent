"""隔离运行 GPU 开发脚本：python3 -m utils.run tune_kernel.py [脚本参数...]。

输入：当前工作区内辅助 .py 文件及字符串参数；输出：GPU 沙箱的运行日志。
固定工具把请求交给宿主调度，复用源码一致的已编译扩展；脚本不接触隐藏样例，
不计入正式正确率或加速比。选择好配置后，仍需修改 ModelNew 并正式验证。
"""

import argparse
from .client import request


def main():
    """读取脚本路径和其余参数；发出 run_script 请求，返回工具退出码。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("script")
    parser.add_argument("args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    return request("run_script", {"script": args.script, "args": args.args})


if __name__ == "__main__":
    raise SystemExit(main())

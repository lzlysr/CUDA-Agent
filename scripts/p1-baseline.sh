#!/usr/bin/env bash
# 作用：单独查看/测量 PyTorch eager 和 torch.compile，不加载 Qwen。
# 列题：bash scripts/p1-baseline.sh --list --split train --limit 12
# 测时：bash scripts/p1-baseline.sh --task-id <ID> --gpu 3 
#       bash scripts/p1-baseline.sh --split train --limit 1 --gpu 3
# 输入：data/processed/tasks.jsonl；输出：新建 runs/p1/baseline-<时间> 目录。
# --list/--dry-run 不启动 GPU；实际测时必须显式传 --gpu，且会检查占用。

# 启用严格模式：命令失败或变量未定义时退出，管道中任一命令失败也视为失败。
set -euo pipefail
R=/home/likun/junjxu/zhuolin/workspace/CUDA-Agent
cd "$R"
# 在当前 Bash 中加载 Conda 的 Shell 函数，让下一行可以执行 conda activate。
source /home/likun/miniconda3/etc/profile.d/conda.sh
conda activate "$R/.runtime/conda/learner"
# 在当前 Shell 中加载项目的 CUDA、缓存和临时目录配置。
source scripts/set-env.sh
# 让宿主控制进程看不到 GPU；实际基线评测由容器按命令行 --gpu 指定的编号使用 GPU。
export CUDA_VISIBLE_DEVICES=''
# 将项目 src 放在 Python 模块搜索路径前面；已有 PYTHONPATH 非空时将其保留在后面。
export PYTHONPATH="$R/src${PYTHONPATH:+:$PYTHONPATH}"
# 用基线入口替换当前 Shell；-B 禁止写入 pyc，"$@" 原样转交列题、测时和输出目录等参数。
exec python -B -m p1.baseline "$@"

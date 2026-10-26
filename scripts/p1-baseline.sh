#!/usr/bin/env bash
# 作用：单独查看/测量 PyTorch eager 和 torch.compile，不加载 Qwen。
# 列题：bash scripts/p1-baseline.sh --list --split train --limit 12
# 测时：bash scripts/p1-baseline.sh --task-id <ID> --gpu 3 
#       bash scripts/p1-baseline.sh --split train --limit 1 --gpu 3
# 多卡：bash scripts/p1-baseline.sh --split train --limit 100 --gpus 2,3 --run runs/p1/baseline-multi
# 续跑：bash scripts/p1-baseline.sh --resume --run runs/p1/baseline-multi --gpus 2,3
# 输入：data/processed/tasks.jsonl；输出：新建 runs/p1/baseline-<时间> 目录。
# --list/--dry-run 不启动 GPU；不传卡号的新运行沿用默认 GPU 7，建议显式指定。
# Ctrl+C/TERM 停止派题，等当前各卡任务结束并保存；续跑不再传选题参数。

# 启用严格模式：命令失败或变量未定义时退出，管道中任一命令失败也视为失败。
set -euo pipefail
# 指定服务器上的项目根目录，统一相对路径的起点。
R=/home/likun/junjxu/CUDA-Agent
# 切换到项目目录，使模块入口、数据和默认 runs 路径保持一致。
cd "$R"
# 在当前 Bash 中加载 Conda 的 Shell 函数，让下一行可以执行 conda activate。
source /data2/likun/miniconda/etc/profile.d/conda.sh
# 激活宿主控制器使用的 learner；容器中的 GPU 工作进程使用 kernel 环境。
conda activate "$R/.runtime/conda/learner"
# 在当前 Shell 中加载项目的 CUDA、缓存和临时目录配置。
source scripts/set-env.sh
# 让宿主控制进程看不到 GPU；评测容器按 --gpu/--gpus 中的物理编号使用 GPU。
export CUDA_VISIBLE_DEVICES=''
# 将项目 src 放在 Python 模块搜索路径前面；已有 PYTHONPATH 非空时将其保留在后面。
export PYTHONPATH="$R/src${PYTHONPATH:+:$PYTHONPATH}"
# 用基线入口替换当前 Shell；-B 禁止写入 pyc，"$@" 原样转交列题、测时和输出目录等参数。
exec python -B -m p1.baseline "$@"

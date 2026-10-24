#!/usr/bin/env bash
# 作用：激活 learner Conda 环境，使用编号 GPU 2 启动 P1；不是自动恢复脚本。
# 运行：bash scripts/p1-run.sh [--run runs/p1/<新目录>]；已有目录需显式 --resume。
# 输入：configs/p1-pilot.json 与 train pilot；输出：指定 runs 目录的原始轨迹。
# 注意：本脚本会启动模型/GPU；仅查看 eager/compile 请用 p1-baseline.sh。
# 启用严格模式：命令失败或变量未定义时退出，管道中任一命令失败也视为失败。
set -euo pipefail
# 记录服务器上的项目根目录，供后续定位环境、源码和输出目录。
R=/home/likun/junjxu/zhuolin/workspace/CUDA-Agent
# 切换到项目根目录，使后续 scripts、configs、data 等相对路径指向本项目。
cd "$R"
# 在当前 Bash 中加载 Conda 的 Shell 函数，让下一行可以执行 conda activate。
source /home/likun/miniconda3/etc/profile.d/conda.sh
# 激活项目内的 learner Conda 环境，使用其中的 Python 和控制器依赖。
conda activate "$R/.runtime/conda/learner"
# 在当前 Shell 中加载项目的 CUDA、缓存和临时目录配置；下一步会覆盖默认 GPU 设置。
source scripts/set-env.sh
# 覆盖 set-env.sh 的 P0 默认值，让模型生成进程使用 GPU 2；评测 GPU 由 P1 配置另行指定。
export CUDA_VISIBLE_DEVICES=2
# 清除可能继承的旧 GPU UUID 变量，后续按 GPU 编号选择设备。
unset CUDA_AGENT_GPU_UUID
# 将 FlashInfer 的工作缓存目录设置到项目 .runtime 下，集中保存其构建缓存。
export FLASHINFER_WORKSPACE_BASE="$R/.runtime/cache/flashinfer"
# 将 Ray 使用的临时目录设置到项目 .runtime 下。
export RAY_TMPDIR="$R/.runtime/tmp/ray"
# 将项目 src 放在 Python 模块搜索路径前面；已有 PYTHONPATH 非空时将其保留在后面。
export PYTHONPATH="$R/src${PYTHONPATH:+:$PYTHONPATH}"
# 确保临时目录和 P1 实验输出根目录存在；目录已存在时保留其中内容。
mkdir -p "$TMPDIR" "$R/runs/p1"
# 用指定 learner Python 启动 P1 并替换当前 Shell；-B 禁写 pyc，-u 实时输出日志，"$@" 原样转交用户参数。
exec "$R/.runtime/conda/learner/bin/python" -B -u -m p1.run "$@"

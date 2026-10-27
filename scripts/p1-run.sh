#!/usr/bin/env bash
# 作用：激活 learner Conda 环境，启动 P1；不是自动恢复脚本。
# 运行：bash scripts/p1-run.sh [--run runs/p1/<新目录>]；已有目录需显式 --resume。
# 输入：configs/p1-pilot.json 与 train pilot；输出：指定 runs 目录的原始轨迹。
# 注意：本脚本会启动模型/GPU；仅查看 eager/compile 请用 p1-baseline.sh。

# 启用严格模式：命令失败或变量未定义时退出，管道中任一命令失败也视为失败。
set -euo pipefail
R=/home/likun/junjxu/CUDA-Agent
cd "$R"
source /data2/likun/miniconda/etc/profile.d/conda.sh
conda activate "$R/.runtime/conda/learner"
source scripts/set-env.sh
# 父进程不使用 GPU；模型子进程按 p1-pilot.json 的编号独立选择生成卡。
export CUDA_VISIBLE_DEVICES=""
# 清除可能继承的旧 GPU UUID 变量，后续按 GPU 编号选择设备。
unset CUDA_AGENT_GPU_UUID
# 将 FlashInfer 的工作缓存目录设置到项目 .runtime 下，集中保存其构建缓存。
export FLASHINFER_WORKSPACE_BASE="$R/.runtime/cache/flashinfer"
# 将 Ray 使用的临时目录设置到项目 .runtime 下。
export RAY_TMPDIR="$R/.runtime/tmp/ray"
export PYTHONPATH="$R/src${PYTHONPATH:+:$PYTHONPATH}"
# 建立公共临时目录和运行目录；--dry-run 不建立目录。
if [[ " ${*} " != *" --dry-run "* ]]; then
  # 为真实运行准备临时文件和输出父目录。
  mkdir -p "$TMPDIR" "$R/runs/p1"
fi
# 用指定 learner Python 启动 P1 并替换当前 Shell；-B 禁写 pyc，-u 实时输出日志，"$@" 原样转交用户参数。
exec "$R/.runtime/conda/learner/bin/python" -B -u -m p1.run "$@"

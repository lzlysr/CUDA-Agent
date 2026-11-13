#!/usr/bin/env bash
# 作用：直接启动原始 Qwen3.5-9B 的多轮 Agent + GRPO。
# 完整训练：bash scripts/p3-train.sh --run runs/p3/agent-grpo
# 断点续训：bash scripts/p3-train.sh --run runs/p3/agent-grpo --resume
# 接线短测：bash scripts/p3-train.sh --run runs/p3/wire --wire
# 配置预览：bash scripts/p3-train.sh --run runs/p3/preview --dry-run
# 原生参数覆盖示例：末尾添加 data.train_max_samples=100 data.val_max_samples=8
# 实时曲线：.runtime/conda/learner/bin/tensorboard --logdir runs/p3 --host 127.0.0.1 --port 6006

# 任一命令失败、未定义变量或管道失败时退出。
set -euo pipefail
# 根据脚本位置确定项目根目录，不依赖调用者当前目录。
P3_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$P3_ROOT"
source scripts/set-env.sh
export PATH="$P3_ROOT/.runtime/conda/learner/bin:$PATH"
export TZ=Asia/Shanghai PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$P3_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
# 允许原生 verl 将更新后的 LoRA 权重同步给 vLLM。
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=True
# 固定全局 Python 哈希种子；不会为每次模型请求重置 seed。
export PYTHONHASHSEED=42
# 在控制器导入框架时避免提前建立 CUDA 上下文。
export PYTORCH_NVML_BASED_CUDA_CHECK=1
exec "$P3_ROOT/.runtime/conda/learner/bin/python" -B -m p3.train "$@"

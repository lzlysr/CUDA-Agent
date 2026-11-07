#!/usr/bin/env bash
# 作用：轻量检查原始 train、保存或复用运行清单，再启动 P2 verl V1 同步在线 PPO。
# 完整 P2 训练：
# bash scripts/p2-train.sh --run runs/p2/single-ppo-warmup
# 断点续训：
# bash scripts/p2-train.sh --run runs/p2/single-ppo-warmup --resume
# 只读预览完整训练的数据及配置：
# bash scripts/p2-train.sh --run runs/p2/single-ppo-warmup --dry-run
# 输入：已有 train.jsonl 和配置；输出：run 下的轻量清单、运行记录及检查点。
# --prepare-only 仅检查并保存清单；--dry-run 只读预览；--resume 恢复训练。
# 在服务器执行以下命令，然后在 VS Code 转发 6006 端口查看实时曲线：
# .runtime/conda/learner/bin/tensorboard --logdir runs/p2 --host 127.0.0.1 --port 6006

set -euo pipefail
P2_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$P2_ROOT"
source scripts/set-env.sh
export PATH="$P2_ROOT/.runtime/conda/learner/bin:$PATH"
# 新日志统一北京时间，设置源码路径，并禁止 Ray 子进程在源码下写字节码缓存。
export TZ=Asia/Shanghai PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$P2_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
# 允许 verl 原生的运行时 LoRA 更新；实际 GPU 编号由 train.py 在导入 torch 前设置。
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=True
# 固定基础 Python 哈希种子，避免进程间集合遍历产生额外随机变化。
export PYTHONHASHSEED=42
# 控制器用 NVML 检查设备，避免仅导入框架就在宿主进程提前创建 CUDA 上下文。
export PYTORCH_NVML_BASED_CUDA_CHECK=1
# 直接调用 learner Python；无需 activate.sh，也不启动已有外部 Ray 集群。
exec "$P2_ROOT/.runtime/conda/learner/bin/python" -B -m p2.train "$@"

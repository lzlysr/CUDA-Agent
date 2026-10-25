# 作用：设置本项目的 CUDA、缓存和临时目录；不激活 Conda、不启动 GPU。
# 运行：在 Bash 中 source scripts/set-env.sh
# 输入：当前环境变量；输出：导出的路径与 CUDA 配置。单独 source 时默认 GPU 4（P0）。

export CUDA_AGENT_ROOT=/home/likun/junjxu/CUDA-Agent
# 清除可能继承的旧 GPU UUID 变量，后续按 GPU 编号选择设备。
unset CUDA_AGENT_GPU_UUID
# 让 CUDA 按 PCI 总线顺序枚举设备，保持设备枚举顺序稳定。
export CUDA_DEVICE_ORDER=PCI_BUS_ID
# 设置 P0 默认使用的 GPU 编号为 4；P1 启动脚本会在加载本文件后覆盖此值。
export CUDA_VISIBLE_DEVICES=4
export CUDA_HOME=/usr/local/cuda-12.9
# 将该 Toolkit 的 bin 目录放到命令搜索路径前面，使 nvcc 等工具优先使用 CUDA 12.9。
export PATH="$CUDA_HOME/bin:$PATH"
export TMPDIR="$CUDA_AGENT_ROOT/.runtime/tmp"
export PIP_CACHE_DIR="$CUDA_AGENT_ROOT/.runtime/cache/pip"
export HF_HOME="$CUDA_AGENT_ROOT/.runtime/cache/huggingface"
# 指定 PyTorch 即时编译 C++/CUDA 扩展时使用的构建与产物缓存目录。
export TORCH_EXTENSIONS_DIR="$CUDA_AGENT_ROOT/.runtime/cache/torch_extensions"
# 指定 Triton 编译内核的磁盘缓存目录，便于重复使用编译结果。
export TRITON_CACHE_DIR="$CUDA_AGENT_ROOT/.runtime/cache/triton"
# 为遵循 XDG 约定的程序设置默认缓存根目录。
export XDG_CACHE_HOME="$CUDA_AGENT_ROOT/.runtime/cache/xdg"
export CONDA_PKGS_DIRS="$CUDA_AGENT_ROOT/.runtime/cache/conda"
# 将 PyTorch 扩展构建工具使用的并行编译任务数限制为 4，控制 CPU 和内存占用。
export MAX_JOBS=4
# 让 PyTorch CUDA 扩展默认面向计算能力 8.0 的 GPU 编译，对应当前 A100。
export TORCH_CUDA_ARCH_LIST=8.0
# 关闭 Hugging Face tokenizer 的内部并行，减少派生进程时的并行冲突与告警。
export TOKENIZERS_PARALLELISM=false
# 将 torch.compile 使用的 TorchInductor 编译缓存放到项目内。
export TORCHINDUCTOR_CACHE_DIR="$CUDA_AGENT_ROOT/.runtime/cache/inductor"
export VLLM_CACHE_ROOT="$CUDA_AGENT_ROOT/.runtime/cache/vllm"
# 将 OpenMP 默认线程数设置为 4，控制采用 OpenMP 的 CPU 计算并行度。
export OMP_NUM_THREADS=4
# 关闭 vLLM 的使用统计上报。
export VLLM_NO_USAGE_STATS=1
# 关闭 Hugging Face Hub 客户端的遥测上报。
export HF_HUB_DISABLE_TELEMETRY=1
# 默认禁用 Weights & Biases 实验跟踪，避免运行时向该服务上报实验日志。
export WANDB_MODE=disabled

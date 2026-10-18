# CUDA-Agent P0 环境与模型验通报告

完成时间：2026-10-17T08:00:45.599852+00:00。本轮仅执行单卡 P0，没有启动正式训练。

服务器：`air-1`；项目目录：`/home/likun/junjxu/zhuolin/workspace/CUDA-Agent`；GPU：A100-SXM4-80GB，编号 4，UUID `GPU-ac246f3e-147f-4145-cac6-ab33881f4308`。

## 最终环境

| 组件 | 实测版本 / 方式 |
|---|---|
| 环境管理 | conda prefix 环境，Python 3.12.15；Python 依赖由 pip 安装 |
| 主模型 | Qwen/Qwen3.5-9B |
| 模型 revision | `c202236235762e1c871ad0ccb60c8ee5ba337b9a` |
| torch | 2.11.0+cu129 |
| CUDA Toolkit / nvcc | 12.9 / 12.9.86 |
| cuDNN | 9.17.1 |
| Transformers / PEFT | 5.9.0 / 0.21.2 |
| vLLM | **0.21.0+cu129 官方发布包** |
| verl | 官方源码 v0.9.1，editable 安装于 `third_party/verl` |
| verl commit | `1876b06d0a3e4e71e06230be10af14492ca8a75b` |
| HF 模型执行 | SDPA；vLLM 执行使用 FlashAttention 2 和 Triton/FLA GDN |
| LoRA | rank 64，alpha 128，dropout 0；仅语言模型线性层，覆盖 Attention、GDN、MLP |
| critic | 独立语言 backbone LoRA 与 Linear(4096, 1) token value head |

learner 与 kernel 的 15 个 CUDA cu12 库包版本全部一致。Docker 使用现有镜像 `rpent-vla-env:libero-pro`，实际 Python/torch 来自只读挂载的项目 kernel 环境，而非镜像自带环境。

## 验证结果

| 检查 | 结果 |
|---|---|
| 两个环境的 pip check | 通过 |
| 模型快照 | 13 个文件全部按官方 revision 元数据校验；4 个权重的 SHA-256 一致 |
| CUDA 扩展编译 | nvcc 12.9 编译成功，15 组数值检查通过 |
| kernel 环境运行扩展 | 通过 |
| Docker 限制执行 | 只读项目、无网络、非 root、去除 capabilities；扩展正确；仅一张 GPU 可见 |
| torch.compile | 3 个随机种子，与 eager 结果一致 |
| verl 实际模块导入 | PPO core、engine workers、vLLM async rollout、Qwen3.5 模块全部通过 |
| HF 基础推理 | thinking 开/关均通过短生成测试 |
| actor LoRA | 单步更新通过；梯度有限；496 个适配器张量改变 |
| actor 保存重载 | logits 最大绝对差 0.0 |
| critic LoRA + value head | 单步更新通过；token value 形状 [1, 22]；梯度有限 |
| critic 保存重载 | token value 最大绝对差 0.0 |
| vLLM 基础与适配器推理 | 两者均生成成功，逐 token logprob 有限 |
| HF / vLLM logprob | 两个模型各 48 个响应 token，检查通过；见下表 |

| 模型 | logprob 平均绝对差 | 最大绝对差 |
|---|---:|---:|
| 基础模型 | 0.003257 | 0.047110 |
| actor 适配器 | 0.003735 | 0.091116 |

logprob 测试预设阈值：平均绝对差 <= 0.1，最大绝对差 <= 0.5。这是短序列 BF16 功能检查，不表示两种后端数值完全相同，也不表示长序列 PPO 的重要性比率已验证。

actor / critic 微步测试峰值显存分别约 **21.50 / 18.69 GiB**。它们是单条短样例，不是正式训练资源估算。测试期间 GPU 4 也有其他 OpenVLA 进程，因此不使用耗时作为性能结论。

## 激活环境

在项目目录内：

```bash
conda activate .runtime/conda/kernel
# 或
conda activate .runtime/conda/learner
```

任意目录均可使用：

```bash
conda activate /home/likun/junjxu/zhuolin/workspace/CUDA-Agent/.runtime/conda/kernel
conda activate /home/likun/junjxu/zhuolin/workspace/CUDA-Agent/.runtime/conda/learner
```

环境提示符已配置为 `(kernel)` / `(learner)`。为支持直接 conda activate，设置了当前用户的 `env_prompt='({name}) '`；未修改全局 envs_dirs、系统 Python、系统 CUDA 或其他项目环境。

P0 测试命令使用 `scripts/set-env.sh` 配置 GPU 4、CUDA 12.9 和项目内缓存。vLLM 0.21 需要数字 CUDA_VISIBLE_DEVICES，已验证编号 4 对应获授权 UUID。

## 重要的兼容性修正

1. 经用户确认，vLLM 候选从 0.24.0 调整到 0.21.0。
2. 仅固定版本号仍不够：0.21.0 默认 wheel 的原生扩展依赖 CUDA 13。已换成[官方 v0.21.0 发布中的 cu129 wheel](https://github.com/vllm-project/vllm/releases/tag/v0.21.0)，并验证 SHA-256 为 `920777691e340df7a8328adfb1e57b9996dbb537edfb654dd32f70844f5f423d`。实测 ELF 依赖 `libcudart.so.12`。
3. 最终 `reports/p0/requirements.txt` 已固定上述 wheel 的官方 URL 与哈希，避免后续重新安装默认 CUDA 13 包。
4. 权重下载经过多个端点尝试；最终使用与 HF 固定 revision 的四个 SHA-256 完全一致的 ModelScope 权重。最终校验涵盖权重、tokenizer、配置与模板，共 13 个文件。

## 当前保留文件

- 本目录仅保留本报告与 `requirements.txt`（固定 Python 依赖版本与 vLLM wheel 来源）。详细验证 JSON、日志、pip freeze 和 conda explicit 快照已按用户要求删除；上面的验证结果为 P0 历史摘要。
- 环境变量：`scripts/set-env.sh`，用 `source scripts/set-env.sh` 加载；Conda 环境使用标准 `conda activate` 激活。
- 模型：`models/Qwen3.5-9B`；Conda 环境：`.runtime/conda/{learner,kernel}`；verl 源码：`third_party/verl`。
- P0 测试脚本、测试适配器、临时环境及缓存已清理。需要复跑时需重新提供测试脚本。
- `requirements.txt` 中的 editable verl 路径要求该目录存在且源码版本匹配上表；此文件不包含源码、Python 解释器、系统驱动或完整 CUDA Toolkit。torch 的 cu129 包安装时需要可访问对应的 PyTorch cu129 软件源。

## 本轮边界

本轮通过的是环境与模型的单卡组件检查，以及可信 CUDA fixture 的编译、隔离执行与校验。没有启动 Ops-6K、RFT、value pretraining、正式 PPO、比赛评测或完整 Agent rollout；没有验证多卡 FSDP、分布式权重同步、8K/32K 训练、通用模型生成 kernel 的完整评测闭环，也没有据此声称复现论文指标或解决 reward hacking。后续阶段需另行开展这些工作。

维护更新：已移除 `scripts/activate.sh`，使用标准 conda activate；环境变量脚本名为 `scripts/set-env.sh`。verl 的 Git 元数据及其备份已删除，源码未变，原始提交记录于本报告上表。已删除旧依赖输入文件，锁文件更名为 `requirements.txt`；临时目录及 Conda 包缓存已清理，后续使用时可重新创建。

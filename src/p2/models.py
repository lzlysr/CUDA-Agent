"""Qwen3.5 的项目级 FSDP2 扩展：配置驱动的 LoRA、Critic 价值头与更新审计。

由项目模型配置类型及 external_lib 导入，通过公开 EngineRegistry 注册 NVIDIA 引擎。
不修改 verl、Transformers、PEFT 或 vLLM 源文件。输入为 verl 模型/引擎配置，
输出为标准 LM logits（Actor）或每 token 一个 value（Critic）。
"""

import json
import os
from pathlib import Path
from dataclasses import dataclass

import torch
from peft import LoraConfig, TaskType, get_peft_model
from verl.utils.py_functional import convert_to_regular_types
from verl.workers.engine.base import EngineRegistry
from verl.workers.engine.fsdp.transformer_impl import FSDPEngineWithLMHead, FSDPEngineWithValueHead
from verl.workers.config import HFModelConfig

# Ray 子进程导入模型配置类型时，同时完成单轮生成流程的公开注册。
# 这里只注册类，不加载模型、不创建服务，也不执行采样。
from . import rollout as _rollout
from .common import now

@dataclass
class CudaHFModelConfig(HFModelConfig):
    """项目模型配置类型，使 Ray 子进程在反序列化配置时先导入本模块。

    external_lib 只在配置首次构造时加载；Ray 传递现成的 dataclass 不再执行
    __post_init__。显式配置类型保证每个 GPU worker 的公开引擎注册都已完成，
    随后仍由原生 TrainingWorker 和同步训练器创建引擎，无需复制训练循环。
    """

    # 原生 HFModelConfig 尚无该字段；通过公开配置扩展传入 PEFT，避免写死在引擎内。
    lora_dropout: float = 0.0

    def __post_init__(self):
        """初始化标准配置，P2 的纯文本任务使用 tokenizer 与一维文本位置。"""
        super().__post_init__()
        # P2 没有图像/视频。关闭多模态 processor，使 verl 使用文本 position_ids，
        # 避免四路 M-RoPE 经 TQ/TensorDict 传输时出现错误的 jagged 轴与 offsets。
        # 原生 Qwen3.5 在前向内把二维文本位置扩展为四路，纯文本位置语义一致。
        self.processor = None


class UpdateAudit:
    """保留 LoRA 角色配置和训练审计；模型创建、优化器更新均交给 verl。"""

    def record(self, event, **fields):
        """输入事件和标量；追加本进程审计文件，返回 None。"""
        directory = Path(os.environ["CUDA_AGENT_P2_RUN"]) / "audit"
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{self.model_config.model_type}-rank{torch.distributed.get_rank()}.jsonl"
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"at": now().isoformat(), "event": event, **fields}, ensure_ascii=False) + "\n")

    def _build_lora_module(self, module):
        """输入原生模型；返回 LoRA 模型，Critic 额外完整训练并保存 score 头。"""
        is_value = self.model_config.model_type == "value_model"
        module.enable_input_require_grads()
        # 原生 verl 的通用 LoRA 构造固定使用 CAUSAL_LM，且没有 dropout 字段。
        # 这里只补齐角色和 dropout；适配器创建、目标层筛选及冻结仍使用 PEFT。
        lora_config = LoraConfig(
            task_type=TaskType.TOKEN_CLS if is_value else TaskType.CAUSAL_LM,
            r=self.model_config.lora_rank,
            lora_alpha=self.model_config.lora_alpha,
            lora_dropout=self.model_config.lora_dropout,
            target_modules=convert_to_regular_types(self.model_config.target_modules),
            target_parameters=convert_to_regular_types(self.model_config.target_parameters),
            exclude_modules=convert_to_regular_types(self.model_config.exclude_modules),
            modules_to_save=["score"] if is_value else None,
            bias="none",
        )
        # PEFT 原生开关让 LoRA 沿用基础层 dtype，避免先升为 FP32 再手动转回 BF16。
        module = get_peft_model(module, lora_config, autocast_adapter_dtype=False)
        # target_modules/exclude_modules 直接控制覆盖范围，不强制绑定某一份层名列表。
        trainable = [name for name, parameter in module.named_parameters() if parameter.requires_grad]
        if is_value and not any("score.modules_to_save" in name for name in trainable):
            raise ValueError("Critic 的价值头未设为可训练")
        self.record("trainable_parameters", names=trainable,
                    lora_dropout=self.model_config.lora_dropout,
                    target_modules=self.model_config.target_modules,
                    parameters=sum(p.numel() for p in module.parameters() if p.requires_grad))
        return module

    def optimizer_step(self):
        """无输入；执行原生裁剪/更新，记录本地分片的变化并返回原生梯度范数。"""
        selected = []
        for name, parameter in self.module.named_parameters():
            if parameter.requires_grad:
                local = parameter.to_local() if hasattr(parameter, "to_local") else parameter
                if local.numel():
                    # 每个张量取少量固定元素，避免为审计保留数亿参数副本。
                    selected.append((name, local, local.detach().reshape(-1)[:64].clone()))
        norm = super().optimizer_step()
        changed = [name for name, local, before in selected
                   if not torch.equal(before, local.detach().reshape(-1)[:64])]
        self.record("optimizer_step", grad_norm=float(norm), changed_tensors=len(changed),
                    changed_value_head=any("score" in name for name in changed))
        if not torch.isfinite(torch.tensor(norm)):
            raise FloatingPointError("梯度非有限，本次接线失败；停止而非假装完成 PPO 更新")
        return norm


@EngineRegistry.register("language_model", "fsdp2", device="cuda", vendor="nvidia")
class CudaActorEngine(UpdateAudit, FSDPEngineWithLMHead):
    """使用 verl 原生语言模型前后向，仅添加 LoRA 限定和更新审计。"""


@EngineRegistry.register("value_model", "fsdp2", device="cuda", vendor="nvidia")
class CudaValueEngine(UpdateAudit, FSDPEngineWithValueHead):
    """继承 verl 原生模型加载、价值头创建、前向及训练流程，仅保留 LoRA 与审计。"""

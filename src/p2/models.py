"""Qwen3.5 的项目级 FSDP2 扩展：配置驱动的 LoRA 和 Critic 价值头接线。

由项目模型配置类型及 external_lib 导入，通过公开 EngineRegistry 注册 NVIDIA 引擎。
不修改 verl、Transformers、PEFT 或 vLLM 源文件。输入为 verl 模型/引擎配置，
输出为标准 LM logits（Actor）或每 token 一个 value（Critic）。
"""

from dataclasses import dataclass

from peft import LoraConfig, TaskType, get_peft_model
from verl.utils.py_functional import convert_to_regular_types
from verl.workers.engine.base import EngineRegistry
from verl.workers.engine.fsdp.transformer_impl import FSDPEngineWithValueHead
from verl.workers.config import HFModelConfig

# Ray 子进程导入模型配置类型时，同时完成单轮生成流程的公开注册。
# 这里只注册类，不加载模型、不创建服务，也不执行采样。
from . import rollout as _rollout

@dataclass
class CudaHFModelConfig(HFModelConfig):
    """项目模型配置类型，使 Ray 子进程在反序列化配置时先导入本模块。

    external_lib 只在配置首次构造时加载；Ray 传递现成的 dataclass 不再执行
    __post_init__。显式配置类型保证每个 GPU worker 的公开引擎注册都已完成，
    随后仍由原生 TrainingWorker 和同步训练器创建引擎，无需复制训练循环。
    """

    def __post_init__(self):
        """初始化标准配置，P2 的纯文本任务使用 tokenizer 与一维文本位置。"""
        super().__post_init__()
        # P2 没有图像/视频。关闭多模态 processor，使 verl 使用文本 position_ids，
        # 避免四路 M-RoPE 经 TQ/TensorDict 传输时出现错误的 jagged 轴与 offsets。
        # 原生 Qwen3.5 在前向内把二维文本位置扩展为四路，纯文本位置语义一致。
        self.processor = None


@EngineRegistry.register("value_model", "fsdp2", device="cuda", vendor="nvidia")
class CudaValueEngine(FSDPEngineWithValueHead):
    """仅适配 Critic 的 LoRA 角色；模型加载、价值头和优化器流程使用 verl 原生实现。"""

    def _build_lora_module(self, module):
        """输入原生 Critic；返回 TOKEN_CLS LoRA 模型，完整训练并保存 score 头。"""
        module.enable_input_require_grads()
        # 原生通用构造固定使用 CAUSAL_LM；Critic 需要 TokenClassification 包装。
        # 不额外传 dropout，沿用 PEFT 默认值；适配器创建、筛选及冻结仍使用 PEFT。
        lora_config = LoraConfig(
            task_type=TaskType.TOKEN_CLS,
            r=self.model_config.lora_rank,
            lora_alpha=self.model_config.lora_alpha,
            target_modules=convert_to_regular_types(self.model_config.target_modules),
            target_parameters=convert_to_regular_types(self.model_config.target_parameters),
            exclude_modules=convert_to_regular_types(self.model_config.exclude_modules),
            modules_to_save=["score"],
            bias="none",
        )
        # PEFT 原生开关让 LoRA 沿用基础层 dtype，避免先升为 FP32 再手动转回 BF16。
        module = get_peft_model(module, lora_config, autocast_adapter_dtype=False)
        # target_modules/exclude_modules 直接控制覆盖范围，不强制绑定某一份层名列表。
        trainable = [name for name, parameter in module.named_parameters() if parameter.requires_grad]
        if not any("score.modules_to_save" in name for name in trainable):
            raise ValueError("Critic 的价值头未设为可训练")
        return module

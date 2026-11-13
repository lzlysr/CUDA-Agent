"""保留 Qwen3.5 已验通的纯文本位置适配；不注册 Critic 或手写模型层。

通过 YAML 的模型配置类型在 GPU worker 导入。输入原生 HFModelConfig 参数，
输出关闭多模态 processor 的配置；Actor、LoRA、分块输出头均由 verl 创建。
"""

from dataclasses import dataclass
from verl.workers.config import HFModelConfig

# 配置类型在各 Ray worker 反序列化时导入，确保原生 Agent 注册表包含 CUDA 接线。
from . import agent as _agent


@dataclass
class TextModelConfig(HFModelConfig):
    """纯文本 CUDA 任务使用一维文本位置，避免 TQ 的多模态 jagged 轴错误。"""

    def __post_init__(self):
        """初始化原生配置后关闭 processor；Qwen 模型自行扩展文本位置。"""
        super().__post_init__()
        self.processor = None

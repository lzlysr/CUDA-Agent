"""verl 原生 SingleTurnAgentLoop 的薄扩展：固定 seed 并检查输出。

由模型配置模块导入并通过 register 注册，不单独运行。
输入：原始消息及采样参数；输出：原生 AgentLoopOutput。
提示词处理、实际生成、Continuous Token 合并与性能指标均交给原生单轮流程；
总上下文上限由原生 vLLM 服务依据 max_model_len 控制，不自行重写生成过程。
"""

import json

from p2.failures import RecoverableSampleError, check_quarantine, record_failure
from verl.experimental.agent_loop.agent_loop import register
from verl.experimental.agent_loop.single_turn_agent_loop import SingleTurnAgentLoop


@register("cuda_single")
class CudaSingleTurnLoop(SingleTurnAgentLoop):
    """不增加工具交互或格式修补，只在原生单轮流程前后补充项目约束。"""

    async def run(self, sampling_params, priority=0, **kwargs):
        """输入采样参数和 raw_prompt；固定请求 seed 后返回原生结果，不另存 token 审计。"""
        # UID 来自 verl 原始请求；用于精确关联失败组，避免扫描旧日志误判。
        extra = kwargs["extra_info"]
        task = json.loads(extra["task_json"])
        try:
            check_quarantine(self.config.p2, task["task_id"])
            params = dict(sampling_params, seed=int(self.config.p2.seed))
            output = await super().run(params, priority=int(priority), **kwargs)
            if not output.response_ids:
                raise RecoverableSampleError("rollout 真正返回零 token，丢弃并补采")
        except Exception as error:
            record_failure(
                self.config.p2, kwargs["uid"], task["task_id"], error,
                hidden=bool(extra.get("hidden", False)),
            )
            raise
        # 不对回答重新编码，直接核对原生流程返回的真实 token、mask 和概率长度。
        size = len(output.response_ids)
        if len(output.prompt_ids) + size > int(self.config.p2.context_tokens):
            raise ValueError("rollout 超出总上下文预算")
        if len(output.response_mask) != size:
            raise ValueError("rollout response mask 与实际 response token 不对应")
        if output.response_logprobs is None or len(output.response_logprobs) != size:
            raise ValueError("rollout logprob 与实际 response token 不对应，停止本批次")
        # PPO 直接使用原生返回的 token/mask/logprob，不再额外落盘重复数组。
        output.extra_fields = dict(output.extra_fields or {}, request_seed=int(self.config.p2.seed))
        return output

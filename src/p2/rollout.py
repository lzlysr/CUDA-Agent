"""verl 单轮 rollout 扩展：固定每次请求 seed，使用真实 token 并限制总上下文。

由项目模型配置模块导入并通过 verl 的 register 注册；输入原始消息与采样参数，
输出标准 AgentLoopOutput，无需额外的生成流程 YAML。
没有工具交互，没有输出格式修补；所有生成 token 都进入 PPO 的 response mask。
"""

from uuid import uuid4
import os
from pathlib import Path
from time import perf_counter

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput, register
from p2.common import now, write_json


@register("cuda_single")
class CudaSingleTurnLoop(AgentLoopBase):
    """一次采样一个最终候选；不把 P1 历史最佳结果替换成本次奖励。"""

    async def run(self, sampling_params, priority=0, **kwargs):
        """输入采样参数和 raw_prompt；返回不重新编码的 prompt/response token。"""
        prompt_ids = await self.ct_build_initial_tokens(list(kwargs["raw_prompt"]))
        remaining = int(self.config.p2.context_tokens) - len(prompt_ids)
        if remaining <= 0:
            raise ValueError("prompt 超出 P2 总上下文预算")
        params = dict(sampling_params, seed=int(self.config.p2.seed), max_tokens=remaining)
        request_id = uuid4().hex
        started = perf_counter()
        output = await self.server_manager.generate(
            request_id=request_id, prompt_ids=prompt_ids, sampling_params=params, priority=int(priority),
        )
        if not output.token_ids or len(output.token_ids) > remaining:
            raise ValueError("rollout 返回空输出或超出总上下文预算")
        if output.log_probs is None or len(output.log_probs) != len(output.token_ids):
            raise ValueError("rollout logprob 与实际 response token 不对应，停止本批次")
        # 小规模审计保留原始 token/采样概率/版本；训练直接消费 TQ 张量，不重读此文件。
        write_json(Path(os.environ["CUDA_AGENT_P2_RUN"]) / "rollouts" / f"{request_id}.json", {
            "at": now().isoformat(), "prompt_ids": prompt_ids, "response_ids": output.token_ids,
            "response_mask": [1] * len(output.token_ids), "rollout_log_probs": output.log_probs,
            "policy_version": output.extra_fields, "seed": int(self.config.p2.seed),
        })
        # 单轮直接使用 vLLM 返回的 token，避免先 decode 再 tokenize 改变 PPO 对齐关系。
        return AgentLoopOutput(
            prompt_ids=prompt_ids, response_ids=output.token_ids,
            response_mask=[1] * len(output.token_ids), response_logprobs=output.log_probs,
            metrics={"generate_sequences": perf_counter() - started,
                     "num_preempted": output.num_preempted if output.num_preempted is not None else -1},
            num_turns=2, extra_fields=dict(output.extra_fields or {}, request_seed=int(self.config.p2.seed)),
        )

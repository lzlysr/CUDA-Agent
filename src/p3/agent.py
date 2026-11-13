"""原生 ToolAgentLoop 的 CUDA 接线：生命周期、终局奖励及错误传播。

由 verl 注册加载，不重写生成状态机、工具解析、token 合并、mask 或 GRPO。
输入原生采样参数/题目；输出携带可信终局奖励的 AgentLoopOutput。
工具 schema 由启动入口从既有 P1 定义注入唯一运行配置。
"""

import asyncio
import json
import re
import subprocess

from verl.experimental.agent_loop.agent_loop import register
from verl.experimental.agent_loop.tool_agent_loop import AgentState, ToolAgentLoop
from verl.tools.base_tool import BaseTool
from verl.tools.schemas import ToolResponse

from p1.evaluation import InfrastructureError, ReferenceTaskError, dump
from p1.helper.pilot_parallel import blocking_lock
from p2.failures import RecoverableSampleError

from .environment import Workspace


def record_failure(config, uid, directory, error):
    """输入组 ID/异常；保存每轨迹现场及组错误，致命错误不被另一个可恢复错误覆盖。"""
    from pathlib import Path
    record = {
        'recoverable': isinstance(error, RecoverableSampleError),
        'error_type': type(error).__name__, 'error': str(error),
        'detail_path': str(getattr(error, 'detail_path', directory)),
    }
    dump(directory / 'failure.json', record)
    target = Path(config.p3.run) / 'scheduler/failures' / f'{uid}.json'
    with blocking_lock(target.with_suffix('.lock')):
        if target.exists() and not json.loads(target.read_text())['recoverable']:
            return
        dump(target, record)


class WorkspaceTool(BaseTool):
    """通用薄工具适配器，复用 WorkspaceTools 和 SandboxSession，不复制工具实现。"""

    async def execute(self, instance_id, parameters, **kwargs):
        """输入原生工具参数和 AgentData；返回真实工具反馈，额外工具奖励恒为零。"""
        context = kwargs['agent_data'].cuda_context
        async with context.tool_lock:
            if context.fatal is not None:
                raise context.fatal
            context.record({'type': 'tool-call', 'name': self.name, 'arguments': parameters})
            try:
                result = await asyncio.to_thread(context.dispatch, self.name, parameters)
            except (InfrastructureError, ReferenceTaskError, RecoverableSampleError,
                    subprocess.SubprocessError) as error:
                context.fatal = error
                raise
            except OSError as error:
                # 文件路径/权限等工具使用错误可反馈给模型；设备/空间/I/O 故障不能吞掉。
                if error.errno not in {2, 13, 17, 20, 21}:
                    context.fatal = error
                raise
            except (ValueError, LookupError, TypeError, re.error):
                # 参数、编辑目标及正则表达式错误使用框架原生工具错误反馈。
                raise
            except Exception as error:
                context.fatal = error
                raise
            context.record({'type': 'tool-response', 'name': self.name, 'result': result})
            return ToolResponse(text=json.dumps(result, ensure_ascii=False)), 0.0, {}


@register('cuda_grpo')
class CudaAgentLoop(ToolAgentLoop):
    """仅扩展真实环境边界；策略生成、工具观察和概率路径完全使用原生实现。"""

    async def run(self, sampling_params, **kwargs):
        """输入一条组内请求；运行完整 Agent，选最佳源码并返回原生 token/mask/logprob。"""
        task = kwargs['extra_info']['task']
        self.context = Workspace(self.config, task, kwargs['uid'], kwargs['session_id'])
        self.context.tool_lock = asyncio.Lock()
        # V1 在派发前移除 validate 字段；原始划分字段仍可靠地区分 test 验证。
        validation = bool(kwargs.get('validate', task.get('split') == 'test'))
        self.max_assistant_turns = int(self.config.p3.validation_turns if validation else self.config.p3.training_turns)
        try:
            # 只固定框架/引擎的全局 seed；不再给每个同题请求重复设置相同 seed。
            params = dict(sampling_params)
            params.pop('seed', None)
            output = await super().run(params, **kwargs)
            if not any(output.response_mask):
                raise RecoverableSampleError('No model-generated tokens; discard the whole group')
            result = await asyncio.to_thread(self.context.finish)
            output.reward_score = float(result['reward'])
            output.extra_fields['reward_extra_info'] = {
                'acc': int(result['correct']), 'reward_valid': True,
                'speedup_eager': result.get('speedup_eager', 0.0),
                'speedup_compile': result.get('speedup_compile', 0.0),
                'detail': str(self.context.directory / 'result.json'),
            }
            return output
        except Exception as error:
            record_failure(self.config, kwargs['uid'], self.context.directory, error)
            raise
        finally:
            await asyncio.to_thread(self.context.close)

    async def _handle_pending_state(self, agent_data, sampling_params):
        """输入原生初始状态；注入本轨迹工具上下文，并用实际 prompt 长度限制总上下文。"""
        agent_data.cuda_context = self.context
        state = await super()._handle_pending_state(agent_data, sampling_params)
        available = min(
            int(self.config.p3.context_tokens) - len(agent_data.prompt_ids),
            self.response_length - len(agent_data.response_mask),
        )
        if available <= 0:
            raise ValueError('Prompt exceeds configured total context')
        self.response_length = min(self.response_length, available)
        return state

    async def _handle_generating_state(self, agent_data, sampling_params, ignore_termination=False):
        """输入原生回合状态；只限剩余 token 预算，原样保留生成 token 与采样概率。"""
        available = min(
            int(self.config.p3.context_tokens) - len(agent_data.prompt_ids),
            self.response_length - len(agent_data.response_mask),
        )
        if available <= 0:
            return AgentState.TERMINATED
        params = dict(sampling_params)
        params['max_tokens'] = min(int(params.get('max_tokens') or available), available)
        state = await super()._handle_generating_state(agent_data, params, ignore_termination)
        self.context.record({
            'type': 'content', 'turn': agent_data.assistant_turns,
            'text': self.tokenizer.decode(agent_data.response_ids, skip_special_tokens=False),
        })
        return state

    async def _handle_processing_tools_state(self, agent_data):
        """执行原生工具阶段；不静默丢弃同回合调用，submit 后结束本轨迹。"""
        state = await super()._handle_processing_tools_state(agent_data)
        return AgentState.TERMINATED if self.context.submitted else state

    def _build_assistant_message(self, content, agent_data):
        """输入原生解析结果；先保留全部调用，使消息记录与实际执行始终对应。"""
        self.max_parallel_calls = max(self.max_parallel_calls, len(agent_data.tool_calls))
        return super()._build_assistant_message(content, agent_data)

    async def _call_tool(self, tool_call, tools_kwargs, agent_data):
        """复用原生参数错误反馈；将被原生工具层捕获的致命环境异常重新抛到采样层。"""
        result = await super()._call_tool(tool_call, tools_kwargs, agent_data)
        if self.context.fatal is not None:
            raise self.context.fatal
        return result

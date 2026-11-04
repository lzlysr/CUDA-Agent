"""P1 单次生成与 Agent 开发流程；使用原始 Qwen，不训练权重。

运行：bash scripts/p1-run.sh --run runs/p1/<新目录>。
输入：configs/p1-pilot.json、train pilot 和本地模型；输出：原始轨迹与结果。
GPU 按配置的数字编号使用；--dry-run 只预览。本模块导入本身不会加载 vLLM 模型。"""

import argparse
from contextlib import ExitStack
import multiprocessing
import signal
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import threading
import time
import traceback

from .data import ROOT, sha
from .helper.baseline_state import (
    archive_incomplete, atomic_json as dump, beijing_now, cleanup_reference_artifacts,
    exclusive_lock, run_lock_path, guard_active_containers,
)
from .helper.pilot_parallel import archive_partial_trajectory, gpu_roles
from .helper.progress import PilotProgress
from .helper.turn_log import TurnLog
from .helper.tool_calls import answer_body, parse_calls, parse_single
from .helper.workspace import workspace_path
from .evaluation import Evaluator, InfrastructureError

from .tools import TOOLS, WorkspaceTools
from .sandbox import SandboxSession, ShellSessionClosed


CONTRACT = """You optimize PyTorch tasks for NVIDIA A100 sm_80 using CUDA C++.
Environment: CUDA 12.9, torch 2.11.0+cu129, cuDNN 9.17, cuBLAS 12.9.
Keep the ModelNew constructor and state_dict fully compatible with Model.
Preserve the original parameter/module hierarchy, parameter names and buffers exactly.
Python forward may allocate tensors, inspect metadata and call cuda_extension,
but must not compute with PyTorch, torch.compile, Triton or other libraries.
Calling an existing nn.Module's forward is also a PyTorch compute fallback.
Use raw CUDA kernels; cuBLAS only for GEMM and cuDNN only for convolution are allowed.
Never alter global precision or evaluation state. Both reference and candidate allow
TF32; atol=rtol=1e-2. Do not cache outputs for input values.
Implement model_new.py plus kernels/*.cu and *_binding.cpp. The supplied binding.cpp
and binding_registry.h are fixed: do not return or overwrite these files.
Include ../binding_registry.h and register bindings with REGISTER_BINDING(name, function).
Use torch/types.h and torch/csrc/utils/pybind.h, c10::cuda::getCurrentCUDAStream(),
allocate outputs and launch CUDA. Do not include your own PYBIND11_MODULE.
Use positional Python arguments unless m.def explicitly names arguments with py::arg.
For every binding, match the argument count, order, types and return structure.
Obtain dimensions from the actual input and preserve output shapes, dtypes and tuple order.
Guard every global-memory access, including vectorized-load tails and reduction indices.
Output creation and nn.Module parameter declarations in __init__ are allowed.
Use direct cuda_extension calls in forward. Inspect installed cuDNN headers rather than
guessing API signatures: use named cudnn enum constants, not untyped integer arguments.
Aim to beat eager and torch.compile by >5%."""


def safe_path(workspace, name, writing=False):
    """输入工作区、相对路径和写标记；返回不逃逸且符合写范围的绝对路径，违规抛错。"""
    return workspace_path(workspace, name, writing=writing)


def write_files(workspace, files):
    """输入工作区及路径→源码字典；先验证全部文件再写入，返回已写路径。"""
    if not isinstance(files, dict) or not files:
        raise ValueError("files must be a nonempty path->source dictionary")
    prepared = []
    # 任一文件越界或内容无效时，整次提交不写入，避免留下一半新、一半旧的源码。
    for name, text in files.items():
        p = safe_path(workspace, name, True)
        if not isinstance(text, str) or len(text.encode()) > 1000000:
            raise ValueError("invalid source file or file too large")
        prepared.append((p, text))
    for p, text in prepared:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
    return {"written": list(files)}


def parse(text):
    """兼容旧的单调用解析入口；多调用必须使用 parse_calls，不能静默只取第一个。"""
    try:
        actions = parse_calls(text)
    except (ValueError, TypeError, KeyError):
        return None
    if len(actions) > 1:
        raise ValueError("multiple tool calls: use parse_calls")
    return actions[0] if actions else None


def generation_budget(cfg, agent, prompt_tokens):
    """输入配置、是否 Agent 及前缀 token 数；返回本次最大输出数，零表示上下文耗尽。

    generation_tokens=None 时使用当前上下文余量，不再默默限制为 4096。
    单轮 32768/Agent 131072 对齐论文的窗口数字；单次输出上限未在论文披露。
    """
    context = cfg["context_tokens"] if agent else cfg["single_context_tokens"]
    remaining = max(0, context - prompt_tokens)
    return min(remaining, cfg.get("generation_tokens") or remaining)


def summarize_error(error):
    """输入完整编译/执行错误；返回保留关键诊断的短文本，不修改原始日志。"""
    if "incompatible function arguments" in error and "Invoked with:" in error:
        # pybind 会打印大型输入张量；优先保留支持的签名，而非只截取张量打印尾部。
        signature, _, arguments = error.partition("Invoked with:")
        kwargs = arguments.rpartition("kwargs:")[-1] if "kwargs:" in arguments else "none shown"
        return signature[:3500] + "\n[Tensor values omitted]\nKeyword arguments: " + kwargs[:1000]
    lines = error.splitlines()
    diagnostics = []
    for index, line in enumerate(lines):
        if any(word in line.lower() for word in ["error:", "fatal error", "undefined reference", "undefined symbol"]):
            diagnostics.extend(lines[max(0, index - 1):index + 3])
    if diagnostics:
        # 重复模板实例化错误只保留第一次，避免同一条信息占满上下文。
        return "\n".join(dict.fromkeys(diagnostics))[:5000]
    if len(error) <= 5000:
        return error
    # 通用异常保留首尾：异常类型/签名通常在开头，具体参数或路径可能在结尾。
    return error[:3000] + "\n[Middle omitted; full error is in the evaluation log]\n" + error[-1500:]


class Pilot:
    def __init__(self, cfg, run, *, generation_gpu=None, evaluator=None):
        """输入公共配置、run 及可选本进程卡号/评测器；初始化状态，不加载模型。"""
        self.cfg = cfg
        self.run = Path(run)
        generation, evaluation = gpu_roles(cfg)
        self.generation_gpu = generation[0] if generation_gpu is None else generation_gpu
        self.ev = evaluator if evaluator is not None else Evaluator(cfg, self.run, gpu=evaluation[0])
        self.llm = None
        self.tokenizer = None
        self.results = []
        self.session = None
        self._search_evals = []
        self._tool_lock = threading.RLock()
        self.stop_event = None
        self.progress = None

    def model(self):
        """无参数；在本进程指定编号卡加载完整 BF16 模型及 tokenizer，返回 None。"""
        if self.llm is not None:
            return
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.generation_gpu)
        os.environ["VLLM_NO_USAGE_STATS"] = "1"
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        from transformers import AutoTokenizer
        from vllm import LLM

        self.model_started = time.monotonic()
        self.tokenizer = AutoTokenizer.from_pretrained(
            ROOT / self.cfg["model"], local_files_only=True
        )
        self.llm = LLM(
            model=str(ROOT / self.cfg["model"]),
            dtype="bfloat16",
            max_model_len=self.cfg["context_tokens"],
            max_num_seqs=1,
            gpu_memory_utilization=0.85,
            enforce_eager=True,
            enable_prefix_caching=False,
            limit_mm_per_prompt={"image": 0, "video": 0},
            seed=self.cfg["seed"],
        )

    def close_model(self):
        """无输入；退出前关闭本模型的推理引擎并释放引用，不影响其他进程的模型。"""
        if self.llm is not None:
            self.llm.llm_engine.engine_core.shutdown()
            self.llm = None
        self.tokenizer = None

    def generate(self, messages, tools, seed):
        """输入消息、是否启用工具和种子；返回文本与生成统计，由调用方写入统一轮次日志。"""
        from vllm import SamplingParams

        text = self.tokenizer.apply_chat_template(
            messages,
            tools=TOOLS if tools else None,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        capacity = generation_budget(self.cfg, tools, len(ids))
        if capacity <= 0:
            return {
                "stop_reason": "context_exhausted",
                "prompt_tokens": len(ids),
                "completion_tokens": 0,
                "text": "",
            }
        start = time.monotonic()
        res = self.llm.generate(
            [{"prompt_token_ids": ids}],
            SamplingParams(
                temperature=self.cfg["temperature"],
                top_p=self.cfg["top_p"],
                max_tokens=capacity,
                seed=seed,
            ),
            use_tqdm=False,
        )[0].outputs[0]
        result = {
            "text": res.text,
            "prompt_tokens": len(ids),
            "completion_tokens": len(res.token_ids),
            "finish_reason": res.finish_reason,
            "stop_reason": res.stop_reason,
            "seconds": time.monotonic() - start,
        }
        return result

    def prompt(self, task, agent):
        """输入题目及 Agent 标记；返回规则/SKILL 系统消息和题目/交付要求用户消息。"""
        binding = (ROOT / "upstream/CUDA-Agent/agent_workdir/binding.cpp").read_text()
        header = (ROOT / "upstream/CUDA-Agent/agent_workdir/binding_registry.h").read_text()
        interface = (
            f"\nFixed binding.cpp:\n```cpp\n{binding}\n```\n"
            f"Fixed binding_registry.h:\n```cpp\n{header}\n```\n"
            f'Reference task (read-only):\n```python\n{task["code"]}\n```'
        )
        skill = (ROOT / "upstream/CUDA-Agent/agent_workdir/SKILL.md").read_text()
        # 稳定的开发规则放在系统消息中，官方 SKILL 正文保持原样。
        # 本地覆盖说明置于 SKILL 之后，明确以实际硬件及沙箱权限为准。
        system = (
            CONTRACT
            + "\n\n## Official CUDA development SKILL\n"
            + skill
            + "\n\n## Local environment overrides and runtime rules\n"
            + "Local environment overrides SKILL examples: target sm_80, not sm_90; do not use sudo. "
            "The supplied binding and model/utils files are read-only. "
            "These CUDA restrictions and optimization guidelines apply to both modes.\n"
        )
        if agent:
            # 工具的固定使用方式属于运行规范；题目、预算和本次目标留在用户消息。
            system += (
                "Bash is persistent. Original commands bash utils/compile.sh, "
                "python3 -m utils.verification, and python3 -m utils.profiling dispatch jobs "
                "to the GPU sandbox. Before creating model_new.py, use "
                "python3 -m utils.profiling --reference-only or profile_reference. "
                "Read existing files before editing. No web/network tools.\n"
                "Bash has no GPU; use the provided utils commands for GPU work. "
                "For custom GPU tuning, write an auxiliary .py file and use "
                "python3 -m utils.run tune_kernel.py [args]. "
                "Scripts reuse compiled extensions when source is unchanged, and never count "
                "as scored solutions. Background Bash jobs use BashOutput/KillBash. "
                "Multiple tool calls execute sequentially; submit must be last. "
                "Fix the reported failure stage before further performance optimization.\n"
            )
            user = (
                "Use the tools to analyze the original performance first, write files, "
                "compile/check, improve, then submit. "
                f"Up to {self.cfg['max_turns']} interactions within {self.cfg['context_tokens']} tokens. "
                "Final scoring selects the fastest correct trajectory candidate. "
                "Search feedback uses five inputs; hidden results are never provided.\n"
                + interface
            )
        else:
            # 单次生成没有执行工具；在系统层解释 SKILL 的交互步骤，避免角色间冲突。
            system += (
                "The SKILL workflow is guidance only for this single mode: "
                "there are no tools or execution feedback.\n"
            )
            user = (
                'Generate one complete solution with no execution feedback. Return only a JSON object '
                '{"files":{"model_new.py":"...","kernels/kernel.cu":"...","kernels/kernel_binding.cpp":"..."}} '
                'after your reasoning. All file contents must be complete. model_new.py must define '
                'class ModelNew and import cuda_extension directly. Only model_new.py and source files '
                'directly under kernels/ are writable; omit binding.cpp, binding_registry.h, model.py, '
                'utils/, setup.py and other files. Escape source newlines inside JSON strings. '
                'Do not replace the JSON object with Markdown Python/CUDA code blocks.\n'
                "Deliver only the allowed candidate files; do not include auxiliary scripts/notebooks. "
                "Invalid/incomplete JSON is a failed single attempt; no automatic repair or extra generation.\n"
                + interface
            )
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

    def feedback(self, result):
        """输入完整评测结果；返回给模型的诊断摘要，完整日志仍保留在评测目录。"""
        keys = [
            "evaluation_id",
            "status",
            "correct",
            "reward",
            "timing_valid",
            "speedup_eager",
            "speedup_compile",
            "issues",
            "audit_warnings",
            "reference_only",
            "eager",
            "compile",
            "compile_first_call_seconds",
            "compile_correct",
            "profiles",
            "profile",
            "runtime_fallbacks",
            "error",
            "error_type",
            "failure_kind",
            "missing_module",
        ]
        feedback = {key: result[key] for key in keys if key in result}
        hints = []
        for section in ["compilation", "execution"]:
            detail = result.get(section, {})
            error = detail.get("error", "")
            if error:
                feedback["failure_stage"] = section
                feedback[section + "_error_type"] = detail.get("error_type")
                feedback[section + "_error"] = summarize_error(error)
                if "incompatible function arguments" in error:
                    hints.append(
                        "Check binding argument count/order/types. For an unnamed m.def binding, "
                        "use positional arguments; keyword config= requires py::arg(\"config\") in C++."
                    )
                if "state_dict" in error:
                    hints.append(
                        "Keep the reference nn.Module hierarchy, parameter names and buffers. "
                        "Do not rename nested weights or omit buffers; strict state_dict loading is required."
                    )
                if "illegal memory access" in error:
                    hints.append(
                        "Check launch sizes, tensor dimensions/strides, pointer types and every "
                        "read/write bound, including vectorized tails. Compilation success "
                        "does not establish memory safety."
                    )
                if "shape" in error and "invalid for input of size" in error:
                    hints.append(
                        "Derive the output allocation and reshape dimensions from the original operation. "
                        "view/reshape cannot change the tensor element count."
                    )
                if "cudnn" in error.lower():
                    hints.append(
                        "Match the installed cuDNN 9 header signatures and use typed enum constants "
                        "such as CUDNN_CROSS_CORRELATION and CUDNN_DATA_FLOAT."
                    )
        if result.get("issues"):
            feedback["failure_stage"] = "contract"
        if hints:
            feedback["diagnostic_hints"] = list(dict.fromkeys(hints))
        checks = result.get("checks", [])
        if checks:
            if any(not check["passed"] for check in checks) and "failure_stage" not in feedback:
                feedback["failure_stage"] = "correctness"
            feedback.update(
                checks_passed=sum(check["passed"] for check in checks),
                checks_total=len(checks),
                failed_checks=[check for check in checks if not check["passed"]][:5],
            )
        if result.get("timing"):
            feedback["timing"] = result["timing"].get("timing")
            feedback["profile"] = result["timing"].get("profile", [])[:20]
        if feedback.get("profiles"):
            feedback["profiles"] = {key: value[:20] for key, value in feedback["profiles"].items()}
        return feedback

    def tool_action(self, task, workspace, action, arguments=None):
        """输入动作和参数；串行处理前台/后台 GPU 作业，避免同一轨迹污染评测状态。"""
        with self._tool_lock:
            if action == "run_script":
                return self.ev.run_script(task, workspace, **(arguments or {}))
            if action == "profile_reference":
                result = self.ev.profile_reference(task)
            elif action == "compile":
                result = self.ev.compile_only(task, workspace)
            elif action in {"verify", "profile", "evaluate"}:
                result = self.ev.evaluate(task, workspace, measure=action != "verify")
                self._search_evals.append(result)
            else:
                raise ValueError("unsupported evaluation action")
            return self.feedback(result)

    def get_session(self, task, workspace):
        """输入当前任务及工作区；惰性创建该轨迹的持久终端，返回会话对象。"""
        if self.session is None:
            self.session = SandboxSession(
                self.cfg, workspace,
                lambda action, arguments=None: self.tool_action(task, workspace, action, arguments),
            )
        return self.session

    def shell(self, task, workspace, command, timeout=None):
        """输入命令与可选秒数；复用持久 CPU Bash，返回该命令输出和返回码。"""
        return self.get_session(task, workspace).run(command, timeout)

    def trajectory(self, task, mode, index):
        """输入题目、模式及稳定序号；运行一条轨迹，返回结果；无论成功失败均清理会话。"""
        self._search_evals = []
        try:
            return self._trajectory(task, mode, index)
        finally:
            if self.session is not None:
                self.session.close()
                self.session = None

    def _trajectory(self, task, mode, index):
        """输入固定题目、模式和序号；执行有预算的开发轨迹并隐藏验收，返回完整结果。"""
        started = time.monotonic()
        out = self.run / "trajectories" / task["task_id"] / mode
        work = out / "workspace"
        work.mkdir(parents=True, exist_ok=True)
        for name in ["binding.cpp", "binding_registry.h"]:
            shutil.copy2(ROOT / "upstream/CUDA-Agent/agent_workdir" / name, work / name)
        (work / "model.py").write_text(task["code"])
        shutil.copy2(ROOT / "upstream/CUDA-Agent/agent_workdir/SKILL.md", work / "SKILL.md")
        shutil.copytree(ROOT / "src/p1/agent_utils", work / "utils", dirs_exist_ok=True)
        toolbox = WorkspaceTools(
            work,
            lambda command, timeout=None: self.shell(task, work, command, timeout),
            lambda action: self.tool_action(task, work, action),
            background=lambda command, timeout: self.get_session(task, work).start_background(command, timeout),
            bash_output=lambda ident: self.get_session(task, work).background_output(ident),
            kill_bash=lambda ident: self.get_session(task, work).kill_background(ident),
        )
        messages = self.prompt(task, mode == "agent")
        tokens = 0
        prompt_tokens = 0
        tool_count = 0
        evals = self._search_evals
        termination = "turn_limit"
        events = []
        max_turns = 1 if mode == "single" else self.cfg["max_turns"]
        for turn in range(max_turns):
            if self.stop_event is not None and self.stop_event.is_set():
                raise InterruptedError("停止当前未完成轨迹，保留中间文件供恢复归档")
            if (
                self.cfg.get("trajectory_seconds") is not None
                and time.monotonic() - started >= self.cfg["trajectory_seconds"]
            ):
                termination = "time_limit"
                break
            if self.session is not None:
                try:
                    self.session.check()
                except ShellSessionClosed as error:
                    # 会话硬超时或显式取消只结束当前轨迹，避免继续几十轮无效调用。
                    termination = "shell_session_closed"
                    events.append({"error_type": type(error).__name__, "error": str(error)})
                    break
            turn_log = TurnLog(out, turn, mode)
            try:
                g = self.generate(
                    messages,
                    mode == "agent",
                    self.cfg["seed"] + index * 1000 + turn,
                )
                turn_log.generation(g)
                tokens += g["completion_tokens"]
                prompt_tokens += g["prompt_tokens"]
                if not g["text"]:
                    termination = g.get("stop_reason") or "empty_generation"
                    turn_log.record["status"] = termination
                    break
                # single 严格交付：解析失败直接结束，不补括号、不重新生成。
                if mode == "single":
                    try:
                        files = parse_single(g["text"])
                    except (ValueError, TypeError, KeyError) as error:
                        termination = (
                            "generation_limit" if g.get("finish_reason") == "length"
                            else "invalid_output_format"
                        )
                        turn_log.record["parse_error"] = str(error)
                    else:
                        try:
                            written = write_files(work, files)
                            turn_log.record["artifact_result"] = written
                            termination = "single_generated"
                        except Exception as error:
                            termination = "invalid_artifacts"
                            turn_log.record["error"] = str(error)
                            events.append({"error": str(error)})
                    turn_log.record["status"] = termination
                    break

                try:
                    actions = parse_calls(g["text"])
                    if any(name == "submit" for name, _ in actions[:-1]):
                        raise ValueError("submit must be the last tool call in an answer")
                except (ValueError, TypeError, KeyError) as error:
                    actions = []
                    turn_log.record["parse_error"] = str(error)

                if not actions:
                    # 无合法调用时，保留原文和解析结果，再给模型明确的格式修复提示。
                    messages.append(
                        {
                            "role": "assistant",
                            "content": (
                                g["text"].split("</think>", 1)[-1]
                                if "</think>" in g["text"] else ""
                            ),
                            "reasoning_content": g["text"].split("</think>")[0],
                        }
                    )
                    repair = (
                        "The entire call batch was not executed because its format is invalid. "
                        + turn_log.record.get("parse_error", "No valid tool call was found.") + " "
                        "Call one of the provided tools using "
                        "its native function/parameter format. Use Edit/MultiEdit for focused "
                        "patches instead of repeating complete files."
                    )
                    messages.append({"role": "user", "content": repair})
                    turn_log.record.update(
                        status="invalid_tool_call",
                        format_error=repair,
                    )
                    events.append({"turn": turn, "error": "invalid_tool_call"})
                    continue

                # 一轮可含多个调用。先完整登记，再逐个执行并按 ID 返回，绝不静默丢弃。
                calls = []
                for call_index, (name, args) in enumerate(actions):
                    ident = f"p1-{index}-{turn}-{call_index}"
                    calls.append({"id": ident, "type": "function",
                                  "function": {"name": name, "arguments": args}})
                    turn_log.call(ident, name, args)
                body = answer_body(g["text"])
                thought = g["text"].rsplit("</think>", 1)[0] if "</think>" in g["text"] else ""
                messages.append({
                    "role": "assistant", "content": body.split("<tool_call>", 1)[0],
                    "reasoning_content": thought.removeprefix("<think>"), "tool_calls": calls,
                })
                for call in calls:
                    if self.stop_event is not None and self.stop_event.is_set():
                        raise InterruptedError("停止本轮剩余调用，保留尚未执行的请求")
                    ident = call["id"]
                    name = call["function"]["name"]
                    args = call["function"]["arguments"]
                    turn_log.begin(ident)
                    tool_count += 1
                    if name == "submit":
                        feedback = {"status": "submitted"}
                        termination = "submitted"
                    else:
                        try:
                            feedback = toolbox.dispatch(name, args)
                        except InfrastructureError:
                            raise
                        except ShellSessionClosed as error:
                            feedback = {"error_type": type(error).__name__, "error": str(error)}
                            termination = "shell_session_closed"
                        except Exception as error:
                            feedback = {"error_type": type(error).__name__, "error": str(error)}
                    raw = json.dumps(feedback, ensure_ascii=False)
                    ids = self.tokenizer.encode(raw, add_special_tokens=False)
                    feedback_limit = self.cfg.get("tool_feedback_tokens")
                    was_truncated = feedback_limit is not None and len(ids) > feedback_limit
                    returned_text = self.tokenizer.decode(ids[:feedback_limit]) if was_truncated else raw
                    turn_log.response(feedback, returned_text, len(ids), was_truncated, ident=ident)
                    messages.append({"role": "tool", "tool_call_id": ident,
                                     "name": name, "content": returned_text})
                    events.append({"turn": turn, "tool": name, "tool_call_id": ident,
                                   "elapsed": time.monotonic() - started})
                    if termination == "shell_session_closed":
                        break
                if termination in {"submitted", "shell_session_closed"}:
                    turn_log.record["status"] = termination
                    break
            except BaseException as error:
                # 基础设施故障/中断也保存当前轮，不能因没有工具返回而丢失模型输出。
                turn_log.record.update(
                    status=(
                        "interrupted" if isinstance(error, (InterruptedError, KeyboardInterrupt))
                        else "error"
                    ),
                    error_type=type(error).__name__,
                    error=str(error),
                )
                raise
            finally:
                turn_log.finish()
                if mode == "agent":
                    dump(out / "messages.json", messages)
        # 终止剩余后台 CPU 任务，并等已派发 GPU 作业收尾，再冻结最终候选。
        if self.session is not None:
            self.session.close()
            # CPU 会话已明确终止时仍要检查 GPU 调度错误，但不再次抛已记录的会话错误。
            self.session.check(shell=False)
            self.session = None
        search_seconds = time.monotonic() - started
        if self.stop_event is not None and self.stop_event.is_set():
            raise InterruptedError("停止后不再启动最终候选及隐藏验收")
        # 最终工作区也先验收；随后在所有测时有效的正确候选中按 compile 加速排序。
        final_work = self.ev.evaluate(task, work)
        measured, hidden = self.select_candidate(task, evals + [final_work])
        final_correct = bool(hidden and hidden.get("correct"))
        best = max(
            (item for item in evals if item.get("correct") and item.get("timing_valid")),
            key=lambda item: item.get("speedup_compile", 0),
            default=None,
        )
        if measured is None:
            measured = final_work
        result = {
            "selection_rule": "best_correct_speedup_compile",
            "final_work_evaluation": final_work,
            "task_id": task["task_id"],
            "mode": mode,
            "generation_gpu": self.generation_gpu,
            "task_index": index,
            "termination": termination,
            "correct": final_correct,
            "feedback_correct": measured.get("correct", False),
            "timing_valid": measured.get("timing_valid", False),
            "speedup_eager": measured.get("speedup_eager") if final_correct else None,
            "speedup_compile": measured.get("speedup_compile") if final_correct else None,
            "final_evaluation": measured,
            "hidden_evaluation": hidden,
            "best_search_evaluation": best,
            "generated_tokens": tokens,
            "prefill_tokens_sum": prompt_tokens,
            "tool_calls": tool_count,
            "search_evaluations": len(evals),
            "search_seconds": search_seconds,
            "total_seconds": time.monotonic() - started,
            "events": events,
        }
        dump(out / "result.json", result)
        return result

    def select_candidate(self, task, candidates):
        """输入题目及反馈候选；按速度降序隐藏验收，返回最快合格候选和隐藏结果。

        同源码只验一次；隐藏失败不反馈给模型，继续检查下一个候选。无正确且
        测时有效的候选时返回 (None, None)，不把隐藏结果用于继续搜索。
        """
        ordered = sorted(
            (item for item in candidates if item.get("correct") and item.get("timing_valid")),
            key=lambda item: item.get("speedup_compile") or 0,
            reverse=True,
        )
        seen = set()
        last_hidden = None
        for item in ordered:
            digest = item["candidate_hash"]
            if digest in seen:
                continue
            seen.add(digest)
            workspace = self.run / "evaluations" / item["evaluation_id"] / "candidate"
            hidden = self.ev.evaluate(task, workspace, final=True)
            last_hidden = hidden
            if hidden.get("correct"):
                return item, hidden
        return None, last_hidden

    def save_result_summaries(self):
        """将内存中的逐题结果写为摘要和详情路径；完整轨迹继续保存在各自 result.json。

        只保留标量字段，省去 final_evaluation、hidden_evaluation、events 等嵌套副本。
        详情路径相对于本次 run 根目录；不修改内存结果，保证后续统计仍可使用完整数据。
        """
        rows = []
        for result in self.results:
            row = {
                key: value
                for key, value in result.items()
                if value is None or isinstance(value, (str, int, float, bool))
            }
            if result["mode"] == "reference":
                detail = self.run / "references" / result["task_id"] / "preflight.json"
            else:
                detail = self.run / "trajectories" / result["task_id"] / result["mode"] / "result.json"
            row["detail_path"] = detail.relative_to(self.run).as_posix()
            rows.append(row)
        dump(self.run / "results.json", rows)
        # 结果原子落盘后再推进显示；子进程不创建进度条，也不写根汇总。
        progress = getattr(self, "progress", None)
        if progress is not None:
            progress.sync(self.results)

    def make_manifest(self, tasks):
        """输入固定题单；返回配置、源码指纹和北京时间运行清单，不加载模型。"""
        return {
            "config": self.cfg,
            "started_at": beijing_now().isoformat(timespec="seconds"),
            "git_head": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            "gpu_snapshot": subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=index,uuid,name,memory.used,utilization.gpu,temperature.gpu,power.draw,clocks.sm",
                    "--format=csv",
                ],
                text=True,
            ),
            "code_sha256": {
                str(p.relative_to(ROOT)): sha(p.read_bytes())
                for p in (ROOT / "src/p1").rglob("*")
                if p.is_file() and p.suffix in {".py", ".sh"}
            },
            "task_count": len(tasks),
            "task_ids": [t["task_id"] for t in tasks],
            "protocol": self.cfg.get("protocol_version", "controlled-tf32"),
            "status": "running",
            "no_training": True,
        }

    def finish_summary(self, tasks, eligible, manifest, *, costs=None, complete=True):
        """输入题单、有效题、清单及可选并行成本；统一统计并写结果，返回 None。"""
        summary = {
            "selected_tasks": len(tasks),
            "reference_valid": len(eligible),
            "invalid_tasks": len(tasks) - len(eligible),
            "modes": {},
        }
        for mode in ["single", "agent"]:
            items = [x for x in self.results if x["mode"] == mode]
            correct = [x for x in items if x["correct"]]
            timed = [x for x in correct if x["timing_valid"]]
            summary["modes"][mode] = {
                "attempted": len(items),
                "correct": len(correct),
                "pass_rate_on_reference_valid": len(correct) / len(eligible) if eligible else None,
                "pass_rate_on_selected": len(correct) / len(tasks) if tasks else None,
                "timing_valid_correct": len(timed),
                "faster_eager": sum(x["speedup_eager"] > 1 for x in timed),
                "faster_compile": sum(x["speedup_compile"] > 1 for x in timed),
                "gm_speedup_eager": (
                    math.exp(statistics.mean(math.log(x["speedup_eager"]) for x in timed))
                    if timed
                    else None
                ),
                "gm_speedup_compile": (
                    math.exp(statistics.mean(math.log(x["speedup_compile"]) for x in timed))
                    if timed
                    else None
                ),
                "generated_tokens": sum(x["generated_tokens"] for x in items),
                "prefill_tokens_sum": sum(x["prefill_tokens_sum"] for x in items),
                "tool_calls": sum(x["tool_calls"] for x in items),
                "search_evaluations": sum(x["search_evaluations"] for x in items),
                "search_seconds": sum(x["search_seconds"] for x in items),
                "total_seconds": sum(x["total_seconds"] for x in items),
            }
        if costs is None:
            costs = {
                "container_seconds_this_invocation": self.ev.seconds,
                "compiles_this_invocation": self.ev.compiles,
                "generation_gpu_reserved_seconds": (
                    time.monotonic() - self.model_started if hasattr(self, "model_started") else 0
                ),
            }
        summary.update(costs)
        summary["generation_gpu_reserved_hours"] = summary["generation_gpu_reserved_seconds"] / 3600
        dump(self.run / "summary.json", summary)
        manifest.update(
            status="complete" if complete else "interrupted",
            finished_at=beijing_now().isoformat(timespec="seconds")
        )
        dump(self.run / "manifest.json", manifest)
        progress = getattr(self, "progress", None)
        if progress is not None:
            progress.close()
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)

    def execute(self, prepare_only=False):
        """输入是否仅预检；主控制器显示进度，并在正常结束或异常退出时关闭进度条。"""
        self.progress = PilotProgress()
        try:
            return self._execute(prepare_only)
        finally:
            self.progress.close()
            self.progress = None

    def _execute(self, prepare_only=False):
        """输入是否仅预检；保存题单、评测两种模式并汇总，不改变原有数据和预算。"""
        generation, evaluation = gpu_roles(self.cfg)
        if len(generation) > 1 or len(evaluation) > 1:
            from .helper.pilot_parallel import execute_parallel
            return execute_parallel(self, prepare_only, self.stop_event)
        task_file = self.run / "tasks.json"
        tasks = json.loads(
            (task_file if task_file.exists() else ROOT / "data/processed/pilot.json").read_text()
        )
        dump(task_file, tasks)
        manifest = self.make_manifest(tasks)
        dump(self.run / "manifest.json", manifest)
        eligible = []
        self.progress.start_references(tasks)
        for task in tasks:
            if self.stop_event is not None and self.stop_event.is_set():
                raise InterruptedError("停止参考预检")
            dest = self.run / "references" / task["task_id"] / "preflight.json"
            pre = json.loads(dest.read_text()) if dest.exists() else None
            # 只有成功或明确的任务拒绝才是终态；环境故障不能缓存成“题目无效”。
            if pre is None or pre.get("status") not in {"ok", "task_invalid"}:
                archive_incomplete(self.run, task["task_id"])
                pre = self.ev.prepare(task)
            # 新预检索引由 Evaluator.prepare 写入；完整终态缓存只读复用。
            if pre["status"] == "ok":
                eligible.append(task)
            else:
                self.results.append(
                    {
                        "task_id": task["task_id"],
                        "mode": "reference",
                        "status": "task_invalid",
                        "detail_path": dest.relative_to(self.run).as_posix(),
                    }
                )
            self.progress.reference_done(task["task_id"], pre["status"])
        # 预检失败也在根目录保留逐题摘要，仅准备模式不丢失状态记录。
        self.save_result_summaries()
        # 预检拒绝的题不会进入 single/agent；状态保存后即可释放其参数和缓存。
        for result in self.results:
            if result["mode"] == "reference" and result["status"] == "task_invalid":
                cleanup_reference_artifacts(self.run, result["task_id"])
        if prepare_only:
            manifest.update(status="prepared", updated_at=beijing_now().isoformat(timespec="seconds"))
            dump(self.run / "manifest.json", manifest)
            return
        # 只读已有最终轨迹，续跑时先恢复进度与统计，再等待模型加载。
        completed = []
        for task in eligible:
            for mode in ("single", "agent"):
                result_path = self.run / "trajectories" / task["task_id"] / mode / "result.json"
                if result_path.exists():
                    completed.append(json.loads(result_path.read_text()))
        self.progress.start_modes(eligible, completed)
        del completed
        needs_model = any(
            not (self.run / "trajectories" / task["task_id"] / mode / "result.json").exists()
            for task in eligible for mode in ("single", "agent")
        )
        if needs_model:
            self.model()
        for index, task in enumerate(eligible):
            for mode in ["single", "agent"]:
                if self.stop_event is not None and self.stop_event.is_set():
                    raise InterruptedError("停止派发新轨迹")
                dest = self.run / "trajectories" / task["task_id"] / mode / "result.json"
                if dest.exists():
                    res = json.loads(dest.read_text())
                else:
                    self.progress.write("START", task["task_id"], mode)
                    # 与并行路径一致：中断轨迹先归档，再从第零轮重新开始，保留旧日志。
                    archive_partial_trajectory(self.run, task["task_id"], mode)
                    res = self.trajectory(task, mode, index)
                self.results.append(res)
                self.save_result_summaries()
            # single 和 agent 共用参数，须等两条轨迹及隐藏验收结束且摘要保存后再清理。
            # 轨迹中断或仅 --prepare-only 时不会到这里，保留后续运行需要的参数。
            cleanup_reference_artifacts(self.run, task["task_id"])
        self.finish_summary(tasks, eligible, manifest)


def validate_resume(cfg, output, resume):
    """输入配置、目录与续跑标志；检查配置及已有源码指纹，GPU 调度字段可换。"""
    manifest = output / "manifest.json"
    if not manifest.exists():
        if resume:
            raise RuntimeError("--resume 指定目录没有 manifest.json")
        if output.exists() and any(output.iterdir()):
            raise RuntimeError("输出目录非空且没有清单，请使用新的运行目录")
        return
    if not resume:
        raise RuntimeError("已有运行目录；恢复请显式传入 --resume")
    saved = json.loads(manifest.read_text())
    if saved.get("mode") == "reference_only":
        raise RuntimeError("纯基线目录不能作为 Qwen pilot 续跑，请使用新的运行目录")
    previous = saved["config"]
    scheduling = {"generation_gpu", "evaluation_gpu"}
    protocol = lambda value: {key: item for key, item in value.items() if key not in scheduling}
    if protocol(previous) != protocol(cfg):
        raise RuntimeError("模型、预算或评测配置与旧 run 不同，请使用新的运行目录")
    saved_hashes = saved.get("code_sha256")
    if saved_hashes:
        current_hashes = {
            str(path.relative_to(ROOT)): sha(path.read_bytes())
            for path in (ROOT / "src/p1").rglob("*")
            if path.is_file() and path.suffix in {".py", ".sh"}
        }
        if current_hashes != saved_hashes:
            raise RuntimeError(
                "P1 源码、提示词或日志实现与原 run 不同，不能混合新旧结果；请使用新的运行目录"
            )


def check_gpu_devices(devices):
    """输入将使用的物理编号；只读查询显存占用，编号不存在或占用超 100 MiB 时拒绝。"""
    rows = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
        text=True, timeout=30,
    )
    available = {int(parts[0]): int(parts[1])
                 for parts in (line.split(",") for line in rows.splitlines())}
    for gpu in devices:
        if gpu not in available:
            raise RuntimeError(f"GPU 编号不存在：{gpu}")
        if available[gpu] > 60000:
            raise RuntimeError(f"GPU {gpu} 已占用 {available[gpu]} MiB；不会停止现有进程")


def main(argv=None):
    """输入可选 CLI 参数列表；预览或持目录/卡号锁运行，正常返回 0，安全停止返回 130。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="runs/p1/pilot-train-v3-seed42")
    parser.add_argument("--task-id", help="只运行指定 task_id；不修改公共 pilot 题单")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="只读预览，不查询或启动 GPU")
    args = parser.parse_args(argv)
    cfg = json.loads((ROOT / "configs/p1-pilot.json").read_text())
    generation, evaluation = gpu_roles(cfg)
    output = (ROOT / args.run).resolve()
    validate_resume(cfg, output, args.resume)
    selected_tasks = None
    if args.task_id:
        # 新运行从公共 pilot 中选题；续跑只能确认原题单，不得悄悄删掉其他题。
        task_path = output / "tasks.json"
        source_tasks = json.loads(
            (task_path if task_path.exists() else ROOT / "data/processed/pilot.json").read_text()
        )
        selected_tasks = [task for task in source_tasks if task["task_id"] == args.task_id]
        if len(selected_tasks) != 1:
            parser.error(f"题单中未找到唯一的 task_id：{args.task_id}")
        if args.resume and len(source_tasks) != 1:
            parser.error("--task-id 不能将已有多题运行改为单题；请使用新的 --run 目录")
    if args.dry_run:
        task_path = output / "tasks.json"
        tasks = selected_tasks if selected_tasks is not None else json.loads(
            (task_path if task_path.exists() else ROOT / "data/processed/pilot.json").read_text()
        )
        print(json.dumps({"run": str(output), "resume": args.resume,
                          "prepare_only": args.prepare_only, "task_count": len(tasks),
                          "generation_gpus": generation, "evaluation_gpus": evaluation,
                          "generation_strategy": "one_model_replica_per_gpu"},
                         ensure_ascii=False, indent=2))
        return 0
    if shutil.disk_usage(ROOT).free < 80 * 1024**3:
        raise RuntimeError("项目所在磁盘剩余空间不足 80 GiB")
    lock_root = ROOT / ".runtime/locks"
    # 与纯基线共用卡号锁，防止同项目的两种入口同时占用一张卡。
    devices = evaluation if args.prepare_only else sorted(set(generation + evaluation))
    with ExitStack() as locks:
        locks.enter_context(exclusive_lock(run_lock_path(output, lock_root)))
        if (output / "controller.lock").exists():
            locks.enter_context(exclusive_lock(output / "controller.lock"))
        validate_resume(cfg, output, args.resume)
        if args.resume:
            guard_active_containers(output)
        for gpu in devices:
            locks.enter_context(exclusive_lock(lock_root / f"p1-baseline-gpu-{gpu}.lock"))
        check_gpu_devices(devices)
        stop = multiprocessing.get_context("spawn").Event()
        previous_signals = {}
        for signum in (signal.SIGINT, signal.SIGTERM):
            previous_signals[signum] = signal.signal(signum, lambda number, frame: stop.set())
        pilot = None
        try:
            if selected_tasks is not None and not args.resume:
                # 取得运行锁并完成环境检查后，只在本次 run 中冻结选出的题目。
                # 串行与多卡入口都会读取这份 tasks.json，公共 pilot.json 保持不变。
                dump(output / "tasks.json", selected_tasks)
            pilot = Pilot(cfg, output)
            pilot.stop_event = stop
            pilot.execute(args.prepare_only)
            return 130 if stop.is_set() else 0
        except InterruptedError:
            path = output / "manifest.json"
            if path.exists():
                manifest = json.loads(path.read_text())
                manifest.update(status="interrupted", updated_at=beijing_now().isoformat(timespec="seconds"))
                dump(path, manifest)
            return 130
        finally:
            try:
                if pilot is not None:
                    pilot.close_model()
            finally:
                for signum, handler in previous_signals.items():
                    signal.signal(signum, handler)


if __name__ == "__main__":
    raise SystemExit(main())

"""P3 的工具执行环境和可信终局评分，复用已有 Docker/Evaluator。

由原生 ToolAgentLoop 的薄适配层创建；每条轨迹独立目录和持久 Bash。
输入文件工具/GPU 工具动作，输出诊断；终局只选择本轨迹已正确测时的最佳源码。
不解析最终文本、不运行隐藏验收、不改写候选、不把基础设施故障记成负奖励。
"""

from hashlib import sha256
import json
from pathlib import Path
import shutil
import threading

from p1.evaluation import InfrastructureError, ReferenceTaskError, dump
from p1.helper.baseline_state import beijing_now, cleanup_reference_artifacts
from p1.helper.pilot_parallel import EvaluationPool
from p1.run import Pilot
from p1.sandbox import SandboxSession
from p1.tools import WorkspaceTools
from p2.failures import RecoverableSampleError
from p2.reward import P2Evaluator

from .data import ROOT


class KernelEvaluator(P2Evaluator):
    """复用现有参考有限重试；每条轨迹只准备一次，不自动增加初始基线测量。"""

    def prepare(self, task):
        """输入原题；只准备可信参数，参考性能由显式分析工具或后测基线计算。"""
        ref = self.run / 'references' / task['task_id']
        target = ref / 'cases/prepare_result.json'
        if not target.exists():
            ref.mkdir(parents=True, exist_ok=True)
            (ref / 'model.py').write_text(task['code'])
            with self.operation():
                result = self.container('reference_prepare', ref / 'cases', reference=ref, timeout=300)
        else:
            result = json.loads(target.read_text())
        if result['status'] != 'ok':
            raise RecoverableSampleError('reference_prepare failed', target, quarantine=True)
        return ref

    def profile_reference(self, task):
        """输入原题；按需测一次原始 eager/compile 并提供 profile，可复用本轨迹结果。"""
        ref = self.prepare(task)
        target = ref / 'initial-bench/bench_result.json'
        if target.exists():
            return json.loads(target.read_text())
        with self.operation():
            result = self.container(
                'reference_profile', ref / 'initial-bench', reference=ref,
                inputs=ref / 'cases/input-spec', timeout=300,
            )
        if result['status'] != 'ok':
            raise RecoverableSampleError('reference profiling failed', target, quarantine=True)
        return dict(result, reference_only=True)


class Workspace:
    """单条轨迹的文件、Bash、评分缓存及最佳候选状态，不持有模型或优化器。"""

    def __init__(self, config, task, uid, session_id):
        """输入完整配置、原题及框架组/轨迹编号；创建独立工作区，不启动 GPU。"""
        self.config = config
        self.task = task
        self.uid = uid
        self.directory = Path(config.p3.run) / 'trajectories' / uid / str(session_id)
        self.work = self.directory / 'workspace'
        self.work.mkdir(parents=True, exist_ok=True)
        official = ROOT / 'upstream/CUDA-Agent/agent_workdir'
        for name in ('binding.cpp', 'binding_registry.h', 'SKILL.md'):
            shutil.copyfile(official / name, self.work / name)
        (self.work / 'model.py').write_text(task['code'])
        shutil.copytree(ROOT / 'src/p1/agent_utils', self.work / 'utils', dirs_exist_ok=True)
        from omegaconf import OmegaConf
        self.kernel = OmegaConf.to_container(config.p3.kernel, resolve=True)
        # 没有隐藏验收，也不为隐藏种子构建额外参考参数。
        self.kernel.update(hidden_init_seeds=[], hidden_cases_per_init=0)
        pool = EvaluationPool(self.kernel['evaluation_gpu'], Path(config.p3.run) / 'scheduler', cfg=self.kernel)
        dump(self.directory / 'kernel/worker-config.json', self.kernel)
        self.evaluator = KernelEvaluator(self.kernel, self.directory / 'kernel', pool)
        self.session = None
        self.fatal = None
        self.submitted = False
        self.cache = {}
        self.gpu_lock = threading.RLock()
        self.toolbox = WorkspaceTools(
            self.work, self.shell, self.action,
            background=lambda command, timeout: self.get_session().start_background(command, timeout),
            bash_output=lambda ident: self.get_session().background_output(ident),
            kill_bash=lambda ident: self.get_session().kill_background(ident),
        )

    def record(self, event):
        """输入一条对话/工具事件；追加北京时间 JSONL，不重复落盘 token 数组。"""
        with (self.directory / 'trajectory.jsonl').open('a') as stream:
            stream.write(json.dumps(dict(at=beijing_now().isoformat(), **event), ensure_ascii=False) + '\n')

    def get_session(self):
        """无输入；惰性创建并复用持久 CPU Bash，返回会话。"""
        if self.session is None:
            self.session = SandboxSession(self.kernel, self.work, self.action)
        return self.session

    def shell(self, command, timeout=None):
        """输入命令/等待秒数；返回真实持久 Bash 输出，不在宿主执行模型命令。"""
        return self.get_session().run(command, timeout)

    def source_key(self):
        """无输入；返回候选源文件内容摘要；遇到链接时禁用缓存，让原审查明确拒绝。"""
        if any(path.is_symlink() for path in self.work.rglob('*')):
            return None
        sources = [self.work / 'model_new.py'] + sorted((self.work / 'kernels').glob('*'))
        digest = sha256()
        for path in sources:
            if path.is_file() and path.suffix in {'.py', '.cu', '.cpp', '.h', '.cuh'}:
                digest.update(str(path.relative_to(self.work)).encode())
                digest.update(path.read_bytes())
        return digest.hexdigest()

    def evaluate(self, measure=True):
        """输入是否测时；复用同源码结果，已正确验算的快照可直接晋升为正式测时。"""
        try:
            return self._evaluate(measure)
        except ReferenceTaskError as error:
            # 参考侧有限重试已由共享控制器执行；连续失败隔离整组，不算模型答错。
            raise RecoverableSampleError(str(error), error.detail_path, quarantine=True) from error

    def _evaluate(self, measure):
        """输入测时标记；在同一可信参数、源码快照及编译产物上复用阶段结果。"""
        self.evaluator.prepare(self.task)
        key = self.source_key()
        previous = self.cache.get(key) if key else None
        if previous is not None:
            if not measure or previous.get('scored') or not previous.get('correct'):
                return previous
            # 五组验算已通过且源码未变，复用同一只读快照与可信编译缓存，不再验算一遍。
            ev = self.evaluator.run / 'evaluations' / previous['evaluation_id']
            snap = ev / 'candidate'
            cache_key = previous['compilation']['cache_key']
            build = ROOT / '.runtime/cache/p1-builds' / cache_key / 'build'
            with self.evaluator.operation():
                result = self.evaluator._measure_candidate(self.task, snap, build, ev, dict(previous), False, 0)
        else:
            result = self.evaluator.evaluate(self.task, self.work, measure=measure)
        if result.get('failure_kind') == 'environment':
            # 无效计时等环境结果不能在终局被默认为模型 -1；阶段证据由原控制器保存。
            detail = self.evaluator.run / 'evaluations' / result['evaluation_id'] / 'result.json'
            raise InfrastructureError(f'Invalid environment result: {result.get("reward_reason")}; {detail}')
        if key:
            self.cache[key] = result
        return result

    def action(self, name, arguments=None):
        """输入固定 GPU 动作及脚本参数；返回诊断，环境故障由调用方保留并向训练层抛出。"""
        with self.gpu_lock:
            if name == 'profile_reference':
                result = self.evaluator.profile_reference(self.task)
            elif name == 'compile':
                result = self.evaluator.compile_only(self.task, self.work)
            elif name == 'run_script':
                return self.evaluator.run_script(self.task, self.work, **(arguments or {}))
            elif name in {'verify', 'profile', 'evaluate'}:
                result = self.evaluate(measure=name != 'verify')
            else:
                raise ValueError(f'Unknown GPU action: {name}')
            # 复用既有诊断摘要，不把 Pilot 的生成循环或模型服务搬入训练。
            return Pilot.feedback(self, result)

    def dispatch(self, name, arguments):
        """输入工具名/参数；调用原有文件工具，submit 只标记结束，不接受模型自报分数。"""
        if name == 'submit':
            self.submitted = True
            return {'status': 'submitted'}
        return self.toolbox.dispatch(name, arguments)

    def finish(self):
        """无输入；验收尚未评分的最终源码，选择最大 compile 加速比，保存最佳快照及结果。"""
        if self.session is not None:
            self.session.close()
            self.session.check(shell=False)
            self.session = None
        if self.fatal is not None:
            raise self.fatal
        # 同源码已评分时命中缓存；文件实际变化才重新编译/验证/测时。
        if (self.work / 'model_new.py').exists():
            self.evaluate()
        candidates = [item for item in self.cache.values() if item.get('correct') and item.get('timing_valid')]
        best = max(candidates, key=lambda item: item['speedup_compile'], default=None)
        result = {'task_id': self.task['task_id'], 'reward': -1, 'correct': False}
        if best is not None:
            source = self.evaluator.run / 'evaluations' / best['evaluation_id'] / 'candidate'
            shutil.copytree(source, self.directory / 'best', dirs_exist_ok=True)
            result.update(
                reward=best['reward'], correct=True, evaluation_id=best['evaluation_id'],
                speedup_eager=best['speedup_eager'], speedup_compile=best['speedup_compile'],
                candidate_hash=best['candidate_hash'], candidate_path='best',
            )
        result['at'] = beijing_now().isoformat()
        dump(self.directory / 'result.json', result)
        return result

    def close(self):
        """无输入；结束本轨迹后台任务并清理输入规格/参考编译缓存，保留源码和评分证据。"""
        if self.session is not None:
            self.session.close()
            self.session = None
        cleanup_reference_artifacts(self.evaluator.run, self.task['task_id'])

"""P1 多卡调度：每张生成卡一个模型副本，共享完整评测作业的 GPU 池。

由 p1.run 调用，不单独启动。配置的 GPU 字段接受整数或整数列表。
父进程独占写根汇总；子进程只写各自题目的轨迹和原有 evaluations 数字目录。
文件锁置于 .runtime/locks，所有卡只读同一份 worker-config.json。
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
import fcntl
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
from queue import Empty
import signal
import threading
import time
import traceback
import uuid

from .baseline_state import (
    archive_incomplete, atomic_json, beijing_now, cleanup_reference_artifacts,
)
from ..evaluation import Evaluator

LOCK_ROOT = Path(__file__).resolve().parents[3] / ".runtime/locks"
MODES = ("single", "agent")


def gpu_numbers(value, name):
    """输入单个编号或列表和字段名；返回非空、无重复的非负编号列表，非法输入抛错。"""
    devices = [value] if type(value) is int else value
    if (not isinstance(devices, list) or not devices
            or any(type(device) is not int or device < 0 for device in devices)
            or len(devices) != len(set(devices))):
        raise ValueError(f"{name} 必须为非负 GPU 编号或无重复的编号列表")
    return list(devices)


def gpu_roles(cfg):
    """输入配置；返回生成/评测编号列表，拒绝两种用途共享同一卡以免干扰测时。"""
    generation = gpu_numbers(cfg.get("generation_gpu", 2), "generation_gpu")
    evaluation = gpu_numbers(cfg.get("evaluation_gpu", 3), "evaluation_gpu")
    if set(generation) & set(evaluation):
        raise ValueError("generation_gpu 与 evaluation_gpu 不能包含相同 GPU")
    return generation, evaluation


def scheduling_directory(run):
    """输入 run；返回该运行的宿主调度锁目录，保持 run 内原有输出布局。"""
    identity = hashlib.sha256(str(Path(run).resolve()).encode()).hexdigest()
    return LOCK_ROOT / f"p1-pilot-{identity}"


@contextmanager
def blocking_lock(path):
    """输入锁路径；跨进程等待并持锁，异常或进程退出自动释放，不删除锁文件。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


class EvaluationPool:
    """用文件锁租用整张评测卡；不存放令牌，进程崩溃不会永久丢失 GPU 配额。"""

    def __init__(self, devices, directory, stop=None):
        """输入编号列表、锁目录和可选停止事件；初始化轮询起点，不启动 GPU。"""
        self.devices = gpu_numbers(devices, "evaluation_gpu")
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.stop = stop
        self.offset = 0

    @contextmanager
    def lease(self):
        """无输入；等待空闲卡并返回物理编号，退出时释放；停止时不再启动新作业。"""
        while True:
            if self.stop is not None and self.stop.is_set():
                raise InterruptedError("P1 已请求停止，取消等待评测卡")
            order = self.devices[self.offset:] + self.devices[:self.offset]
            for gpu in order:
                stream = (self.directory / f"evaluation-gpu-{gpu}.lock").open("a+")
                try:
                    fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    stream.close()
                    continue
                self.offset = (self.devices.index(gpu) + 1) % len(self.devices)
                try:
                    yield gpu
                finally:
                    fcntl.flock(stream, fcntl.LOCK_UN)
                    stream.close()
                return
            time.sleep(0.05)


class ParallelEvaluator(Evaluator):
    """给原评测器增加卡池和快照编号锁；数值检查及测时仍使用原 Evaluator。"""

    def __init__(self, cfg, run, pool):
        """输入公共配置、run 和卡池；只读复用父进程写好的公共 worker 配置。"""
        super().__init__(cfg, run, gpu=pool.devices[0], write_config=False)
        self.pool = pool
        self._operation_lock = threading.RLock()
        self._leased = False

    @contextmanager
    def operation(self):
        """无输入；完整作业持有同一张卡，嵌套 prepare/重试复用已有租约。"""
        with self._operation_lock:
            if self._leased:
                yield
                return
            with self.pool.lease() as gpu:
                self.gpu = gpu
                self._leased = True
                try:
                    yield
                finally:
                    self._leased = False

    def prepare(self, task):
        """输入任务；参数预检及初始基线作为一个作业使用同一卡，返回原预检索引。"""
        with self.operation():
            return super().prepare(task)

    def profile_reference(self, task):
        """输入任务；参考性能分析使用一个租约，返回原性能结果。"""
        with self.operation():
            return super().profile_reference(task)

    def compile_only(self, task, workspace):
        """输入任务/工作区；隔离编译期间持卡，返回原编译反馈。"""
        with self.operation():
            return super().compile_only(task, workspace)

    def evaluate(self, task, workspace, *args, **kwargs):
        """输入任务、候选和原评测选项；整组编译/检查/前后测时固定使用一张卡。"""
        with self.operation():
            return super().evaluate(task, workspace, *args, **kwargs)

    def snapshot(self, workspace):
        """输入工作区；跨进程串行分配 evaluations 编号，返回原快照与审查结果。"""
        with blocking_lock(self.pool.directory / "evaluation-id.lock"):
            self.counter = max(
                (int(path.name) for path in (self.run / "evaluations").iterdir()
                 if path.is_dir() and path.name.isdigit()),
                default=0,
            ) if (self.run / "evaluations").exists() else 0
            return super().snapshot(workspace)


def saved_trajectory(run, task_id, mode):
    """输入 run、题号、模式；返回完整结果的标量摘要，缺文件返回 None，损坏时拒绝覆盖。"""
    path = Path(run) / "trajectories" / task_id / mode / "result.json"
    if not path.exists():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("task_id") != task_id or value.get("mode") != mode or "correct" not in value:
        raise ValueError(f"轨迹结果身份不匹配：{path}")
    return compact_result(value)


def compact_result(value):
    """输入完整轨迹结果；返回根汇总所需标量，避免跨进程复制完整候选和工具日志。"""
    return {key: item for key, item in value.items()
            if item is None or isinstance(item, (str, int, float, bool))}


def archive_partial_trajectory(run, task_id, mode):
    """输入待重做轨迹；归档没有最终结果的旧工作区，返回 None，避免遗留源码影响新生成。"""
    old = Path(run) / "trajectories" / task_id / mode
    if not old.exists():
        return
    if (old / "result.json").exists():
        raise ValueError("不能归档已有最终结果的轨迹")
    stamp = beijing_now().strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:8]
    target = Path(run) / "interrupted" / stamp / task_id / mode
    target.parent.mkdir(parents=True, exist_ok=True)
    old.rename(target)


def generation_worker(cfg, run, gpu, directory, jobs, events, stop, pilot_factory=None):
    """子进程入口：输入公共配置、卡号和队列；常驻一个模型副本，输出逐轨迹事件及成本。

    整题派发保证同题 single/agent 不并发；种子使用父进程给出的固定题目 index。
    该进程不写根汇总或清理参考参数，父进程收到完成事件并保存后负责清理。
    pilot_factory 仅供 CPU 测试替换真实模型，不改变生产调用方式。
    """
    # 独立会话避免终端 Ctrl+C 直接打断 vLLM；停止请求由父进程事件传递。
    if pilot_factory is None:
        os.setsid()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    if threading.current_thread() is threading.main_thread():
        signal.signal(signal.SIGTERM, lambda signum, frame: stop.set())
    if pilot_factory is None:
        from ..run import Pilot
        pilot_factory = Pilot
    pilot = None
    finished = 0
    try:
        pool = EvaluationPool(gpu_roles(cfg)[1], directory, stop)
        evaluator = ParallelEvaluator(cfg, run, pool)
        pilot = pilot_factory(cfg, Path(run), generation_gpu=gpu, evaluator=evaluator)
        pilot.stop_event = stop
        while not stop.is_set():
            job = jobs.get()
            if job is None:
                break
            index, task = job
            for mode in MODES:
                if stop.is_set():
                    raise InterruptedError("P1 已请求停止")
                row = saved_trajectory(run, task["task_id"], mode)
                if row is None:
                    archive_partial_trajectory(run, task["task_id"], mode)
                    pilot.model()
                    row = compact_result(pilot.trajectory(task, mode, index))
                events.put(("result", row))
            finished += 1
            events.put(("task_complete", task["task_id"]))
    except InterruptedError:
        stop.set()
    except BaseException:
        stop.set()
        events.put(("error", {"generation_gpu": gpu, "traceback": traceback.format_exc()}))
    finally:
        reserved = (time.monotonic() - pilot.model_started
                    if pilot is not None and hasattr(pilot, "model_started") else 0)
        # 先释放模型引擎，再报告退出；释放失败也必须让父进程获知。
        if pilot is not None:
            try:
                pilot.close_model()
            except BaseException:
                stop.set()
                events.put(("error", {"generation_gpu": gpu, "traceback": traceback.format_exc()}))
        events.put(("worker_done", {
            "generation_gpu": gpu, "tasks_completed": finished,
            "generation_gpu_reserved_seconds": reserved,
            "container_seconds": pilot.ev.seconds if pilot is not None else 0,
            "compiles": pilot.ev.compiles if pilot is not None else 0,
        }))


def run_generation_workers(cfg, run, pending, directory, stop, on_event,
                           worker_target=generation_worker):
    """输入待处理整题、停止事件和父进程回调；spawn 模型进程并收集结果，返回成本列表。

    worker_target 可注入 CPU 假进程。显式检查退出码，进程异常退出不能无限等待队列。
    """
    context = multiprocessing.get_context("spawn")
    jobs = context.Queue()
    events = context.Queue()
    devices = gpu_roles(cfg)[0][:len(pending)]
    for job in pending:
        jobs.put(job)
    for _ in devices:
        jobs.put(None)
    workers = [context.Process(target=worker_target,
                               args=(cfg, str(run), gpu, str(directory), jobs, events, stop),
                               name=f"p1-generation-gpu-{gpu}") for gpu in devices]
    statistics = []
    errors = []
    dead_reported = set()
    callback_failed = False
    try:
        for worker in workers:
            try:
                worker.start()
            except BaseException:
                stop.set()
                raise
        while len(statistics) < len(workers):
            try:
                kind, payload = events.get(timeout=0.5)
            except Empty:
                dead = [worker for worker in workers if worker.exitcode not in (None, 0)]
                if dead:
                    stop.set()
                    for worker in dead:
                        if worker.name not in dead_reported:
                            errors.append(f"{worker.name} 异常退出：{worker.exitcode}")
                            dead_reported.add(worker.name)
                if all(worker.exitcode is not None for worker in workers):
                    # 退出前 put 的事件会由 multiprocessing feeder 刷入；再等一轮读取尾部。
                    try:
                        kind, payload = events.get(timeout=1)
                    except Empty:
                        if len(statistics) != len(workers):
                            errors.append("生成进程退出但缺少最终成本事件")
                        break
                    else:
                        pass
                else:
                    continue
            if kind == "worker_done":
                statistics.append(payload)
            elif kind == "error":
                errors.append(payload["traceback"])
                stop.set()
            elif not callback_failed:
                try:
                    on_event(kind, payload)
                except Exception:
                    # 汇总写入失败后不再处理整题清理事件，保留参数供下次恢复。
                    callback_failed = True
                    errors.append(traceback.format_exc())
                    stop.set()
    finally:
        for worker in workers:
            if worker.pid is not None:
                worker.join()
        # 未消费完的任务无需刷入管道，否则中断退出时 feeder 可能等待已离开的消费者。
        jobs.cancel_join_thread()
        jobs.close()
        events.close()
    for worker in workers:
        if worker.exitcode not in (None, 0) and worker.name not in dead_reported:
            errors.append(f"{worker.name} 异常退出：{worker.exitcode}")
    if errors:
        raise RuntimeError("P1 多卡运行中断，已保存完成结果：\n" + "\n".join(errors))
    return statistics


def execute_parallel(pilot, prepare_only=False, stop=None):
    """输入父 Pilot 和预检选项；并行预检、派发整题、统一落盘并在完成题后清理。

    父进程不加载 Qwen；完整轨迹保存在原路径，可从最终结果恢复而无需重新生成。
    """
    context = multiprocessing.get_context("spawn")
    stop = stop if stop is not None else context.Event()
    generation, evaluation = gpu_roles(pilot.cfg)
    directory = scheduling_directory(pilot.run)
    task_file = pilot.run / "tasks.json"
    from ..data import ROOT
    tasks = json.loads((task_file if task_file.exists() else ROOT / "data/processed/pilot.json").read_text())
    ids = [task["task_id"] for task in tasks]
    if len(ids) != len(set(ids)):
        raise ValueError("pilot 题单含重复 task_id，拒绝并行覆盖结果")
    atomic_json(task_file, tasks)
    manifest = pilot.make_manifest(tasks)
    manifest["execution"] = {"generation_gpus": generation, "evaluation_gpus": evaluation,
                             "generation_strategy": "one_model_replica_per_gpu"}
    atomic_json(pilot.run / "manifest.json", manifest)
    records = {}
    indices = {task["task_id"]: index for index, task in enumerate(tasks)}
    eligible = []
    progress = getattr(pilot, "progress", None)
    if progress is not None:
        progress.start_references(tasks)
    evaluators = []
    local = threading.local()

    def persist():
        """按固定题单顺序由父进程唯一写入摘要；子进程不会竞争 results.json。"""
        keys = [(task["task_id"], mode) for task in tasks for mode in ("reference", *MODES)]
        pilot.results = [records[key] for key in keys if key in records]
        pilot.save_result_summaries()

    def prepare(task):
        """读取已有预检或租卡准备参考；缺少最终索引的旧参考目录先归档。"""
        index = pilot.run / "references" / task["task_id"] / "preflight.json"
        if index.exists():
            cached = json.loads(index.read_text())
            # 缺依赖等基础设施故障可在环境修复后重试，不能当作永久拒绝跳过。
            if cached.get("status") in {"ok", "task_invalid"}:
                return cached
        if stop.is_set():
            raise InterruptedError("停止参考预检")
        if not hasattr(local, "evaluator"):
            pool = EvaluationPool(evaluation, directory, stop)
            local.evaluator = ParallelEvaluator(pilot.cfg, pilot.run, pool)
            evaluators.append(local.evaluator)
        archive_incomplete(pilot.run, task["task_id"])
        return local.evaluator.prepare(task)

    workers = []
    try:
        with ThreadPoolExecutor(max_workers=len(evaluation), thread_name_prefix="p1-preflight") as pool:
            futures = {pool.submit(prepare, task): task for task in tasks}
            for future in as_completed(futures):
                task = futures[future]
                try:
                    preflight = future.result()
                except BaseException:
                    # 必须在等待线程池退出前通知其他预检停止，取消尚未开始的任务。
                    stop.set()
                    for queued in futures:
                        queued.cancel()
                    raise
                if preflight["status"] == "ok":
                    eligible.append(task)
                    for mode in MODES:
                        row = saved_trajectory(pilot.run, task["task_id"], mode)
                        if row is not None:
                            records[(task["task_id"], mode)] = row
                else:
                    records[(task["task_id"], "reference")] = {
                        "task_id": task["task_id"], "mode": "reference", "status": "task_invalid",
                    }
                persist()
                if progress is not None:
                    progress.reference_done(task["task_id"], preflight["status"])
                if preflight["status"] != "ok":
                    cleanup_reference_artifacts(pilot.run, task["task_id"])
        eligible.sort(key=lambda task: indices[task["task_id"]])
        if prepare_only:
            manifest.update(status="prepared", updated_at=beijing_now().isoformat(timespec="seconds"))
            atomic_json(pilot.run / "manifest.json", manifest)
            return
        if progress is not None:
            progress.start_modes(eligible, pilot.results)
        pending = []
        for index, task in enumerate(eligible):
            if all((task["task_id"], mode) in records for mode in MODES):
                cleanup_reference_artifacts(pilot.run, task["task_id"])
                continue
            inputs = pilot.run / "references" / task["task_id"] / "cases/input-spec"
            required = [inputs / "model.py"] + [inputs / f"state-{seed}.pt"
                                                for seed in [71001, *pilot.cfg.get("hidden_init_seeds", [])]]
            if not all(path.is_file() for path in required):
                raise RuntimeError(f"待完成题缺少参考参数，请重新预检或新建运行：{task['task_id']}")
            pending.append((index, task))

        def on_event(kind, payload):
            """接收轨迹或整题完成事件；先保存结果，再清理该题参考参数。"""
            if kind == "result":
                records[(payload["task_id"], payload["mode"])] = payload
                persist()
            elif kind == "task_complete":
                if not all((payload, mode) in records for mode in MODES):
                    raise RuntimeError("整题完成事件早于两条轨迹结果")
                cleanup_reference_artifacts(pilot.run, payload)

        if pending:
            workers = run_generation_workers(pilot.cfg, pilot.run, pending, directory, stop, on_event)
        costs = {
            "container_seconds_this_invocation": sum(item.seconds for item in evaluators)
                                                 + sum(item["container_seconds"] for item in workers),
            "compiles_this_invocation": sum(item.compiles for item in evaluators)
                                        + sum(item["compiles"] for item in workers),
            "generation_gpu_reserved_seconds": sum(item["generation_gpu_reserved_seconds"] for item in workers),
            "generation_workers": workers,
        }
        pilot.finish_summary(tasks, eligible, manifest, costs=costs, complete=not stop.is_set())
    except BaseException as error:
        stop.set()
        persist()
        manifest.update(status="interrupted" if isinstance(error, (InterruptedError, KeyboardInterrupt)) else "failed",
                        updated_at=beijing_now().isoformat(timespec="seconds"), error=str(error))
        atomic_json(pilot.run / "manifest.json", manifest)
        raise

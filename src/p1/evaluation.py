"""外部评测控制器：候选快照、隔离调度、数值比较及公平测时。

输入：配置、固定题目和候选工作区；输出：每次评测 JSON、原始日志及奖励。
运行：由 python -m p1.run 或 python -m p1.baseline 调用，不单独执行。
参考和候选分容器，控制器在 CPU 比较；基础设施故障不记作模型失败。"""

# 阅读路线：prepare/profile_reference 准备并测量原始 PyTorch 模型；
# snapshot 固定本次候选源码；compile_only 仅编译；evaluate 完成正式评测。
# evaluate 的顺序是：静态合同 → 独立编译 → 外部数值比较 → 公平测时 → 奖励。
# 本文件负责宿主机上的调度和判定；GPU 运算、编译及 profiler 采样由 worker.py 完成。
# 这里的容器是限制候选访问范围的执行边界，不能把候选自行报告的正确性当作结论。

import ast
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import statistics
import subprocess
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
import torch
from .data import ROOT, sha
from .baseline_state import atomic_json as dump

# 限制宿主 PyTorch 的 CPU 算子线程数，避免外部比较器占用过多 CPU。
torch.set_num_threads(4)


# 专门区分基础设施错误与候选代码失败。
# 例如参考执行失败或参考/候选案例顺序错乱时，不能直接给模型一个失败奖励。
class InfrastructureError(RuntimeError):
    # 只定义异常类型，不额外改写 RuntimeError 的默认行为。
    pass


# audit_contract 返回阻断项和复核提示；contract 保留为既有兼容导出。
from .contract import audit_contract, contract


# 一、外部正确性比较：输入可以是 Tensor、列表、元组、字典或标量组成的输出树。
# actual 是候选输出，expected 是原始 PyTorch 参考输出；调用方负责把输出放在 CPU。
# atol 是绝对容差，rtol 是相对容差；本函数不运行模型，也不接触候选的判定逻辑。
# 返回字典供控制器记录正确性、最大误差、非有限值情况与错误发生的位置。
def compare(actual, expected, atol=0.01, rtol=0.01):
    """输入候选/参考输出树及绝对/相对容差；返回 passed、误差摘要及逐层错误。"""
    # 积累各输出节点的错误；递归遇到不匹配时记录路径，而不是立即终止整棵树的检查。
    errors = []
    # 整棵输出树中浮点/复数张量的最大绝对误差，初始为 0。
    max_abs = 0.0
    # 整棵输出树中浮点/复数张量的最大相对误差，初始为 0。
    max_rel = 0.0
    # 累计所有已检查浮点/复数张量是否有限；任一处 NaN/Inf 都使其变为 False。
    finite = True

    # a/b 是当前一对节点；key 是类似 output.0.values 的诊断路径。
    def walk(a, b, key="output"):
        """输入一对输出节点及路径；递归比较并更新外层错误/误差统计，返回 None。"""
        # 允许递归内部更新 compare 外层的三个统计变量，不另建局部同名变量。
        nonlocal max_abs, max_rel, finite
        # 先严格核对 Python 类型；例如 list 和 tuple 即使内容相同也不视为相同输出结构。
        if type(a) != type(b):
            # 记录类型不匹配的节点位置。
            errors.append(key + ": output type differs")
            # 该节点类型不同，无法继续按同一种结构递归，因此结束当前节点检查。
            return
        # 张量分支：结构先检查形状和 dtype，再检查数值。
        if isinstance(a, torch.Tensor):
            # 相同数值但形状或精度类型不同，也不符合原模型的输出合同。
            if a.shape != b.shape or a.dtype != b.dtype:
                errors.append(key + ": shape/dtype differs")
                return
            # 仅浮点/复数张量统计连续数值误差；整数等类型仍会经过下方 assert_close。
            if a.is_floating_point() or a.is_complex():
                # 候选和参考两边都必须有限；&= 让之前发现的非有限值不会被后续正常节点覆盖。
                finite &= bool(torch.isfinite(a).all() and torch.isfinite(b).all())
                # 逐元素取候选减参考的绝对值；对于复数，这里得到差值的模。
                delta = (a - b).abs()
                # 空张量没有最大元素，按 0 处理，避免对空张量调用 max 报错。
                max_abs = max(max_abs, float(delta.max()) if delta.numel() else 0.0)
                # 相对误差以参考值绝对值为分母；加 1e-8 避免参考值为零时直接除零。
                max_rel = max(
                    max_rel, float((delta / (b.abs() + 1e-8)).max()) if delta.numel() else 0.0
                )
            # 使用 PyTorch 的正式数值断言决定是否通过；上面的 max_abs/max_rel 主要用于诊断。
            try:
                # 容差判定使用 atol/rtol；equal_nan=False 明确禁止两边同为 NaN 时算通过。
                torch.testing.assert_close(a, b, atol=atol, rtol=rtol, equal_nan=False)
            # 数值断言失败只转成当前节点的错误记录，让其他输出节点仍可继续检查。
            except AssertionError:
                errors.append(key + ": numerical mismatch")
        # 列表和元组分支：先检查长度，然后按位置逐项递归。
        elif isinstance(a, (list, tuple)):
            # 长度不同意味着输出结构不完整或多出内容，无需继续 zip 截断比较。
            if len(a) != len(b):
                errors.append(key + ": length differs")
                return
            # 同时遍历候选和参考的对应元素，索引 i 用来拼接诊断路径。
            for i, (v, w) in enumerate(zip(a, b)):
                # 例如第一层第 2 个元素的位置会表示为 output.1。
                walk(v, w, f"{key}.{i}")
        # 字典分支：要求键集合一致，再按参考字典的键逐项比较值。
        elif isinstance(a, dict):
            # 不要求字典的插入顺序一致，但要求没有缺失键或额外键。
            if a.keys() != b.keys():
                errors.append(key + ": keys differ")
                return
            # 以参考输出为遍历依据，把键名追加到错误位置中。
            for k in b:
                walk(a[k], b[k], key + "." + str(k))
        # 其他标量走精确相等判断；这里不会对普通 Python 数字额外套用 atol/rtol。
        elif a != b:
            errors.append(key + ": scalar differs")

    # 从整棵输出树根节点开始递归，初始诊断路径为 output。
    walk(actual, expected)
    # 汇总此次外部比较的结论；返回值本身不会被候选代码控制。
    return {
        # 必须同时满足没有结构/数值错误，以及所有已检查浮点/复数值有限。
        "passed": not errors and finite,
        # 报告所有张量节点中的最大绝对误差。
        "max_abs": max_abs,
        # 报告所有张量节点中的最大相对误差；它不是独立的通过阈值。
        "max_rel": max_rel,
        # 单独保留有限性标志，便于区分 NaN/Inf 与一般误差超限。
        "finite": finite,
        # 对外只保留前 10 条诊断，避免大型嵌套输出产生过长反馈。
        "errors": errors[:10],
    }


# 二、里程碑奖励：输入耗时必须使用相同单位，调用方传入 CUDA 毫秒数。
# 这里仅计算奖励，不验证计时是否有效；有效性检查在 evaluate 中先完成。
def reward(correct, eager_ms, compile_ms, candidate_ms):
    """输入正确性和 eager/compile/候选耗时；返回论文里程碑奖励 -1、1、2 或 3。"""
    # 候选结果错误时直接给 -1，不允许靠错误结果获得速度奖励。
    if not correct:
        return -1
    # 候选耗时必须严格小于 eager 的 95%，才进入更高奖励档位。
    if candidate_ms < 0.95 * eager_ms:
        # 已跨过 eager 门槛后，再跨过 compile 的同一门槛给 3，否则给 2。
        return 3 if candidate_ms < 0.95 * compile_ms else 2
    # 结果正确但未跨过 eager 耗时门槛，给基础奖励 1。
    return 1


# 三、评测调度器：管理同一个 run 的参考文件、候选快照、容器及累计成本。
# 不负责语言模型推理；Agent 和纯 eager/compile 基线都可以复用这个类。
class Evaluator:
    def __init__(self, cfg, run, *, gpu=None, write_config=True):
        """输入公共配置、run、可选实际卡号及是否写配置；初始化控制器，无返回值。

        所有实例只读挂载同一份 worker-config.json。多卡基线由主控制器写一次，
        本实例用 gpu 选择 Docker 设备，write_config=False 禁止各卡重复覆盖公共文件。
        其他入口不传新增参数时，仍按 cfg 中 evaluation_gpu 选卡并生成公共配置。
        """
        self.cfg = cfg
        # 实际卡号属于宿主调度状态，不修改 cfg，也不要求写进每张卡自己的 JSON。
        self.gpu = cfg.get("evaluation_gpu", 7) if gpu is None else gpu
        self.run = Path(run)
        # 累加评测容器的运行耗时，单位是秒（不是算子耗时，也不能拿来计算 kernel 加速比；算子性能使用结果中的 cuda_ms。）
        self.seconds = 0.0  
        # 该计数器记录本 Evaluator 实例发起的候选编译尝试，不代表编译成功数。
        self.compiles = 0
        # 读取已有评测目录的最大编号，默认 0。 为每次候选快照分配新的编号，避免覆盖已有记录
        self.counter = max(
            (
                int(p.name)
                for p in (self.run / "evaluations").glob("*")
                if p.is_dir() and p.name.isdigit()
            ),
            default=0,
        )
        self.run.mkdir(parents=True, exist_ok=True)
        # 固定原文件名；多卡主控制器保证在线程启动前写好，各评测器只读复用。
        self.config = self.run / "worker-config.json"
        if write_config:
            dump(self.config, cfg)
        elif not self.config.is_file():
            raise FileNotFoundError(f"主控制器尚未生成公共配置：{self.config}")
        self.env = ROOT / ".runtime/conda/kernel"
        # 读取当前 Linux 用户 ID，避免容器输出文件默认归 root 所有。
        self.uid = os.getuid()
        # 同样保留当前 Linux 用户的主组 ID。
        self.gid = os.getgid()

    # 四、单次容器调用：所有 reference/compile/candidate 模式共用这一入口。
    # mode：worker.py 的执行模式；out：此模式的宿主日志/结果目录。
    # reference/candidate/inputs：按需挂载的参考源码、候选源码、固定参数与输入定义目录。
    # build：编译产物目录，允许容器写入；phase：feedback 或 hidden 测试阶段。
    # timeout：可选秒数，未提供时采用 cfg 中的 evaluation_seconds。
    # 返回 worker 的结果字典，并追加 controller_seconds；不会在这里计算模型奖励。
    def container(
        self,
        mode,
        out,
        reference=None,
        candidate=None,
        inputs=None,
        build=None,
        phase="feedback",
        timeout=None,
    ):
        """输入模式、输出目录、只读挂载和超时；在指定编号 GPU 隔离执行，返回状态/日志/耗时。"""
        # 统一输出路径类型，并建立可写的宿主结果目录。
        out = Path(out)
        out.mkdir(parents=True, exist_ok=True)
        # 生成短随机容器名，供结束或超时时精确清理本次容器。
        name = "cuda-agent-p1-" + uuid.uuid4().hex[:12]
        # 用参数列表构造 docker 命令，不经 Shell 拼接执行，避免路径和参数被解释为 Shell 代码。
        c = [
            "docker",
            "run",
            "--name",
            name,
            "--rm",
            # 只使用服务器已有镜像，禁止缺镜像时自动联网拉取。
            "--pull=never",
            # 禁用外部网络连接，减少候选下载外部内容或访问网络服务的机会。
            "--network=none",
            "--read-only",
            # 不让 Docker 守护进程持久保存容器日志；stdout 仍由下方 subprocess 捕获。
            "--log-driver=none",
            # 移除容器默认附带的 Linux capabilities，减少额外系统权限。
            "--cap-drop=ALL",
            # 禁止进程通过 setuid 等机制获得新的权限。
            "--security-opt=no-new-privileges",
            # 限制容器中的进程/线程任务数量，避免无限派生任务。
            "--pids-limit=256",
            # 限制宿主内存为 24 GiB；这不是 GPU 显存上限。
            "--memory=24g",
            # 限制 CPU 配额为约 4 个逻辑 CPU 的算力，并非绑定到固定的 4 个核心。
            "--cpus=4",
            # 指定容器进程的 UID:GID，下一项传入当前宿主用户与组。
            "--user",
            f"{self.uid}:{self.gid}",
            # 创建容器内的内存临时文件系统，下一项描述挂载位置和选项。
            "--tmpfs",
            # /tmp 可写、禁用 setuid 语义，大小上限 8 GiB，供编译器及库写临时文件。
            "/tmp:rw,nosuid,size=8g",
            # 将容器共享内存 /dev/shm 的大小设为 1 GiB。
            "--shm-size=1g",
            "--gpus",
            "device=" + str(self.gpu),
        ]
        # 挂载表每项为（宿主路径，容器路径，ro/rw）；只开放此模式需要的文件。
        mounts = [
            # 把 kernel 环境只读挂到相同绝对路径，保留 Python/入口脚本所需的安装前缀。
            (self.env, str(self.env), "ro"),
            # 使用宿主 CUDA 12.9 Toolkit 的编译工具和库，镜像不必自行提供同版本 Toolkit。
            (Path("/usr/local/cuda-12.9"), "/usr/local/cuda-12.9", "ro"),
            # 只挂载可信 worker.py 单文件；候选不需要访问整个项目源码。
            (ROOT / "src/p1/worker.py", "/worker.py", "ro"),
            # 把配置快照以只读形式提供给 worker。
            (self.config, "/config.json", "ro"),
            (out, "/outputs", "rw"),
        ]
        # 参考性能测量专用缓存；同一题前后基线可以复用编译缓存，减少重复冷启动成本。
        if mode == "reference_bench" and reference is not None:
            cache = Path(reference) / "compile-cache"
            cache.mkdir(exist_ok=True)
            # 参考 bench 容器通过 /refcache 读写持久缓存。
            mounts.append((cache, "/refcache", "rw"))
        # 根据本次模式实际提供的参数，追加参考、候选、输入与构建目录。
        for p, d in [
            (reference, "/reference"),
            # 冻结后的候选源码目录；不是正在被 Agent 修改的工作目录。
            (candidate, "/candidate"),
            # 固定参数快照和输入生成定义；不包含参考输出的共享目录。
            (inputs, "/inputs"),
            # 独立的扩展构建目录；正确性和性能容器从这里加载新编译的扩展。
            (build, "/build"),
        ]:
            # None 表示本次模式不需要该目录，因此完全不挂载。
            if p is not None:
                # 只有 /build 可写，参考源码、候选源码与输入定义都按只读挂载。
                mounts.append((Path(p), d, "rw" if d == "/build" else "ro"))
        # 把挂载表逐项转换成 Docker --mount 参数。
        for p, d, perm in mounts:
            c += [
                "--mount",
                (
                    # resolve 得到宿主绝对路径；readonly 明确禁止容器回写这一挂载。
                    f"type=bind,src={p.resolve()},dst={d},readonly"
                    if perm == "ro"
                    # 可写 bind mount 不附加 readonly，允许输出目录和构建目录落盘。
                    else f"type=bind,src={p.resolve()},dst={d}"
                ),
            ]
        # 显式传入运行环境，减少镜像默认变量对实际 Python、CUDA 和缓存位置的影响。
        for k, v in {
            "CUDA_HOME": "/usr/local/cuda-12.9",
            "PATH": f"{self.env}/bin:/usr/local/cuda-12.9/bin:/usr/bin:/bin",
            # 把用户主目录指向临时空间，避免库尝试写只读镜像的默认 home。
            "HOME": "/tmp",
            # 声明给库读取的用户名标签；真正的系统权限由上方 --user 的数字 UID 决定。
            "USER": "sandbox",
            # 同样提供登录名标签，不会创建真实的 sandbox 系统账户。
            "LOGNAME": "sandbox",
            # 让临时文件进入可写的 /tmp。
            "TMPDIR": "/tmp",
            # 让遵循 XDG 规范的缓存进入临时目录。
            "XDG_CACHE_HOME": "/tmp/cache",
            # reference_bench 模式下使用可复用的 Inductor、Triton 缓存；其他模式使用本次输出下的独立缓存。
            "TORCHINDUCTOR_CACHE_DIR": (
                "/refcache/inductor" if mode == "reference_bench" else "/outputs/inductor"
            ),
            "TRITON_CACHE_DIR": (
                "/refcache/triton" if mode == "reference_bench" else "/outputs/triton"
            ),
            # 将 PyTorch C++/CUDA 扩展的构建根目录固定为 /build。
            "TORCH_EXTENSIONS_DIR": "/build",
            # 按计算能力 8.0 编译扩展，对应当前实验目标 A100。
            "TORCH_CUDA_ARCH_LIST": "8.0",
            # 限制扩展构建时的并行编译任务数，控制 CPU 和内存开销。
            "MAX_JOBS": "4",
            # 限制 OpenMP 默认线程数，减少 CPU 线程过度竞争。
            "OMP_NUM_THREADS": "4",
            # 禁止容器往只读环境或源码目录写 .pyc 字节码缓存。
            "PYTHONDONTWRITEBYTECODE": "1",
            # 按需加载 CUDA 模块，相关冷启动成本仍由 worker 的预热/计时流程处理。
            "CUDA_MODULE_LOADING": "LAZY",
        }.items():
            c += ["-e", f"{k}={v}"]
        # Docker 选项之后依次是镜像、容器内程序及其参数。
        c += [
            self.cfg["docker_image"],
            str(self.env / "bin/python"),
            # Python 隔离模式：忽略 PYTHON* 环境变量、用户 site-packages 等隐式导入来源。
            "-I",
            # 再次明确禁止写字节码缓存。
            "-B",
            # 运行只读挂载的可信 worker 入口。
            "/worker.py",
            # 由 worker 根据 mode 分派到预检、编译、流式验算或测时分支。
            mode,
            # 接下来的 phase 参数区分反馈样例和隐藏验收样例。
            "--phase",
            phase,
        ]
        # 参考预检和参考测时各自使用明确的文件名，其余模式维持 result.json。
        result_name = {
            "reference_prepare": "prepare_result.json",
            "reference_bench": "bench_result.json",
        }.get(mode, "result.json")
        result_path = out / result_name
        started = time.monotonic()
        try:
            # 同步等待这个 Docker 命令完成；并行性由调用方的线程池决定。
            p = subprocess.run(
                c,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                # 让前台 Ctrl+C 只请求控制器停止派题，不直接中断当前 Docker 客户端。
                start_new_session=True,
                timeout=timeout or self.cfg["evaluation_seconds"],
            )
            # 落盘最多最后 100000 个字符，保留故障尾部并限制日志大小。
            (out / "worker.log").write_text(p.stdout[-100000:])
            # 优先读取 worker 落盘的结构化结果；这里不以候选打印的成功字符串作判定。
            result = (
                json.loads(result_path.read_text())
                if result_path.exists()
                else {"status": "error", "error": p.stdout[-6000:]}
            )
        # Docker 命令超时后进入此分支；下方 finally 仍会尝试强制删除容器。
        except subprocess.TimeoutExpired as e:
            result = {"status": "timeout", "error": f"{mode} exceeded timeout"}
        # 无论正常、超时还是传播其他异常，都执行容器清理和成本累计。
        finally:
            # --rm 之外的兜底清理：防止客户端退出后 GPU 容器仍留在后台。
            subprocess.run(
                ["docker", "rm", "-f", name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
            elapsed = time.monotonic() - started
            # 累加容器调用成本；参考与候选并发时，这个和会包含重叠时间。
            self.seconds += elapsed
        # 把本次控制器耗时附到结构化结果；kernel 加速比应使用 cuda_ms。
        result["controller_seconds"] = elapsed
        # 持久保存实际使用的物理编号；多卡恢复不依赖可能损坏的根摘要推断来源。
        result["evaluation_gpu"] = self.gpu
        # 补齐控制器耗时后覆盖同一份阶段结果；超时/缺少 worker 输出也保留此记录。
        dump(result_path, result)
        return result

    # 五、准备一题的可信参考：task 至少含 task_id 和原始 PyTorch code。
    # 成功时创建参考源码、参数/输入定义及初始 eager/compile 基线；返回预检详情。
    # 此方法会启动参考 GPU 容器，但不加载 Qwen，也不执行候选代码。
    def prepare(self, task):
        """输入原始任务；生成参考参数和基线，返回仅含状态、阶段结果路径的预检索引。

        所有结果路径均相对于本次 run 根目录；完整预检/测时只保存在各阶段目录。
        """
        ref = self.run / "references" / task["task_id"]
        ref.mkdir(parents=True, exist_ok=True)
        (ref / "model.py").write_text(task["code"])
        # 预检结果由 container 写入 cases/prepare_result.json，包含 controller_seconds。
        prepared = self.container("reference_prepare", ref / "cases", reference=ref, timeout=300)
        preflight = {
            "status": "task_invalid",
            "prepare_result_path": (ref / "cases/prepare_result.json").relative_to(self.run).as_posix(),
        }
        # 仅预检正常且没有近常数退化时继续测量，具体拒绝原因保留在预检结果中。
        if prepared["status"] == "ok" and not prepared.get("near_constant_sampled_output"):
            bench = self.container(
                "reference_bench",
                ref / "initial-bench",
                reference=ref,
                inputs=ref / "cases/input-spec",
                timeout=300,
            )
            preflight["bench_result_path"] = (
                (ref / "initial-bench/bench_result.json").relative_to(self.run).as_posix()
            )
            preflight["status"] = "ok" if bench["status"] == "ok" else "task_invalid"
        # 索引不再嵌套复制参数审计、profile 或计时数据。
        dump(ref / "preflight.json", preflight)
        return preflight

    def profile_reference(self, task):
        """输入固定题目；返回 eager/compile 性能及相对 run 的 result_path。

        首次调用完成参考准备并读取初始测量；后续复用参数、重新测量性能。
        返回的完整结果可供工具反馈使用，汇总文件只保存摘要和详情路径。
        """
        ref = self.run / "references" / task["task_id"]
        if not (ref / "cases/input-spec").exists():
            preflight = self.prepare(task)
            if preflight["status"] != "ok":
                # 拒绝结果指向预检索引，沿索引可找到完整阶段错误或近常数标记。
                return {
                    "status": "reference_rejected",
                    "result_path": (ref / "preflight.json").relative_to(self.run).as_posix(),
                }
            # 首次基线已在 prepare 内完成，不再重复测量或复制整份 JSON。
            result_path = preflight["bench_result_path"]
            result = json.loads((self.run / result_path).read_text())
        else:
            # 再次分析时保留新的测时目录，避免覆盖 initial-bench 的首次结果。
            output = ref / ("profile-" + uuid.uuid4().hex[:10])
            result = self.container(
                "reference_bench", output, reference=ref, inputs=ref / "cases/input-spec", timeout=300
            )
            result_path = (output / "bench_result.json").relative_to(self.run).as_posix()
        return dict(result, reference_only=True, result_path=result_path)

    # 七、冻结候选源码：workspace 是 Agent 开发目录，快照只复制允许的源码文件。
    # 返回 (ev, snap, result, issues)：评测目录、快照目录、初始结果和明确违规项。
    # 这里不会信任工作区现成的 .so；快照在后续容器中只读挂载，并非在宿主上 chmod 只读。
    def snapshot(self, workspace):
        """输入开发工作区；创建递增编号的只读候选快照，返回评测目录、快照和摘要。"""
        # 每提交一次快照就递增一次编号，包括静态审查未通过的提交。
        self.counter += 1
        # 编号补齐到至少五位，如 evaluations/00001，便于按名称浏览。
        ev = self.run / "evaluations" / f"{self.counter:05d}"
        # 本次候选快照位于该评测目录的 candidate 子目录。
        snap = ev / "candidate"
        # 创建全新的快照目录；不使用 exist_ok，避免误复用同编号旧快照。
        snap.mkdir(parents=True)
        # 递归检查工作区文件，但不把整个工作区原样复制进评测。
        for path in Path(workspace).rglob("*"):
            # 按文件名允许模型入口和两份固定绑定基础设施，下一项再放行 kernels 源码。
            allowed = path.name in {"model_new.py", "binding.cpp", "binding_registry.h"}
            # |= 表示在已有允许条件上追加：直属父目录名为 kernels，且后缀为以下源码类型。
            allowed |= path.parent.name == "kernels" and path.suffix in {
                # 允许 CUDA 源文件。
                ".cu",
                # 允许 C++ 源文件。
                ".cpp",
                # 允许普通头文件。
                ".h",
                # 允许 CUDA 头文件。
                ".cuh",
            }
            # 仅复制普通文件且拒绝文件本身是符号链接；完整链接审查还由 audit_contract 负责。
            if path.is_file() and not path.is_symlink() and allowed:
                # 保留相对于工作区的目录结构，避免同名源码在快照中互相覆盖。
                destination = snap / path.relative_to(workspace)
                # 按需要创建源码的父目录。
                destination.parent.mkdir(parents=True, exist_ok=True)
                # 按字节复制内容，不改写源码，也不复制工作区编译好的扩展二进制。
                destination.write_bytes(path.read_bytes())
        # 为快照中每个文件计算内容摘要，方便查重、审计和核对提交身份。
        hashes = {
            # 字典键是快照相对路径，值是文件字节的 SHA 摘要。
            str(path.relative_to(snap)): sha(path.read_bytes())
            for path in snap.rglob("*")
            if path.is_file()
        }
        # 静态审查只读取快照源码，返回明确违规 issues 和需要人工复核的 warnings。
        audit = audit_contract(snap)
        # 两份绑定文件是可信固定基础设施，不能让候选自由修改或省略。
        for name in ["binding.cpp", "binding_registry.h"]:
            # 以已经下载的官方 agent_workdir 文件为逐字节比较基准。
            fixed = ROOT / "upstream/CUDA-Agent/agent_workdir" / name
            # 缺少固定文件，或文件内容与官方版本不同，都加入明确阻断项。
            if not (snap / name).exists() or (snap / name).read_bytes() != fixed.read_bytes():
                audit["issues"].append("fixed infrastructure modified: " + name)
        # 先构造保守的结果骨架，后续步骤成功后再更新，不预先假定候选正确。
        result = {
            # 评测 ID 对应 evaluations 的数字目录名。
            "evaluation_id": ev.name,
            # 排序文件哈希表后再序列化，得到与文件遍历顺序无关的整体候选摘要。
            "candidate_hash": sha(json.dumps(hashes, sort_keys=True)),
            # 保留逐文件摘要，便于定位究竟哪个源码发生变化。
            "source_hashes": hashes,
            # warnings 是人工复核线索；本函数只把 issues 作为直接阻断项返回。
            "audit_warnings": audit["warnings"],
            # 尚未编译和执行，正确性默认 False。
            "correct": False,
            # 默认保留失败奖励占位；compile_only 会另用 scored=False 表明不计分。
            "reward": -1,
        }
        # 把静态阻断项单独返回给调用者，决定是否继续启动编译容器。
        return ev, snap, result, audit["issues"]

    # 八、仅编译工具：workspace 是开发目录；task 保留为统一调用接口参数。
    # 输出编译日志、编译状态和可选扩展路径，不得把编译成功理解为数值正确或更快。
    def compile_only(self, task, workspace):
        """输入题目和工作区；只检查合同并编译，返回日志，不产生正确性或性能结论。"""
        # 先冻结源码并检查合同，让仅编译工具与正式评测使用同样的源码边界。
        ev, snap, result, issues = self.snapshot(workspace)
        # 标注动作类型并明确不计分；结果骨架中的 reward 不能用于训练或性能结论。
        result.update(action="compile_only", scored=False)
        # 有明确违规项就直接反馈，不浪费 GPU 容器和编译时间。
        if issues:
            # 把违规列表放进结果，供 Agent 定位需要修改的源码。
            result.update(status="contract_violation", issues=issues)
        else:
            # 本次编译使用评测目录下全新的 build，不复用工作区已有二进制。
            build = ev / "build"
            build.mkdir()
            # 在隔离容器中编译冻结源码，编译上限使用单独的 compile_seconds。
            compiled = self.container(
                "compile",
                ev / "compilation",
                candidate=snap,
                build=build,
                timeout=self.cfg["compile_seconds"],
            )
            # 只要实际发起容器编译就记一次尝试，不论最终成功还是失败。
            self.compiles += 1
            # 保留 worker 编译细节，同时转换成仅编译工具对外的状态名称。
            result.update(
                compilation=compiled,
                status="compiled" if compiled["status"] == "ok" else "compilation_failed",
            )
            # 编译成功才尝试把扩展发布回工作区，让 Agent 继续调试调用。
            if compiled["status"] == "ok":
                # 与原 compile.sh 一样提供工作区扩展；正式验收仍从源码独立重编译，
                # 不信任 Agent 工作区已有的 .so，也不暴露参考结果。
                # 隔离 build 内本次实际产出的扩展文件。
                binary = build / "cuda_extension.so"
                # Agent 工作区里扩展的固定发布位置。
                destination = Path(workspace) / "cuda_extension.so"
                # 拒绝覆盖符号链接，避免把扩展写到链接指向的其他宿主路径。
                if destination.is_symlink():
                    raise ValueError("extension output must not be a symlink")
                # 确认编译确实生成目标文件，才执行复制。
                if binary.is_file():
                    # copy2 复制扩展并尽量保留元数据；正式验收仍会从源码重新编译。
                    shutil.copy2(binary, destination)
                    # 只返回工作区相对文件名，不把参考输出或参考状态路径暴露给工具。
                    result["artifact"] = "cuda_extension.so"
        # 无论静态阻断还是编译结果，都为本次提交保存 result.json。
        dump(ev / "result.json", result)
        # 返回工具反馈；这个入口不做数值比较，也不做加速比测量。
        return result

    # 九、逐例外部验算：参考与候选分别计算，把输出写到各自独立的挂载目录。
    # task 决定参考参数；snap/build 决定候选源码与扩展；ev 保存这次比较的文件。
    # phase 决定使用 feedback 还是 hidden 案例；返回 (checks, reference_result, candidate_result)。
    # worker 每次原子发布 item.pt 后等待控制器删除文件，删除即确认，随后才输出下一例。
    # 因此两边无需把所有大张量同时堆在宿主内存里，也不能直接读取对方的输出文件。
    def streamed_checks(self, task, snap, build, ev, phase):
        """输入题目、快照、构建与阶段；外部逐例比较，返回 checks、参考状态、候选状态。"""
        # 定位已准备好的可信参考及固定参数。
        ref = self.run / "references" / task["task_id"]
        # 参考容器只把结果写入 reference-stream。
        expected = ev / "reference-stream"
        # 候选容器只把结果写入 candidate-stream，不挂载参考输出目录。
        actual = ev / "candidate-stream"
        # 为两端各创建全新输出目录，避免读取上一轮残留的 item.pt。
        expected.mkdir()
        actual.mkdir()
        # 按实际配对顺序保存每一个案例的比较结论。
        checks = []
        # 控制器循环也设总等待预算，避免无限等待输出握手。
        started = time.monotonic()
        # 两个线程分别同步等待各自容器；主线程负责读取和比较两边输出。
        with ThreadPoolExecutor(max_workers=2) as pool:
            # rf 是参考容器任务的 Future，供后续检查完成状态并取回结构化结果。
            rf = pool.submit(
                self.container,
                "reference_stream",
                expected,
                reference=ref,
                # 双方读同一套固定参数/输入定义，从种子生成相同测试输入。
                inputs=ref / "cases/input-spec",
                phase=phase,
            )
            # cf 是候选容器任务的 Future；该容器可见候选快照和本次构建产物。
            cf = pool.submit(
                self.container,
                "candidate_stream",
                actual,
                candidate=snap,
                inputs=ref / "cases/input-spec",
                build=build,
                phase=phase,
            )
            # 轮询预算为评测秒数再加 5 秒余量；容器本身还受 container 的超时限制。
            while time.monotonic() - started < self.cfg["evaluation_seconds"] + 5:
                # 每端只有一个待确认的 item.pt 槽位，worker 在该文件被删除前不会发布下一例。
                a, b = expected / "item.pt", actual / "item.pt"
                # 只有参考和候选都发布当前结果，才进行配对比较。
                if a.exists() and b.exists():
                    try:
                        # weights_only=True 限制反序列化可接受的对象，不使用允许任意对象的普通 pickle 加载。
                        r = torch.load(a, weights_only=True)
                        # 读取候选的输出树；worker 发布前已把张量转到 CPU。
                        c = torch.load(b, weights_only=True)
                        # 案例标签必须相同；顺序错乱属于评测协议错误，不能比较两份不同输入的输出。
                        if r["id"] != c["id"]:
                            raise InfrastructureError("stream case order differs")
                        # 调用外部可信比较器，候选代码没有机会自行声称 assert 已通过。
                        check = compare(
                            c["output"], r["output"], self.cfg["atol"], self.cfg["rtol"]
                        )
                        # 将比较结论与具体案例 ID 一起保存，失败时可以定位到种子和检查类型。
                        checks.append(dict(check, case=r["id"]))
                        # 及时释放两棵输出树的引用，避免多案例累积占用宿主内存。
                        del r, c
                    # 协议层基础设施异常继续向上传播，不把它包装成模型数值错误。
                    except InfrastructureError:
                        raise
                    # 其他读取/比较异常转为一条失败检查，保留错误文字供诊断。
                    except Exception as e:
                        checks.append(
                            {"passed": False, "errors": [str(e)], "case": "unreadable-output"}
                        )
                    # 删除参考输出，确认这一例已被控制器消费，允许参考 worker 继续。
                    a.unlink()
                    # 同样确认候选输出；即便本例比较失败，也继续收集后续可用证据。
                    b.unlink()
                # worker 要等待外部确认；若另一端提前失败，仍要释放本端待确认输出。
                # 清理未配对输出只用于结束握手，不把它当作比较通过。
                # 任一端的失败状态仍会阻止正确性通过。
                # 若候选已失败，参考可能仍堵在等待确认，删除它的待确认文件以释放握手。
                if cf.done() and cf.result().get("status") != "ok":
                    # 文件可能尚未出现或已删除，因此使用 missing_ok=True；这不新增通过记录。
                    a.unlink(missing_ok=True)
                # 若参考已失败，也释放候选端待确认输出，避免另一端被动耗尽超时。
                if rf.done() and rf.result().get("status") != "ok":
                    b.unlink(missing_ok=True)
                # 两边容器任务都结束时退出；worker 完成前最后一例也必须先被外部确认。
                if rf.done() and cf.done():
                    break
                # 每轮休眠 10 毫秒，避免在 CPU 上持续忙等文件。
                time.sleep(0.01)
            # 取回两端最终状态；result() 会等待尚未完成的任务，并传播线程里的异常。
            rr, cr = rf.result(), cf.result()
        # 返回全部逐例检查与两端执行状态；最终通过条件由 evaluate 统一判断。
        return checks, rr, cr

    # 十、正式候选评测入口，按合同、编译、正确性、性能顺序逐级放行。
    # task 是固定题目，workspace 是待验收源码目录；也可传上一轮冻结快照用于原样重试。
    # final=True 使用隐藏案例，只验正确性；False 使用反馈案例并可测时。
    # attempt 是控制器内部的一次重试计数；measure=False 可跳过性能测试。
    # 返回 result 字典并写盘；参考基础设施连续失败则抛 InfrastructureError，不给模型错误归因。
    def evaluate(self, task, workspace, final=False, attempt=0, measure=True):
        """输入题目、工作区及隐藏/重试/测时标记；返回合同、编译、执行、数值与性能结果。"""
        # 定位本题参考状态；调用者应先完成 prepare 或 profile_reference。
        ref = self.run / "references" / task["task_id"]
        # 先冻结本次源码，后面即使 Agent 改了工作区，也不影响这次候选身份。
        ev, snap, result, issues = self.snapshot(workspace)
        # 记录是否为最终隐藏验收，方便汇总时区分反馈结果与隐藏结果。
        result["final"] = final
        # 第一道关：静态合同明确违规时，立即记录并返回，不编译、不执行。
        if issues:
            result.update(status="contract_violation", issues=issues)
            dump(ev / "result.json", result)
            return result
        # 第二道关：建立独立 build，从冻结源码重新编译候选扩展。
        build = ev / "build"
        build.mkdir()
        # 正式评测不加载 Agent 工作区现成 .so，以免混入与快照不一致的二进制。
        compiled = self.container(
            "compile",
            ev / "compilation",
            candidate=snap,
            build=build,
            timeout=self.cfg["compile_seconds"],
        )
        # 累计本次实际候选编译尝试。
        self.compiles += 1
        # 把完整编译结果收入本次评测，保留错误原因和控制器耗时。
        result["compilation"] = compiled
        # 候选编译失败是候选结果，保留默认的 correct=False、reward=-1 并提前返回。
        if compiled["status"] != "ok":
            result.update(status="compilation_failed")
            dump(ev / "result.json", result)
            return result
        # 第三道关：反馈与隐藏使用不同案例协议，隐藏还会检查重复调用和输入/权重变化。
        phase = "hidden" if final else "feedback"
        # 两端分别执行，宿主逐例核对输出；ran 是候选 worker 最终返回的状态。
        checks, reference, ran = self.streamed_checks(task, snap, build, ev, phase)
        # 先判断参考是否正常；参考出问题时，尚不能据此判断候选算错。
        if reference["status"] != "ok":
            # 单独保存参考端故障与候选端状态，保留基础设施证据。
            dump(ev / "infrastructure-error.json", {"reference": reference, "candidate": ran})
            # 首次参考故障允许对同一份快照完整重试一次；新调用会生成新的评测目录。
            if attempt == 0:
                # 重试保留 final 和 measure，不能把隐藏验收或仅正确性检查意外变成其他模式。
                return self.evaluate(task, snap, final, attempt=1, measure=measure)
            # 第二次参考仍失败就抛基础设施异常，由上层决定如何处理，不输出模型失败奖励。
            raise InfrastructureError("reference stream failed twice; do not assign model reward")
        # 精确核对预期比较次数，防止少执行案例或提前退出却因已通过的少量样例而获通过。
        expected_n = (
            # 反馈阶段每个案例只发布一次普通输出。
            self.cfg["feedback_cases"]
            if not final
            # 隐藏阶段：每个初始化种子 × 每种输入各比较普通/重复/原地址改输入三次；
            # 另外加两次乱序调用、两次重新加载权重后的调用，所以再加 4。
            else self.cfg["hidden_cases_per_init"] * len(self.cfg["hidden_init_seeds"]) * 3 + 4
        )
        # 保留候选执行状态与逐例证据，并据此计算严格的正确性结论。
        result.update(
            # execution 是候选 worker 的结构化返回，包括失败和精度状态变化等信息。
            execution=ran,
            # checks 保存外部比较器的所有逐例结果。
            checks=checks,
            # 此处 status=ok 表示已到达汇总阶段，并不单独代表 correct=True。
            status="ok",
            # 候选必须同时满足下面四项条件，不能只检查部分案例通过。
            correct=(
                # 候选 worker 本身正常结束，没有执行错误或超时。
                ran["status"] == "ok"
                # 比较条数必须和当前协议要求完全一致。
                and len(checks) == expected_n
                # 每一条外部数值比较都通过。
                and all(c["passed"] for c in checks)
                # 没有被 worker 检出的精度状态变更，例如运行后修改 TF32 设置。
                and not ran.get("state_mutations")
            ),
        )
        # 错误候选、最终隐藏验收、或显式禁用测时，均在正确性阶段结束。
        if not result["correct"] or final or not measure:
            # 这三类路径只给正确性基础奖励：正确为 1，错误为 -1，不生成速度奖励。
            result["reward"] = 1 if result["correct"] else -1
            # 保存此阶段结论，使没有性能结果的评测仍有完整记录。
            dump(ev / "result.json", result)
            return result
        # 候选独占测时前后都重测参考基线，以检测资源干扰和计时漂移。
        # 第四道关先测参考，再测候选，最后再测参考；三个调用依次执行而非同时测时。
        before = self.container(
            "reference_bench",
            ev / "baseline-before",
            reference=ref,
            inputs=ref / "cases/input-spec",
            timeout=300,
        )
        # 候选测时复用本次新构建的扩展，但在独立的新容器中进行。
        measured = self.container(
            "candidate_bench",
            ev / "timing",
            candidate=snap,
            inputs=ref / "cases/input-spec",
            build=build,
        )
        # 候选结束后再测一次参考，用前后差异诊断 GPU 负载、频率或资源干扰带来的漂移。
        after = self.container(
            "reference_bench",
            ev / "baseline-after",
            reference=ref,
            inputs=ref / "cases/input-spec",
            timeout=300,
        )
        # 参考前测或后测任一个失败，都不能计算可信的候选速度奖励。
        if before["status"] != "ok" or after["status"] != "ok":
            # 保存前后两次参考故障的原始结果。
            dump(ev / "infrastructure-error.json", {"before": before, "after": after})
            # 首次故障完整重试同一源码；第二次仍失败抛基础设施异常。
            if attempt == 0:
                return self.evaluate(task, snap, final, attempt=1)
            raise InfrastructureError("reference timing failed twice; do not assign model reward")
        # 候选自身测时执行失败属于候选失败；与上方参考基础设施失败分开处理。
        if measured["status"] != "ok":
            # 标记 timing_failed 并令候选不合格，不保留先前数值通过带来的正奖励。
            result.update(status="timing_failed", correct=False, reward=-1)
            dump(ev / "result.json", result)
            return result
        # 动态 profiler 若发现不允许的 ATen 计算回退，即使输出正确也不能作为合规 CUDA 候选。
        if measured.get("runtime_fallbacks"):
            # 记录实际回退事件，方便区分真实 CUDA/允许库调用与 PyTorch 计算捷径。
            result.update(
                status="runtime_fallback_detected",
                correct=False,
                reward=-1,
                runtime_fallbacks=measured["runtime_fallbacks"],
            )
            dump(ev / "result.json", result)
            return result
        # 收集 5 个 CUDA 毫秒数：候选一次，参考前后各 eager/compile 两次。
        times = [measured["timing"]["cuda_ms"]] + [
            baseline[key]["cuda_ms"] for baseline in [before, after] for key in ["eager", "compile"]
        ]
        # 每个耗时都必须是有限且严格大于 0 的数，先防止除零和虚假无限加速。
        if any(
            not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0
            for value in times
        ):
            # 无效计时用 reward=None，表示不能拿来打速度分，而非把零耗时当成极快。
            result.update(
                status="invalid_timing",
                timing_valid=False,
                reward=None,
                # 同时保留候选计时及前后基线，便于检查零值、NaN/Inf 等异常。
                timing=measured,
                baseline_before=before,
                baseline_after=after,
            )
            # 无效数值计时直接返回该证据记录；下面的漂移重试只处理数值合法但漂移大的情况。
            dump(ev / "result.json", result)
            return result
        # 分别计算 eager 和 compile 的前后相对漂移绝对值：abs(后测/前测 - 1)。
        drift = {
            k: abs(after[k]["cuda_ms"] / before[k]["cuda_ms"] - 1) for k in ["eager", "compile"]
        }
        # 两种参考基线都必须不超过配置 baseline_drift_limit，才认可这次性能比较。
        valid = max(drift.values()) <= self.cfg["baseline_drift_limit"]
        # 无论漂移是否通过，都先把完整测时证据写入结果对象。
        result.update(
            baseline_before=before,
            baseline_after=after,
            # 保存 eager/compile 各自的漂移比例，便于定位是哪条基线不稳定。
            baseline_drift=drift,
            # timing_valid 单独表示测时可信度，不能与数值正确性 correct 混淆。
            timing_valid=valid,
            timing=measured,
        )
        # 候选在性能运行中也不能更改 TF32 等精度状态，防止只在测速时降低精度。
        if measured.get("state_mutations"):
            # 精度状态违规会令候选不合格并给 -1，即使此前正确性和耗时看起来正常。
            result.update(correct=False, reward=-1, status="precision_state_violation")
        # 只有数值正确、没有状态违规且基线稳定，才计算最终加速比和里程碑奖励。
        elif valid:
            # 分别用参考前/后两个 CUDA 耗时的中位数作为该基线的代表耗时。
            baseline = {
                # 这里恰好只有两个值，statistics.median 的结果等于它们的算术平均值。
                k: statistics.median([before[k]["cuda_ms"], after[k]["cuda_ms"]])
                for k in ["eager", "compile"]
            }
            # 取候选自身的 CUDA 毫秒数，不使用 controller_seconds 或编译首调时间。
            ms = measured["timing"]["cuda_ms"]
            # 加速比定义为参考耗时/候选耗时：大于 1 表示候选更快，小于 1 表示更慢。
            result.update(
                # 相对 PyTorch eager 的加速比。
                speedup_eager=baseline["eager"] / ms if ms > 0 else None,
                # 相对 torch.compile 的加速比；两种基线分别报告，不能混成一个数。
                speedup_compile=baseline["compile"] / ms if ms > 0 else None,
            )
            # 在可信性能结果上按上方 reward 的严格耗时门槛赋予 1/2/3 奖励。
            result["reward"] = reward(True, baseline["eager"], baseline["compile"], ms)
        # 基线漂移过大时不打速度分，保留 None，避免噪声被当成模型能力。
        else:
            result["reward"] = None
        # 先保存当前尝试，重试也不会覆盖这份原始故障/漂移证据。
        dump(ev / "result.json", result)
        # 数值合法但漂移超限，且尚未重试时，允许对同一快照再做一次完整评测。
        if not valid and attempt == 0:
            # 重试复用候选源码身份，但重新编号、重新编译并重新执行各检查。
            retried = self.evaluate(task, snap, final, attempt=1)
            # 在返回对象中补上来源评测编号；此处不再次改写递归调用已经保存的 result.json。
            retried["retry_of"] = ev.name
            # 优先返回重试的结果，让上层看到最后一次尝试的结论。
            return retried
        # 正常完成或已经用过一次漂移重试后，返回当前结果；不会无限递归重试。
        return result

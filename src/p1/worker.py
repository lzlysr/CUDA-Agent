"""可信容器工作进程：输入重建、CUDA 编译、验证、计时与 profiler。

运行：由 Evaluator.container 在隔离容器内执行 /worker.py <模式>。
输入：只读 /config.json、/reference 或 /candidate、/inputs；输出：/outputs。
不要在宿主直接运行；CPU 单测只导入纯辅助函数，不调用 GPU。"""

import argparse
import hashlib
from datetime import datetime, timedelta, timezone
import importlib.util
import json
from pathlib import Path
import random
import runpy
import shutil
import statistics
import sys
import time
import traceback
import numpy as np
import torch
from torch.utils._python_dispatch import TorchDispatchMode

# 保存 CUDA 同步函数的引用；调用时等待当前设备的 CUDA 工作完成。
SYNC = torch.cuda.synchronize
# 保存 CUDA Event 类的引用，后续用事件记录 GPU 时间区间。
EVENT = torch.cuda.Event
# 保存序列化函数的引用，用于写模型参数和逐例输出快照。
SAVE = torch.save
# 保存高精度计时函数，返回秒数差值；它用于测时长，不是日历时间戳。
CLOCK = time.perf_counter


class ReferenceValidationError(ValueError):
    """记录参考程序未通过现有预检的具体原因；不改变数值判定标准。"""

    def __init__(self, kind, message):
        """输入原因代码与可读诊断；供 main 写入阶段结果，无返回值。"""
        super().__init__(message)
        self.kind = kind


class StorageLimitError(ValueError):
    """单个输出或本题参数快照超过已声明预算，避免继续写入大文件。"""


class RuntimeFallbackError(ValueError):
    """模型实际调用了既定允许表以外的 ATen 计算。"""

    def __init__(self, names):
        """输入违规算子名；保存列表，供 main 写入候选失败证据。"""
        super().__init__("PyTorch compute fallback: " + ", ".join(names))
        self.names = names


def storage_bytes(value):
    """输入张量树；返回去重后的底层存储字节数，视图按实际被序列化的存储计数。"""
    storages = {}

    def collect(tensor):
        """输入一片张量；登记其底层存储，返回原张量，不改变计算结果。"""
        storage = tensor.untyped_storage()
        storages[(str(tensor.device), storage._cdata)] = storage.nbytes()
        return tensor

    tree(value, collect)
    return sum(storages.values())


def save_limited(value, path, limit):
    """输入 CPU 张量树、目标路径和剩余字节预算；有界写出，成功返回文件字节数。

    先检查张量存储，再限制序列化器的每次 write；元数据也计入最终文件上限。
    超限时删除半成品并抛 StorageLimitError，不留下可被控制器误读的发布文件。
    """
    path = Path(path)
    if storage_bytes(value) > limit:
        raise StorageLimitError(f"tensor storage exceeds {limit} bytes")

    class BoundedWriter:
        """限制 torch.save 写入总长度；不依靠写完之后才发现文件过大。"""

        def __init__(self, stream):
            """输入新建文件对象；记录文件和是否发生超限，无返回值。"""
            self.stream = stream
            self.exceeded = False

        def write(self, data):
            """输入待写字节；检查文件末端后写入并返回写入长度。"""
            if self.stream.tell() + len(data) > limit:
                self.exceeded = True
                raise StorageLimitError(f"serialized file exceeds {limit} bytes")
            return self.stream.write(data)

        def flush(self):
            """无输入；把序列化缓冲写入文件，无返回值。"""
            self.stream.flush()

    try:
        with path.open("wb") as stream:
            writer = BoundedWriter(stream)
            try:
                SAVE(value, writer)
            except Exception:
                # ZIP 收尾异常可能覆盖 write 的异常；仍保留明确的容量超限归因。
                if writer.exceeded:
                    raise StorageLimitError(f"serialized file exceeds {limit} bytes") from None
                raise
        return path.stat().st_size
    except BaseException:
        path.unlink(missing_ok=True)
        raise


# 返回两个布尔值，顺序固定为矩阵乘法 TF32、cuDNN TF32，供执行前后对照。
def flags():
    """无输入；返回 matmul 和 cuDNN 的 TF32 状态列表。"""
    return [torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32]


# 统一每个新 worker 的初始计算设置；通过修改库配置生效，无返回值。
def settings():
    """无输入；固定 TF32、线程及 cuDNN benchmark 设置，返回 None。"""
    # 允许 float32 矩阵乘法使用 TF32 路径，参考与候选从相同设置开始。
    torch.backends.cuda.matmul.allow_tf32 = True
    # 同样允许 cuDNN 中符合条件的卷积使用 TF32。
    torch.backends.cudnn.allow_tf32 = True
    # 关闭 cuDNN 按输入动态 benchmark 选算法的模式。
    torch.backends.cudnn.benchmark = False
    # 限制 PyTorch CPU 算子的线程数，避免占用过多宿主 CPU 资源。
    torch.set_num_threads(4)


# n 是整数种子；同步下面四类随机源，便于重建相同初始化和输入。
def seed(n):
    """输入整数种子；同步 Python、NumPy、torch 和 CUDA 随机源，返回 None。"""
    random.seed(n)
    np.random.seed(n)
    torch.manual_seed(n)
    torch.cuda.manual_seed_all(n)


# x 是嵌套数据，fn 是对单个 Tensor 的转换函数；递归后返回转换后的数据。
def tree(x, fn):
    """输入嵌套输出树及 Tensor 回调；返回结构对应的转换结果，tuple 统一为普通 tuple。"""
    # 张量是递归的叶节点，直接执行调用方传入的转换。
    if isinstance(x, torch.Tensor):
        return fn(x)
    # 列表逐项递归，返回新的列表。
    if isinstance(x, list):
        return [tree(v, fn) for v in x]
    # 元组逐项递归，返回普通 tuple。
    if isinstance(x, tuple):
        return tuple(tree(v, fn) for v in x)
    # 字典保留键，对每个值递归转换，返回普通字典。
    if isinstance(x, dict):
        return {k: tree(v, fn) for k, v in x.items()}
    # 数字、字符串、None 等非张量叶节点保持原值。
    return x


# 把接口返回值统一成可用 * 展开的参数序列；已有列表/元组不再额外嵌套。
def as_list(x):
    """输入题目接口返回值；将单值包装成列表，列表/元组原样返回。"""
    return x if isinstance(x, (list, tuple)) else [x]


# path 是模块文件路径，name 是本次加载使用的模块名；返回加载后的模块对象。
def load(path, name):
    """输入只读 Python 文件及模块名；加载并返回任务或候选模块，仅在隔离 worker 使用。"""
    # 建立指定文件的模块加载说明，不要求源码位于标准 Python 包目录。
    s = importlib.util.spec_from_file_location(name, path)
    # 按加载说明先创建一个模块对象。
    m = importlib.util.module_from_spec(s)
    # 执行文件的模块级代码，将 Model、get_inputs 等定义填入模块对象。
    s.loader.exec_module(m)
    # 把模块返回给调用者，之后通过 mod.Model、mod.get_inputs 等访问其中接口。
    return m


# 输入任意张量树；对每个张量解除梯度关联并搬到 CPU，返回对应结构。
def cpu(x):
    """输入输出树；返回 detach 后复制到 CPU 的同结构内容，供外部比较。"""
    # CPU 输出可以交给宿主比较器，也可以作为模型参数快照保存。
    return tree(x, lambda t: t.detach().cpu())


# model 是可调用模型，inputs 是位置参数序列；返回模型的实际输出。
def call(model, inputs):
    """输入模型和输入列表；在 no_grad 下执行 forward 并返回结果。"""
    # 评测无需反向传播，因此关闭梯度记录，避免建立无用的计算图。
    with torch.no_grad():
        return model(*inputs)


# mod 提供原始题目的 get_inputs，n 决定输入随机种子；返回 GPU 输入树。
def inputs_for(mod, n):
    """输入任务模块和输入种子；重建输入并复制到 GPU，返回输入列表。"""
    # 先重置随机源，再生成输入，使参考和候选可独立重建同一组数据。
    seed(n)
    # 调用原始题目的 get_inputs，统一参数序列后把其中的张量逐个搬到 GPU。
    return tree(as_list(mod.get_inputs()), lambda t: t.cuda())


# phase 区分反馈和隐藏阶段，cfg 提供种子/数量；返回基础案例列表。
def case_list(phase, cfg):
    """输入反馈/隐藏阶段及配置；返回案例 ID、初始化种子和输入种子列表。"""
    inits = [71001] if phase == "feedback" else cfg["hidden_init_seeds"]
    n = cfg["feedback_cases"] if phase == "feedback" else cfg["hidden_cases_per_init"]
    # 为每个初始化种子生成 n 个案例，每项保存标签和两类种子。
    return [
        {
            "id": f"{phase}-{s}-{i:02d}",
            "init_seed": s,
            # 反馈输入从 72000 递增；隐藏输入按初始化种子分组编号。
            "input_seed": 72000 + i if phase == "feedback" else s * 100 + i,
        }
        for s in inits
        for i in range(n)
    ]


# mod 是模型模块，klass 是 Model/ModelNew，init_seed 指定参数快照；返回 GPU 模型。
def load_model(mod, klass, init_seed, input_spec):
    """输入模型模块、类名、种子和可信参考模块；重建构造参数并严格恢复权重，返回 GPU 模型。"""
    # 快照仅含权重；保持 weights_only=True，不反序列化任意配置类或 Python 对象。
    d = torch.load(Path("/inputs") / f"state-{init_seed}.pt", weights_only=True)
    # 构造参数来自已经允许执行的只读参考接口；不能让候选自行提供 get_init_inputs。
    seed(init_seed)
    init = as_list(input_spec.get_init_inputs())
    m = getattr(mod, klass)(*init).eval().cuda()
    # 严格恢复固定 state_dict；参数键缺失、多余或张量形状不兼容会触发错误。
    m.load_state_dict(d["state"], strict=True)
    return m


# 参考准备入口：cfg 是配置字典，源码来自 /reference/model.py。
# 输出文件写入 /outputs/input-spec；返回输出审计、近常数标记及案例数量等信息。
def prepare(cfg):
    """输入配置；核查参考容差内稳定性/有限值/近常数，写输入规格及参数，返回审计元数据。"""
    mod = load("/reference/model.py", "reference_task")
    metadata = []
    out = Path("/outputs")
    (out / "input-spec").mkdir(exist_ok=True)
    (out / "input-spec/model.py").write_text(Path("/reference/model.py").read_text())
    for s in [71001] + cfg["hidden_init_seeds"]:
        seed(s)
        # 调用题目的 get_init_inputs，取得 Model 构造函数的位置参数。
        init = as_list(mod.get_init_inputs())
        # 实例化题目的原始 Model，切到推理模式并移动到 GPU。
        m = mod.Model(*init).eval().cuda()
        # 只保存 CPU 权重；BrosConfig 等构造参数按相同种子从可信参考接口重建，避免不安全 pickle。
        # 同题全部参数快照共享这份预算；它与逐例输出预算分别计算，不是显存限制。
        target = out / "input-spec" / f"state-{s}.pt"
        limit = cfg.get("case_storage_limit_bytes", 8 * 1024**3)
        used = sum(p.stat().st_size for p in target.parent.glob("state-*.pt") if p != target)
        if used + storage_bytes(m.state_dict()) > limit:
            raise StorageLimitError(f"reference parameter snapshots exceed {limit} bytes")
        save_limited({"state": cpu(m.state_dict())}, target, limit - used)
        # 先检查初始化、非有限值与确定性；更完整的输入验收在候选评测阶段执行。
        outputs = []
        # 用两个固定输入种子检查参考行为。
        for input_seed in [72000, 72001]:
            # 重新设置输入种子，调用 get_inputs 并将输入搬到 GPU。
            inp = inputs_for(mod, input_seed)
            # 使用输入张量的副本，避免模型原位修改污染待复用的原始输入。
            a = call(m, tree(inp, lambda t: t.clone()))
            b = call(m, tree(inp, lambda t: t.clone()))

            # x/y 是同一个参考模型两次执行的输出；递归比较，不匹配时抛出异常。
            def equal(x, y, path="output"):
                """输入两次参考输出及节点路径；浮点按配置容差比较，失败保留路径/种子，返回 None。"""
                context = f"{path}; init_seed={s}, input_seed={input_seed}"
                if type(x) is not type(y):
                    raise ReferenceValidationError(
                        "reference_nondeterministic", f"Reference output type changed at {context}"
                    )
                # 张量节点先检查有限性，再按数据类型选择重复输出的比较容差。
                if isinstance(x, torch.Tensor):
                    floating = x.is_floating_point() or x.is_complex()
                    # 显式检查浮点输出中是否存在 NaN 或 Inf。
                    if floating and not (torch.isfinite(x).all() and torch.isfinite(y).all()):
                        raise ReferenceValidationError(
                            "reference_nonfinite", f"Reference has nonfinite output at {context}"
                        )
                    # 浮点/复数读取同一份数值验算配置；整数和布尔张量仍要求完全相等。
                    atol = cfg["atol"] if floating else 0
                    rtol = cfg["rtol"] if floating else 0
                    try:
                        torch.testing.assert_close(x, y, atol=atol, rtol=rtol, equal_nan=False)
                    except AssertionError as error:
                        raise ReferenceValidationError(
                            "reference_nondeterministic",
                            f"Reference repeated output differs at {context}; "
                            f"dtype={x.dtype}, shape={tuple(x.shape)}, atol={atol}, rtol={rtol}\n{error}",
                        ) from error
                # 列表或元组按对应位置递归检查。
                elif isinstance(x, (list, tuple)):
                    if len(x) != len(y):
                        raise ReferenceValidationError(
                            "reference_nondeterministic", f"Reference output length changed at {context}"
                        )
                    for index, (v, w) in enumerate(zip(x, y)):
                        equal(v, w, f"{path}[{index}]")
                # 字典按当前输出的键递归检查对应值。
                elif isinstance(x, dict):
                    if x.keys() != y.keys():
                        raise ReferenceValidationError(
                            "reference_nondeterministic", f"Reference output keys changed at {context}"
                        )
                    for k in x:
                        equal(x[k], y[k], f"{path}[{k!r}]")
                # 普通标量使用精确相等断言。
                else:
                    if isinstance(x, (float, complex, np.floating, np.complexfloating)) and not (
                        np.isfinite(x) and np.isfinite(y)
                    ):
                        raise ReferenceValidationError(
                            "reference_nonfinite", f"Reference has nonfinite scalar at {context}"
                        )
                    if x != y:
                        raise ReferenceValidationError(
                            "reference_nondeterministic", f"Reference scalar output changed at {context}"
                        )

            # 确认这组输入的两次参考输出一致后，再采样记录输出统计。
            equal(a, b)
            # 收集该输出树中每个张量的形状、dtype、大小和采样值。
            stats = []
            # 利用 tree 遍历输出树；转换的返回值不使用，这里通过 append 收集统计。
            tree(
                a,
                lambda t: stats.append(
                    {
                        "shape": list(t.shape),
                        "dtype": str(t.dtype),
                        "numel": t.numel(),
                        # 先展平，再按步长抽样并截取最多 2048 个元素，控制审计记录大小。
                        "sample": t.flatten()[:: max(1, t.numel() // 2048)][:2048]
                        # 采样值不再关联梯度图。
                        .detach()
                        .float()
                        .cpu()
                        .tolist(),
                    }
                ),
            )
            # 加入当前输入的统计；两次循环后 outputs 包含两组输入记录。
            outputs.append(stats)
            # 及时移除 GPU 输入和输出引用，避免在预检中累积占用显存。
            del inp, a, b
        metadata.append(
            {
                "init_seed": s,
                "output_audits": outputs,
                "same_output_samples_across_inputs": outputs[0] == outputs[1],
            }
        )
        # 本种初始化检查完毕，释放模型引用。
        del m
        # 释放缓存分配器中已不用的显存；不会释放仍有引用的活跃张量。
        torch.cuda.empty_cache()
    # 输出采样仅保留审计摘要，不再据此判断或拒绝近常数题。
    for entry in metadata:
        for audits in entry["output_audits"]:
            for audit in audits:
                values = audit.pop("sample")
                audit.update(
                    sample_count=len(values),
                    sample_min=min(values) if values else None,
                    sample_max=max(values) if values else None,
                    sample_sha256=hashlib.sha256(
                        json.dumps(values, separators=(",", ":")).encode("utf-8")
                    ).hexdigest(),
                )
    # 返回准备阶段的审计结果，供宿主决定是否接受这道题。
    return {
        # 保留逐初始化、逐输入的形状、dtype 和采样摘要。
        "metadata": metadata,
        # 反馈阶段的基础案例数量。
        "cases_feedback": len(case_list("feedback", cfg)),
        # 隐藏阶段的基础案例数量；不包含 stream 追加的重复/变更等检查。
        "cases_hidden": len(case_list("hidden", cfg)),
        # 记录准备结束时两项 TF32 配置。
        "tf32": flags(),
    }


# 编译入口无显式参数，使用约定挂载路径和构建目录，成功返回 compiled=True。
def compile_candidate():
    """无参数；使用固定源文件/编译参数生成扩展，返回 compiled=True，失败抛错。"""
    # 与原 load 使用同一 Ninja 构建器，但不导入 .so，也不运行候选静态初始化。
    # 当前固定 torch 版本已核对该接口；升级 torch 时须一并核对构建器签名。
    from torch.utils.cpp_extension import _write_ninja_file_and_build_library

    # 收集固定绑定源文件和候选 kernels 下的 CUDA/C++ 源码。
    sources = (
        # 绑定层负责把 C++/CUDA 接口暴露给 Python 模块。
        [Path("/candidate/binding.cpp")]
        # 按文件名排序 CUDA 源文件，使源码列表顺序稳定。
        + sorted(Path("/candidate/kernels").glob("*.cu"))
        # 追加同目录下的 C++ 实现文件。
        + sorted(Path("/candidate/kernels").glob("*.cpp"))
    )
    # 扩展构建器按 basename 产生对象文件。同名 root/kernels 源文件需要独立名称，
    # 只改变隔离 build 中的编译路径，不修改模型生成的内容或正式候选快照。
    names = [path.name for path in sources]
    if len(names) != len(set(names)):
        staged = Path("/build/sources")
        shutil.copytree("/candidate", staged, dirs_exist_ok=True)
        staged_sources = []
        for index, path in enumerate(sources):
            relative = path.relative_to("/candidate")
            staged_path = staged / relative
            if names.count(path.name) > 1:
                renamed = staged_path.with_name(f"p1_source_{index}_{path.name}")
                renamed.write_bytes(staged_path.read_bytes())
                staged_path = renamed
            staged_sources.append(staged_path)
        sources = staged_sources
    # 合法 cuBLAS/cuDNN GPU 包装可能只有 .cpp；仍要求 kernels 下存在实现/绑定源码。
    if not any(path.parent.name == "kernels" for path in sources):
        raise ValueError("no kernel CUDA/C++ source")
    # 从实际导入的 torch 路径定位 kernel 环境的 site-packages。
    site = Path(torch.__file__).resolve().parents[1]
    # 允许扩展使用该环境中 NVIDIA cuDNN/cuBLAS 包提供的头文件。
    includes = [
        # 固定注册头在候选根目录；同时兼容 binding_registry.h 和 ../binding_registry.h。
        "/candidate",
        "/candidate/kernels",
        str(site / "nvidia/cudnn/include"),
        str(site / "nvidia/cublas/include"),
    ]
    # 指定 cuDNN/cuBLAS 动态库的实际文件路径，供链接阶段使用。
    libs = [site / "nvidia/cudnn/lib/libcudnn.so.9", site / "nvidia/cublas/lib/libcublas.so.12"]
    # 只做 CPU 编译/链接。扩展的加载及 CUDA 初始化在 GPU 验算容器中检查，
    # 避免纯 CPU 编译因合法的 CUDA 静态初始化而失败，也避免编译抢占测时 GPU。
    _write_ninja_file_and_build_library(
        # 固定扩展名，候选 Python 可通过 import cuda_extension 使用。
        name="cuda_extension",
        # 把 Path 列表转换为构建工具接受的路径字符串列表。
        sources=[str(p) for p in sources],
        # 所有构建中间文件和生成扩展进入独立的可写 /build。
        build_directory="/build",
        # 明确启用 CUDA 扩展构建。
        with_cuda=True,
        # 本项目只构建 CUDA，不启用 SYCL；保留当前构建器要求的显式参数。
        with_sycl=False,
        extra_sycl_cflags=[],
        # 关闭构建过程的详细命令打印，错误仍会通过异常反馈。
        verbose=False,
        # 向编译器添加上方 NVIDIA 库的头文件搜索路径。
        extra_include_paths=includes,
        # 把指定动态库作为附加链接参数传给链接器。
        extra_ldflags=[str(p) for p in libs],
        # C++ 编译采用 O3 优化级别和 C++17 标准。
        extra_cflags=["-O3", "-std=c++17"],
        # CUDA 编译采用 O3 与 fast-math，结果正确性仍需后续数值检查确认。
        extra_cuda_cflags=["-O3", "--use_fast_math"],
    )
    # 这里只表示编译/链接成功；加载、数值正确性和性能分别在后续 GPU 阶段检查。
    return {"compiled": True}


# model 是待测模型，sample 是当前唯一一组原始 GPU 输入。
# iterations 是本组在每轮中分摊的测量次数；输入复制始终在计时区间之外。
def timing(model, sample, cfg, iterations):
    """输入模型、单组输入、配置和次数；返回每轮的原始 CUDA/墙钟时间列表。

    每次调用都使用新的克隆，避免原地计算污染同组的后续测量和另一条基线。
    返回原始时间供所有输入组共同汇总，不能先取各组中位数再取中位数。
    """
    # 预热副本与计时副本都只保留到本次调用结束，不随输入组数累积。
    warmup_inputs = tree(sample, lambda tensor: tensor.clone())
    call(model, warmup_inputs)
    SYNC()
    del warmup_inputs
    rounds = []
    for _ in range(cfg["timing_rounds"]):
        events, walls = [], []
        for _ in range(iterations):
            # 重置输入内容并等待复制完成；复制开销不计入 GPU Event 或墙钟测时。
            inputs = tree(sample, lambda tensor: tensor.clone())
            SYNC()
            start, end = EVENT(enable_timing=True), EVENT(enable_timing=True)
            started = CLOCK()
            start.record()
            output = call(model, inputs)
            end.record()
            SYNC()
            walls.append((CLOCK() - started) * 1e3)
            events.append(start.elapsed_time(end))
            # 必须在下一次 clone 前释放，否则赋值右侧计算期间会同时保留两个副本。
            del output, inputs
        rounds.append({"cuda_ms": events, "wall_ms": walls})
    return rounds


def summarize_timing(rounds):
    """输入合并所有输入组后的逐轮原始时间；返回兼容现有结果格式的中位数统计。"""
    # 先在每轮全部测量中取中位数，再跨轮取中位数，保持原先的聚合规则。
    medians = [
        {key: statistics.median(row[key]) for key in ("cuda_ms", "wall_ms")}
        for row in rounds
    ]
    return {
        "cuda_ms": statistics.median(row["cuda_ms"] for row in medians),
        "wall_ms": statistics.median(row["wall_ms"] for row in medians),
        "rounds": medians,
    }


# 输入模型和一组 GPU 样本；返回按设备耗时排序的摘要，以及完整事件名列表。
def profile_model(model, sample):
    """输入可调用模型及一组 GPU 输入；返回算子统计与全部事件名，不参与计时。"""
    # 在 profiler 外准备副本，避免输入原地修改污染当前组，复制不进入算子审计。
    inputs = tree(sample, lambda tensor: tensor.clone())
    SYNC()
    # 启用 profiler 执行单次模型调用；采样独立于上方 timing，不混入性能计时。
    with torch.profiler.profile(
        # 同时记录 CPU 与 CUDA 活动，便于识别运算调用及其设备开销。
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    ) as profiler:
        # 执行传入样本，不在此处计算输出正确性。
        output = call(model, inputs)
        # 等待 CUDA 工作完成，让 profiler 捕获该次调用产生的设备活动。
        SYNC()
    del output, inputs
    # 按事件键聚合采样结果，取得每类事件的调用次数与累计耗时。
    events = list(profiler.key_averages())
    # 构建可序列化的事件摘要。
    rows = [
        # name 是事件名，calls 是聚合调用数，device_us 是累计设备时间（微秒）。
        {"name": event.key, "calls": event.count, "device_us": event.device_time_total}
        for event in events
        # 展示摘要只保留有设备耗时的事件。
        if event.device_time_total > 0
    ]
    # 摘要按设备耗时降序保留前 80 项；第二个返回值仍包含所有事件名。
    return sorted(rows, key=lambda row: row["device_us"], reverse=True)[:80], [
        event.key for event in events
    ]


# 输入 profiler 事件名列表，返回当前允许表之外的 ATen 事件名，用于运行期回退检查。
def runtime_fallbacks(event_names):
    """输入 profiler 事件名；返回超出张量创建/元数据允许范围的 ATen 计算事件。"""
    # 下面允许创建/初始化张量、视图、复制及元数据相关操作；其他 aten:: 事件会被报告。
    allowed = {
        # 创建未初始化张量。
        "empty",
        # 按已有张量的属性创建未初始化张量。
        "empty_like",
        # 按指定形状和步长创建张量。
        "empty_strided",
        # 按排列布局创建未初始化张量。
        "empty_permuted",
        # 基于现有张量的 dtype/device 等创建未初始化张量。
        "new_empty",
        # 创建零初始化张量。
        "zeros",
        # 创建与现有张量属性对应的零张量。
        "zeros_like",
        # 将已有张量原位置零。
        "zero_",
        # 创建所有元素为 1 的张量。
        "ones",
        # 按已有张量属性创建全 1 张量。
        "ones_like",
        # 用指定标量填充已有张量。
        "fill_",
        # 创建视图，调整张量的逻辑形状。
        "view",
        # 允许框架内部使用的低层视图操作。
        "_unsafe_view",
        # 调整形状，按实际布局可能生成视图或副本。
        "reshape",
        # 按指定步长描述张量视图。
        "as_strided",
        # 解除梯度关联。
        "detach",
        # 创建共享存储的张量别名。
        "alias",
        # 转换设备或数据类型。
        "to",
        # 允许转换过程中产生的显式复制事件。
        "_to_copy",
        # 将数据复制到已有张量。
        "copy_",
        # 解析复数张量的共轭标记。
        "resolve_conj",
        # 解析张量的取负标记。
        "resolve_neg",
    }
    # 用集合去重，再排序，得到稳定的回退事件列表。
    return sorted(
        {
            name
            for name in event_names
            # 只检查 aten:: 前缀，截取命名空间后的操作名，与上方 allowed 表对照。
            if name.startswith("aten::") and name.split("::", 1)[1] not in allowed
        }
    )


class FallbackGuard(TorchDispatchMode):
    """复用既定 ATen 允许表，覆盖当前模型调用里的每个可分派运算。"""

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        """输入 PyTorch 算子及实参；违规时抛错，否则原样调用并返回结果。"""
        forbidden = runtime_fallbacks([func._schema.name])
        if forbidden:
            raise RuntimeFallbackError(forbidden)
        return func(*args, **(kwargs or {}))


# cfg 提供计时与容差配置，candidate 决定测原始模型还是候选；返回测时和审计字典。
def bench(cfg, candidate=False):
    """逐组重建可信输入，测量所有目标后释放；返回原有性能、profile 和状态字段。

    五组输入不再同时驻留 GPU；eager/compile 共享当前组的原始输入，各次调用用独立副本。
    每轮仍总共执行 timing_iterations 次测量，按原来的 i % 组数分配到各组。
    """
    inpmod = load("/inputs/model.py", "input_spec")
    if candidate:
        # 候选仍使用可信输入定义和固定参数，只加载冻结的 ModelNew。
        sys.path[:0] = ["/build", "/candidate"]
        mod = load("/candidate/model_new.py", "candidate_model")
        model = load_model(mod, "ModelNew", 71001, inpmod)
        models = {"timing": model}
    else:
        model = load_model(inpmod, "Model", 71001, inpmod)
        # compile 在第一组输入就绪后初始化，单列首次调用成本。
        models = {"eager": model, "compile": None}
    cases = case_list("feedback", cfg)
    # 仅保留标量测时结果；这里不会保存任意组的 GPU 张量。
    measurements = {
        name: [{"cuda_ms": [], "wall_ms": []} for _ in range(cfg["timing_rounds"])]
        for name in models
    }
    before = flags()
    result = {} if candidate else {"profiles": {}, "compile_mode": "default"}
    for index, case in enumerate(cases):
        # 同一组只生成一次，供两个参考基线共用；上一组已在 finally 中释放。
        sample = inputs_for(inpmod, case["input_seed"])
        try:
            # 例如每轮 10 次、5 组时，每组恰好 2 次；不能变成每组各测 10 次。
            iterations = len(range(index, cfg["timing_iterations"], len(cases)))
            for name in models:
                if name == "compile" and models[name] is None:
                    # 保护原始输入；准备副本和同步均放在编译首次调用计时之外。
                    first_inputs = tree(sample, lambda tensor: tensor.clone())
                    SYNC()
                    started = CLOCK()
                    models[name] = torch.compile(model, mode="default")
                    first_output = call(models[name], first_inputs)
                    SYNC()
                    result["compile_first_call_seconds"] = CLOCK() - started
                    del first_output, first_inputs
                rows = timing(models[name], sample, cfg, iterations)
                for total, current in zip(measurements[name], rows):
                    total["cuda_ms"].extend(current["cuda_ms"])
                    total["wall_ms"].extend(current["wall_ms"])
                # 与旧实现相同，只对第一组做 profiler；不额外生成大输入。
                if index == 0:
                    profile, names = profile_model(models[name], sample)
                    if candidate:
                        result["profile"] = profile
                        result["runtime_fallbacks"] = runtime_fallbacks(names)
                    else:
                        result["profiles"][name] = profile
        finally:
            # 及时解除当前组引用；缓存分配器可在下一组复用这些显存，无需清空整个缓存。
            del sample
    for name, rounds in measurements.items():
        result[name] = summarize_timing(rounds)
    result.update(
        tf32_before=before,
        tf32_after=flags(),
        state_mutations=[] if before == flags() == [True, True] else ["precision flags changed"],
    )
    return result


# 输入为张量树；修改其中浮点张量的内容但保持对象/存储地址，用于后续重新验算。
def perturb(inp):
    """输入张量树；原地址内更改浮点内容，返回修改后的输入，检查缓存投机。"""

    # 单张量转换函数，返回值仍是输入张量本身。
    def change(t):
        """输入一个张量；原位翻转批次或改变标量符号，返回相同张量对象。"""
        # 只改浮点张量，整数、布尔等类型原样保留。
        if t.is_floating_point():
            # 非标量沿第 0 维翻转，标量取负，再 copy_ 回原存储；特殊数据可能变换后仍相同。
            t.copy_(t.flip(0) if t.ndim else -t)
        # 保持张量对象不变，让检查覆盖依赖旧地址缓存结果的行为。
        return t

    # 递归应用原位修改，返回与输入结构对应的张量树。
    return tree(inp, change)


# cfg 提供种子，phase 指定反馈/隐藏阶段，candidate 指定参考/候选；返回输出次数和状态。
def stream(cfg, phase, candidate):
    """输入配置、阶段和参考/候选标记；逐例输出并等待控制器确认，返回次数及状态变化。"""
    def progress(stage, case=None):
        """输入阶段和可选案例 ID；立即打印北京时间诊断行，不参与计时或正确性判定。"""
        print(json.dumps({
            "event": "stream_progress",
            "at": datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="milliseconds"),
            "role": "candidate" if candidate else "reference",
            "phase": phase, "case": case, "stage": stage,
        }), flush=True)

    progress("load_input_spec")
    # 加载可信 get_inputs 定义，参考和候选都从这里重建测试输入。
    inpmod = load("/inputs/model.py", "input_spec")
    if candidate:
        # 候选需要从 /build 导入本次扩展，从 /candidate 导入候选模块。
        sys.path[:0] = ["/build", "/candidate"]
        # 加载冻结后的 model_new.py，而非正在编辑的工作区文件。
        mod = load("/candidate/model_new.py", "candidate_model")
        # 候选按约定使用 ModelNew 类。
        klass = "ModelNew"
    else:
        # 参考直接使用输入定义文件中保存的原始题目模块。
        mod = inpmod
        # 参考按约定使用 Model 类。
        klass = "Model"
    # 按初始化种子缓存模型实例；同一种初始化的多个输入会复用同一实例。
    models = {}
    # 统计已经被外部消费的输出条数，不代表数值通过条数。
    count = 0
    # 记录在哪些输出之后发现 TF32 设置不符合统一要求。
    mutations = []
    # 规定这两个标志的期望初始值，与 settings 保持一致。
    before = [True, True]

    def checked_call(model, inputs):
        """输入模型与一组输入；在原验算调用上检查回退，不另跑一遍模型。"""
        if candidate:
            with FallbackGuard():
                return call(model, inputs)
        return call(model, inputs)

    # label 是案例 ID，value 是输出树；发布一例输出并等待宿主确认，无返回值。
    def emit(label, value):
        """输入案例标签和输出值；原子写文件并等待外部确认，累积检查次数，返回 None。"""
        # 嵌套函数需要修改 stream 外层的输出计数器。
        nonlocal count
        # 宿主比较器轮询读取的固定文件名。
        channel = Path("/stream") if Path("/stream").is_dir() else Path("/outputs")
        target = channel / "item.pt"
        # 先写临时文件，避免宿主读到尚未写完的序列化内容。
        tmp = channel / "item.tmp"
        # 输出先转 CPU，再与案例 ID 一起序列化；参考和候选各有独立的输出挂载目录。
        progress("serialize_output", label)
        limit = cfg.get("case_storage_limit_bytes", 8 * 1024**3)
        # 先在 GPU 输出上估算，避免超大输出先复制到宿主内存；写文件仍有独立硬检查。
        if storage_bytes(value) > limit:
            raise StorageLimitError(f"output tensor storage exceeds {limit} bytes")
        save_limited({"id": label, "output": cpu(value)}, tmp, limit)
        # 在同一目录原子替换为 item.pt，表示这份输出已写完、可以读取。
        tmp.replace(target)
        progress("waiting_for_ack", label)
        # 只有外部比较器删除文件后才算确认，避免候选自行声称比较通过。
        # 文件还存在就继续等待；只有宿主消费后删除它，worker 才发布下一例。
        while target.exists():
            # 每次等待 10 毫秒，避免空转占用 CPU；容器总超时由控制器负责。
            time.sleep(0.01)
        # 外部确认本例已消费后再累加计数；数值是否通过由外部比较器单独记录。
        count += 1
        progress("acknowledged", label)
        # 在本次执行之后检查 TF32 状态是否仍符合统一设置。
        if flags() != before:
            # 保存发生状态变更的案例标签，交给宿主决定该候选是否合格。
            mutations.append(label)

    # 生成基础案例列表；隐藏阶段的附加检查在循环内及循环后追加。
    cases = case_list(phase, cfg)
    # 逐个重建案例、执行模型并发布输出，不一次性保存所有大张量结果。
    for case in cases:
        # 取得当前案例对应的参数初始化种子。
        s = case["init_seed"]
        progress("prepare_case", case["id"])
        # 首次遇到该种子时，才创建并恢复对应模型实例。
        if s not in models:
            models[s] = load_model(mod, klass, s, inpmod)
        # 后续同一种子的输入复用已加载模型。
        m = models[s]
        # 按本例输入种子调用可信 get_inputs，重建 GPU 输入。
        inp = inputs_for(inpmod, case["input_seed"])
        progress("inference", case["id"])
        # 执行该案例的正常模型调用。
        out = checked_call(m, inp)
        # 发布普通输出，等待外部逐例比较。
        emit(case["id"], out)
        # 这一例已被消费后，释放输出引用。
        del out
        # 只有隐藏验收阶段追加重复执行和原地址改输入检查。
        if phase == "hidden":
            # 对同一模型实例和同一输入再次调用，检查重复调用结果。
            progress("inference", case["id"] + "-repeat")
            repeated = checked_call(m, inp)
            # 用 -repeat 后缀区分这条比较记录。
            emit(case["id"] + "-repeat", repeated)
            del repeated
            # 在已有输入张量地址上修改浮点内容。
            perturb(inp)
            # 对改过内容的输入重新调用模型。
            progress("inference", case["id"] + "-changed")
            changed = checked_call(m, inp)
            # 用 -changed 后缀发布结果，外部参考端也执行相同变换并给出对应输出。
            emit(case["id"] + "-changed", changed)
            del changed
        # 普通及附加检查结束后，释放该案例的输入引用。
        del inp
    # 全部基础案例完成后，隐藏阶段再追加四条检查。
    if phase == "hidden":
        # 检查调用顺序变化，以及同一实例重新加载权重后是否使用了新参数。
        # 按最后一个、再第一个基础案例的顺序执行两次，检查调用顺序改变后的结果。
        for case in reversed([cases[0], cases[-1]]):
            # 取对应初始化下已经用过的模型，而非每次新建实例。
            m = models[case["init_seed"]]
            # 根据案例种子重新生成原始输入，避免复用刚被 perturb 改过的输入。
            inp = inputs_for(inpmod, case["input_seed"])
            progress("inference", case["id"] + "-reordered")
            out = checked_call(m, inp)
            # 用 -reordered 后缀发布乱序执行的结果。
            emit(case["id"] + "-reordered", out)
        # 选第一个初始化对应的同一模型实例，继续测试重新加载参数。
        m = models[cases[0]["init_seed"]]
        # 依次加载最后一个、再第一个案例对应的参数状态。
        for case in [cases[-1], cases[0]]:
            # 从可信输入目录读取该初始化的固定权重。
            d = torch.load(Path("/inputs") / f'state-{case["init_seed"]}.pt', weights_only=True)
            # 在同一实例中严格替换权重，检查模型是否真正使用更新后的参数。
            m.load_state_dict(d["state"], strict=True)
            # 重建与当前参数检查对应的输入。
            inp = inputs_for(inpmod, case["input_seed"])
            # 运行重新加载参数后的模型。
            progress("inference", case["id"] + "-reloaded")
            out = checked_call(m, inp)
            # 用 -reloaded 后缀发布结果，与外部参考输出对应。
            emit(case["id"] + "-reloaded", out)
    # 所有输出得到外部确认后，返回本次流式执行的摘要。
    return {
        # 本阶段实际发布并被消费的输出次数，宿主还会核对预期总数。
        "comparisons": count,
        # 非空表示至少一次输出后发现 TF32 状态不符合要求。
        "state_mutations": mutations,
        # 保留期望的初始精度状态。
        "tf32_before": before,
        # 保留最终状态，供宿主记录和复核。
        "tf32_after": flags(),
    }


def run_development_script(script, arguments):
    """输入冻结脚本相对路径和参数；在本 GPU 容器执行，返回不计分的完成状态。"""
    if not isinstance(script, str):
        raise ValueError("missing development script")
    root = Path("/experiment").resolve()
    source = (root / script).resolve()
    if not source.is_relative_to(root) or source.suffix != ".py" or not source.is_file():
        raise ValueError("script must stay inside /experiment")
    if not isinstance(arguments, list) or not all(isinstance(item, str) for item in arguments):
        raise ValueError("script arguments must be strings")
    # import cuda_extension 优先读取控制器构建的 .so；不导入原始参考模型。
    sys.path[:0] = ["/build", "/candidate", str(source.parent), str(root)]
    sys.argv = [str(source), *arguments]
    try:
        runpy.run_path(str(source), run_name="__main__")
    except SystemExit as error:
        if error.code not in {None, 0}:
            raise RuntimeError(f"development script exited with {error.code}") from error
    SYNC()
    return {"scored": False, "action": "run_script", "script": script}


def main():
    """读取容器模式和阶段；固定环境并运行 worker，写对应阶段 JSON，返回 0/1。"""
    p = argparse.ArgumentParser()
    p.add_argument("mode")
    p.add_argument("--phase", default="feedback")
    p.add_argument("--script")
    p.add_argument("--script-args", default="[]")
    a = p.parse_args()
    cfg = json.loads(Path("/config.json").read_text())
    # 在执行选定模式之前统一 TF32、cuDNN 和线程设置。
    settings()
    started = CLOCK()
    try:
        result = {
            # compile：从候选源码构建并加载 cuda_extension。
            "compile": compile_candidate,
            # reference_prepare：准备参数快照并审计原始模型输出。
            "reference_prepare": lambda: prepare(cfg),
            # reference_bench：测量原始模型 eager/compile 基线。
            "reference_bench": lambda: bench(cfg),
            # candidate_bench：测量候选扩展模型，并收集运行期回退事件。
            "candidate_bench": lambda: bench(cfg, True),
            # reference_stream：参考端按指定阶段逐例发布输出。
            "reference_stream": lambda: stream(cfg, a.phase, False),
            # candidate_stream：候选端执行同一案例协议，供宿主配对比较。
            "candidate_stream": lambda: stream(cfg, a.phase, True),
            # 仅执行开发脚本，不计算 reward、不调用参考实现或隐藏输入。
            "script": lambda: run_development_script(a.script, json.loads(a.script_args)),
        }[a.mode]()
        result.update(
            status="ok",
            elapsed_seconds=CLOCK() - started,
            gpu_uuid=(None if a.mode == "compile" else str(torch.cuda.get_device_properties(0).uuid)),
        )
    except Exception as e:
        result = {
            "status": "error",
            "error_type": type(e).__name__,
            "error": str(e),
            "traceback": traceback.format_exc()[-10000:],
            "elapsed_seconds": CLOCK() - started,
        }
        # 环境缺依赖与参考程序自身不稳定分别记录，避免把二者都解释成数据无效。
        if isinstance(e, (ModuleNotFoundError, ImportError)):
            result.update(failure_kind="missing_dependency", missing_module=getattr(e, "name", None))
        elif isinstance(e, ReferenceValidationError):
            result["failure_kind"] = e.kind
        elif isinstance(e, StorageLimitError):
            result["failure_kind"] = "storage_limit"
        elif isinstance(e, RuntimeFallbackError):
            result["runtime_fallbacks"] = e.names
        elif isinstance(e, OSError) and e.errno in {28, 122}:
            # 磁盘/配额不足是环境问题，区别于候选超过事先约定的单输出上限。
            result["failure_kind"] = "storage_unavailable"
    # 与控制器使用相同的阶段文件名；宿主读取后会在同一文件补上 controller_seconds。
    result_name = {
        "reference_prepare": "prepare_result.json",
        "reference_bench": "bench_result.json",
    }.get(a.mode, "result.json")
    (Path("/outputs") / result_name).write_text(json.dumps(result, indent=2))
    # 详情已在 JSON 中保存；终端只输出阶段摘要与错误，避免 worker.log 重复整份结果。
    summary = {"mode": a.mode, "phase": a.phase, "result_file": result_name}
    for key in ("status", "elapsed_seconds", "error_type", "error", "traceback", "failure_kind"):
        if key in result:
            summary[key] = result[key]
    print(json.dumps(summary), flush=True)
    return 0 if result["status"] == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())

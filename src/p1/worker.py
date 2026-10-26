"""可信容器工作进程：输入重建、CUDA 编译、验证、计时与 profiler。

运行：由 Evaluator.container 在隔离容器内执行 /worker.py <模式>。
输入：只读 /config.json、/reference 或 /candidate、/inputs；输出：/outputs。
不要在宿主直接运行；CPU 单测只导入纯辅助函数，不调用 GPU。"""

import argparse
import importlib.util
import json
from pathlib import Path
import random
import statistics
import sys
import time
import traceback
import numpy as np
import torch

# 保存 CUDA 同步函数的引用；调用时等待当前设备的 CUDA 工作完成。
SYNC = torch.cuda.synchronize
# 保存 CUDA Event 类的引用，后续用事件记录 GPU 时间区间。
EVENT = torch.cuda.Event
# 保存序列化函数的引用，用于写模型参数和逐例输出快照。
SAVE = torch.save
# 保存高精度计时函数，返回秒数差值；它用于测时长，不是日历时间戳。
CLOCK = time.perf_counter


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
def load_model(mod, klass, init_seed):
    """输入模块、类名和初始化种子；严格加载固定参数，返回 eval 模式 GPU 模型。"""
    # 读取可信 input-spec 中对应种子的快照，包含构造参数 init 和权重 state。
    d = torch.load(Path("/inputs") / f"state-{init_seed}.pt", weights_only=True)
    # 按类名取得模型类，用相同构造参数建模，再切换 eval 模式并放到 GPU。
    m = getattr(mod, klass)(*d["init"]).eval().cuda()
    # 严格恢复固定 state_dict；参数键缺失、多余或张量形状不兼容会触发错误。
    m.load_state_dict(d["state"], strict=True)
    return m


# 参考准备入口：cfg 是配置字典，源码来自 /reference/model.py。
# 输出文件写入 /outputs/input-spec；返回输出审计、近常数标记及案例数量等信息。
def prepare(cfg):
    """输入配置；核查参考确定性/有限值/近常数，写输入规格及参数，返回审计元数据。"""
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
        # 保存构造参数和 CPU 版 state_dict，后续所有模型都严格恢复这份参数。
        SAVE({"init": init, "state": cpu(m.state_dict())}, out / "input-spec" / f"state-{s}.pt")
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
            def equal(x, y):
                """输入同一参考程序的两次输出；递归确认完全一致，不一致抛错，返回 None。"""
                # 张量节点执行有限性检查和零容差数值比较。
                if isinstance(x, torch.Tensor):
                    # 显式检查浮点输出中是否存在 NaN 或 Inf。
                    if x.is_floating_point() and not torch.isfinite(x).all():
                        raise ValueError("reference has nonfinite output")
                    # atol/rtol 都为 0，要求数值完全相等，且不接受 NaN 与 NaN 相等。
                    torch.testing.assert_close(x, y, atol=0, rtol=0, equal_nan=False)
                # 列表或元组按对应位置递归检查。
                elif isinstance(x, (list, tuple)):
                    for v, w in zip(x, y):
                        equal(v, w)
                # 字典按当前输出的键递归检查对应值。
                elif isinstance(x, dict):
                    for k in x:
                        equal(x[k], y[k])
                # 普通标量使用精确相等断言。
                else:
                    assert x == y

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
    # 展开各初始化/输入/输出张量的采样数组，只保留非空张量。
    arrays = [v["sample"] for m in metadata for o in m["output_audits"] for v in o if v["numel"]]
    # 没有可检查的张量采样时，默认不标记近常数。
    constant = False
    # 存在有效采样数组时，再执行近常数判断。
    if arrays:
        # 以第一份采样的第一个值作为共同参照值。
        anchor = arrays[0][0]
        # 检查所有采样值是否都落在同一个参照值的绝对/相对容差范围内。
        constant = all(
            abs(v - anchor) <= cfg["atol"] + cfg["rtol"] * abs(anchor) for a in arrays for v in a
        )
    # 返回准备阶段的审计结果，供宿主决定是否接受这道题。
    return {
        # 这是基于采样值的近常数标记，宿主据此过滤退化题。
        "near_constant_sampled_output": constant,
        # 保留上述逐初始化、逐输入的输出审计。
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
    # PyTorch 扩展加载器负责调用编译工具，生成并加载 Python 可导入的原生扩展。
    from torch.utils.cpp_extension import load as extload

    # 收集固定绑定源文件和候选 kernels 下的 CUDA/C++ 源码。
    sources = (
        # 绑定层负责把 C++/CUDA 接口暴露给 Python 模块。
        [Path("/candidate/binding.cpp")]
        # 按文件名排序 CUDA 源文件，使源码列表顺序稳定。
        + sorted(Path("/candidate/kernels").glob("*.cu"))
        # 追加同目录下的 C++ 实现文件。
        + sorted(Path("/candidate/kernels").glob("*.cpp"))
    )
    # 至少要求存在一个 .cu 文件，避免完全没有 CUDA 源码的提交进入编译。
    if not any(p.suffix == ".cu" for p in sources):
        raise ValueError("no CUDA source")
    # 从实际导入的 torch 路径定位 kernel 环境的 site-packages。
    site = Path(torch.__file__).resolve().parents[1]
    # 允许扩展使用该环境中 NVIDIA cuDNN/cuBLAS 包提供的头文件。
    includes = [str(site / "nvidia/cudnn/include"), str(site / "nvidia/cublas/include")]
    # 指定 cuDNN/cuBLAS 动态库的实际文件路径，供链接阶段使用。
    libs = [site / "nvidia/cudnn/lib/libcudnn.so.9", site / "nvidia/cublas/lib/libcublas.so.12"]
    # 开始构建并加载扩展；发生编译、链接或加载错误时异常由 main 收集。
    extload(
        # 固定扩展名，候选 Python 可通过 import cuda_extension 使用。
        name="cuda_extension",
        # 把 Path 列表转换为构建工具接受的路径字符串列表。
        sources=[str(p) for p in sources],
        # 所有构建中间文件和生成扩展进入独立的可写 /build。
        build_directory="/build",
        # 明确启用 CUDA 扩展构建。
        with_cuda=True,
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
    # 这里只表示扩展编译/加载成功，不表示数值正确或性能达标。
    return {"compiled": True}


# model 是待测模型，samples 是 GPU 输入列表，cfg 提供轮数和每轮迭代数。
# 返回 cuda_ms、wall_ms 和逐轮结果；计时单位为毫秒，预热和输入复制在测量区间外。
def timing(model, samples, cfg):
    """输入可调用模型、多组 GPU 输入和配置；预热并重复同步测量，返回 CUDA/墙钟中位数。"""
    # 先让每一组输入各执行一次，完成首次调用和相关运行准备。
    for x in samples:
        call(model, tree(x, lambda t: t.clone()))
    # 等待预热及此前 CUDA 工作全部结束。
    SYNC()
    # 保存各轮的代表耗时，最后再跨轮取中位数。
    rounds = []
    for _ in range(cfg["timing_rounds"]):
        # 本轮每次迭代的 CUDA Event 耗时列表。
        events = []
        # 本轮每次迭代的宿主墙钟耗时列表。
        walls = []
        for i in range(cfg["timing_iterations"]):
            x = tree(samples[i % len(samples)], lambda t: t.clone())
            # 在计时开始前等待输入复制及已有 GPU 工作结束。
            SYNC()
            # 创建开始/结束事件，并启用 GPU 时间统计。
            a, b = EVENT(enable_timing=True), EVENT(enable_timing=True)
            t = CLOCK()
            # 在当前 CUDA 流记录开始事件。
            a.record()
            # 执行一次模型调用；这是本次测量的主体。
            out = call(model, x)
            # 在当前 CUDA 流记录结束事件。
            b.record()
            # 等待 GPU 工作完成，使墙钟和事件时间都可读取完整结果。
            SYNC()
            walls.append((CLOCK() - t) * 1e3)
            events.append(a.elapsed_time(b))
            # 测量完成后释放输出引用，不把输出销毁放在本次计时区间内。
            del out
        rounds.append({"cuda_ms": statistics.median(events), "wall_ms": statistics.median(walls)})
    return {
        # 各轮 CUDA 代表耗时的中位数，用于控制器计算加速比。
        "cuda_ms": statistics.median(x["cuda_ms"] for x in rounds),
        # 各轮墙钟代表耗时的中位数，与 CUDA 时间分别报告。
        "wall_ms": statistics.median(x["wall_ms"] for x in rounds),
        "rounds": rounds,
    }


# 输入模型和一组 GPU 样本；返回按设备耗时排序的摘要，以及完整事件名列表。
def profile_model(model, sample):
    """输入可调用模型及一组 GPU 输入；返回算子统计与全部事件名，不参与计时。"""
    # 启用 profiler 执行单次模型调用；采样独立于上方 timing，不混入性能计时。
    with torch.profiler.profile(
        # 同时记录 CPU 与 CUDA 活动，便于识别运算调用及其设备开销。
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    ) as profiler:
        # 执行传入样本，不在此处计算输出正确性。
        call(model, sample)
        # 等待 CUDA 工作完成，让 profiler 捕获该次调用产生的设备活动。
        SYNC()
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


# cfg 提供计时与容差配置，candidate 决定测原始模型还是候选；返回测时和审计字典。
def bench(cfg, candidate=False):
    """输入配置和是否候选；返回 eager/compile 或候选耗时、profile、精度状态和运行期回退。"""
    # 始终从可信输入定义生成样本，即使测候选也不使用候选自定义的输入生成器。
    inpmod = load("/inputs/model.py", "input_spec")
    # 候选分支加载 ModelNew，参考分支加载原始 Model。
    if candidate:
        # 把扩展构建目录和候选目录放到导入搜索路径前面，以便找到 cuda_extension。
        sys.path[:0] = ["/build", "/candidate"]
        # 动态加载冻结的候选 Python 入口文件。
        mod = load("/candidate/model_new.py", "candidate_model")
        # 恢复反馈初始化种子 71001 对应的固定参数。
        model = load_model(mod, "ModelNew", 71001)
    else:
        # 参考分支用同一份固定参数恢复原始 Model。
        model = load_model(inpmod, "Model", 71001)
    # 性能测试统一使用 feedback 案例的输入种子，先生成全部 GPU 样本。
    samples = [inputs_for(inpmod, c["input_seed"]) for c in case_list("feedback", cfg)]
    # 在测时开始前记录两项 TF32 设置。
    before = flags()
    result = {}
    if candidate:
        result["timing"] = timing(model, samples, cfg)
        # 额外对第一组输入做 profiler 采样，取得展示摘要和完整事件名。
        result["profile"], names = profile_model(model, samples[0])
        # 把不允许的 ATen 事件列入 runtime_fallbacks，供宿主判断候选是否合规。
        result["runtime_fallbacks"] = runtime_fallbacks(names)
    else:
        # 先测未经 torch.compile 包装的 PyTorch eager 模型。
        result["eager"] = timing(model, samples, cfg)
        # 另外统计本次创建编译包装器并完成首次调用的耗时。
        started = CLOCK()
        # 使用 torch.compile 的 default 模式包装原始模型。
        compiled = torch.compile(model, mode="default")
        call(compiled, samples[0])
        SYNC()
        # 首调成本单独以秒记录，不混入下面的稳态毫秒时间。
        result["compile_first_call_seconds"] = CLOCK() - started
        # 再通过相同 timing 流程测量编译模型的稳态性能。
        result["compile"] = timing(compiled, samples, cfg)
        # 基线本身也要核对语义；编译/首调成本与稳态 kernel 时间分开报告。
        # 对每组样本核对编译模型与 eager 模型，确认基线本身也保持题目语义。
        for sample in samples:
            torch.testing.assert_close(
                call(compiled, sample), call(model, sample), atol=cfg["atol"], rtol=cfg["rtol"]
            )
        # 只有全部样本的编译/eager 比较都通过，才写入 compile_correct=True。
        result["compile_correct"] = True
        result["profiles"] = {
            "eager": profile_model(model, samples[0])[0],
            "compile": profile_model(compiled, samples[0])[0],
        }
        # 在结果中明确记录这次编译基线使用的模式。
        result["compile_mode"] = "default"
    # 无论参考或候选，都附上运行前后的 TF32 设置及变更标记。
    result.update(
        tf32_before=before,
        tf32_after=flags(),
        # 前后必须都等于 [True, True]；否则记录精度标志发生变化。
        state_mutations=[] if before == flags() == [True, True] else ["precision flags changed"],
    )
    # 把性能及状态交回 main，再由 main 补充成功状态和执行时长。
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

    # label 是案例 ID，value 是输出树；发布一例输出并等待宿主确认，无返回值。
    def emit(label, value):
        """输入案例标签和输出值；原子写文件并等待外部确认，累积检查次数，返回 None。"""
        # 嵌套函数需要修改 stream 外层的输出计数器。
        nonlocal count
        # 宿主比较器轮询读取的固定文件名。
        target = Path("/outputs/item.pt")
        # 先写临时文件，避免宿主读到尚未写完的序列化内容。
        tmp = Path("/outputs/item.tmp")
        # 输出先转 CPU，再与案例 ID 一起序列化；参考和候选各有独立的输出挂载目录。
        SAVE({"id": label, "output": cpu(value)}, tmp)
        # 在同一目录原子替换为 item.pt，表示这份输出已写完、可以读取。
        tmp.replace(target)
        # 只有外部比较器删除文件后才算确认，避免候选自行声称比较通过。
        # 文件还存在就继续等待；只有宿主消费后删除它，worker 才发布下一例。
        while target.exists():
            # 每次等待 10 毫秒，避免空转占用 CPU；容器总超时由控制器负责。
            time.sleep(0.01)
        # 外部确认本例已消费后再累加计数；数值是否通过由外部比较器单独记录。
        count += 1
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
        # 首次遇到该种子时，才创建并恢复对应模型实例。
        if s not in models:
            models[s] = load_model(mod, klass, s)
        # 后续同一种子的输入复用已加载模型。
        m = models[s]
        # 按本例输入种子调用可信 get_inputs，重建 GPU 输入。
        inp = inputs_for(inpmod, case["input_seed"])
        # 执行该案例的正常模型调用。
        out = call(m, inp)
        # 发布普通输出，等待外部逐例比较。
        emit(case["id"], out)
        # 这一例已被消费后，释放输出引用。
        del out
        # 只有隐藏验收阶段追加重复执行和原地址改输入检查。
        if phase == "hidden":
            # 对同一模型实例和同一输入再次调用，检查重复调用结果。
            repeated = call(m, inp)
            # 用 -repeat 后缀区分这条比较记录。
            emit(case["id"] + "-repeat", repeated)
            del repeated
            # 在已有输入张量地址上修改浮点内容。
            perturb(inp)
            # 对改过内容的输入重新调用模型。
            changed = call(m, inp)
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
            out = call(m, inp)
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
            out = call(m, inp)
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


def main():
    """读取容器模式和阶段；固定环境并运行 worker，写对应阶段 JSON，返回 0/1。"""
    p = argparse.ArgumentParser()
    p.add_argument("mode")
    p.add_argument("--phase", default="feedback")
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
        }[a.mode]()
        result.update(
            status="ok",
            elapsed_seconds=CLOCK() - started,
            gpu_uuid=str(torch.cuda.get_device_properties(0).uuid),
        )
    except Exception as e:
        result = {
            "status": "error",
            "error_type": type(e).__name__,
            "error": str(e),
            "traceback": traceback.format_exc()[-10000:],
            "elapsed_seconds": CLOCK() - started,
        }
    # 与控制器使用相同的阶段文件名；宿主读取后会在同一文件补上 controller_seconds。
    result_name = {
        "reference_prepare": "prepare_result.json",
        "reference_bench": "bench_result.json",
    }.get(a.mode, "result.json")
    (Path("/outputs") / result_name).write_text(json.dumps(result, indent=2))
    print(json.dumps(result))
    return 0 if result["status"] == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())

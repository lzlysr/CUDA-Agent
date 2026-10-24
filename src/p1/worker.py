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

SYNC = torch.cuda.synchronize
EVENT = torch.cuda.Event
SAVE = torch.save
CLOCK = time.perf_counter


def flags():
    """无输入；返回 matmul 和 cuDNN 的 TF32 状态列表。"""
    return [torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32]


def settings():
    """无输入；固定 TF32、线程及 cuDNN benchmark 设置，返回 None。"""
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = False
    torch.set_num_threads(4)


def seed(n):
    """输入整数种子；同步 Python、NumPy、torch 和 CUDA 随机源，返回 None。"""
    random.seed(n)
    np.random.seed(n)
    torch.manual_seed(n)
    torch.cuda.manual_seed_all(n)


def tree(x, fn):
    """输入嵌套输出树及 Tensor 回调；返回结构对应的转换结果，tuple 统一为普通 tuple。"""
    if isinstance(x, torch.Tensor):
        return fn(x)
    if isinstance(x, list):
        return [tree(v, fn) for v in x]
    if isinstance(x, tuple):
        return tuple(tree(v, fn) for v in x)
    if isinstance(x, dict):
        return {k: tree(v, fn) for k, v in x.items()}
    return x


def as_list(x):
    """输入题目接口返回值；将单值包装成列表，列表/元组原样返回。"""
    return x if isinstance(x, (list, tuple)) else [x]


def load(path, name):
    """输入只读 Python 文件及模块名；加载并返回任务或候选模块，仅在隔离 worker 使用。"""
    s = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


def cpu(x):
    """输入输出树；返回 detach 后复制到 CPU 的同结构内容，供外部比较。"""
    return tree(x, lambda t: t.detach().cpu())


def call(model, inputs):
    """输入模型和输入列表；在 no_grad 下执行 forward 并返回结果。"""
    with torch.no_grad():
        return model(*inputs)


def inputs_for(mod, n):
    """输入任务模块和输入种子；重建输入并复制到 GPU，返回输入列表。"""
    seed(n)
    return tree(as_list(mod.get_inputs()), lambda t: t.cuda())


def case_list(phase, cfg):
    """输入反馈/隐藏阶段及配置；返回案例 ID、初始化种子和输入种子列表。"""
    inits = [71001] if phase == "feedback" else cfg["hidden_init_seeds"]
    n = cfg["feedback_cases"] if phase == "feedback" else cfg["hidden_cases_per_init"]
    return [
        {
            "id": f"{phase}-{s}-{i:02d}",
            "init_seed": s,
            "input_seed": 72000 + i if phase == "feedback" else s * 100 + i,
        }
        for s in inits
        for i in range(n)
    ]


def load_model(mod, klass, init_seed):
    """输入模块、类名和初始化种子；严格加载固定参数，返回 eval 模式 GPU 模型。"""
    d = torch.load(Path("/inputs") / f"state-{init_seed}.pt", weights_only=True)
    m = getattr(mod, klass)(*d["init"]).eval().cuda()
    m.load_state_dict(d["state"], strict=True)
    return m


def prepare(cfg):
    """输入配置；核查参考确定性/有限值/近常数，写输入规格及参数，返回审计元数据。"""
    mod = load("/reference/model.py", "reference_task")
    metadata = []
    out = Path("/outputs")
    (out / "input-spec").mkdir(exist_ok=True)
    (out / "input-spec/model.py").write_text(Path("/reference/model.py").read_text())
    for s in [71001] + cfg["hidden_init_seeds"]:
        seed(s)
        init = as_list(mod.get_init_inputs())
        m = mod.Model(*init).eval().cuda()
        SAVE({"init": init, "state": cpu(m.state_dict())}, out / "input-spec" / f"state-{s}.pt")
        # 先检查初始化、非有限值与确定性；更完整的输入验收在候选评测阶段执行。
        outputs = []
        for input_seed in [72000, 72001]:
            inp = inputs_for(mod, input_seed)
            a = call(m, tree(inp, lambda t: t.clone()))
            b = call(m, tree(inp, lambda t: t.clone()))

            def equal(x, y):
                """输入同一参考程序的两次输出；递归确认完全一致，不一致抛错，返回 None。"""
                if isinstance(x, torch.Tensor):
                    if x.is_floating_point() and not torch.isfinite(x).all():
                        raise ValueError("reference has nonfinite output")
                    torch.testing.assert_close(x, y, atol=0, rtol=0, equal_nan=False)
                elif isinstance(x, (list, tuple)):
                    for v, w in zip(x, y):
                        equal(v, w)
                elif isinstance(x, dict):
                    for k in x:
                        equal(x[k], y[k])
                else:
                    assert x == y

            equal(a, b)
            stats = []
            tree(
                a,
                lambda t: stats.append(
                    {
                        "shape": list(t.shape),
                        "dtype": str(t.dtype),
                        "numel": t.numel(),
                        "sample": t.flatten()[:: max(1, t.numel() // 2048)][:2048]
                        .detach()
                        .float()
                        .cpu()
                        .tolist(),
                    }
                ),
            )
            outputs.append(stats)
            del inp, a, b
        metadata.append(
            {
                "init_seed": s,
                "output_audits": outputs,
                "same_output_samples_across_inputs": outputs[0] == outputs[1],
            }
        )
        del m
        torch.cuda.empty_cache()
    arrays = [v["sample"] for m in metadata for o in m["output_audits"] for v in o if v["numel"]]
    constant = False
    if arrays:
        anchor = arrays[0][0]
        constant = all(
            abs(v - anchor) <= cfg["atol"] + cfg["rtol"] * abs(anchor) for a in arrays for v in a
        )
    return {
        "near_constant_sampled_output": constant,
        "metadata": metadata,
        "cases_feedback": len(case_list("feedback", cfg)),
        "cases_hidden": len(case_list("hidden", cfg)),
        "tf32": flags(),
    }


def compile_candidate():
    """无参数；使用固定源文件/编译参数生成扩展，返回 compiled=True，失败抛错。"""
    from torch.utils.cpp_extension import load as extload

    sources = (
        [Path("/candidate/binding.cpp")]
        + sorted(Path("/candidate/kernels").glob("*.cu"))
        + sorted(Path("/candidate/kernels").glob("*.cpp"))
    )
    if not any(p.suffix == ".cu" for p in sources):
        raise ValueError("no CUDA source")
    site = Path(torch.__file__).resolve().parents[1]
    includes = [str(site / "nvidia/cudnn/include"), str(site / "nvidia/cublas/include")]
    libs = [site / "nvidia/cudnn/lib/libcudnn.so.9", site / "nvidia/cublas/lib/libcublas.so.12"]
    extload(
        name="cuda_extension",
        sources=[str(p) for p in sources],
        build_directory="/build",
        with_cuda=True,
        verbose=False,
        extra_include_paths=includes,
        extra_ldflags=[str(p) for p in libs],
        extra_cflags=["-O3", "-std=c++17"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
    )
    return {"compiled": True}


def timing(model, samples, cfg):
    """输入可调用模型、多组 GPU 输入和配置；预热并重复同步测量，返回 CUDA/墙钟中位数。"""
    for x in samples:
        call(model, tree(x, lambda t: t.clone()))
    SYNC()
    rounds = []
    for _ in range(cfg["timing_rounds"]):
        events = []
        walls = []
        for i in range(cfg["timing_iterations"]):
            x = tree(samples[i % len(samples)], lambda t: t.clone())
            SYNC()
            a, b = EVENT(enable_timing=True), EVENT(enable_timing=True)
            t = CLOCK()
            a.record()
            out = call(model, x)
            b.record()
            SYNC()
            walls.append((CLOCK() - t) * 1e3)
            events.append(a.elapsed_time(b))
            del out
        rounds.append({"cuda_ms": statistics.median(events), "wall_ms": statistics.median(walls)})
    return {
        "cuda_ms": statistics.median(x["cuda_ms"] for x in rounds),
        "wall_ms": statistics.median(x["wall_ms"] for x in rounds),
        "rounds": rounds,
    }


def profile_model(model, sample):
    """输入可调用模型及一组 GPU 输入；返回算子统计与全部事件名，不参与计时。"""
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    ) as profiler:
        call(model, sample)
        SYNC()
    events = list(profiler.key_averages())
    rows = [
        {"name": event.key, "calls": event.count, "device_us": event.device_time_total}
        for event in events
        if event.device_time_total > 0
    ]
    return sorted(rows, key=lambda row: row["device_us"], reverse=True)[:80], [
        event.key for event in events
    ]


def runtime_fallbacks(event_names):
    """输入 profiler 事件名；返回超出张量创建/元数据允许范围的 ATen 计算事件。"""
    allowed = {
        "empty",
        "empty_like",
        "empty_strided",
        "empty_permuted",
        "new_empty",
        "zeros",
        "zeros_like",
        "zero_",
        "ones",
        "ones_like",
        "fill_",
        "view",
        "_unsafe_view",
        "reshape",
        "as_strided",
        "detach",
        "alias",
        "to",
        "_to_copy",
        "copy_",
        "resolve_conj",
        "resolve_neg",
    }
    return sorted(
        {
            name
            for name in event_names
            if name.startswith("aten::") and name.split("::", 1)[1] not in allowed
        }
    )


def bench(cfg, candidate=False):
    """输入配置和是否候选；返回 eager/compile 或候选耗时、profile、精度状态和运行期回退。"""
    inpmod = load("/inputs/model.py", "input_spec")
    if candidate:
        sys.path[:0] = ["/build", "/candidate"]
        mod = load("/candidate/model_new.py", "candidate_model")
        model = load_model(mod, "ModelNew", 71001)
    else:
        model = load_model(inpmod, "Model", 71001)
    samples = [inputs_for(inpmod, c["input_seed"]) for c in case_list("feedback", cfg)]
    before = flags()
    result = {}
    if candidate:
        result["timing"] = timing(model, samples, cfg)
        result["profile"], names = profile_model(model, samples[0])
        result["runtime_fallbacks"] = runtime_fallbacks(names)
    else:
        result["eager"] = timing(model, samples, cfg)
        started = CLOCK()
        compiled = torch.compile(model, mode="default")
        call(compiled, samples[0])
        SYNC()
        result["compile_first_call_seconds"] = CLOCK() - started
        result["compile"] = timing(compiled, samples, cfg)
        # 基线本身也要核对语义；编译/首调成本与稳态 kernel 时间分开报告。
        for sample in samples:
            torch.testing.assert_close(
                call(compiled, sample), call(model, sample), atol=cfg["atol"], rtol=cfg["rtol"]
            )
        result["compile_correct"] = True
        result["profiles"] = {
            "eager": profile_model(model, samples[0])[0],
            "compile": profile_model(compiled, samples[0])[0],
        }
        result["compile_mode"] = "default"
    result.update(
        tf32_before=before,
        tf32_after=flags(),
        state_mutations=[] if before == flags() == [True, True] else ["precision flags changed"],
    )
    return result


def perturb(inp):
    """输入张量树；原地址内更改浮点内容，返回修改后的输入，检查缓存投机。"""

    def change(t):
        """输入一个张量；原位翻转批次或改变标量符号，返回相同张量对象。"""
        if t.is_floating_point():
            t.copy_(t.flip(0) if t.ndim else -t)
        return t

    return tree(inp, change)


def stream(cfg, phase, candidate):
    """输入配置、阶段和参考/候选标记；逐例输出并等待控制器确认，返回次数及状态变化。"""
    inpmod = load("/inputs/model.py", "input_spec")
    if candidate:
        sys.path[:0] = ["/build", "/candidate"]
        mod = load("/candidate/model_new.py", "candidate_model")
        klass = "ModelNew"
    else:
        mod = inpmod
        klass = "Model"
    models = {}
    count = 0
    mutations = []
    before = [True, True]

    def emit(label, value):
        """输入案例标签和输出值；原子写文件并等待外部确认，累积检查次数，返回 None。"""
        nonlocal count
        target = Path("/outputs/item.pt")
        tmp = Path("/outputs/item.tmp")
        SAVE({"id": label, "output": cpu(value)}, tmp)
        tmp.replace(target)
        # 只有外部比较器删除文件后才算确认，避免候选自行声称比较通过。
        while target.exists():
            time.sleep(0.01)
        count += 1
        if flags() != before:
            mutations.append(label)

    cases = case_list(phase, cfg)
    for case in cases:
        s = case["init_seed"]
        if s not in models:
            models[s] = load_model(mod, klass, s)
        m = models[s]
        inp = inputs_for(inpmod, case["input_seed"])
        out = call(m, inp)
        emit(case["id"], out)
        del out
        if phase == "hidden":
            repeated = call(m, inp)
            emit(case["id"] + "-repeat", repeated)
            del repeated
            perturb(inp)
            changed = call(m, inp)
            emit(case["id"] + "-changed", changed)
            del changed
        del inp
    if phase == "hidden":
        # 检查调用顺序变化，以及同一实例重新加载权重后是否使用了新参数。
        for case in reversed([cases[0], cases[-1]]):
            m = models[case["init_seed"]]
            inp = inputs_for(inpmod, case["input_seed"])
            out = call(m, inp)
            emit(case["id"] + "-reordered", out)
        m = models[cases[0]["init_seed"]]
        for case in [cases[-1], cases[0]]:
            d = torch.load(Path("/inputs") / f'state-{case["init_seed"]}.pt', weights_only=True)
            m.load_state_dict(d["state"], strict=True)
            inp = inputs_for(inpmod, case["input_seed"])
            out = call(m, inp)
            emit(case["id"] + "-reloaded", out)
    return {
        "comparisons": count,
        "state_mutations": mutations,
        "tf32_before": before,
        "tf32_after": flags(),
    }


def main():
    """读取容器模式和阶段；固定环境并运行 worker，写 result.json，返回 0/1。"""
    p = argparse.ArgumentParser()
    p.add_argument("mode")
    p.add_argument("--phase", default="feedback")
    a = p.parse_args()
    cfg = json.loads(Path("/config.json").read_text())
    settings()
    started = CLOCK()
    try:
        result = {
            "compile": compile_candidate,
            "reference_prepare": lambda: prepare(cfg),
            "reference_bench": lambda: bench(cfg),
            "candidate_bench": lambda: bench(cfg, True),
            "reference_stream": lambda: stream(cfg, a.phase, False),
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
    Path("/outputs/result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result))
    return 0 if result["status"] == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())

"""外部评测控制器：候选快照、隔离调度、数值比较及公平测时。

输入：配置、固定题目和候选工作区；输出：每次评测 JSON、原始日志及奖励。
运行：由 python -m p1.run 或 python -m p1.baseline 调用，不单独执行。
参考和候选分容器，控制器在 CPU 比较；基础设施故障不记作模型失败。"""

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
from .data import ROOT, dump, sha

torch.set_num_threads(4)


class InfrastructureError(RuntimeError):
    pass


from .contract import audit_contract, contract


def compare(actual, expected, atol=0.01, rtol=0.01):
    """输入候选/参考输出树及绝对/相对容差；返回 passed、误差摘要及逐层错误。"""
    errors = []
    max_abs = 0.0
    max_rel = 0.0
    finite = True

    def walk(a, b, key="output"):
        """输入一对输出节点及路径；递归比较并更新外层错误/误差统计，返回 None。"""
        nonlocal max_abs, max_rel, finite
        if type(a) != type(b):
            errors.append(key + ": output type differs")
            return
        if isinstance(a, torch.Tensor):
            if a.shape != b.shape or a.dtype != b.dtype:
                errors.append(key + ": shape/dtype differs")
                return
            if a.is_floating_point() or a.is_complex():
                finite &= bool(torch.isfinite(a).all() and torch.isfinite(b).all())
                delta = (a - b).abs()
                max_abs = max(max_abs, float(delta.max()) if delta.numel() else 0.0)
                max_rel = max(
                    max_rel, float((delta / (b.abs() + 1e-8)).max()) if delta.numel() else 0.0
                )
            try:
                torch.testing.assert_close(a, b, atol=atol, rtol=rtol, equal_nan=False)
            except AssertionError:
                errors.append(key + ": numerical mismatch")
        elif isinstance(a, (list, tuple)):
            if len(a) != len(b):
                errors.append(key + ": length differs")
                return
            for i, (v, w) in enumerate(zip(a, b)):
                walk(v, w, f"{key}.{i}")
        elif isinstance(a, dict):
            if a.keys() != b.keys():
                errors.append(key + ": keys differ")
                return
            for k in b:
                walk(a[k], b[k], key + "." + str(k))
        elif a != b:
            errors.append(key + ": scalar differs")

    walk(actual, expected)
    return {
        "passed": not errors and finite,
        "max_abs": max_abs,
        "max_rel": max_rel,
        "finite": finite,
        "errors": errors[:10],
    }


def reward(correct, eager_ms, compile_ms, candidate_ms):
    """输入正确性和 eager/compile/候选耗时；返回论文里程碑奖励 -1、1、2 或 3。"""
    if not correct:
        return -1
    if candidate_ms < 0.95 * eager_ms:
        return 3 if candidate_ms < 0.95 * compile_ms else 2
    return 1


class Evaluator:
    def __init__(self, cfg, run):
        """输入配置和 run 路径；初始化计数器、只读配置及 kernel 环境路径，无返回值。"""
        self.cfg = cfg
        self.run = Path(run)
        self.seconds = 0.0
        self.compiles = 0
        self.counter = max(
            (
                int(p.name)
                for p in (self.run / "evaluations").glob("*")
                if p.is_dir() and p.name.isdigit()
            ),
            default=0,
        )
        self.run.mkdir(parents=True, exist_ok=True)
        self.config = self.run / "worker-config.json"
        dump(self.config, cfg)
        self.env = ROOT / ".runtime/conda/kernel"
        self.uid = os.getuid()
        self.gid = os.getgid()

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
        out = Path(out)
        out.mkdir(parents=True, exist_ok=True)
        name = "cuda-agent-p1-" + uuid.uuid4().hex[:12]
        c = [
            "docker",
            "run",
            "--name",
            name,
            "--rm",
            "--pull=never",
            "--network=none",
            "--read-only",
            "--log-driver=none",
            "--cap-drop=ALL",
            "--security-opt=no-new-privileges",
            "--pids-limit=256",
            "--memory=24g",
            "--cpus=4",
            "--user",
            f"{self.uid}:{self.gid}",
            "--tmpfs",
            "/tmp:rw,nosuid,size=8g",
            "--shm-size=1g",
            "--gpus",
            "device=" + str(self.cfg["evaluation_gpu"]),
        ]
        mounts = [
            (self.env, str(self.env), "ro"),
            (Path("/usr/local/cuda-12.9"), "/usr/local/cuda-12.9", "ro"),
            (ROOT / "src/p1/worker.py", "/worker.py", "ro"),
            (self.config, "/config.json", "ro"),
            (out, "/outputs", "rw"),
        ]
        if mode == "reference_bench" and reference is not None:
            cache = Path(reference) / "compile-cache"
            cache.mkdir(exist_ok=True)
            mounts.append((cache, "/refcache", "rw"))
        for p, d in [
            (reference, "/reference"),
            (candidate, "/candidate"),
            (inputs, "/inputs"),
            (build, "/build"),
        ]:
            if p is not None:
                mounts.append((Path(p), d, "rw" if d == "/build" else "ro"))
        for p, d, perm in mounts:
            c += [
                "--mount",
                (
                    f"type=bind,src={p.resolve()},dst={d},readonly"
                    if perm == "ro"
                    else f"type=bind,src={p.resolve()},dst={d}"
                ),
            ]
        for k, v in {
            "CUDA_HOME": "/usr/local/cuda-12.9",
            "PATH": f"{self.env}/bin:/usr/local/cuda-12.9/bin:/usr/bin:/bin",
            "HOME": "/tmp",
            "USER": "sandbox",
            "LOGNAME": "sandbox",
            "TMPDIR": "/tmp",
            "XDG_CACHE_HOME": "/tmp/cache",
            "TORCHINDUCTOR_CACHE_DIR": (
                "/refcache/inductor" if mode == "reference_bench" else "/outputs/inductor"
            ),
            "TRITON_CACHE_DIR": (
                "/refcache/triton" if mode == "reference_bench" else "/outputs/triton"
            ),
            "TORCH_EXTENSIONS_DIR": "/build",
            "TORCH_CUDA_ARCH_LIST": "8.0",
            "MAX_JOBS": "4",
            "OMP_NUM_THREADS": "4",
            "PYTHONDONTWRITEBYTECODE": "1",
            "CUDA_MODULE_LOADING": "LAZY",
        }.items():
            c += ["-e", f"{k}={v}"]
        c += [
            self.cfg["docker_image"],
            str(self.env / "bin/python"),
            "-I",
            "-B",
            "/worker.py",
            mode,
            "--phase",
            phase,
        ]
        started = time.monotonic()
        try:
            p = subprocess.run(
                c,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=timeout or self.cfg["evaluation_seconds"],
            )
            (out / "worker.log").write_text(p.stdout[-100000:])
            result = (
                json.loads((out / "result.json").read_text())
                if (out / "result.json").exists()
                else {"status": "error", "error": p.stdout[-6000:]}
            )
        except subprocess.TimeoutExpired as e:
            result = {"status": "timeout", "error": f"{mode} exceeded timeout"}
        finally:
            subprocess.run(
                ["docker", "rm", "-f", name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
            elapsed = time.monotonic() - started
            self.seconds += elapsed
        result["controller_seconds"] = elapsed
        return result

    def prepare(self, task):
        """输入原始任务；核查参考确定性和退化输出，保存参数及基线，返回预检状态。"""
        ref = self.run / "references" / task["task_id"]
        ref.mkdir(parents=True, exist_ok=True)
        (ref / "model.py").write_text(task["code"])
        prepared = self.container("reference_prepare", ref / "cases", reference=ref, timeout=300)
        dump(ref / "prepare.json", prepared)
        if prepared["status"] != "ok":
            return {"status": "task_invalid", "detail": prepared}
        if prepared.get("near_constant_sampled_output"):
            return {
                "status": "task_invalid",
                "reason": "near_constant_sampled_output_across_two_inputs_and_three_parameter_seeds",
                "detail": prepared,
            }
        bench = self.container(
            "reference_bench",
            ref / "initial-bench",
            reference=ref,
            inputs=ref / "cases/input-spec",
            timeout=300,
        )
        dump(ref / "baseline.json", bench)
        return {
            "status": "ok" if bench["status"] == "ok" else "task_invalid",
            "baseline": bench,
            "prepared": prepared,
        }

    def profile_reference(self, task):
        """输入固定题目；返回 eager/compile 耗时和算子 profile，不要求候选存在。

        首次调用准备参考参数；后续复用该 run 的参数快照，但重新测量性能。
        该入口不加载 Qwen，也不会把参考输出交给开发容器。
        """
        ref = self.run / "references" / task["task_id"]
        if not (ref / "cases/input-spec").exists():
            prepared = self.prepare(task)
            dump(ref / "preflight.json", prepared)
            if prepared["status"] != "ok":
                return {"status": "reference_rejected", "detail": prepared}
            return dict(prepared["baseline"], reference_only=True)
        output = ref / ("profile-" + uuid.uuid4().hex[:10])
        result = self.container(
            "reference_bench", output, reference=ref, inputs=ref / "cases/input-spec", timeout=300
        )
        return dict(result, reference_only=True)

    def snapshot(self, workspace):
        """输入开发工作区；创建递增编号的只读候选快照，返回评测目录、快照和摘要。"""
        self.counter += 1
        ev = self.run / "evaluations" / f"{self.counter:05d}"
        snap = ev / "candidate"
        snap.mkdir(parents=True)
        for path in Path(workspace).rglob("*"):
            allowed = path.name in {"model_new.py", "binding.cpp", "binding_registry.h"}
            allowed |= path.parent.name == "kernels" and path.suffix in {
                ".cu",
                ".cpp",
                ".h",
                ".cuh",
            }
            if path.is_file() and not path.is_symlink() and allowed:
                destination = snap / path.relative_to(workspace)
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(path.read_bytes())
        hashes = {
            str(path.relative_to(snap)): sha(path.read_bytes())
            for path in snap.rglob("*")
            if path.is_file()
        }
        audit = audit_contract(snap)
        for name in ["binding.cpp", "binding_registry.h"]:
            fixed = ROOT / "upstream/CUDA-Agent/agent_workdir" / name
            if not (snap / name).exists() or (snap / name).read_bytes() != fixed.read_bytes():
                audit["issues"].append("fixed infrastructure modified: " + name)
        result = {
            "evaluation_id": ev.name,
            "candidate_hash": sha(json.dumps(hashes, sort_keys=True)),
            "source_hashes": hashes,
            "audit_warnings": audit["warnings"],
            "correct": False,
            "reward": -1,
        }
        return ev, snap, result, audit["issues"]

    def compile_only(self, task, workspace):
        """输入题目和工作区；只检查合同并编译，返回日志，不产生正确性或性能结论。"""
        ev, snap, result, issues = self.snapshot(workspace)
        result.update(action="compile_only", scored=False)
        if issues:
            result.update(status="contract_violation", issues=issues)
        else:
            build = ev / "build"
            build.mkdir()
            compiled = self.container(
                "compile",
                ev / "compilation",
                candidate=snap,
                build=build,
                timeout=self.cfg["compile_seconds"],
            )
            self.compiles += 1
            result.update(
                compilation=compiled,
                status="compiled" if compiled["status"] == "ok" else "compilation_failed",
            )
            if compiled["status"] == "ok":
                # 与原 compile.sh 一样提供工作区扩展；正式验收仍从源码独立重编译，
                # 不信任 Agent 工作区已有的 .so，也不暴露参考结果。
                binary = build / "cuda_extension.so"
                destination = Path(workspace) / "cuda_extension.so"
                if destination.is_symlink():
                    raise ValueError("extension output must not be a symlink")
                if binary.is_file():
                    shutil.copy2(binary, destination)
                    result["artifact"] = "cuda_extension.so"
        dump(ev / "result.json", result)
        return result

    def streamed_checks(self, task, snap, build, ev, phase):
        """输入题目、快照、构建与阶段；外部逐例比较，返回 checks、参考状态、候选状态。"""
        ref = self.run / "references" / task["task_id"]
        expected = ev / "reference-stream"
        actual = ev / "candidate-stream"
        expected.mkdir()
        actual.mkdir()
        checks = []
        started = time.monotonic()
        with ThreadPoolExecutor(max_workers=2) as pool:
            rf = pool.submit(
                self.container,
                "reference_stream",
                expected,
                reference=ref,
                inputs=ref / "cases/input-spec",
                phase=phase,
            )
            cf = pool.submit(
                self.container,
                "candidate_stream",
                actual,
                candidate=snap,
                inputs=ref / "cases/input-spec",
                build=build,
                phase=phase,
            )
            while time.monotonic() - started < self.cfg["evaluation_seconds"] + 5:
                a, b = expected / "item.pt", actual / "item.pt"
                if a.exists() and b.exists():
                    try:
                        r = torch.load(a, weights_only=True)
                        c = torch.load(b, weights_only=True)
                        if r["id"] != c["id"]:
                            raise InfrastructureError("stream case order differs")
                        check = compare(
                            c["output"], r["output"], self.cfg["atol"], self.cfg["rtol"]
                        )
                        checks.append(dict(check, case=r["id"]))
                        del r, c
                    except InfrastructureError:
                        raise
                    except Exception as e:
                        checks.append(
                            {"passed": False, "errors": [str(e)], "case": "unreadable-output"}
                        )
                    a.unlink()
                    b.unlink()
                # worker 要等待外部确认；若另一端提前失败，仍要释放本端待确认输出。
                # 清理未配对输出只用于结束握手，不把它当作比较通过。
                # 任一端的失败状态仍会阻止正确性通过。
                if cf.done() and cf.result().get("status") != "ok":
                    a.unlink(missing_ok=True)
                if rf.done() and rf.result().get("status") != "ok":
                    b.unlink(missing_ok=True)
                if rf.done() and cf.done():
                    break
                time.sleep(0.01)
            rr, cr = rf.result(), cf.result()
        return checks, rr, cr

    def evaluate(self, task, workspace, final=False, attempt=0, measure=True):
        """输入题目、工作区及隐藏/重试/测时标记；返回合同、编译、执行、数值与性能结果。"""
        ref = self.run / "references" / task["task_id"]
        ev, snap, result, issues = self.snapshot(workspace)
        result["final"] = final
        if issues:
            result.update(status="contract_violation", issues=issues)
            dump(ev / "result.json", result)
            return result
        build = ev / "build"
        build.mkdir()
        compiled = self.container(
            "compile",
            ev / "compilation",
            candidate=snap,
            build=build,
            timeout=self.cfg["compile_seconds"],
        )
        self.compiles += 1
        result["compilation"] = compiled
        if compiled["status"] != "ok":
            result.update(status="compilation_failed")
            dump(ev / "result.json", result)
            return result
        phase = "hidden" if final else "feedback"
        checks, reference, ran = self.streamed_checks(task, snap, build, ev, phase)
        if reference["status"] != "ok":
            dump(ev / "infrastructure-error.json", {"reference": reference, "candidate": ran})
            if attempt == 0:
                return self.evaluate(task, snap, final, attempt=1, measure=measure)
            raise InfrastructureError("reference stream failed twice; do not assign model reward")
        expected_n = (
            self.cfg["feedback_cases"]
            if not final
            else self.cfg["hidden_cases_per_init"] * len(self.cfg["hidden_init_seeds"]) * 3 + 4
        )
        result.update(
            execution=ran,
            checks=checks,
            status="ok",
            correct=(
                ran["status"] == "ok"
                and len(checks) == expected_n
                and all(c["passed"] for c in checks)
                and not ran.get("state_mutations")
            ),
        )
        if not result["correct"] or final or not measure:
            result["reward"] = 1 if result["correct"] else -1
            dump(ev / "result.json", result)
            return result
        # 候选独占测时前后都重测参考基线，以检测资源干扰和计时漂移。
        before = self.container(
            "reference_bench",
            ev / "baseline-before",
            reference=ref,
            inputs=ref / "cases/input-spec",
            timeout=300,
        )
        measured = self.container(
            "candidate_bench",
            ev / "timing",
            candidate=snap,
            inputs=ref / "cases/input-spec",
            build=build,
        )
        after = self.container(
            "reference_bench",
            ev / "baseline-after",
            reference=ref,
            inputs=ref / "cases/input-spec",
            timeout=300,
        )
        if before["status"] != "ok" or after["status"] != "ok":
            dump(ev / "infrastructure-error.json", {"before": before, "after": after})
            if attempt == 0:
                return self.evaluate(task, snap, final, attempt=1)
            raise InfrastructureError("reference timing failed twice; do not assign model reward")
        if measured["status"] != "ok":
            result.update(status="timing_failed", correct=False, reward=-1)
            dump(ev / "result.json", result)
            return result
        if measured.get("runtime_fallbacks"):
            result.update(
                status="runtime_fallback_detected",
                correct=False,
                reward=-1,
                runtime_fallbacks=measured["runtime_fallbacks"],
            )
            dump(ev / "result.json", result)
            return result
        times = [measured["timing"]["cuda_ms"]] + [
            baseline[key]["cuda_ms"] for baseline in [before, after] for key in ["eager", "compile"]
        ]
        if any(
            not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0
            for value in times
        ):
            result.update(
                status="invalid_timing",
                timing_valid=False,
                reward=None,
                timing=measured,
                baseline_before=before,
                baseline_after=after,
            )
            dump(ev / "result.json", result)
            return result
        drift = {
            k: abs(after[k]["cuda_ms"] / before[k]["cuda_ms"] - 1) for k in ["eager", "compile"]
        }
        valid = max(drift.values()) <= self.cfg["baseline_drift_limit"]
        result.update(
            baseline_before=before,
            baseline_after=after,
            baseline_drift=drift,
            timing_valid=valid,
            timing=measured,
        )
        if measured.get("state_mutations"):
            result.update(correct=False, reward=-1, status="precision_state_violation")
        elif valid:
            baseline = {
                k: statistics.median([before[k]["cuda_ms"], after[k]["cuda_ms"]])
                for k in ["eager", "compile"]
            }
            ms = measured["timing"]["cuda_ms"]
            result.update(
                speedup_eager=baseline["eager"] / ms if ms > 0 else None,
                speedup_compile=baseline["compile"] / ms if ms > 0 else None,
            )
            result["reward"] = reward(True, baseline["eager"], baseline["compile"], ms)
        else:
            result["reward"] = None
        dump(ev / "result.json", result)
        if not valid and attempt == 0:
            retried = self.evaluate(task, snap, final, attempt=1)
            retried["retry_of"] = ev.name
            return retried
        return result

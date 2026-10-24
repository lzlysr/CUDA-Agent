"""CPU 回归：划分、预算、候选选择、工具语义和误拒修正。

运行：PYTHONPATH=src python -B -m unittest discover -s tests/p1 -v。
输入：临时工作区和假评测结果；输出：unittest 断言，不运行 Docker/GPU/模型。"""

import json
from pathlib import Path
import subprocess
import tempfile
import unittest

from p1.baseline import run_baselines, select_tasks
from p1.contract import audit_contract, contract
from p1.data import split_groups
from p1.run import Pilot, generation_budget
from p1.sandbox import BashProcess
from p1.tools import WorkspaceTools
from p1.worker import runtime_fallbacks
from p1.data import ROOT
from p1.evaluation import Evaluator


class AlignmentTests(unittest.TestCase):
    """覆盖以前已观察到的问题；所有输入均为小型 CPU 测试数据。"""

    def test_compile_command_publishes_binary_but_snapshots_only_source(self):
        """输入假编译产物；断言发布 .so、快照不信任它，并拒绝符号链接输出。"""

        class FakeCompiler(Evaluator):
            """只写占位产物的 CPU 编译替身。"""

            def container(self, mode, out, build=None, **kwargs):
                """输入编译模式与构建目录；写占位 .so 并返回成功，不执行 nvcc。"""
                (Path(build) / "cuda_extension.so").write_bytes(b"fake-binary")
                return {"status": "ok"}

        with tempfile.TemporaryDirectory() as directory:
            cfg = json.loads((ROOT / "configs/p1-pilot.json").read_text())
            workspace = Path(directory) / "workspace"
            (workspace / "kernels").mkdir(parents=True)
            for name in ["binding.cpp", "binding_registry.h"]:
                (workspace / name).write_bytes(
                    (ROOT / "upstream/CUDA-Agent/agent_workdir" / name).read_bytes()
                )
            (workspace / "model_new.py").write_text(
                "import cuda_extension\nclass ModelNew:\n    def forward(self,x):return cuda_extension.f(x)\n"
            )
            (workspace / "kernels/k.cu").write_text("__global__ void k() {}")
            evaluator = FakeCompiler(cfg, Path(directory) / "run")
            result = evaluator.compile_only({"task_id": "fixture"}, workspace)
            self.assertEqual(result["status"], "compiled")
            self.assertEqual((workspace / "cuda_extension.so").read_bytes(), b"fake-binary")
            self.assertNotIn("cuda_extension.so", result["source_hashes"])
            external = Path(directory) / "external.so"
            external.write_bytes(b"untouched")
            (workspace / "cuda_extension.so").unlink()
            (workspace / "cuda_extension.so").symlink_to(external)
            with self.assertRaises(ValueError):
                evaluator.compile_only({"task_id": "fixture"}, workspace)
            self.assertEqual(external.read_bytes(), b"untouched")

    def test_zero_timing_is_not_rewarded_as_fast(self):
        """输入数值正确但 CUDA event 为零的假候选；断言耗时无效且没有速度奖励。"""

        class ZeroEvaluator(Evaluator):
            """跳过真实容器，只模拟数值通过和无效计时。"""

            def container(self, mode, out, **kwargs):
                """输入 worker 模式；返回零候选耗时和正基线耗时，不使用 GPU。"""
                if mode == "compile":
                    return {"status": "ok"}
                if mode == "candidate_bench":
                    return {"status": "ok", "timing": {"cuda_ms": 0}}
                return {"status": "ok", "eager": {"cuda_ms": 2}, "compile": {"cuda_ms": 1}}

            def streamed_checks(self, task, snap, build, ev, phase):
                """输入流式参数；返回五例通过的假结果，不创建输出流。"""
                return ([{"passed": True}] * 5, {"status": "ok"}, {"status": "ok"})

        with tempfile.TemporaryDirectory() as directory:
            cfg = json.loads((ROOT / "configs/p1-pilot.json").read_text())
            workspace = Path(directory) / "workspace"
            (workspace / "kernels").mkdir(parents=True)
            for name in ["binding.cpp", "binding_registry.h"]:
                (workspace / name).write_bytes(
                    (ROOT / "upstream/CUDA-Agent/agent_workdir" / name).read_bytes()
                )
            (workspace / "model_new.py").write_text(
                "import cuda_extension\nclass ModelNew:\n    def forward(self,x):return cuda_extension.f(x)\n"
            )
            (workspace / "kernels/k.cu").write_text("__global__ void k() {}")
            result = ZeroEvaluator(cfg, Path(directory) / "run").evaluate(
                {"task_id": "fixture"}, workspace
            )
            self.assertTrue(result["correct"])
            self.assertFalse(result["timing_valid"])
            self.assertIsNone(result["reward"])
            self.assertEqual(result["status"], "invalid_timing")

    def test_group_split_is_reproducible_and_has_only_two_sets(self):
        """输入含重复行的组；断言 80/20、无跨集合、顺序改变不影响结果。"""
        groups = {
            str(index): [{"data_source": "a" if index < 13 else "b"}] * (index % 3 + 1)
            for index in range(25)
        }
        first = split_groups(groups)
        second = split_groups(dict(reversed(list(groups.items()))))
        self.assertEqual(first, second)
        self.assertEqual(set(first.values()), {"train", "test"})
        self.assertEqual(sum(value == "train" for value in first.values()), 20)
        self.assertEqual(sum(value == "test" for value in first.values()), 5)

    def test_generation_uses_remaining_window_not_4096(self):
        """输入论文窗口及无单次上限配置；断言单轮/Agent 分别使用正确余量。"""
        cfg = {"context_tokens": 131072, "single_context_tokens": 32768, "generation_tokens": None}
        self.assertEqual(generation_budget(cfg, False, 1000), 31768)
        self.assertEqual(generation_budget(cfg, True, 1000), 130072)
        self.assertEqual(generation_budget(cfg, True, 131072), 0)

    def test_best_candidate_skips_hidden_failure_and_duplicates(self):
        """输入最快但隐藏失败的候选；断言选择下一个合格项，重复源码只验一次。"""
        pilot = Pilot.__new__(Pilot)
        pilot.run = Path("/not-used")
        calls = []

        class FakeEvaluator:
            """记录候选顺序，不执行文件或 GPU。"""

            def evaluate(self, task, workspace, final=False):
                """输入候选路径；候选 1 失败、候选 2 成功，返回假隐藏验收。"""
                ident = workspace.parent.name
                calls.append(ident)
                return {"correct": ident == "00002"}

        pilot.ev = FakeEvaluator()
        candidates = [
            dict(
                correct=True,
                timing_valid=True,
                candidate_hash="fast",
                evaluation_id="00001",
                speedup_compile=3,
            ),
            dict(
                correct=True,
                timing_valid=True,
                candidate_hash="fast",
                evaluation_id="00003",
                speedup_compile=2.5,
            ),
            dict(
                correct=True,
                timing_valid=True,
                candidate_hash="good",
                evaluation_id="00002",
                speedup_compile=2,
            ),
        ]
        selected, hidden = pilot.select_candidate({"task_id": "fixture"}, candidates)
        self.assertEqual(calls, ["00001", "00002"])
        self.assertEqual(selected["candidate_hash"], "good")
        self.assertTrue(hidden["correct"])

    def test_helpers_types_unused_imports_and_comments_do_not_false_reject(self):
        """输入纯扩展辅助方法、未使用 F 和 ArrayRef；应允许，但实际 F.relu 仍拒绝。"""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "kernels").mkdir()
            (root / "kernels/k.cu").write_text(
                "// at::matmul(a,b) is only a comment\nat::Tensor f(at::Tensor x, at::ArrayRef<long> shape){return x;}\n__global__ void k() {}"
            )
            source = "import torch\nimport torch.nn as nn\nimport torch.nn.functional as F\nimport cuda_extension as ext\nclass ModelNew(nn.Module):\n    def __init__(self):\n        super().__init__()\n    def forward(self,x):\n        return self.helper(x)\n    def helper(self,x):\n        return ext.kernel(x)\n"
            (root / "model_new.py").write_text(source)
            self.assertEqual(contract(root), [])
            (root / "model_new.py").write_text(
                source.replace("return ext.kernel(x)", "return F.relu(x)")
            )
            self.assertTrue(contract(root))

    def test_unresolved_cpp_is_reviewed_but_compute_is_blocked(self):
        """输入未知类型与实际 add 调用；前者给人工提示，后者明确阻断。"""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "kernels").mkdir()
            (root / "model_new.py").write_text(
                "import cuda_extension\nclass ModelNew:\n    def forward(self,x): return cuda_extension.f(x)\n"
            )
            source = root / "kernels/k.cu"
            source.write_text("at::UnknownType marker;\n__global__ void k() {}")
            audit = audit_contract(root)
            self.assertFalse(audit["issues"])
            self.assertTrue(audit["warnings"])
            source.write_text(
                "at::Tensor f(at::Tensor x){ return at::add(x,x); }\n__global__ void k() {}"
            )
            self.assertTrue(contract(root))

    def test_runtime_profiler_distinguishes_allocation_from_compute(self):
        """输入 profiler 事件；只允许分配/元数据，aten::add 被标记为回退。"""
        self.assertEqual(
            runtime_fallbacks(["aten::empty_like", "aten::view", "my_cuda_kernel"]), []
        )
        self.assertEqual(runtime_fallbacks(["aten::empty", "aten::add"]), ["aten::add"])

    def test_read_before_write_and_atomic_multiedit(self):
        """输入临时源码；先读后写生效，MultiEdit 中任一步失败不得部分写入。"""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "model_new.py").write_text("alpha = 1\nbeta = 2\n")
            tools = WorkspaceTools(root, lambda command: {}, lambda action: {})
            with self.assertRaises(ValueError):
                tools.dispatch("Write", {"path": "model_new.py", "content": "bad"})
            tools.dispatch("Read", {"path": "model_new.py"})
            with self.assertRaises(ValueError):
                tools.dispatch(
                    "MultiEdit",
                    {
                        "path": "model_new.py",
                        "edits": [
                            {"old_string": "alpha = 1", "new_string": "alpha = 3"},
                            {"old_string": "absent", "new_string": "x"},
                        ],
                    },
                )
            self.assertEqual((root / "model_new.py").read_text(), "alpha = 1\nbeta = 2\n")
            tools.dispatch(
                "Edit",
                {"path": "model_new.py", "old_string": "alpha = 1", "new_string": "alpha = 3"},
            )
            self.assertIn("alpha = 3", (root / "model_new.py").read_text())
            (root / "kernels").mkdir()
            (root / "kernels/k.cu").write_text("// nested-search-needle\n")
            matches = tools.dispatch("Grep", {"pattern": "nested-search-needle"})
            self.assertEqual(matches["matches"][0]["path"], "kernels/k.cu")

    def test_bash_preserves_directory_export_function_and_files(self):
        """输入连续 Bash 命令；cwd、export、函数和临时文件跨调用保留。"""
        with tempfile.TemporaryDirectory() as directory:
            process = subprocess.Popen(
                ["/bin/bash", "--noprofile", "--norc"],
                cwd=directory,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            bash = BashProcess(process)
            try:
                first = bash.run(
                    "mkdir sub; cd sub; export P1_TEST_VALUE=kept; function p1_test_fn(){ echo function-kept; }; echo file-kept > saved"
                )
                self.assertEqual(first["returncode"], 0)
                second = bash.run('printf "%s\\n" "$P1_TEST_VALUE"; p1_test_fn; cat saved; pwd')
                self.assertIn("kept\nfunction-kept\nfile-kept", second["output"])
                self.assertIn("/sub", second["output"])
            finally:
                bash.close()

    def test_baseline_cli_can_be_tested_without_gpu_or_qwen(self):
        """输入假基线结果；生成 JSON/CSV/报告，不调用真实 GPU 或语言模型。"""
        records = [
            {"task_id": "a", "split": "train"},
            {"task_id": "a", "split": "train"},
            {"task_id": "b", "split": "test"},
        ]
        self.assertEqual(len(select_tasks(records)), 1)
        with self.assertRaises(ValueError):
            select_tasks(records, "train", ["b"])

        class FakeEvaluator:
            """只返回固定 CPU 数据的评测替身。"""

            def __init__(self, cfg, output):
                """接受配置与目录但不创建任何进程；无返回值。"""
                pass

            def profile_reference(self, task):
                """输入任务；返回 eager=2ms、compile=1ms 的假记录。"""
                return {
                    "status": "ok",
                    "eager": {"cuda_ms": 2, "wall_ms": 3},
                    "compile": {"cuda_ms": 1, "wall_ms": 2},
                }

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "baseline"
            results = run_baselines({}, select_tasks(records), output, FakeEvaluator)
            self.assertEqual(results[0]["compile_over_eager_speedup"], 2)
            self.assertTrue((output / "results.csv").exists())
            self.assertFalse(json.loads((output / "manifest.json").read_text())["loads_qwen"])


if __name__ == "__main__":
    unittest.main()

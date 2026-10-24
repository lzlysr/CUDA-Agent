"""CPU 流式握手回归：一方提前失败时，另一方仍能正常释放输出。

运行：PYTHONPATH=src python -B -m unittest discover -s tests/p1 -v。
输入：假 worker 与 CPU 张量；输出：错误分类及不死锁断言。"""

import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
import torch
from p1.data import ROOT
from p1.evaluation import Evaluator


class PeerReplay(Evaluator):
    def __init__(self, cfg, run, failed_peer, paired_prefix=False):
        """输入配置、目录、失败端及前缀选项；初始化 CPU 握手模拟状态，无返回值。"""
        super().__init__(cfg, run)
        self.failed_peer = failed_peer
        self.paired_prefix = paired_prefix
        self.reference_ready = threading.Event()

    def container(self, mode, out, **kwargs):
        """输入假模式和输出目录；模拟单例输出/提前失败，返回 worker 状态，不创建容器。"""
        if mode == "compile":
            return {"status": "ok"}
        out = Path(out)
        out.mkdir(parents=True, exist_ok=True)
        is_reference = mode == "reference_stream"
        should_fail = (is_reference and self.failed_peer == "reference") or (
            not is_reference and self.failed_peer == "candidate"
        )
        if is_reference:
            self.reference_ready.set()
        else:
            self.reference_ready.wait(1)
        if should_fail and not self.paired_prefix:
            return {
                "status": "error",
                "error_type": "RuntimeError",
                "error": "replayed worker execution error",
            }
        count = 2 if self.paired_prefix and not should_fail else 1
        for i in range(count):
            target = out / "item.pt"
            tmp = out / "item.tmp"
            torch.save({"id": str(i), "output": torch.tensor([1.0])}, tmp)
            tmp.replace(target)
            deadline = time.monotonic() + 0.25
            while target.exists() and time.monotonic() < deadline:
                time.sleep(0.002)
            if target.exists():
                return {"status": "timeout", "error": "peer stopped acknowledging outputs"}
        if should_fail:
            return {
                "status": "error",
                "error_type": "RuntimeError",
                "error": "replayed worker execution error",
            }
        return {"status": "ok", "state_mutations": []}


class StreamFailureTests(unittest.TestCase):
    def replay(self, root, peer, prefix=False):
        """输入临时目录、失败端与前缀；运行 CPU 流式比较，返回 checks、参考状态和候选状态。"""
        cfg = json.loads((ROOT / "configs/p1-pilot.json").read_text())
        cfg["evaluation_seconds"] = 0.5
        e = PeerReplay(cfg, root, peer, prefix)
        dest = Path(root) / "stream"
        dest.mkdir()
        return e.streamed_checks({"task_id": "replay"}, Path(root), Path(root), dest, "feedback")

    def test_candidate_failure_does_not_turn_into_reference_timeout(self):
        """无外部输入；模拟候选提前失败，断言参考端正常结束、不误报超时。无返回值，仅断言。"""
        with tempfile.TemporaryDirectory(dir=ROOT / ".runtime/tmp") as d:
            checks, ref, candidate = self.replay(d, "candidate")
            self.assertEqual(candidate["status"], "error")
            self.assertEqual(ref["status"], "ok")
            self.assertEqual(checks, [])

    def test_failure_after_one_result_releases_remaining_reference(self):
        """无外部输入；模拟比较一例后候选失败，断言参考剩余输出被确认并结束。无返回值，仅断言。"""
        with tempfile.TemporaryDirectory(dir=ROOT / ".runtime/tmp") as d:
            checks, ref, candidate = self.replay(d, "candidate", True)
            self.assertEqual(candidate["status"], "error")
            self.assertEqual(ref["status"], "ok")
            self.assertEqual(len(checks), 1)
            self.assertTrue(checks[0]["passed"])

    def test_reference_failure_remains_an_infrastructure_error(self):
        """无外部输入；模拟参考失败，断言仍标记基础设施错误，不归责模型。无返回值，仅断言。"""
        with tempfile.TemporaryDirectory(dir=ROOT / ".runtime/tmp") as d:
            checks, ref, candidate = self.replay(d, "reference")
            self.assertEqual(ref["status"], "error")
            self.assertEqual(candidate["status"], "ok")

    def test_successful_pair_is_still_compared(self):
        """无外部输入；模拟正常双端，断言输出确实比较而非直接放行。无返回值，仅断言。"""
        with tempfile.TemporaryDirectory(dir=ROOT / ".runtime/tmp") as d:
            checks, ref, candidate = self.replay(d, None)
            self.assertEqual(ref["status"], "ok")
            self.assertEqual(candidate["status"], "ok")
            self.assertEqual(len(checks), 1)
            self.assertTrue(checks[0]["passed"])

    def test_failed_candidate_is_reported_without_retrying_reference(self):
        """无外部输入；模拟执行失败，断言奖励 -1 且不进行虚假的参考重试。无返回值，仅断言。"""
        with tempfile.TemporaryDirectory(dir=ROOT / ".runtime/tmp") as d:
            cfg = json.loads((ROOT / "configs/p1-pilot.json").read_text())
            cfg["evaluation_seconds"] = 0.5
            e = PeerReplay(cfg, Path(d) / "run", "candidate")
            w = Path(d) / "workspace"
            (w / "kernels").mkdir(parents=True)
            for name in ["binding.cpp", "binding_registry.h"]:
                (w / name).write_bytes(
                    (ROOT / "upstream/CUDA-Agent/agent_workdir" / name).read_bytes()
                )
            (w / "model_new.py").write_text(
                "import cuda_extension\nclass ModelNew:\n    def forward(self,x):return cuda_extension.forward(x)\n"
            )
            (w / "kernels/kernel.cu").write_text("__global__ void kernel() {}")
            result = e.evaluate({"task_id": "replay"}, w)
            self.assertFalse(result["correct"])
            self.assertEqual(result["reward"], -1)
            self.assertEqual(result["execution"]["status"], "error")
            self.assertEqual(e.counter, 1)


if __name__ == "__main__":
    unittest.main()

"""CPU 协议回归：奖励边界、数值比较、路径、工具格式及已知回退。

运行：PYTHONPATH=src python -B -m unittest discover -s tests/p1 -v。
输入：临时小样本；输出：unittest 断言；不运行 Docker、GPU 或 Qwen。"""

import json
from pathlib import Path
import tempfile
import unittest
import torch
from p1.evaluation import compare, contract, reward
from p1.run import parse, write_files


class ProtocolTests(unittest.TestCase):
    def test_reward_boundaries(self):
        """无外部输入；给定临界耗时，断言奖励严格遵守超过 5% 的边界。无返回值，仅断言。"""
        self.assertEqual(reward(False, 10, 8, 1), -1)
        self.assertEqual(reward(True, 10, 8, 9.5), 1)
        self.assertEqual(reward(True, 10, 8, 9.49), 2)
        self.assertEqual(reward(True, 10, 8, 7.6), 2)
        self.assertEqual(reward(True, 10, 8, 7.59), 3)

    def test_external_comparison(self):
        """无外部输入；给定错误、NaN 和合法输出树，断言外部比较器正确判定。无返回值，仅断言。"""
        self.assertFalse(compare(torch.ones(3), torch.zeros(3))["passed"])
        self.assertFalse(
            compare(torch.tensor([float("nan")]), torch.tensor([float("nan")]))["passed"]
        )
        self.assertFalse(compare({"a": torch.ones(1), "extra": 0}, {"a": torch.ones(1)})["passed"])
        self.assertTrue(compare((torch.ones(2), 3), (torch.ones(2), 3))["passed"])

    def test_path_traversal_and_fixed_file(self):
        """无外部输入；给定逃逸路径和固定设施路径，断言写入被拒绝且不修改文件。无返回值，仅断言。"""
        with tempfile.TemporaryDirectory() as d:
            for name in ["../escape.py", "binding.cpp", "model.py", "kernels/../model.py"]:
                with self.assertRaises(ValueError):
                    write_files(d, {name: "x"})

    def test_no_hidden_tool_in_thinking(self):
        """无外部输入；给定 thinking 内外工具文本，断言只执行已结束思考后的工具。无返回值，仅断言。"""
        self.assertIsNone(parse('<think><tool_call>{"name":"submit"}</tool_call>'))
        self.assertEqual(
            parse('<think>x</think><tool_call>{"name":"submit","arguments":{}}</tool_call>')[0],
            "submit",
        )

    def test_torch_fallback_and_state_hacks(self):
        """无外部输入；给定 PyTorch 计算和精度修改，断言违规被阻断，纯扩展调用被允许。无返回值，仅断言。"""
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            (p / "kernels").mkdir()
            (p / "kernels/kernel.cu").write_text("__global__ void k() {}")
            for body in [
                "return torch.matmul(x,x)",
                "torch.backends.cuda.matmul.allow_tf32 = False\n        return x",
                "return torch.compile(self.m)(x)",
            ]:
                (p / "model_new.py").write_text(
                    "import torch\nclass ModelNew:\n    def forward(self,x):\n        "
                    + body
                    + "\n"
                )
                self.assertTrue(contract(p))
            (p / "model_new.py").write_text(
                "import torch\nimport cuda_extension\nclass ModelNew:\n    def forward(self,x):\n        return cuda_extension.forward(x)\n"
            )
            self.assertEqual(contract(p), [])
            (p / "kernels/kernel.cu").write_text("void f(){ torch::matmul(a,b); }")
            self.assertTrue(contract(p))

    def test_extension_aliases_are_legal(self):
        """无外部输入；给定扩展别名和 torch 计算导入，断言合法别名不误拒，实际计算不放行。无返回值，仅断言。"""
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            (p / "kernels").mkdir()
            (p / "kernels/k.cu").write_text("__global__ void k() {}")
            for imp, call in [
                ("import cuda_extension as ext", "ext.kernel(x)"),
                ("from cuda_extension import kernel as run", "run(x)"),
            ]:
                (p / "model_new.py").write_text(
                    imp
                    + "\nclass ModelNew:\n    def forward(self,x):\n        return "
                    + call
                    + "\n"
                )
                self.assertEqual(contract(p), [])
            (p / "model_new.py").write_text(
                "from torch import max\nclass ModelNew:\n    def forward(self,x):\n        return max(x)\n"
            )
            self.assertTrue(contract(p))

    def test_aten_tensor_type_is_legal_but_compute_is_not(self):
        """无外部输入；给定 Tensor 类型与 add 调用，断言类型允许、计算回退拒绝。无返回值，仅断言。"""
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            (p / "kernels").mkdir()
            (p / "model_new.py").write_text(
                "import cuda_extension\nclass ModelNew:\n    def forward(self,x):\n        return cuda_extension.kernel(x)\n"
            )
            (p / "kernels/k.cu").write_text(
                "at::Tensor f(at::Tensor x){ return x; }\n__global__ void k() {}"
            )
            self.assertEqual(contract(p), [])
            (p / "kernels/k.cu").write_text(
                "at::Tensor f(at::Tensor x){ return at::add(x,x); }\n__global__ void k() {}"
            )
            self.assertTrue(contract(p))

    def test_native_tool_format(self):
        """无外部输入；给定 Qwen 原生 function/parameter 文本，断言解析出工具及 JSON 参数。无返回值，仅断言。"""
        text = '<think>x</think><tool_call>\n<function=write_files>\n<parameter=files>\n{"model_new.py":"import torch"}\n</parameter>\n</function>\n</tool_call>'
        self.assertEqual(parse(text), ("write_files", {"files": {"model_new.py": "import torch"}}))


if __name__ == "__main__":
    unittest.main()

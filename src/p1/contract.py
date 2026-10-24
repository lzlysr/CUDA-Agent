"""候选源码的可解释静态审查；不执行候选，也不初始化 GPU。

运行：PYTHONPATH=src python -m unittest discover -s tests/p1 -v。
输入：包含 model_new.py 与 kernels 的候选目录。
输出：明确的阻断原因与待人工复核提示。静态审查不能替代数值和运行期检查。"""

import ast
from pathlib import Path
import re


ALLOCATIONS = {"empty", "empty_like", "zeros", "zeros_like", "ones", "ones_like"}
METADATA = {"size", "dim", "numel", "stride", "element_size", "view", "reshape", "is_contiguous"}
SCALARS = {"len", "int", "float", "bool", "tuple", "list", "range", "min", "max", "abs"}
CPP_TYPES = {
    "Tensor",
    "TensorOptions",
    "ArrayRef",
    "IntArrayRef",
    "SymIntArrayRef",
    "Scalar",
    "ScalarType",
    "Device",
    "DeviceType",
    "Layout",
    "MemoryFormat",
    "SymInt",
    "TensorList",
}


def audit_contract(directory):
    """输入候选目录；返回 {issues: 明确违规列表, warnings: 人工复核提示列表}。

    沿 forward→self.helper/模块函数追踪调用；只导入 F 不视为执行回退。
    C++ 类型（例如 at::ArrayRef）与实际调用分开判断，避免按陌生名字一律拒绝。
    """
    directory = Path(directory)
    issues = []
    warnings = []
    for path in directory.rglob("*"):
        if path.is_symlink():
            issues.append(f"symlink forbidden: {path.name}")
    source = directory / "model_new.py"
    if not source.exists():
        return {"issues": ["missing model_new.py"], "warnings": []}

    try:
        tree = ast.parse(source.read_text(encoding="utf-8"))
    except SyntaxError as error:
        return {"issues": [f"Python syntax: {error}"], "warnings": []}

    aliases = {"torch": "torch", "cuda_extension": "cuda_extension"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for item in node.names:
                if item.name not in {
                    "torch",
                    "torch.nn",
                    "torch.nn.functional",
                    "cuda_extension",
                    "math",
                }:
                    issues.append("forbidden import " + item.name)
                aliases[item.asname or item.name.split(".")[0]] = (
                    item.name if item.asname else item.name.split(".")[0]
                )
        elif isinstance(node, ast.ImportFrom):
            if node.module not in {
                "torch",
                "torch.nn",
                "torch.nn.functional",
                "cuda_extension",
                "math",
            }:
                issues.append("forbidden from-import " + str(node.module))
            for item in node.names:
                if item.name == "*":
                    issues.append("wildcard imports prevent static audit")
                aliases[item.asname or item.name] = f"{node.module}.{item.name}"
        elif isinstance(node, ast.Call):
            if ast.unparse(node.func) in {
                "eval",
                "exec",
                "compile",
                "open",
                "getattr",
                "setattr",
                "__import__",
                "globals",
                "locals",
                "vars",
            }:
                issues.append("forbidden call " + ast.unparse(node.func))
        elif isinstance(node, ast.Attribute):
            dotted = ast.unparse(node)
            if node.attr != "__init__" and any(part.startswith("__") for part in dotted.split(".")):
                issues.append("forbidden Python dunder access " + dotted)

    def resolve(expression):
        """输入调用表达式 AST；返回展开 import 别名后的点分名称。"""
        name = ast.unparse(expression)
        head, separator, tail = name.partition(".")
        return aliases.get(head, head) + (separator + tail if separator else "")

    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            dotted = resolve(node)
            if any(
                dotted.startswith(prefix)
                for prefix in ("torch.backends", "torch.cuda", "torch.ops", "torch._")
            ):
                issues.append("forbidden Python access " + dotted)

    globals_ = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    classes = [
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "ModelNew"
    ]
    if not classes:
        issues.append("missing ModelNew class")
    for cls in classes:
        methods = {node.name: node for node in cls.body if isinstance(node, ast.FunctionDef)}
        if "forward" not in methods:
            issues.append("missing ModelNew.forward")
            continue
        seen = set()

        def check_function(function, active):
            """输入可达函数与递归栈；检查计算调用并累积 issues，返回 None。"""
            if function.name in active:
                issues.append("recursive helper cannot be audited: " + function.name)
                return
            if id(function) in seen:
                return
            seen.add(id(function))
            for call in ast.walk(function):
                if not isinstance(call, ast.Call):
                    continue
                original = ast.unparse(call.func)
                name = resolve(call.func)
                helper = (
                    methods.get(original[5:])
                    if original.startswith("self.")
                    else globals_.get(original)
                )
                if helper is not None:
                    check_function(helper, active | {function.name})
                    continue
                permitted = (
                    name.startswith("cuda_extension.")
                    or name in SCALARS
                    or name in {"torch." + item for item in ALLOCATIONS}
                    or name
                    in {"math." + item for item in ("ceil", "floor", "sqrt", "log", "log2", "exp")}
                    or (isinstance(call.func, ast.Attribute) and call.func.attr in METADATA)
                )
                if not permitted:
                    issues.append("forward computation outside extension: " + name)

        check_function(methods["forward"], set())

    sources = list((directory / "kernels").glob("*.cu")) + list(
        (directory / "kernels").glob("*.cpp")
    )
    if not any(path.suffix == ".cu" for path in sources):
        issues.append("missing kernels/*.cu")
    for path in sources:
        raw = path.read_text(encoding="utf-8")
        # 先去掉注释，避免注释里的 torch::matmul 被误识别为实际回退。
        text = re.sub(r"/\*.*?\*/|//[^\n]*", "", raw, flags=re.S)
        for forbidden in (
            "system(",
            "popen(",
            "fopen(",
            "getenv(",
            "PyRun",
            "pybind11::eval",
            "dlopen(",
            "torch/nn",
            "ATen/ops",
            "ATen/native",
            "/reference",
            "/inputs",
            "/outputs",
        ):
            if forbidden in text:
                issues.append("forbidden C++ access " + forbidden)
        for namespace, symbol, suffix in re.findall(r"\b(torch|at)::(\w+)(\s*\()?", text):
            if symbol in CPP_TYPES or re.fullmatch(r"k[A-Z][A-Za-z0-9_]*", symbol):
                continue
            if symbol in ALLOCATIONS:
                continue
            if suffix.strip() == "(":
                issues.append(f"forbidden {namespace} C++ compute call {symbol}")
            else:
                warnings.append(f"unresolved C++ symbol requires review: {namespace}::{symbol}")

    return {"issues": sorted(set(issues)), "warnings": sorted(set(warnings))}


def contract(directory):
    """输入候选目录；返回明确阻断原因列表，保持旧控制器和测试的接口。"""
    return audit_contract(directory)["issues"]

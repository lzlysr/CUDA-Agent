"""CPU Docker 集成验证：持久 shell、utils 请求路由和固定文件只读。

运行：PYTHONPATH=src python -B tests/p1/sandbox_smoke.py。
输入：现有 Docker 镜像、临时工作区和假评测回调；输出：终端 PASS。
不挂载 GPU、不加载 Qwen、不执行真实 eager/compile 性能测试。"""

import json
from pathlib import Path
import tempfile

from p1.data import ROOT, dump
from p1.sandbox import SandboxSession


def main():
    """创建临时 CPU 容器并检查会话语义；返回 None，退出时自动清理容器。"""
    cfg = json.loads((ROOT / "configs/p1-pilot.json").read_text())
    calls = []

    def dispatch(action):
        """输入客户端动作；返回标明 fake_reply 的假结果，不执行 GPU 评测。"""
        calls.append(action)
        if action == "compile":
            return {"status": "compiled", "correct": False, "fake_reply": True}
        if action == "verify":
            return {"status": "ok", "correct": True, "fake_reply": True}
        return {"status": "ok", "reference_only": True, "fake_reply": True}

    with tempfile.TemporaryDirectory(dir=ROOT / ".runtime/tmp") as directory:
        work = Path(directory) / "workspace"
        work.mkdir()
        (work / "model.py").write_text("# CPU integration fixture\n")
        for name in ["binding.cpp", "binding_registry.h", "SKILL.md"]:
            (work / name).write_bytes(
                (ROOT / "upstream/CUDA-Agent/agent_workdir" / name).read_bytes()
            )
        session = SandboxSession(cfg, work, dispatch)
        try:
            first = session.run(
                "export P1_PERSIST=kept; cd /tmp; echo saved > p1-persist-file; function p1_check(){ echo function-kept; }"
            )
            assert first["returncode"] == 0, first
            second = session.run('echo "$P1_PERSIST"; pwd; cat p1-persist-file; p1_check')
            assert "kept\n/tmp\nsaved\nfunction-kept" in second["output"], second
            routed = session.run("python3 -m utils.profiling --reference-only")
            assert routed["returncode"] == 0 and "fake_reply" in routed["output"], routed
            compiled = session.run("cd /workspace; bash utils/compile.sh")
            assert compiled["returncode"] == 0, compiled
            verified = session.run("python3 -m utils.verification")
            assert verified["returncode"] == 0, verified
            assert calls == ["profile_reference", "compile", "verify"], calls
            protected = session.run(
                "test ! -e /dev/nvidia0 && ! touch /workspace/utils/__write_probe__"
            )
            assert protected["returncode"] == 0, protected
            result = {
                "passed": True,
                "cpu_only": True,
                "gpu_jobs": 0,
                "persistent_state": True,
                "utils_rpc": calls,
                "fixed_utils_readonly": True,
            }
            dump(ROOT / "reports/p1/sandbox-v3-cpu-smoke.json", result)
            print(json.dumps(result, indent=2))
        finally:
            session.close()


if __name__ == "__main__":
    main()

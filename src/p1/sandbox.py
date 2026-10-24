"""持久 Bash 开发会话与受限评测请求通道。

输入：固定任务工作区、配置及仅接受 compile/verify/profile 的宿主回调。
输出：每条命令的返回码和终端输出；cwd、export、函数及 /tmp 均跨调用保留。
终端容器无 GPU、无网络、无 Docker socket；评测由宿主固定回调派发。
CPU 回归：PYTHONPATH=src python -m unittest discover -s tests/p1 -v。"""

import base64
import json
import os
from pathlib import Path
import queue
import subprocess
import threading
import time
import uuid

from .data import ROOT, dump


class BashProcess:
    """对已有 Bash 进程执行命令；会话状态不因工具调用结束而重置。"""

    def __init__(self, process):
        """输入带 stdin/stdout 文本管道的 Popen；启动读取线程，无返回值。"""
        self.process = process
        self.lines = queue.Queue()
        self.reader = threading.Thread(target=self._read, daemon=True)
        self.reader.start()

    def _read(self):
        """无参数；将终端输出逐行入队，退出时写入结束标记。"""
        try:
            for line in self.process.stdout:
                self.lines.put(line)
        finally:
            self.lines.put(None)

    def run(self, command, timeout=600, poll=None):
        """输入命令、秒数与可选调度回调；返回 returncode/output，不启动新 Bash。"""
        marker = "P1_" + uuid.uuid4().hex
        encoded = base64.b64encode(command.encode()).decode()
        # eval 在既有 shell 内执行，避免 bash -c 子进程丢失 cwd/export/函数。
        script = f"builtin printf '\\n{marker}_BEGIN\\n'\neval \"$(printf '%s' '{encoded}' | base64 -d)\"\n"
        script += f"__p1_status=$?\nbuiltin printf '\\n{marker}_END:%s\\n' \"$__p1_status\"\n"
        if self.process.poll() is not None:
            raise RuntimeError("Bash 会话已经退出；不自动创建丢失状态的新会话")
        self.process.stdin.write(script)
        self.process.stdin.flush()
        deadline = time.monotonic() + timeout
        started = False
        output = []
        while time.monotonic() < deadline:
            if poll is not None:
                poll()
            try:
                line = self.lines.get(timeout=0.05)
            except queue.Empty:
                continue
            if line is None:
                raise RuntimeError("Bash 提前退出：" + "".join(output)[-2000:])
            if line.strip() == marker + "_BEGIN":
                started = True
                continue
            if line.startswith(marker + "_END:"):
                return {
                    "returncode": int(line.split(":", 1)[1]),
                    "output": "".join(output)[-20000:],
                }
            if started:
                output.append(line)
        self.close()
        raise TimeoutError("Bash 命令超时，会话已关闭，避免后台命令污染下一次调用")

    def close(self):
        """无参数；关闭终端进程及管道，重复调用安全，无返回值。"""
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=3)
        for stream in (self.process.stdin, self.process.stdout):
            if stream is not None:
                stream.close()


class SandboxSession:
    """每条 Agent 轨迹一个 Docker 容器、一个 Bash；离开轨迹时整体清理。"""

    def __init__(self, cfg, workspace, dispatch):
        """输入配置、工作区和固定评测回调；创建 CPU 容器与受限文件请求通道。"""
        self.cfg = cfg
        self.workspace = Path(workspace).resolve()
        self.dispatch = dispatch
        self.name = "cuda-agent-p1-shell-" + uuid.uuid4().hex[:12]
        self.rpc = self.workspace.parent / "tool-rpc"
        self.rpc.mkdir(exist_ok=True)
        env = ROOT / ".runtime/conda/kernel"
        options = [
            "docker",
            "run",
            "-d",
            "--name",
            self.name,
            "--rm",
            "--pull=never",
            "--network=none",
            "--read-only",
            "--log-driver=none",
            "--cap-drop=ALL",
            "--security-opt=no-new-privileges",
            "--user",
            f"{os.getuid()}:{os.getgid()}",
            "--pids-limit=128",
            "--memory=8g",
            "--cpus=4",
            "--tmpfs",
            "/tmp:rw,nosuid,size=2g",
            "--workdir",
            "/workspace",
            "--entrypoint",
            "/bin/sleep",
        ]
        mounts = [
            (self.workspace, "/workspace", False),
            (self.rpc, "/rpc", False),
            (env, str(env), True),
            (Path("/usr/local/cuda-12.9"), "/usr/local/cuda-12.9", True),
            (ROOT / "src/p1/agent_utils", "/workspace/utils", True),
        ]
        for name in ["model.py", "binding.cpp", "binding_registry.h", "SKILL.md"]:
            mounts.append((self.workspace / name, "/workspace/" + name, True))
        for source, target, readonly in mounts:
            options += [
                "--mount",
                f"type=bind,src={source},dst={target}" + (",readonly" if readonly else ""),
            ]
        variables = {
            "PATH": f"{env}/bin:/usr/local/cuda-12.9/bin:/usr/bin:/bin",
            "CUDA_HOME": "/usr/local/cuda-12.9",
            "HOME": "/tmp",
            "USER": "sandbox",
            "LOGNAME": "sandbox",
            "NVIDIA_VISIBLE_DEVICES": "void",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONPATH": "/workspace",
            "P1_TOOL_RPC": "/rpc",
        }
        for key, value in variables.items():
            options += ["-e", f"{key}={value}"]
        options += [cfg["docker_image"], "infinity"]
        self.bash = None
        try:
            subprocess.run(
                options, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
            )
            process = subprocess.Popen(
                ["docker", "exec", "-i", self.name, "/bin/bash", "--noprofile", "--norc"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            self.bash = BashProcess(process)
        except Exception:
            self.close()
            raise

    def poll_requests(self):
        """无参数；处理本轨迹请求。只接受动作枚举，不接受宿主路径或 shell 命令。"""
        for path in sorted(self.rpc.glob("request-*.json")):
            if path.is_symlink() or path.stat().st_size > 4096:
                raise ValueError("无效评测请求文件")
            request = json.loads(path.read_text(encoding="utf-8"))
            action = request.get("action")
            if action not in {"compile", "verify", "profile", "profile_reference"}:
                response = {"status": "error", "error": "unsupported action"}
            else:
                try:
                    response = self.dispatch(action)
                except Exception as error:
                    # 基础设施异常应停止整个轨迹，不伪装成模型负奖励。
                    if error.__class__.__name__ == "InfrastructureError":
                        raise
                    response = {"status": "error", "error": str(error)}
            reply = self.rpc / path.name.replace("request-", "reply-", 1)
            temporary = reply.with_suffix(".tmp")
            dump(temporary, response)
            temporary.replace(reply)
            path.unlink()

    def run(self, command):
        """输入 Bash 命令；在持久会话执行并派发受限评测请求，返回终端反馈。"""
        if len(command) > 100000:
            raise ValueError("命令超过 100000 字符")
        return self.bash.run(
            command, timeout=self.cfg.get("shell_seconds", 600), poll=self.poll_requests
        )

    def close(self):
        """无参数；清理 Bash 及该轨迹容器，保留宿主工作区和日志，无返回值。"""
        if self.bash is not None:
            self.bash.close()
        subprocess.run(
            ["docker", "rm", "-f", self.name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )

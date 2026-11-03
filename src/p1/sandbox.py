"""持久 Bash 开发会话与受限评测请求通道。

输入：固定任务工作区、配置及受限 GPU 作业回调。
输出：每条命令的返回码和终端输出；cwd、export、函数及 /tmp 均跨调用保留。
终端容器无 GPU、无网络、无 Docker socket；评测由宿主固定回调派发。
CPU 回归：PYTHONPATH=src python -m unittest discover -s tests/p1 -v。"""

import base64
import codecs
from contextlib import contextmanager
import json
import os
from pathlib import Path
import queue
import stat
import subprocess
import threading
import time
import uuid

from .data import ROOT, dump
from .helper.baseline_state import beijing_now


class BashProcess:
    """对已有 Bash 进程执行命令；会话状态不因工具调用结束而重置。"""

    def __init__(self, process):
        """输入带 stdin/stdout 文本管道的 Popen；启动读取线程，无返回值。"""
        self.process = process
        self.lines = queue.Queue()
        self.lock = threading.RLock()
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
        # 同一条持久 shell 的输入和标记读取必须串行，后台任务另行重定向输出。
        with self.lock:
            return self._run(command, timeout, poll)

    def _run(self, command, timeout, poll):
        """输入已锁定的命令；执行前后标记协议，返回该命令自身的输出。"""
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
        (self.rpc / "jobs").mkdir(exist_ok=True)
        self.jobs = {}
        self.jobs_lock = threading.RLock()
        self.rpc_lock = threading.Lock()
        self.stop = threading.Event()
        self.rpc_error = None
        self.rpc_thread = None
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
            # rg 与 Python 一起由只读挂载的 kernel Conda 环境提供。
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
            # 后台 utils 请求在模型生成期间也会处理，而非等到下一次前台 Bash 才执行。
            self.rpc_thread = threading.Thread(target=self._serve, daemon=True)
            self.rpc_thread.start()
        except Exception:
            self.close()
            raise

    def poll_requests(self):
        """无参数；处理本轨迹请求。只接受动作枚举，不接受宿主路径或 shell 命令。"""
        with self.rpc_lock:
            return self._poll_requests()

    def _poll_requests(self):
        """无参数；串行消费原子发布的请求，脚本参数仅作为数据传给隔离作业。"""
        for path in sorted(self.rpc.glob("request-*.json")):
            if self.stop.is_set():
                return
            if path.is_symlink() or path.stat().st_size > 4096:
                raise ValueError("无效评测请求文件")
            request = json.loads(path.read_text(encoding="utf-8"))
            action = request.get("action")
            if action not in {"compile", "verify", "profile", "profile_reference", "run_script"}:
                response = {"status": "error", "error": "unsupported action"}
            else:
                try:
                    response = (
                        self.dispatch(action, request.get("arguments", {}))
                        if action == "run_script" else self.dispatch(action)
                    )
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

    def _serve(self):
        """无输入；持续处理 RPC 和后台超时，基础设施异常保留并向下一次调用传播。"""
        try:
            while not self.stop.wait(0.05):
                self.poll_requests()
                with self.jobs_lock:
                    expired = [ident for ident, job in self.jobs.items()
                               if self.job_returncode(ident) is None
                               and time.monotonic() - job["started"] >= job["timeout"]]
                for ident in expired:
                    self.kill_background(ident, reason="timeout")
        except BaseException as error:
            self.rpc_error = error

    def check(self):
        """无输入；将后台调度异常重新抛给控制器，避免把基础设施故障算模型失败。"""
        if self.rpc_error is not None:
            from .evaluation import InfrastructureError
            if isinstance(self.rpc_error, InfrastructureError):
                raise self.rpc_error
            raise InfrastructureError("后台工具调度失败：" + str(self.rpc_error)) from self.rpc_error

    def command_timeout(self, timeout):
        """输入可选命令上限；返回不超过当前 shell_seconds 预算的正秒数。"""
        maximum = self.cfg.get("shell_seconds", 600)
        if timeout is None:
            return maximum
        if type(timeout) is not int or not 0 < timeout <= maximum:
            raise ValueError(f"timeout must be an integer in 1..{maximum} seconds")
        return timeout

    @contextmanager
    def job_file(self, ident, name, *, writing=False):
        """输入固定任务 ID/文件名；逐层拒绝符号链接后打开日志，避免越界读取宿主文件。"""
        # /rpc 中的文件由不可信终端写入，不能对其直接使用 Path.read_text 跟随链接。
        directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
        descriptors = []
        try:
            root = os.open(self.rpc / "jobs", directory_flags)
            descriptors.append(root)
            directory = os.open(ident, directory_flags, dir_fd=root)
            descriptors.append(directory)
            flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC if writing else os.O_RDONLY
            descriptor = os.open(name, flags | os.O_NOFOLLOW | os.O_NONBLOCK, 0o600, dir_fd=directory)
            if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                os.close(descriptor)
                raise ValueError("background output must be a regular file")
            with os.fdopen(descriptor, "wb" if writing else "rb") as stream:
                yield stream
        finally:
            for descriptor in reversed(descriptors):
                os.close(descriptor)

    def job_returncode(self, ident):
        """输入任务 ID；读取完整退出码，任务尚未退出时返回 None。"""
        try:
            with self.job_file(ident, "exit") as stream:
                return int(stream.read(64).decode().strip())
        except FileNotFoundError:
            return None

    def run(self, command, timeout=None):
        """输入 Bash 命令；在持久会话执行并派发受限评测请求，返回终端反馈。"""
        self.check()
        if len(command) > 100000:
            raise ValueError("命令超过 100000 字符")
        result = self.bash.run(command, timeout=self.command_timeout(timeout), poll=self.check)
        self.check()
        return result

    def start_background(self, command, timeout=None):
        """输入 CPU shell 命令与秒数；继承当前会话状态启动任务，返回 bash_id。"""
        self.check()
        if not isinstance(command, str) or len(command) > 100000:
            raise ValueError("invalid Bash command")
        timeout = self.command_timeout(timeout)
        ident = uuid.uuid4().hex[:12]
        # mkdir 的父目录也通过 O_NOFOLLOW 打开，拒绝模型把 jobs 换成外部链接。
        root = os.open(self.rpc / "jobs", os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            os.mkdir(ident, dir_fd=root)
        finally:
            os.close(root)
        encoded = base64.b64encode(command.encode()).decode()
        target = "/rpc/jobs/" + ident
        # monitor 模式为异步子 shell 建立独立进程组，KillBash 可停止全部子进程。
        # 子 shell 自然继承 cwd、普通变量、export 和函数，不改变父会话的状态。
        launch = (
            "set -m\n(\n"
            "set +m\n"
            f"trap 'builtin printf \"%s\\n\" \"$?\" > {target}/exit.tmp; mv {target}/exit.tmp {target}/exit' EXIT\n"
            f"( eval \"$(printf '%s' '{encoded}' | base64 -d)\" )\n"
            f") > {target}/output.log 2>&1 < /dev/null &\n"
            f"builtin printf '%s\\n' \"$!\" > {target}/pid\n"
        )
        result = self.bash.run(launch, timeout=10, poll=self.check)
        if result["returncode"]:
            raise RuntimeError(result["output"])
        with self.job_file(ident, "pid") as stream:
            pid = int(stream.read(64).decode())
        with self.jobs_lock:
            self.jobs[ident] = {
                "pid": pid,
                "cursor": 0, "started": time.monotonic(), "timeout": timeout,
                "decoder": codecs.getincrementaldecoder("utf-8")(errors="replace"),
            }
        return {"bash_id": ident, "status": "running", "started_at": beijing_now().isoformat(timespec="seconds")}

    def background_output(self, ident):
        """输入本会话任务 ID；返回上次读取之后的新输出及当前状态。"""
        self.check()
        with self.jobs_lock:
            if ident not in self.jobs:
                raise ValueError("unknown bash_id")
            job = self.jobs[ident]
            chunk = b""
            has_more = False
            try:
                with self.job_file(ident, "output.log") as stream:
                    stream.seek(job["cursor"])
                    # 每次返回有限的新内容，余下内容下一次继续读，不直接丢掉。
                    chunk = stream.read(20000)
                    job["cursor"] = stream.tell()
                    has_more = os.fstat(stream.fileno()).st_size > job["cursor"]
            except FileNotFoundError:
                pass
            code = self.job_returncode(ident)
            return {"bash_id": ident, "status": job.get("reason") or ("completed" if code is not None else "running"),
                    "returncode": code, "output": job["decoder"].decode(chunk, final=code is not None and not has_more),
                    "has_more": has_more}

    def kill_background(self, ident, reason="killed"):
        """输入任务 ID；只在该容器内终止对应进程组，返回任务状态。"""
        with self.jobs_lock:
            if ident not in self.jobs:
                raise ValueError("unknown bash_id")
            job = self.jobs[ident]
            if self.job_returncode(ident) is None:
                pid = job["pid"]
                # PID 为本容器内读取的整数，绝不向宿主 PID 发送信号。
                for signal_name in ("TERM", "KILL"):
                    subprocess.run(
                        ["docker", "exec", self.name, "/bin/bash", "-c", f"kill -{signal_name} -- -{pid}"],
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=10,
                    )
                job["reason"] = reason
                if self.job_returncode(ident) is None:
                    with self.job_file(ident, "exit", writing=True) as stream:
                        stream.write(b"124" if reason == "timeout" else b"143")
        return self.background_output(ident)

    def close(self):
        """无参数；清理 Bash 及该轨迹容器，保留宿主工作区和日志，无返回值。"""
        self.stop.set()
        subprocess.run(
            ["docker", "rm", "-f", self.name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        if self.bash is not None:
            self.bash.close()
        # 已派发的单个 GPU 作业依照原预算收尾，之后才允许轨迹进行最终验收。
        if self.rpc_thread is not None and self.rpc_thread is not threading.current_thread():
            self.rpc_thread.join()

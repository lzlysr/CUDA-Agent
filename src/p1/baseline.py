"""只测量原始 PyTorch 的 eager/torch.compile；不加载 Qwen、不生成候选。

查看 train 题单（CPU）：PYTHONPATH=src python -B -m p1.baseline --list --split train。
测一题（GPU）：PYTHONPATH=src python -B -m p1.baseline --task-id <ID> --gpu 3。
多卡：PYTHONPATH=src python -B -m p1.baseline --limit 100 --gpus 2,3 --run runs/p1/multi。
续跑：PYTHONPATH=src python -B -m p1.baseline --resume --run runs/p1/multi --gpus 2,3。
输出：新建的 runs/p1/baseline-<时间>/ 下的 JSON、CSV、Markdown 与参考日志。
--dry-run 只打印将使用的配置；仅真正执行测时时会启动 GPU 容器。
Ctrl+C/TERM 停止派新题，等待当前各卡题目结束并保存；默认新运行卡号仍为 7。"""

import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import ExitStack
import csv
import io
import json
import math
import signal
import statistics
from pathlib import Path
import subprocess
import sys
from threading import Event

from .data import ROOT
from .baseline_state import (
    archive_incomplete, atomic_json as dump, atomic_text, beijing_now,
    cleanup_reference_artifacts, completed_reference, exclusive_lock,
    guard_active_containers, guard_legacy_controller,
    protocol_fingerprints, protocol_matches, run_lock_path, task_fingerprints,
)

# 同项目不同 --run 目录也共享编号卡锁，避免两个新版基线进程抢同一张卡。
GPU_LOCK_ROOT = ROOT / ".runtime/locks"


def select_tasks(records, split="train", task_ids=None, limit=1):
    """输入记录、集合、ID 列表与数量；按组去重后返回确定性排序的任务列表。"""
    unique = {item["task_id"]: item for item in records if item["split"] == split}
    requested = set(task_ids or [])
    missing = requested - unique.keys()
    if missing:
        raise ValueError(f"任务不在 {split} 集合：{sorted(missing)}")
    tasks = [unique[key] for key in sorted(unique) if not requested or key in requested]
    return tasks if requested else tasks[:limit]


def gpu_available(index):
    """输入 GPU 编号；沿用当前仓库的 60000 MiB 显存占用阈值，超限则拒绝。"""
    value = subprocess.check_output(
        ["nvidia-smi", f"--id={index}", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        text=True,
    )
    if int(value.strip()) > 60000:
        raise RuntimeError(f"GPU {index} 已被占用，未启动测时")


# 汇总的数值指标及展示名称；CUDA/墙钟单位为毫秒，首调/控制器成本单位为秒。
METRICS = {
    "eager_cuda_ms": "eager CUDA ms",
    "compile_cuda_ms": "compile CUDA ms",
    "eager_wall_ms": "eager wall ms",
    "compile_wall_ms": "compile wall ms",
    "compile_over_eager_speedup": "eager/compile",
    "compile_first_call_seconds": "compile 首调 s",
    "controller_seconds": "控制器 s",
}


def baseline_mean(results):
    """输入逐题摘要；返回算术均值行和处理数量，不把拒绝或缺失指标补成零。

    每列只统计 status=ok 的有限数值；加速比取逐题比值的均值，不是两个耗时均值相除。
    已有 MEAN 行不参与重算；整列没有有效值时返回 None。
    """
    samples = [row for row in results if row.get("task_id") != "MEAN"]
    valid = [row for row in samples if row.get("status") == "ok"]
    mean = {
        "task_id": "MEAN",
        "split": "all",
        "status": "summary",
        "sample_count": len(samples),
        "valid_sample_count": len(valid),
        "rejected_sample_count": len(samples) - len(valid),
        "detail_path": "",
    }
    for key in METRICS:
        values = [
            row[key] for row in valid
            if isinstance(row.get(key), (int, float))
            and not isinstance(row[key], bool)
            and math.isfinite(row[key])
        ]
        mean[key] = statistics.fmean(values) if values else None
    return mean


def rejection_summary(output, detail_path):
    """输入 run 根目录和预检索引路径；读取阶段证据，返回简短的拒绝阶段与原因。

    只解释已有结果，不修改预检条件、容差或资源设置；完整错误仍留在阶段详情文件中。
    """
    try:
        index = json.loads((Path(output) / detail_path).read_text())
        for stage, key in [("reference_prepare", "prepare_result_path"),
                           ("reference_bench", "bench_result_path")]:
            if key not in index:
                continue
            record = json.loads((Path(output) / index[key]).read_text())
            constant = stage == "reference_prepare" and record.get("near_constant_sampled_output")
            if record.get("status") == "ok" and not constant:
                continue
            error = record.get("error", "")
            if constant:
                reason = "采样输出近常数，未进入性能测试"
            elif record.get("status") == "timeout":
                reason = "参考执行超时"
            elif "out of memory" in error.lower():
                reason = "CUDA 显存不足"
            elif "same device" in error.lower() or "two devices" in error.lower():
                reason = "CPU/GPU 张量设备不一致"
            elif "nonfinite" in error.lower():
                reason = "原始模型输出包含 NaN/Inf"
            elif record.get("error_type") == "UnpicklingError":
                reason = "参数快照不兼容 weights_only 安全加载"
            elif record.get("error_type") == "AssertionError":
                reason = ("原始模型重复运行未通过零容差一致性检查" if stage == "reference_prepare"
                          else "torch.compile 与 eager 的输出比较未通过")
            elif error.startswith("Comparing"):
                reason = "数值比较器内部异常，需查看阶段日志中的异常链"
            else:
                first_line = error.splitlines()[0] if error else record.get("status", "unknown")
                reason = f"{record.get('error_type', 'error')}: {first_line[:180]}"
            return {"rejection_stage": stage, "rejection_reason": reason}
    except (OSError, json.JSONDecodeError, TypeError):
        return {"rejection_reason": "预检未通过，阶段详情缺失或无法读取"}
    return {"rejection_reason": "预检未通过，请查看详情索引"}


def write_baseline_outputs(output, results):
    """输入 run 目录和逐题摘要；更新 JSON/CSV/Markdown，首个数据行均为 MEAN。

    保留 CSV 和 Markdown 的表头。此函数只读写汇总，不调用模型、容器或 GPU。
    """
    output = Path(output)
    samples = [row for row in results if row.get("task_id") != "MEAN"]
    mean = baseline_mean(samples)
    rows = [mean] + samples
    dump(output / "results.json", rows)
    fields = [
        "task_id", "split", "status", "sample_count", "valid_sample_count", "rejected_sample_count",
        *METRICS, "evaluation_gpu", "detail_path", "rejection_stage", "rejection_reason",
    ]
    # 先在内存生成完整 CSV，再原子替换，避免中断时只剩半个文件。
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    atomic_text(output / "results.csv", stream.getvalue())
    lines = [
        "# PyTorch 原始性能基线", "",
        (f"已处理 {mean['sample_count']} 题；成功 {mean['valid_sample_count']} 题；"
         f"拒绝/失败 {mean['rejected_sample_count']} 题。"),
        "首行 MEAN 为各指标的算术均值，只统计成功样本中的有限数值；缺失值和拒绝样本不补零，无有效数据记为 —。",
        "加速比的均值是逐题 eager/compile 比值的均值，不是两列耗时均值相除。",
        "不加载 Qwen；编译首调成本不混入稳态耗时。", "",
        "| 任务 | 状态 | " + " | ".join(METRICS.values()) + " |",
        "|---|---|" + "---:|" * len(METRICS),
    ]
    for row in rows:
        values = [str(row.get(key)) if row.get(key) is not None else "—" for key in METRICS]
        lines.append(f"| {row['task_id']} | {row['status']} | " + " | ".join(values) + " |")
    rejected = [row for row in samples if row.get("status") != "ok"]
    if rejected:
        lines += ["", "## 被拒绝的样本", "", "| 任务 | 阶段 | 原因 | 详情路径 |", "|---|---|---|---|"]
        for row in rejected:
            # 错误描述可含换行或竖线，写入 Markdown 前处理，避免破坏表格结构。
            reason = row.get("rejection_reason", "请查看详情").replace("|", r"\|").replace("\n", " ")
            lines.append(
                f"| {row['task_id']} | {row.get('rejection_stage', '—')} | {reason} | "
                f"`{row.get('detail_path', '')}` |"
            )
    atomic_text(output / "REPORT.md", "\n".join(lines) + "\n")
    return rows


def result_row(task, result, output, gpu):
    """输入题目、评测结果、run 和卡号；返回不嵌套详情的逐题摘要，支持恢复后重建。"""
    row = {
        "task_id": task["task_id"],
        "split": task["split"],
        "status": result["status"],
        "detail_path": result["result_path"],
        "evaluation_gpu": result.get("evaluation_gpu") if result.get("evaluation_gpu") is not None else gpu,
    }
    if result["status"] == "ok":
        eager = result["eager"]["cuda_ms"]
        compile_ms = result["compile"]["cuda_ms"]
        row.update(
            eager_cuda_ms=eager,
            compile_cuda_ms=compile_ms,
            eager_wall_ms=result["eager"]["wall_ms"],
            compile_wall_ms=result["compile"]["wall_ms"],
            compile_over_eager_speedup=eager / compile_ms if compile_ms > 0 else None,
            compile_first_call_seconds=result.get("compile_first_call_seconds"),
            controller_seconds=result.get("controller_seconds"),
        )
    else:
        row.update(rejection_summary(output, result["result_path"]))
    return row


def parse_gpus(value):
    """输入逗号分隔的物理卡号文本；返回不重复的非负整数列表，错误时报告参数问题。"""
    try:
        devices = [int(item.strip()) for item in value.split(",")]
    except ValueError as error:
        raise argparse.ArgumentTypeError("--gpus 应为卡号列表，例如 2,3") from error
    if not devices or any(device < 0 for device in devices) or len(devices) != len(set(devices)):
        raise argparse.ArgumentTypeError("GPU 编号必须非负且不可重复")
    return devices


def validate_resume(cfg, tasks, manifest, output):
    """输入续跑配置、题目、原 manifest 和目录；验证数据、协议一致，失败时不改结果。"""
    if manifest.get("mode") != "reference_only":
        raise ValueError("该目录不是纯基线运行，不能用 p1.baseline 恢复")
    ids = [task["task_id"] for task in tasks]
    if ids != manifest.get("task_ids") or manifest.get("task_count", len(ids)) != len(ids):
        raise ValueError("续跑必须沿用 manifest 中完全相同的题单和顺序")
    # 卡号属于执行资源，可在续跑时改变；其余配置保留原值，不能混入不同测时协议。
    def settings(value):
        """输入完整配置；返回不含可变卡号的协议配置，用于比较而不修改原字典。"""
        return {key: item for key, item in value.items() if key != "evaluation_gpu"}

    if settings(cfg) != settings(manifest["config"]):
        raise ValueError("续跑配置与原 manifest 不一致，请使用原配置或新建运行目录")
    if ("task_fingerprints" in manifest
            and manifest["task_fingerprints"] != task_fingerprints(tasks)):
        raise ValueError("任务源码或 train/test 归属已经变化，拒绝混合结果")
    if ("protocol_fingerprints" in manifest
            and not protocol_matches(manifest["protocol_fingerprints"],
                                     protocol_fingerprints(Path(__file__).parent))):
        raise ValueError("评测实现已发生变化，拒绝以原协议的名义继续测量")
    for task in tasks:
        model = Path(output) / "references" / task["task_id"] / "model.py"
        if model.exists() and model.read_text(encoding="utf-8") != task["code"]:
            raise ValueError(f"原参考源码与当前数据不同：{task['task_id']}")


def run_baselines(cfg, tasks, output, evaluator_factory=None, *, resume=False, gpus=None, stop_event=None):
    """输入配置、题单和目录；单卡或多卡测基线，逐题持久化并返回已完成摘要列表。

    resume=True 时复用原题单/终态阶段结果；中断题先归档再从参考预检开始。
    gpus 为物理编号列表，一卡一次只执行一题；stop_event 置位后不再派发新题，
    等当前题结束再退出。evaluator_factory(cfg, output) 供 CPU 测试注入，不用 GPU。
    返回值按原题单排序且不含 MEAN；文件首行仍为 MEAN。异常时保留已完成结果。
    每题汇总保存后删除其 input-spec 和 compile-cache；仅保留结果、源码与日志。
    """
    output = Path(output).resolve()
    devices = list(gpus) if gpus is not None else [cfg.get("evaluation_gpu", 7)]
    if (not devices
            or any(type(device) is not int or device < 0 for device in devices)
            or len(devices) != len(set(devices))):
        raise ValueError("GPU 必须为不重复的非负整数编号")
    ids = [task["task_id"] for task in tasks]
    if len(ids) != len(set(ids)):
        raise ValueError("题单存在重复任务")
    if resume:
        if not output.is_dir():
            raise ValueError(f"续跑目录不存在：{output}")
    else:
        output.mkdir(parents=True, exist_ok=False)
    real_evaluator = evaluator_factory is None
    if evaluator_factory is None:
        from .evaluation import Evaluator

        def evaluator_factory(config, run):
            """输入含调度卡号的配置和 run；复用公共配置，卡号单独传给宿主评测器。"""
            return Evaluator(
                manifest["config"], run, gpu=config["evaluation_gpu"], write_config=False,
            )

    stop_event = stop_event if stop_event is not None else Event()
    with ExitStack() as locks:
        # 运行锁也放在 .runtime/locks，续跑旧目录不再新增或改名其中的文件。
        locks.enter_context(exclusive_lock(run_lock_path(output, GPU_LOCK_ROOT)))
        if (output / "controller.lock").exists():
            # 兼容上一版已创建的锁：只复用现有文件，不删除或重新命名。
            locks.enter_context(exclusive_lock(output / "controller.lock"))
        if resume:
            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            validate_resume(cfg, tasks, manifest, output)
            if "execution" not in manifest:
                guard_legacy_controller(output, ROOT)
        else:
            # task_count 紧邻并位于 task_ids 之前，表示所选去重任务总数。
            manifest = {
                "mode": "reference_only", "config": dict(cfg),
                "task_count": len(tasks), "task_ids": ids, "loads_qwen": False,
            }
        manifest.setdefault("task_fingerprints", task_fingerprints(tasks))
        # 已验证兼容时更新指纹，旧版多卡 run 不会因配置文件布局调整而被迫重跑。
        manifest["protocol_fingerprints"] = protocol_fingerprints(Path(__file__).parent)
        completed = {}
        if resume:
            for task in tasks:
                result = completed_reference(output, task)
                if result is not None:
                    completed[task["task_id"]] = result_row(
                        task, result, output, manifest["config"].get("evaluation_gpu", 7)
                    )
        pending = [task for task in tasks if task["task_id"] not in completed]
        execution = {
            "status": "running", "gpus": devices,
            "started_at": beijing_now().isoformat(timespec="seconds"),
            "resumed": resume, "reused_task_count": len(completed),
        }
        manifest["execution"] = execution

        def save(status="running"):
            """输入执行状态；由唯一控制线程按固定顺序原子更新汇总和进度，返回 None。"""
            rows = [completed[ident] for ident in ids if ident in completed]
            write_baseline_outputs(output, rows)
            execution.update(
                status=status,
                completed_task_count=len(rows),
                pending_task_count=len(tasks) - len(rows),
                updated_at=beijing_now().isoformat(timespec="seconds"),
            )
            dump(output / "manifest.json", manifest)

        save()
        # 恢复上次在“结果写完、清理之前”中断的情况；完成题不再需要参数快照。
        for task_id in completed:
            cleanup_reference_artifacts(output, task_id)
        # 先落盘可恢复的题单；争卡失败时也保留合法 manifest，而非留下无法恢复的空目录。
        try:
            if pending:
                if resume and real_evaluator:
                    guard_active_containers(output)
                # 卡锁只协调同项目新版基线，不控制其他用户的 GPU 进程。
                for gpu in sorted(devices):
                    locks.enter_context(exclusive_lock(GPU_LOCK_ROOT / f"p1-baseline-gpu-{gpu}.lock"))
        except Exception as error:
            execution["last_error"] = f"{type(error).__name__}: {error}"
            save("failed")
            raise
        if pending and real_evaluator:
            # 所有卡共享同一份参数；只在启动各卡线程前写一次，沿用原来的文件名。
            # 实际卡号由 Evaluator.gpu 决定，cfg 中的 evaluation_gpu 仍是单卡默认值。
            dump(output / "worker-config.json", manifest["config"])
        print(
            f"基线：已完成/复用 {len(completed)}/{len(tasks)}，待处理 {len(pending)}，GPU {devices}",
            flush=True,
        )
        queue = iter(pending)
        active = {}
        failure = None
        evaluators = {}
        # 不提前排入全部任务：每张卡最多一个 Future，停止后队列中的题完全没有启动。
        with ThreadPoolExecutor(max_workers=len(devices), thread_name_prefix="p1-baseline") as pool:
            def submit(gpu):
                """输入空闲卡号；若仍可派题，提交一题并登记 Future，无返回值。"""
                if stop_event.is_set() or failure is not None:
                    return
                task = next(queue, None)
                if task is None:
                    return
                if gpu not in evaluators:
                    config = dict(manifest["config"], evaluation_gpu=gpu)
                    evaluators[gpu] = evaluator_factory(config, output)
                archived = archive_incomplete(output, task["task_id"])
                if archived:
                    print(f"归档未完成题 {task['task_id']}：{archived}", flush=True)
                active[pool.submit(evaluators[gpu].profile_reference, task)] = (gpu, task)

            try:
                for gpu in devices:
                    submit(gpu)
                while active:
                    done, _ = wait(active, timeout=0.2, return_when=FIRST_COMPLETED)
                    for future in done:
                        gpu, task = active.pop(future)
                        try:
                            row = result_row(task, future.result(), output, gpu)
                            completed[task["task_id"]] = row
                            save()
                            # 必须先保存可恢复的结果，之后才能删本题的参数和编译缓存。
                            cleanup_reference_artifacts(output, task["task_id"])
                            print(json.dumps(row, ensure_ascii=False), flush=True)
                        except Exception as error:
                            # 基础设施/控制器异常不伪装成题目失败；其他卡已在执行的结果仍收齐。
                            failure = failure or error
                            execution["last_error"] = f"{type(error).__name__}: {error}"
                    if failure is None:
                        # 已完成的卡已从 active 移除，重新派发不会与其上一次测时重叠。
                        busy = {gpu for gpu, _ in active.values()}
                        for gpu in devices:
                            if gpu not in busy:
                                submit(gpu)
            except Exception as error:
                failure = failure or error
                execution["last_error"] = f"{type(error).__name__}: {error}"
                # 构造评测器/归档异常也停止派题，等待其余已启动任务并保留可用结果。
                for future, (gpu, task) in list(active.items()):
                    try:
                        completed[task["task_id"]] = result_row(task, future.result(), output, gpu)
                        save()
                        cleanup_reference_artifacts(output, task["task_id"])
                    except Exception:
                        pass
        if failure is not None:
            status = "failed"
        elif len(completed) == len(tasks):
            status = "completed"
        else:
            status = "interrupted"
        save(status)
        if failure is not None:
            raise failure
        return [completed[ident] for ident in ids if ident in completed]


def main(argv=None):
    """输入可选参数；列题/预览或测量，正常返回 0，收到停止信号时保留结果并返回 130。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=["train", "test"])
    parser.add_argument("--task-id", action="append", default=[])
    parser.add_argument("--limit", type=int)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--gpu", type=int, help="单张 GPU 的物理编号；新运行默认 7")
    group.add_argument("--gpus", type=parse_gpus, help="多卡物理编号，例如 2,3；每卡串行测题")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--run")
    parser.add_argument("--resume", action="store_true", help="从 --run 目录的固定题单和配置恢复")
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit < 1:
        parser.error("--limit 必须为正整数")
    if args.gpu is not None and args.gpu < 0:
        parser.error("--gpu 必须是非负整数编号")
    if args.resume and (not args.run or args.split is not None or args.task_id or args.limit is not None):
        parser.error("--resume 必须指定 --run，且不能再传 --split/--task-id/--limit；题单沿用原 manifest")
    records = [
        json.loads(line)
        for line in (ROOT / "data/processed/tasks.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    # 续跑读取原 manifest，避免当前配置文件或默认 --limit=1 改变原实验。
    output = ROOT / (args.run or beijing_now().strftime("runs/p1/baseline-%Y%m%dT%H%M%S"))
    if args.resume:
        manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
        unique = {task["task_id"]: task for task in records}
        try:
            tasks = [unique[ident] for ident in manifest["task_ids"]]
        except KeyError as error:
            parser.error(f"当前数据缺少原任务：{error}")
        cfg = dict(manifest["config"])
        validate_resume(cfg, tasks, manifest, output)
    else:
        tasks = select_tasks(records, args.split or "train", args.task_id,
                             args.limit if args.limit is not None else 1)
        cfg = json.loads((ROOT / "configs/p1-pilot.json").read_text(encoding="utf-8"))
    if args.list:
        for task in tasks:
            print(task["task_id"], task["ops"])
        return 0
    if args.gpus is not None:
        devices = args.gpus
    elif args.gpu is not None:
        devices = [args.gpu]
    else:
        devices = [cfg.get("evaluation_gpu", 7) if args.resume else 7]
    cfg["evaluation_gpu"] = devices[0]
    reused = sum(completed_reference(output, task) is not None for task in tasks) if args.resume else 0
    if args.dry_run:
        print(
            json.dumps(
                {
                    "tasks": [task["task_id"] for task in tasks],
                    "evaluation_gpu": cfg["evaluation_gpu"],
                    "evaluation_gpus": devices,
                    "resume": args.resume,
                    "completed_task_count": reused,
                    "pending_task_count": len(tasks) - reused,
                    "loads_qwen": False,
                },
                indent=2,
            )
        )
        return 0
    if len(tasks) > reused:
        for gpu in devices:
            gpu_available(gpu)
    stop = Event()

    def request_stop(signum, frame):
        """输入操作系统信号和栈；仅置停止标志，当前各卡的题收尾后再退出。"""
        if not stop.is_set():
            print("收到停止信号：不再派新题，等待当前题完成并保存；之后可用 --resume 继续。", file=sys.stderr, flush=True)
        stop.set()

    previous = {number: signal.signal(number, request_stop) for number in (signal.SIGINT, signal.SIGTERM)}
    try:
        run_baselines(cfg, tasks, output, resume=args.resume, gpus=devices, stop_event=stop)
    finally:
        for number, handler in previous.items():
            signal.signal(number, handler)
    return 130 if stop.is_set() else 0


if __name__ == "__main__":
    raise SystemExit(main())

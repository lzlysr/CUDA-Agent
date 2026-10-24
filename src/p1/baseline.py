"""只测量原始 PyTorch 的 eager/torch.compile；不加载 Qwen、不生成候选。

查看 train 题单（CPU）：PYTHONPATH=src python -B -m p1.baseline --list --split train。
测一题（GPU）：PYTHONPATH=src python -B -m p1.baseline --task-id <ID> --gpu 3。
输出：新建的 runs/p1/baseline-<时间>/ 下的 JSON、CSV、Markdown 与参考日志。
--dry-run 只打印将使用的配置；仅真正执行测时时会启动 GPU 容器。"""

import argparse
import csv
import json
from pathlib import Path
import subprocess
import time

from .data import ROOT, dump


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
    """输入 GPU 编号；查询是否占用，若显存超过 100 MiB 则拒绝，返回 None。"""
    value = subprocess.check_output(
        ["nvidia-smi", f"--id={index}", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        text=True,
    )
    if int(value.strip()) > 100:
        raise RuntimeError(f"GPU {index} 已被占用，未启动测时")


def run_baselines(cfg, tasks, output, evaluator_factory=None):
    """输入配置、任务列表、新输出目录；逐题测基线并返回结果列表。

    evaluator_factory 供 CPU 单测注入假评测器。真实调用才导入 Evaluator；
    不导入 p1.run/vLLM，不会创建语言模型。已有输出目录不会被覆盖。
    """
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    if evaluator_factory is None:
        from .evaluation import Evaluator

        evaluator_factory = Evaluator
    evaluator = evaluator_factory(cfg, output)
    dump(
        output / "manifest.json",
        {
            "mode": "reference_only",
            "config": cfg,
            "task_ids": [task["task_id"] for task in tasks],
            "loads_qwen": False,
        },
    )
    results = []
    for task in tasks:
        result = evaluator.profile_reference(task)
        row = {
            "task_id": task["task_id"],
            "split": task["split"],
            "status": result["status"],
            "detail": result,
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
            )
        results.append(row)
        dump(output / "results.json", results)
        print(
            json.dumps(
                {key: value for key, value in row.items() if key != "detail"}, ensure_ascii=False
            ),
            flush=True,
        )
    fields = [
        "task_id",
        "split",
        "status",
        "eager_cuda_ms",
        "compile_cuda_ms",
        "eager_wall_ms",
        "compile_wall_ms",
        "compile_over_eager_speedup",
    ]
    with (output / "results.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    lines = [
        "# PyTorch 原始性能基线",
        "",
        "不加载 Qwen；不生成 CUDA 候选。编译首调成本不混入稳态耗时。",
        "",
        "| 任务 | 状态 | eager CUDA ms | compile CUDA ms | eager/compile |",
        "|---|---|---:|---:|---:|",
    ]
    for row in results:
        lines.append(
            f'| {row["task_id"]} | {row["status"]} | {row.get("eager_cuda_ms", "—")} | {row.get("compile_cuda_ms", "—")} | {row.get("compile_over_eager_speedup", "—")} |'
        )
    (output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return results


def main(argv=None):
    """输入可选命令行参数；列题/预览或测量基线，返回 None；GPU 需显式指定。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=["train", "test"], default="train")
    parser.add_argument("--task-id", action="append", default=[])
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--gpu", type=int, default=7)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--run")
    args = parser.parse_args(argv)
    if args.limit < 1:
        parser.error("--limit 必须为正整数")
    records = [
        json.loads(line)
        for line in (ROOT / "data/processed/tasks.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    tasks = select_tasks(records, args.split, args.task_id, args.limit)
    if args.list:
        for task in tasks:
            print(task["task_id"], task["ops"])
        return
    cfg = json.loads((ROOT / "configs/p1-pilot.json").read_text(encoding="utf-8"))
    if args.gpu is not None:
        cfg["evaluation_gpu"] = args.gpu
    if args.dry_run:
        print(
            json.dumps(
                {
                    "tasks": [task["task_id"] for task in tasks],
                    "evaluation_gpu": cfg["evaluation_gpu"],
                    "loads_qwen": False,
                },
                indent=2,
            )
        )
        return
    if args.gpu is None:
        parser.error("实际测时需显式指定 --gpu，例如 --gpu 3")
    gpu_available(args.gpu)
    output = ROOT / (args.run or time.strftime("runs/p1/baseline-%Y%m%dT%H%M%S", time.gmtime()))
    run_baselines(cfg, tasks, output)


if __name__ == "__main__":
    main()

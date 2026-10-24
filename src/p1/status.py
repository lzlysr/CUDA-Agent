"""只读查询 P1 状态；不初始化 CUDA，也不加载语言模型。

旧结果：PYTHONPATH=src python -B -m p1.status --run runs/p1/pilot-seed42。
新流程：PYTHONPATH=src python -B -m p1.status --run runs/p1/pilot-train-v3-seed42。
--report 将摘要写到所选 run/STATUS.md，不覆盖人工审查报告。
输入：该 run 的 manifest/tasks/逐条 result；输出：终端 JSON 或 Markdown。"""

import argparse
import json
from pathlib import Path
import time

from .data import ROOT, dump

RUN = ROOT / "runs/p1/pilot-seed42"


def read(path, default=None):
    """输入 JSON 路径和缺省值；缺失或尚未写完时返回缺省值，否则返回对象。"""
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def status(run=RUN):
    """输入运行目录；使用 run 自身的题单/配置返回状态，不套用新的 train 划分。"""
    run = Path(run)
    manifest = read(run / "manifest.json", {})
    cfg = manifest.get("config", read(ROOT / "configs/p1-pilot.json", {}))
    tasks = read(run / "tasks.json")
    if tasks is None:
        ids = manifest.get("task_ids")
        tasks = (
            [{"task_id": ident} for ident in ids]
            if ids is not None
            else read(ROOT / "data/processed/pilot.json", [])
        )
    rows = []
    for task in tasks:
        preflight = read(run / "references" / task["task_id"] / "preflight.json", {})
        row = {"task_id": task["task_id"], "preflight": preflight.get("status", "pending")}
        for mode in ["single", "agent"]:
            directory = run / "trajectories" / task["task_id"] / mode
            result = read(directory / "result.json")
            if result is not None:
                row[mode] = {
                    "status": "complete",
                    "correct": result["correct"],
                    "termination": result["termination"],
                }
            else:
                row[mode] = {
                    "status": "not_scored",
                    "completed_generations": len(list(directory.glob("generation-*.json"))),
                    "completed_tools": len(list(directory.glob("tool-*.json"))),
                }
        rows.append(row)
    return {
        "run": str(run),
        "manifest_status": manifest.get("status", "not_started"),
        "generation_gpu": cfg.get("generation_gpu"),
        "evaluation_gpu": cfg.get("evaluation_gpu"),
        "completed_trajectories": sum(
            row[mode]["status"] == "complete" for row in rows for mode in ["single", "agent"]
        ),
        "reference_valid": sum(row["preflight"] == "ok" for row in rows),
        "rows": rows,
        "summary": read(run / "summary.json"),
    }


def report(run=RUN, output=None):
    """输入运行目录和可选输出位置；写入简明状态表，返回报告路径。"""
    state = status(run)
    destination = Path(output) if output else Path(run) / "STATUS.md"
    lines = [
        "# P1 运行状态",
        "",
        f'状态：{state["manifest_status"]}；完整轨迹：{state["completed_trajectories"]}。',
        "",
        "| 任务 | 预检 | single | agent |",
        "|---|---|---|---|",
    ]
    for row in state["rows"]:
        cells = []
        for mode in ["single", "agent"]:
            result = row[mode]
            cells.append(
                ("正确" if result["correct"] else "失败") + " / " + result["termination"]
                if result["status"] == "complete"
                else "未计分"
            )
        lines.append(f'| {row["task_id"]} | {row["preflight"]} | {cells[0]} | {cells[1]} |')
    lines.extend(
        [
            "",
            "未完成项不计为失败。部分运行不能作为完整 test/KernelBench 指标。详细分析见人工审查报告。",
            "",
        ]
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(lines), encoding="utf-8")
    return destination


def main(argv=None):
    """输入可选命令行参数；查询状态或等待指定进程结束后写摘要，返回 None。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="runs/p1/pilot-seed42")
    parser.add_argument("--watch-pid", type=int)
    parser.add_argument("--report", action="store_true")
    args = parser.parse_args(argv)
    run = ROOT / args.run
    if args.watch_pid:
        while True:
            try:
                alive = b"p1.run" in Path(f"/proc/{args.watch_pid}/cmdline").read_bytes()
            except FileNotFoundError:
                alive = False
            if not alive:
                break
            time.sleep(15)
        manifest = read(run / "manifest.json", {})
        if manifest.get("status") == "running":
            manifest["status"] = "stopped_before_completion"
            dump(run / "manifest.json", manifest)
        print(report(run))
    elif args.report:
        print(report(run))
    else:
        print(json.dumps(status(run), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

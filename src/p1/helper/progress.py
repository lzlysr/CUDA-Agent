"""P1 终端进度：预检及 single/agent 完成数、通过率和有效加速比几何均值。

由 p1.run 的主控制器调用，不独立运行，不写实验文件，不改变评测结果。
实际调用：bash scripts/p1-run.sh --run runs/p1/<目录>。
非交互终端关闭动态刷新，仍保留逐题 RESULT 日志；模型子进程不创建进度条。
"""

import math
import statistics
import sys

from tqdm import tqdm


MODES = ("single", "agent")


def valid_speedups(row):
    """输入轨迹摘要；返回是否正确、测时有效且两个加速比均为有限正数。"""
    if not row.get("correct") or not row.get("timing_valid"):
        return False
    return all(
        isinstance(row.get(key), (int, float))
        and not isinstance(row[key], bool)
        and math.isfinite(row[key])
        and row[key] > 0
        for key in ("speedup_eager", "speedup_compile")
    )


def speedup_text(value):
    """输入有效加速比或 None；返回两位小数加 x，缺失时显示 --，不以零代替。"""
    return f"{value:.2f}x" if value is not None else "--"


class PilotProgress:
    """主进程唯一持有的显示状态；各模式独立计数，续跑结果按题号去重。"""

    def __init__(self):
        """无输入；初始化空显示状态，不运行模型或评测器。"""
        self.reference = None
        self.reference_ids = set()
        self.rejected = 0
        self.bars = {}
        self.completed = {mode: {} for mode in MODES}
        self.task_ids = set()
        self.interactive = sys.stderr.isatty()

    def start_references(self, tasks):
        """输入固定题单；打开参考预检进度条，已缓存预检也按完成计数。"""
        self.reference = tqdm(
            total=len(tasks), desc="参考预检", unit="题", dynamic_ncols=True,
            file=sys.stderr, disable=not self.interactive,
        )

    def reference_done(self, task_id, status):
        """输入题号与预检状态；去重推进预检进度，显示已拒绝题数。"""
        if task_id in self.reference_ids:
            return
        self.reference_ids.add(task_id)
        self.rejected += status != "ok"
        self.reference.set_postfix_str(f"rejected={self.rejected}", refresh=False)
        self.reference.update(1)
        self.reference.refresh()
        if not self.interactive or status != "ok":
            self.write("PREFLIGHT", task_id, status)

    def start_modes(self, tasks, rows):
        """输入有效题单及已完成轨迹；恢复计数后创建两条进度，不重复打印旧结果。"""
        if self.reference is not None:
            self.reference.close()
        self.task_ids = {task["task_id"] for task in tasks}
        self._remember(rows)
        for position, mode in enumerate(MODES):
            self.bars[mode] = tqdm(
                total=len(tasks), initial=len(self.completed[mode]), desc=mode, unit="题",
                position=position, dynamic_ncols=True, file=sys.stderr,
                disable=not self.interactive,
            )
        self.sync(rows, announce=False)

    def _remember(self, rows):
        """输入根汇总记录；仅缓存有效题的完成轨迹，返回本次首次见到的记录列表。"""
        added = []
        for row in rows:
            mode = row.get("mode")
            task_id = row.get("task_id")
            if mode not in self.completed or task_id not in self.task_ids or "correct" not in row:
                continue
            if task_id not in self.completed[mode]:
                added.append(row)
            # 显示缓存只保留指标，避免续跑时长期持有完整工具日志和候选详情。
            self.completed[mode][task_id] = {
                key: row.get(key)
                for key in ("task_id", "mode", "correct", "timing_valid",
                            "speedup_eager", "speedup_compile")
            }
        return added

    def sync(self, rows, announce=True):
        """输入已成功落盘的轨迹列表；更新通过率/几何均值，并逐条打印新增结果。"""
        if not self.bars:
            return
        added = self._remember(rows)
        rates = {}
        for mode, bar in self.bars.items():
            items = list(self.completed[mode].values())
            correct = sum(bool(row["correct"]) for row in items)
            valid = [row for row in items if valid_speedups(row)]
            rate = f"{correct / len(items):.1%}" if items else "--"
            rates[mode] = f"{rate}({correct}/{len(items)})"
            means = {}
            for key in ("speedup_eager", "speedup_compile"):
                means[key] = (
                    math.exp(statistics.mean(math.log(row[key]) for row in valid))
                    if valid else None
                )
            # GM 只对正确且测时有效的结果统计；通过率对所有已完成结果统计。
            postfix = (
                f"pass_rate={rate}({correct}/{len(items)}) "
                f"speedup_eager={speedup_text(means['speedup_eager'])} "
                f"speedup_compile={speedup_text(means['speedup_compile'])} "
                f"valid={len(valid)}(GM)"
            )
            bar.set_postfix_str(postfix, refresh=False)
            delta = len(items) - bar.n
            if bar.disable:
                bar.n = len(items)
            else:
                bar.update(delta)
                bar.refresh()
        if announce:
            for row in added:
                valid = valid_speedups(row)
                self.write(
                    "RESULT", row["task_id"], row["mode"],
                    f"correct={bool(row['correct'])}", row.get("termination", ""),
                    f"pass_rate={rates[row['mode']]}",
                    f"speedup_eager={speedup_text(row.get('speedup_eager') if valid else None)}",
                    f"speedup_compile={speedup_text(row.get('speedup_compile') if valid else None)}",
                )

    @staticmethod
    def write(*parts):
        """输入若干日志片段；通过 tqdm.write 输出到 stderr，避免覆盖进度条。"""
        tqdm.write(" ".join(str(part) for part in parts), file=sys.stderr)

    def close(self):
        """无输入；正常结束或异常退出时关闭显示，保留最终进度，不修改实验数据。"""
        if self.reference is not None:
            self.reference.close()
        for bar in self.bars.values():
            bar.close()

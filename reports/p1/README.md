# P1 入口

更新日期：2026-10-26。完整方案、环境、工具、评测协议和论文对齐说明统一见 [P1_REPORT.md](P1_REPORT.md)。实际实验结果写入每次独立的 run 目录。新运行的根目录 `results.json` 只保存摘要和 `detail_path`；预检与基线详情分别保存在 `cases/prepare_result.json`、`initial-bench/bench_result.json`，不再生成 `prepare.json` 或 `baseline.json` 副本。

## 只看 eager/torch.compile

```bash
cd /home/likun/junjxu/CUDA-Agent
# 列题，不启动 GPU。
bash scripts/p1-baseline.sh --list --split train --limit 12
# 预览将测量的任务，不启动 GPU。
bash scripts/p1-baseline.sh --dry-run --split train --limit 1 --gpu 3
# 实际测量排序后的第一道 train 题，使用 GPU 3，不加载 Qwen。
bash scripts/p1-baseline.sh --split train --limit 1 --gpu 3
```

结果写入新建的 `runs/p1/baseline-<时间>/`，包含 JSON、CSV、REPORT.md、eager/compile 稳态时间、compile 首调时间及 profiler。`--limit` 选的是排序后的 train 任务，不等同于模型 pilot；指定题目可使用 `--task-id`。

### 多卡与断点续跑

以下卡号为示例，请先确认空闲；不指定卡号的新运行仍默认 GPU 7。

```bash
# 两张卡各自串行处理独立题目，不加载 Qwen，也不是单题多卡计算。
bash scripts/p1-baseline.sh --split train --limit 100 --gpus 2,3 --run runs/p1/baseline-multi
# 恢复预览：只读原题单和阶段文件，不查询或启动 GPU。
bash scripts/p1-baseline.sh --resume --run runs/p1/baseline-multi --gpus 2,3 --dry-run
# 在原目录继续，可更换空闲卡；题单、容差、轮数和种子沿用原 manifest。
bash scripts/p1-baseline.sh --resume --run runs/p1/baseline-multi --gpus 2,3
```

`--gpu 3` 与 `--gpus 2,3` 二选一；只接受编号，重复编号会报错。续跑不再传 `--split/--limit/--task-id`。
多卡仍只保留一份 `worker-config.json`，公共参数只读共享，实际卡号由宿主单独传给 Docker。
原有运行文件不迁移或改名；新的运行锁位于项目 `.runtime/locks`，不在 run 内新增锁文件。
成功和明确拒绝的题都跳过；未完成题先归档至原 run 的 `interrupted/` 再重做。
每题结果持久保存后，自动删除该题的 `cases/input-spec/` 和 `compile-cache/`。
阶段 JSON、源码、日志和汇总保留，续跑依赖这些结果识别完成题，不要求参数快照仍存在。
Ctrl+C 或向控制器发 TERM 会停止派新题，等待各卡当前题结束保存后退出，最坏可能还需等待当前的两个 300 秒阶段。
根汇总损坏或丢失可从完整阶段详情重建；每个 JSON/CSV/Markdown 文件均原子替换，三文件并非跨文件事务。
已有旧版进程不具备新停止机制，不会自动切换多卡；恢复前须先结束原进程。详细行为和限制见报告第 12 节。

## 运行 Qwen single/agent pilot

当前 `data/processed/pilot.json` 是从 train 按 seed=42 确定性抽取的 12 题。启动脚本会激活 learner，使用 GPU 2 生成、GPU 3 评测，均按编号选择。128K GPU 流程仍需实际运行验证。

```bash
cd /home/likun/junjxu/CUDA-Agent
# 启动真实模型实验，使用新的输出目录。
bash scripts/p1-run.sh --run runs/p1/pilot-train-v3
```

已有运行目录需显式 `--resume` 且配置一致；新实验不要恢复其他 run。默认配置为 single/Agent 窗口 32768/131072、200 turns、temperature=0.7/top_p=0.95；没有额外的 4096 token 单次输出默认上限或 30 分钟轨迹上限。其余条件见报告与 `configs/p1-pilot.json`。

同一题的 single、agent 及最终隐藏验收全部结束、两条结果保存后，才删除共享的
`references/<id>/cases/input-spec/`、`compile-cache/`；不删除轨迹、候选源码或结果文件。
single 完成而 agent 中断时仍保留参数。`--prepare-only` 保留有效题参数供后续使用；
预检拒绝的题在拒绝结果保存后清理。清理后的完成题如需重新测量或验收，应新建运行重建参数。

```bash
conda activate .runtime/conda/learner
# 指定正在运行的目录，仅查询状态，不启动 GPU。
PYTHONPATH=src python -B -m p1.status --run runs/p1/pilot-train-v3
```

## 数据审计从哪里来

`reports/p1/data-audit.json` 由 `src/p1/data.py::prepare()` 自动生成，记录数据来源、审计及划分计数、pilot 题单。配置、生成器输出和该 JSON 均不使用 `split_version` 字段。

当前按规范化源码任务组划分 train/test≈8:2：4434/1108 组，对应 4803/1197 行。pilot 只用 train，test 留给正式评测；不按历史 pilot 调整归属。

```bash
conda activate .runtime/conda/learner
# 在项目根目录重生成任务、划分、pilot 和审计，仅使用 CPU。
PYTHONPATH=src python -B -m p1.data
```

该命令也会生成 `reports/p1/kernelbench-similarity.json` 的逐题源码相似性记录，不运行 KernelBench GPU 实验。现有数据无需为了启动 P1 重复生成。

## 测试与目录职责

```bash
# 在已激活的 learner 环境和项目根目录运行 CPU 回归。
PYTHONPATH=src python -B -m unittest discover -s tests/p1 -v
# 验证持久 Shell、utils 请求和只读保护；CPU Docker，零 GPU job。
PYTHONPATH=src python -B tests/p1/sandbox_smoke.py
```

- `src`：数据、评测器、worker、工具、模型循环、基线入口和状态查询等实现。
- `scripts`：环境设置与启动入口；每条命令上一行说明用途。
- `tests`：回归及集成测试代码；`gpu_*.py` 需显式运行，会使用 GPU。
- `runs`：每次运行的固定题单、配置、原始轨迹、候选快照和结果。
- `reports`：当前统一报告、入口说明和数据审计。

本次整理没有启动实验或定时任务。完整 test、KernelBench 和 P2 训练不由以上 pilot 命令自动扩展执行。

## 基线汇总的均值行

新基线运行的 JSON 首条记录、CSV/Markdown 表格首个数据行均为 MEAN，统计各数值指标的算术均值并记录处理/成功/拒绝数量。
均值按成功样本中每列实际存在的有限值计算，不把拒绝或缺失值补零；manifest 在 task_ids 前记录 task_count。
三种汇总逐题共同更新，REPORT 另列拒绝阶段和原因；完整错误仍保存在阶段详情中。

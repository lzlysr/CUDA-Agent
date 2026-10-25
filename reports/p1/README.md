# P1 入口

更新日期：2026-10-24。完整方案、环境、工具、评测协议和论文对齐说明统一见 [P1_REPORT.md](P1_REPORT.md)。实际实验结果写入每次独立的 run 目录。

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

## 运行 Qwen single/agent pilot

当前 `data/processed/pilot.json` 是从 train 按 seed=42 确定性抽取的 12 题。启动脚本会激活 learner，使用 GPU 2 生成、GPU 3 评测，均按编号选择。128K GPU 流程仍需实际运行验证。

```bash
cd /home/likun/junjxu/CUDA-Agent
# 启动真实模型实验，使用新的输出目录。
bash scripts/p1-run.sh --run runs/p1/pilot-train-v3
```

已有运行目录需显式 `--resume` 且配置一致；新实验不要恢复其他 run。默认配置为 single/Agent 窗口 32768/131072、200 turns、temperature=0.7/top_p=0.95；没有额外的 4096 token 单次输出默认上限或 30 分钟轨迹上限。其余条件见报告与 `configs/p1-pilot.json`。

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

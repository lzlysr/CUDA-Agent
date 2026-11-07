# P2 单轮同步在线 PPO

训练算法直接使用 `third_party/verl` v0.9.1 的 V1 同步训练器。
项目代码负责 Ops-6K 数据、CUDA 沙箱奖励和 Qwen3.5 Critic 接线。
完整说明见 [P2_REPORT.md](P2_REPORT.md)。

Actor/Critic 的模型加载均继承 verl 原生实现。Critic 经 load_valuehead_model 和
Transformers 的 AutoModelForTokenClassification 自动创建价值头，项目不再单独实现
Qwen Critic 加载函数。模型扩展只保留 LoRA 角色/dropout 配置、文本输入接线及更新审计。

2026-11-06 已在 `runs/p2/wiring-20261105` 完成两步 8K 接线，含从第一步检查点恢复、
Actor/Critic 更新和策略版本 0→1 的新采样。独立 CUDA 奖励重放也已通过。
四条短测回答均格式失败；本次验证训练接线，不代表正确率或能力提升。

本轮只允许 1–2 步短上下文接线验证；150 步参数仅作为后续配置，未获准自动运行。

唯一启动脚本为 `scripts/p2-train.sh`。入口先对原始 train/test 做 CPU 语法/接口与
提示词长度检查，在 run 下保存 `data-manifest.json`，再训练。
自定义 Dataset 直接读取配置中的原始文件：train.jsonl 用于训练、test.jsonl 用于验证，
取样时补充 prompt 和奖励字段；不二次划分，也不从 train 抽取监控题。
不生成转换后的训练/验证数据副本。清单冻结原始行及顺序、输入摘要和检查结果，
`config.json` 绑定清单路径及摘要，续跑时校验后复用原题单。
准备阶段不执行原题、不启动 Docker/GPU；实际 CUDA 检查与测时仍在奖励阶段执行。
历史已完成 run 保留原 Parquet 及检查点。本轮改变 LoRA 结构和验证协议，需使用新 run，
不能将旧的语言层 LoRA 检查点直接当作全模型 LoRA 续训。
主配置仅保留 `configs/p2-single-ppo-warmup.yaml`，生成流程由 Python 注册。

```bash
# 只读预览数据准备及配置；不写数据或运行文件，不启动 GPU。
bash scripts/p2-train.sh --run runs/p2/my-wire --wire --task-id ops6k-d4413b455033577a ops6k-ac6e648cc33136d2 --dry-run

# 可选：只检查两道题并保存轻量清单及运行配置，不启动训练。
bash scripts/p2-train.sh --run runs/p2/my-wire --wire --task-id ops6k-d4413b455033577a ops6k-ac6e648cc33136d2 --prepare-only

# 直接运行第一步：自动检查或复用运行清单，再训练并保存原生状态。
bash scripts/p2-train.sh --run runs/p2/my-wire --wire --steps 1 --task-id ops6k-d4413b455033577a ops6k-ac6e648cc33136d2

# 从第一步检查点恢复，仅继续到第二步；不是额外再跑两步。
bash scripts/p2-train.sh --run runs/p2/my-wire --wire --steps 2 --resume
```

上述两题由示例的 `--task-id` 显式选定；`--wire` 仅把每步 batch 改为 2，
不会把数据集固定为两题。新运行不传 `--task-id/--limit` 时，从全部 train 开始轻量检查。

当前 Actor/Critic 使用 `flash_attention_2`、`target_modules: all-linear` 和
`lora_dropout: 0.05`。PEFT 覆盖语言与视觉的适用线性层，Critic score 头另外完整训练并保存。
vLLM 使用原生 `enable_tower_connector_lora` 加载视觉适配器；任务仍只有文本，
视觉层不会因挂上 LoRA 就获得有效视觉训练。正式训练按频率分批遍历全部 test，
`--wire` 仅做接线，关闭训练前验证及周期性全量验证。test 已用作训练期间验证，
其指标应标为验证成绩；当前没有另外保留一份未使用的最终测试集。

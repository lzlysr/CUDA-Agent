# P2 单轮同步在线 PPO

训练算法直接使用 `third_party/verl` v0.9.1 的 V1 同步训练器。
项目代码负责 Ops-6K 数据、CUDA 沙箱奖励和 Qwen3.5 Critic 接线。
完整说明见 [P2_REPORT.md](P2_REPORT.md)。

Actor/Critic 的模型加载均继承 verl 原生实现。Critic 经 load_valuehead_model 和
Transformers 的 AutoModelForTokenClassification 自动创建价值头，项目不再单独实现
Qwen Critic 加载函数。Actor 的 LoRA 构造使用 verl 原生实现；模型扩展只保留 Critic 的 LoRA 角色适配和文本输入接线。
更新和标量指标全部使用 verl 原生实现。

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
历史已完成 run 的文件保留不动；当前入口不再续用历史转换数据格式。本轮改变 LoRA 结构和验证协议，需使用新 run，
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

当前 Actor/Critic 使用 `flash_attention_2`、`target_modules: all-linear`，
LoRA dropout 沿用 PEFT 默认值 0，不提供额外配置或自定义 dropout 接线。PEFT 覆盖语言与视觉的适用线性层，Critic score 头另外完整训练并保存。
vLLM 使用原生 `enable_tower_connector_lora` 加载视觉适配器；任务仍只有文本，
视觉层不会因挂上 LoRA 就获得有效视觉训练。正式训练按频率验证从 test 确定性选出的 24 条记录，
`--wire` 仅做接线，关闭训练前验证及周期性全量验证。test 已用作训练期间验证，
其指标应标为验证成绩；当前没有另外保留一份未使用的最终测试集。

当前损失使用 `seq-mean-token-mean`，Actor 的额外 dual-clip 设置为不生效。
训练默认启用 verl 原生 TensorBoard 日志，保存在每个 run 的 `tensorboard/`。
在服务器执行以下命令，然后在 VS Code 转发 6006 端口查看实时曲线：

```bash
.runtime/conda/learner/bin/tensorboard --logdir runs/p2 --host 127.0.0.1 --port 6006
```

实际 PPO loss、梯度范数、学习率、奖励和验证标量来自 verl；不再生成逐 rank 权重变化审计。
本轮改动改变损失及配置身份，应使用新 run；未启动训练验证。

新运行不再生成 `rollouts/*.json` token/logprob 审计副本。每条回答保存在
`samples/<ID>/response.txt`，对应 `result.json` 保存奖励、正确性及反馈/隐藏评测路径。
路径相对于本次 run 根目录。确认源码已完整保存在评测快照后，删除重复的 workspace；
尚无完整快照或发生环境异常时保留现场。P2 评测总结果只保存摘要和 detail_paths，
阶段详情及候选源码继续保存在 kernel/evaluations 下。
参考预检仅落盘采样摘要；worker 日志不再重复打印整份结果 JSON。
检查点保留数量未调整；旧运行文件不迁移、不删除。完整说明见报告第 14 节。

参考测时按官方 profiling.py 执行：eager 与 compile 分别计时，不额外要求二者输出一致。
候选正确性仍以原始 eager 输出为准；不改变五组训练反馈、隐藏验收、容差及奖励门槛。
历史 compile_correct 字段不再生成；基线执行失败、无效耗时仍不能用于评分。

当前 P1 single/P2 的一次交付使用 `<file path="…">` / `</file>` 原始源码文件块，
源码不做 JSON/XML 转义，不带 Markdown 围栏。结束标签缺失或多余可容错，
按结束标签、下一个开始标签或回答结束划分文件；不补源码，不追加生成。
缺开始标签、重复文件和明确闭合后的额外正文仍失败，不接受旧 files JSON。
源码完整性仍需编译与正确性检查。Agent 工具参数继续使用原生协议。
未启用 vLLM 语法约束，verl 的采样、PPO 和结束状态保持不变。
新提示词应使用新 run，旧结果不迁移；完整规则见 [P2_REPORT.md 第 16 节](P2_REPORT.md#16-2026-11-07single-改用原始文件块交付)。
本次 4 项解析检查通过；两条真实隔离生成均耗尽输出预算而交付失败，一条重复生成
kernel，另一条未结束思考。没有补标签、抽取思考区源码或追加生成；不能将此称为
实际格式成功率已提升，完整证据见上述章节。

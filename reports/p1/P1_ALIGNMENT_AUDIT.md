# P1 与 CUDA Agent 原始论文、官方公开代码的全面对照审计

审计日期：2026-10-27，北京时间（Asia/Shanghai）。

同日澄清：本项目直接使用作者发布的最终Ops-6K，它已经包含执行过滤和KernelBench去污染，P1无需重复实现或再次应用这些筛选。本报告已据此修正相关问题分类。TF32公平性及图18冲突按既定本地协议处理，属于已解决事项，不是待对齐缺陷。

**结论先行：当前 P1 是以原生 Qwen3.5-9B 为模型、借鉴 CUDA Agent 的多文件开发约定和 Agent 工作流、自行实现隔离评测器的推理基线系统。它不是原论文训练完成的 CUDA Agent，也没有复现原论文的完整训练或 KernelBench 主实验。** 已经对齐的内容包括官方任务接口、固定绑定基础设施、Agent 中使用官方 SKILL、CUDA 扩展开发、5 例反馈、双基线比较、里程碑奖励公式和按轨迹最佳候选选择等。不能把模型、数据、训练、工具、验证和测时上的差异统一称为“只是工程实现不同”。

本次仅阅读、比对和撰写报告，没有修改实现、启动模型、重跑实验或执行 CPU 回归测试。下文指出的问题是静态代码审计发现；未利用新实验测量问题的发生频率或性能影响。原有测试文件的存在与设计覆盖，也不等于本次测试通过。

## 1. 审计对象、证据边界与判定方法

### 1.1 冻结的三个对象

| 对象 | 本次实际读取的版本 | 用途与边界 |
|---|---|---|
| 用户提供的原始 PDF | `D:/zotero/store/storage/IAQITCQ3/Dai 等 - CUDA Agent Large-Scale Agentic RL for High-Performance CUDA Kernel Generation.pdf`，32 页；arXiv:2602.24286v1，2026-02-27 | 论文设计、训练方法、实验定义和附录示例的第一证据；按 PDF 的 1 起算页码定位 |
| PDF SHA256 | `cdeebc53efb4a25d74b7e9b428e5292fcccd8626aa283c0d8aa9277f516995f9` | 已核对原文件与逐页提取文本所依据文件一致；代码截图另外查看渲染页，不能只靠提取文本 |
| 当前项目 | 服务器 `/home/likun/junjxu/CUDA-Agent`，HEAD `4f9b6f4576148abe06d20215a0c252386c59178b` | 本次实现结论的依据。取样时工作区干净，后续修改不自动包含在本报告中 |
| 官方公开仓库 | `upstream/CUDA-Agent/REVISION`：`473025c8af7878e928525138b5cb327d6bcdf9dd` | 下载相同提交的 GitHub 完整归档，逐文件核对；16 个官方文件无缺失、无内容差异。`REVISION` 是本地额外溯源文件 |
| 数据与环境 | 当前配置、数据审计、数据清单、环境包元信息与硬件信息 | 未重新生成数据，未重新校验全部模型权重字节，未重新验证训练或长上下文运行能力 |

外部一手入口：[原论文](https://arxiv.org/abs/2602.24286)、[官方仓库](https://github.com/BytedTsinghua-SIA/CUDA-Agent)、[官方 Ops-6K 数据集](https://huggingface.co/datasets/BytedTsinghua-SIA/CUDA-Agent-Ops-6K)。**论文 v1、固定提交的公开示例、作者之后的更新是三个不同时间截面的证据，不能拼成一份不存在的“官方完整实现”。** 本报告以用户上传 PDF 和上述固定代码为主，后来的 FlashInfer/Seed 2.1 内容不作为原论文已实现的证据。

### 1.2 如何读代码定位和判定标签

代码位置写作 `src/p1/worker.py:313 timing`，均相对于服务器项目根目录 `/home/likun/junjxu/CUDA-Agent`；行号属于上述冻结版本。核心文件入口为 [run.py](/home/likun/junjxu/CUDA-Agent/src/p1/run.py)、[evaluation.py](/home/likun/junjxu/CUDA-Agent/src/p1/evaluation.py)、[worker.py](/home/likun/junjxu/CUDA-Agent/src/p1/worker.py)。同一功能可能同时有多种判定，例如“工作流对齐、数值协议不同”。

| 标签 | 含义 | 不能据此推出的结论 |
|---|---|---|
| 对齐 | 可在论文或公开代码中定位，当前实现保留对应语义 | 不能推出模型效果相等、私有实现逐行相同 |
| 工程扩展 | 增加调度、记录、隔离或可维护性，且需说明是否影响实验 | 不能把新增验证、改变样本筛选归为不影响指标的改进 |
| 部分对齐 | 目标或接口相同，但实现范围、参数或约束不同 | 不能简称“完全复现” |
| 明确差异 | 有明确来源差别，包括用户主动选择、资源限制、实现缺陷 | 并非所有差异都应撤销，需区分目的 |
| 尚未实现 | 当前 P1 没有执行或实现该阶段 | 安装依赖、写入配置不等于完成实验 |
| 无法确认 | 论文未披露、公开仓库未提供，或本次没有相应实测 | 不能用合理猜测填成“已对齐” |

### 1.3 覆盖范围

本次覆盖 PDF 全部 32 页：正文、参考文献、附录 A–E、图 1–18（以及第 2 页未编号比较图）、表 1–3、式 (1)–(5)。对截图形式的附录代码进行了视觉阅读。官方归档全部 16 文件均核对；三个图片资产也查看了内容。当前 P1 的 20 个 Python/Shell 源文件、3 个项目启动/环境脚本、配置、11 个测试文件及现有 P1 文档纳入结构和证据审查。测试按所约束的行为分析，不把 mock 测试视为真实 GPU 协议证明。

“全面审计”在这里指公开、可见材料的可追踪覆盖，不表示已证明任意模型代码都不可攻击，或已还原未公开的训练器、任务生成器及调度平台。

## 2. 最重要的结论与影响

| 编号 | 结论 | 影响 | 原因/处理性质 |
|---|---|---|---|
| F01 | 没有原论文的单轮 PPO、Actor RFT、Critic Value Pretraining、Agentic PPO | 不能声称复现论文训练或训练收益 | P1 阶段范围；核心训练代码/权重未公开，模型也按用户要求更换 |
| F02 | 原生 Qwen3.5-9B 代替 Seed1.6 230B 总参数/23B 激活参数 MoE | 能力、token 分布和训练推理性质都不同 | 用户明确选择，合理但应写为变体 |
| F03 | 当前做 Ops-6K 自定义 train/test 与 train pilot；论文主评测是 KernelBench 250 题 | 样本难度、分布、指标不可直接与表 1 对比 | 用户明确要求 8:2 和 train pilot；正式 KernelBench 入口尚未完成 |
| F04 | 直接使用已由作者去污染的最终Ops-6K；本地Jaccard只是额外审计 | 无需二次过滤，也不应把未重做作者筛选列为P1缺陷 | 原始数据来源与指纹需保留；额外审计不能冒充同一算法的独立复核 |
| F05 | 最终 `correct` 依赖有效测时并额外经过隐藏验收 | 当前 pass_rate 不是论文单纯“编译+功能正确”的 Pass Rate | `run.py:489` 只选择 `correct && timing_valid`；属于指标语义偏差 |
| F06 | 公开脚本 profiler 累加/平均 GPU 时间；本地 CUDA Event 逐次测量、分轮中位数、前后基线漂移约束 | 合理的稳态评测变体，但数值不能当成原协议复现 | 主动工程设计且改变测量定义 |
| F07 | 工具是自行实现的近似接口，缺 NotebookEdit、BashOutput、KillBash，部分 CLI 也不同 | 不能称 OpenHands 工具完整复刻或官方 utils 原样运行 | 轻量隔离实现；功能差距尚在 |
| F08 | 论文图18与声明约束冲突；本地已固定公平精度条件并拒绝状态污染 | 已解决，以当前本地协议为准 | 保留历史来源说明，不作为待修复项，不要求照搬该案例行为 |
| F09 | 本地确有隔离与反投机设计，但静态头文件、供应商 API 语义、运行期白名单等仍有缺口 | 不能声称彻底解决 reward hacking | 属于可明确定位的审查边界，未做攻击实测 |
| F10 | pilot 续跑主要校验配置；baseline 续跑有更强源码/任务指纹校验 | 修改实现后沿用同一 pilot 目录，可能混入不同协议结果 | 不一致的工程实现，需要统一身份校验 |
| F11 | `case_storage_limit_bytes` 没有实现；若干 revision/selection 字段只是记录或硬编码对应 | 配置不能保证容量上限、实际版本或可切换选择规则 | 死配置/元信息与实际执行之间有差距 |
| F12 | A100/sm80、CUDA 12.9、torch 2.11 代替论文 H20/sm90 环境 | 硬件优化路径和 baseline 性能不同 | 用户现有资源；原论文完整包版本未披露，无法逐项一致 |

## 3. 先建立三层对照，避免对齐目标混淆

### 3.1 论文提出的完整系统

```text
torch/transformers 算子爬取 → LLM 组合合成 → 执行/确定性/非平凡/耗时/去污染过滤 → 6000 训练任务
Seed1.6 base → 单轮 PPO warm-up → Agent 轨迹采样 → Actor RFT + Critic VP → Agentic PPO
训练后模型 + 多文件开发环境 + Agent 工具 + CPU/GPU 解耦池
→ KernelBench L1/L2/L3 共 250 题 → 按轨迹最优候选 → 正确率、Faster Rate、几何平均加速比
```

### 3.2 官方实际公开的内容

```text
SKILL.md + 固定绑定 + 一个 AXPBY 参考/候选例子
+ 编译脚本 + 5 输入验证脚本 + 三模型 profiling 脚本 + 三张说明图
```

没有发布论文完整训练器、原始生成流水线、Agent 控制器、完整工具服务器、128 H20 调度器、训练后的 CUDA Agent 权重或表 1 的完整逐题轨迹。原始 6K 数据另在 Hugging Face 发布。**“upstream 目录下载完整”与“论文所有系统都已开源”完全不同。** 本次归档校验确认前者，不能推出后者。

### 3.3 当前本地 P1

```text
官方 Ops-6K Parquet
→ 源码规范化分组 → 普通确定性 train/test≈8:2 → 从 train 固定抽 12 题 pilot
→ 参考程序预检 + eager/compile 初始测时
→ 每张生成卡一份 Qwen3.5-9B：同题 single 与 agent
→ 候选源码快照/静态检查 → 重新编译 → 5 例反馈验证
→ 候选前后分别测参考 + 测候选 + 运行期回退检查 + 漂移判断
→ 按 compile 加速比选候选 → 124 次隐藏比较 → 逐轨迹结果与汇总
→ 整题完成后删除参考参数快照和 compile-cache
```

另有 `p1-baseline.sh`：只运行参考预检、eager 和 torch.compile，不加载 Qwen、不生成 CUDA 候选，也没有 single/agent 两种轨迹。

## 4. 数据生成、数据接口、划分与去污染

### 4.1 论文数据流水线逐项对照

| 项目 | 论文依据 | 当前实际行为与代码 | 判定与影响 |
|---|---|---|---|
| 种子算子来源 | §3.1，PDF pp.4–5，图 1；torch 与 transformers 的维护算子 | 直接读取官方 `data/raw/ops6k.parquet`，`data.py:87` | 最终数据来源对齐；未复现爬取过程 |
| LLM 顺序组合 | torch 最多组合 5 类；transformers 独立任务 | 使用数据已有 `ops`、`data_source`、`code`；没有合成模型/提示词执行器 | 尚未实现，不能称再生成了 Ops-6K |
| 程序接口 | Appendix A，PDF p.13，图 6 | 要求顶层 `Model/get_inputs/get_init_inputs`，`data.py:99` | 接口对齐；这里只查名称，不在离线准备时证明可执行 |
| 数据规模 | 最终 6000 任务 | 原始 6000 行；规范化后 5542 组 | 原始数据对齐，独立任务数口径不同 |
| 来源分布 | 表 3，PDF p.14 | 204/5026/457/168/74/71 行 | 与表 3 六类比例匹配：3.40/83.77/7.62/2.80/1.23/1.18% |
| 可执行性 | eager 和 compile 都可运行 | 每题运行时预检，`Evaluator.prepare`、`worker.prepare/bench` | 目标对齐；重新筛查会受 A100、包版本、超时、容器配额影响 |
| 无随机性 | 同输入执行确定 | 3 组初始化、每组 2 输入各执行两次，零容差比较，`worker.py:151` | 部分对齐，有限抽样更具体；对微小不确定性也更严格 |
| 有限输出 | 排除不合理任务 | 检查浮点 Tensor 非有限值，后续外部比较也检查 | 工程检查；并不证明所有类型、全部输入都有限 |
| 非平凡、随输入变化 | §3.1，排除近常数/不同输入难区分输出 | 采样每 Tensor 至多 2048 元素，所有样本与一个标量 anchor 比较 | 部分对齐，规则不等价，见下文 |
| eager 耗时 1–100 ms | PDF p.4 的作者数据构建筛选条件 | 使用已过滤数据，在本机测量实际耗时，不再次按此范围剔除 | P1无需重做；A100/软件版本不同，再按范围过滤会改变发布数据集 |
| 不同任务难度支持课程学习 | 图 1/§3.1 的训练动机 | 未做课程采样、分难度训练 | 尚未实现；属于训练阶段 |
| 原始数据锁定 | 公开数据来源 | `data.py:91` 硬编码检查 Parquet SHA256 | 工程改进；比只记录 revision 更可核验 |

**近常数规则的具体问题。** `worker.py:228` 记录 `same_output_samples_across_inputs`，但 `Evaluator.prepare` 没有用它拒绝任务。真正拒绝依据是采样的所有输出元素是否都接近单一 anchor。因此“无论输入如何都返回同一个非均匀向量”可能通过；而在给定容差下接近单一标量的合法结果会被隔离。有限采样也可能漏掉未抽到的变化。应把该过滤描述为“本地采样近常数规则”，不能等同论文完整非平凡性判定。

### 4.2 train/test 与 pilot：遵守用户要求，但不是论文划分

`data.py:31 normalized` 去除 docstring、位置信息，保留变量/属性名称和常量；相同规范化程序归一组。同组不跨 train/test；按来源分层，用 seed=42 与哈希排序确定归属，最大余数分配使 test 的**组数**接近 20%。没有固定旧 pilot、强塞 train、补足 test 或 dev/holdout。

当前审计文件记录：train **4434 组/4803 行**，test **1108 组/1197 行**。5542 组的 8:2 因整数取整为上述数目；原始行比例因组大小不同而不是严格 4800/1200。`tasks.jsonl/train.jsonl/test.jsonl` 保留原始行，baseline 选题与 pilot 会按 `task_id` 去重。因此“6000 行”“5542 个规范化任务”“实际跑多少题”必须区分。

`pilot.json` 是从当前 train 组按固定排序抽取的 **12 题**，不是旧 dev pilot。用户明确要求 test 用于冻结后评估。论文则用全部合成 6K 做训练、用独立 KernelBench 250 题做主评测，没有披露这套 8:2 自划分。当前安排符合用户实验规划，应作为复现变体保留，并在报告中说明。

### 4.3 去污染：继承官方最终数据，不重复筛选

官方[数据卡](https://huggingface.co/datasets/BytedTsinghua-SIA/CUDA-Agent-Ops-6K/blob/main/README.md)明确把执行过滤与去污染列为数据构建的第三阶段，包括排除与KernelBench高度相似的任务。因此，当前使用已发布最终6K进行P1，**不需要再实现一次PythonASTSimilarity并删除任务**。此前将“没有重现作者的数据生成/筛选器”列为P1执行缺陷的表述已更正。

| 维度 | 论文 Appendix A / 图 7（pp.13–14） | 本地 `data.py:48,131–177` |
|---|---|---|
| 被比较代码 | 提取 `Model` 类 | 整个 Python 程序，包括输入生成等 |
| 算法 | PythonASTSimilarity | AST 节点类型/名称连续三元组的集合 Jaccard |
| 聚合 | 每训练题取与 KernelBench 的最大相似度 | 同样取最大值，另记录规范化完全相同项 |
| 处理 | 剔除最大相似度 >0.9 的任务 | 仅生成审计，不按阈值剔除 |
| 当前证据 | 论文直方图是其筛选说明 | `data-audit.json` 有 exact_matches=0；逐题 `kernelbench-similarity.json` 当前未保留 |

上表是在区分作者的数据构建方法与本地额外审计，不是要求P1补做过滤。两个算法的0.9不可直接对应；本地exact_matches=0不是对作者筛选的独立复核。正确安排是继承作者发布的去污染结果，保留版本/原始数据指纹，按用户要求划分train/test；可选Jaccard检查只作补充证据，不改变样本集合。只有未来自行重新合成数据，或增加训练来源时，才需要为新增数据重新设计去污染。

## 5. 模型、精度、上下文与生成参数

### 5.1 模型替换改变了什么

| 项目 | 原论文 | 当前实现 | 判定 |
|---|---|---|---|
| 基座 | Seed1.6，MoE，总 230B/激活 23B，PDF p.8 | `models/Qwen3.5-9B`，Qwen3_5ForConditionalGeneration；P1 只用文本 | 用户主动替换；不是相同基座复现 |
| 训练后权重 | 经完整多阶段 RL 的 CUDA Agent | 原生 Qwen 权重，无本项目训练后的 adapter | 未复现训练成果 |
| 推理 dtype | 文中讨论 BF16/FP16 失配导致 RL 不稳，未给完整推理栈 | `LLM(dtype="bfloat16")`，模型 text config 也标 BF16 | 非量化推理；dtype 数字不能据此证明与论文推理全对齐 |
| 量化/LoRA | 论文未以本项目这种 LoRA 设定定义主结果 | 当前 P1 未传 quantization 或 LoRA 参数，未加载 Actor/Critic adapter | 原生 BF16。P0 曾有适配器试验，不等于 P1 在使用 |
| 推理引擎 | 私有训练/服务实现细节未公开 | vLLM `LLM` 离线本地加载，`run.py:132` | 无法逐项对齐私有服务 |
| 并行 | 论文大规模 Agent/RL 与 GPU 池 | 每生成卡一个完整模型副本，每次一条序列 | 数据级任务并行，不是张量并行或 RL 训练 |
| 视觉分支 | CUDA 编程文本任务 | `limit_mm_per_prompt` 关闭 image/video | 合理文本化使用，不是另一个量化模型 |

BF16 权重标识属于模型数值存储/计算精度，与把 BF16 权重量化为 INT4/INT8/FP8 是不同概念。本次查了当前模型配置与加载逻辑，没有重新对全部权重文件做字节级溯源；原生权重来源的历史校验不能冒充本次重新下载/重哈希。

### 5.2 上下文、单次输出与轮次不能混为一谈

`run.py:105 generation_budget` 按“上下文上限 − 当前完整提示 token 数”计算剩余额度，`generation_tokens=null` 不施加额外 4096 限制。历史 4096 上限已经不在当前默认协议中。

| 参数 | 当前行为 | 与论文关系 |
|---|---|---|
| single 上下文 32768 | 单轮整个输入+输出窗口 | 与论文**单轮 RL 训练窗口**数值相同；不是论文明确披露的单轮评测采样协议 |
| agent 上下文 131072 | 完整累积对话+工具定义+输出共用窗口 | 与 Agentic RL 训练和大上下文描述的数字对齐；不证明 tokenizer/有效预算等价 |
| 每轮输出上限 | 由剩余上下文决定，未另限 4096 | 论文未公开精确单次输出 max_tokens，不能标注完全一致 |
| agent max_turns=200 | 一次生成/解析/工具交互按控制循环计一轮 | 与论文评测上限数字对齐；具体 turn 定义未证明与私有系统相同 |
| 训练 max_turns=150 | 当前不训练 | 未实现；不能用评测 200 替代训练 150 |
| `trajectory_seconds=null` | 无额外整条轨迹墙钟限额 | 私有系统总预算未披露；各工具仍有独立超时 |
| temperature=.7/top_p=.95 | `SamplingParams` 实际使用 | 本地选择，论文未披露可核对数值 |
| thinking | Qwen chat template `enable_thinking=True` | Qwen 适配；不是原论文已知相同模板 |
| seed | 42 + 有效任务索引×1000 + turn | 本地可重复性约定；不是论文采样种子 |

更多引擎参数：`max_num_seqs=1`、`gpu_memory_utilization=.85`、`enforce_eager=True`、关闭 prefix caching（`run.py:149`）。这里 `enforce_eager` 是 vLLM 执行模式，**不是**所测 PyTorch eager 基线；关闭 prefix cache 也不等于证明模型生成不存在任何底层 KV 优化。每轮重新提交完整历史，没有本地显式的跨请求会话 KV 管理。重采样、batching、上下文截断、结束 token 等细节未获得论文私有实现，不能逐项宣称一致。

### 5.3 single 与 agent 不是论文表 2 的直接复刻

single：模型得到统一系统合同、绑定代码和题目，生成一次完整文件 JSON，无工具反馈；控制器在生成后编译/验证/测时/隐藏验收。

agent：同一个原生 Qwen 还得到完整官方 SKILL 和工具 schema，能先分析参考性能，再读写文件、编译、验证、profile、反复修改，直至提交或耗尽预算。

因此当前比较同时改变了**工具反馈、SKILL 提示、上下文上限、交互次数与输出协议**。它可回答“这套单轮方案与这套 Agent 方案在本地如何表现”，不能独立归因于 Agent Loop。论文表 2 的 `w/o Agent Loop` 是有单轮训练背景的消融模型，原论文 §4.1 的各基线模型则都在同一个 Agent Loop 下评测。当前 single/agent 既不是两种 PyTorch baseline，也不是论文训练消融的同义词。

## 6. 开发约定、SKILL 和多文件编译

### 6.1 与官方工作区对照

| 文件/规则 | 官方要求与实现 | 本地实现 | 判定 |
|---|---|---|---|
| `model.py` | 参考程序不可改 | 原题写入只读挂载；输入规格另有可信副本 | 对齐并增强隔离 |
| `binding.cpp` | 固定模块入口 | 逐字复制官方文件，快照时比对，终端只读挂载 | 对齐 |
| `binding_registry.h` | `REGISTER_BINDING` 注册机制 | 逐字复制并固定 | 对齐 |
| `model_new.py` | 同构造参数、可加载参考 state_dict、forward 调扩展 | 固定 state，`strict=True` 加载；允许初始化参数声明 | 对齐核心状态接口 |
| `kernels/*.cu` | 原始 CUDA kernel 与 launcher | 支持、编译，且静态检查强制至少一个 `.cu` | 部分对齐；强制 `.cu` 比官方编译器更严格 |
| `kernels/*.cpp` | vendor 调用与绑定；建议计算/绑定分离 | 编译支持，未强制文件角色语义 | 部分对齐 |
| `.h/.cuh` | 可作为工程辅助 | 快照可复制，但静态 C++ 扫描不覆盖 | 工程支持存在审查缺口 |
| 根目录额外 `.cpp/.cu` | 官方 compile.py 会枚举 | 本地只编固定 binding.cpp 与 kernels 下 cu/cpp | 明确缩小范围，安全/结构约束 |
| 输出扩展模块 | `cuda_extension`，构建后复制 `.so` | 同名模块；独立受控编译目录 | 接口对齐，产物管理不同 |
| 当前 CUDA stream | SKILL 示例和 AXPBY 绑定传当前 stream | 提示词要求，但静态检查不证明每个生成 kernel 都遵守 | 约定对齐，执行约束不完整 |
| 轻量 headers | `torch/types.h`、pybind 头替代 extension.h | 提示词沿用，未强制 | 部分对齐 |
| config 参数调优 | 无重编译切换 kernel 配置 | 可生成同类接口，但缺官方示例的直接 GPU Bash 调优环境 | 表达能力保留，工具流程部分不同 |

**强制 `.cu` 的误拒。** 只需要 cuBLAS GEMM 或 cuDNN Conv 的任务，理论上可完全用 `.cpp` 实现供应商调用；官方 `find_sources` 不要求一定存在 `.cu`。本地 `contract.py:165` 会拒绝这种候选，除非附带 `.cu`。这是可明确定位的额外门槛，不能解释成论文强制规定。

### 6.2 SKILL 内容传递与执行差异

Agent 的初始 user prompt 拼入上游原始 `SKILL.md`；single 没有拼入该文件。优化优先级、fusion/shared memory/coalescing、向量化、warp、occupancy、数值和边界检查、先写性能预期、反复验证、超过最低 5% 后继续优化等内容，Agent 都能看到。**看到指令不等于控制器证明模型做到了。**

| SKILL 细节 | 本地情况 | 结论 |
|---|---|---|
| 禁 Torch 计算、只准 CUDA/cuBLAS GEMM/cuDNN Conv | 系统合同、静态检查、profile 白名单辅助约束 | 原则对齐；供应商用途语义未严格验证 |
| Conv/ConvTranspose 必须 cuDNN | 原始 SKILL 提供，但没有基于原题语义的强制检查 | 提示对齐，执行未保证 |
| `TORCH_CUDA_ARCH_LIST=9.0` | 原文仍在 prompt；实际 worker 强制 8.0 | A100 必需适配，但 prompt 有矛盾，应解释而非照搬 sm90 |
| `sudo python3 -m utils.*` | 本地 prompt 额外说明不用 sudo；容器非 root、禁止提权 | 功能目的保留，原命令不可原样等价执行 |
| 距目标 1.2×内才做参数微调 | 只有文字，没有控制器门禁 | 非强制规则 |
| 调优示例 10 warmup+100 calls | 原始 SKILL 可见，本地正式测时 5 输入预热+3×30 中位数 | 不是相同测时；此例本来也只是调优示例 |
| 最低 5% 优化并持续迭代 | 作为目标；`submit` 不验证已达到 5% | 不等于不正确：论文 Pass Rate 本来也不要求加速，但工作流目标未强制 |
| 完成前 kernels 只留最终版本 | 没有自动确认/清理工作区所有历史 kernel | 未落实强制规则；不能用删除参考缓存替代 |
| 错误诊断、安全 checklist | 可见提示 | 没有 compute-sanitizer、racecheck 或全部 checklist 的自动证明 |

## 7. 工具逐个对齐：哪些相同、哪些只是近似

论文 Appendix B.1（pp.14–15）描述的工具与当前 `tools.py:31`：

| 论文工具 | 本地对应 | 保留内容 | 差异及影响 |
|---|---|---|---|
| Bash | Bash + `BashProcess` | 同一容器、同一持久 Bash 中保留 cd/export/function/文件状态 | 自制会话实现，600s 超时即关闭；输出尾部约 20k 字符；没有完整后台任务管理 |
| Read | Read | 文件阅读；跟踪已读文件 | 限工作区允许路径、扩展名与文件大小，返回格式非官方逐字实现 |
| Write | Write | 可创建文件；覆盖已有文件要求先 Read | 路径/1MB 限制，本地工程安全约束 |
| Edit | Edit | 先读、唯一匹配替换 | 不具备完整 OpenHands 编辑器全部语义 |
| MultiEdit | MultiEdit | 全部编辑在内存中成功后一次写入 | 逻辑原子，未采用临时文件+os.replace 的掉电原子语义 |
| Glob | Glob | 通配查找文件 | pathlib 版本，最多 500 项 |
| Grep | Grep | 正则搜索 | Python re，不是论文提到的 ripgrep；上下文/类型过滤等能力不全，最多 200 匹配、单文件 1MB |
| NotebookEdit | 无 | CUDA 工作区通常不需要 notebook | 明确缺失，不可称完整工具集 |
| BashOutput | 无独立工具 | Bash 当次调用收集输出 | 缺持续查询后台任务输出能力 |
| KillBash | 无独立工具 | 超时/会话关闭可终止 | 缺模型主动管理指定后台会话的工具 |
| 论文未列独立 TodoWrite | TodoWrite | 保存任务列表 | 本地增加；不是核心 CUDA 算法能力 |
| “先分析原始性能”工作流 | profile_reference | 可在没有 model_new.py 时测 eager/compile 和 profile | 合理新增，修补官方 profiling.py 顶层依赖候选的使用不便 |
| 验证/profile 技能脚本 | evaluate + Bash 中 utils RPC | 受控派发 GPU 编译/检查/测时 | 工具 API 是本地实现，官方未公开这套 RPC |
| 结束任务 | submit | 终止模型交互，控制器执行最终选优 | 不强制先调用 profile_reference，不强制最低加速或清理完成 |

工具解析兼容 Qwen XML function-call 与 JSON；`run.py:69` 去掉思考区域后解析，拒绝未闭合 think，只处理首个工具调用。若模型一次返回多个调用，不等价于具备完整多工具并发语义。未知/错误格式会转成反馈供下一轮修复。工具调用计数包含模型发起的交互，不等于底层 Docker 作业数。

200轮是控制器循环的交互上限，不是200次完整“编译+验证+profile”：Read/Write、无效工具格式以及submit都可能消耗轮次。整个pilot在模型启动前已经做过参考预检/初始测时，但初始prompt不直接附上全部profile结果；Agent仍需调用原始分析工具才能得到这些反馈，控制器没有强制第一步调用。`tool_feedback_tokens=null`也不表示原样无限回传：`run.feedback`会筛选失败样本、profile top条目，编译信息会取有限行/尾部文本，Bash与worker日志也有上限。这是上下文工程选择，论文没有披露相同截断规则。

### 7.1 持久 Shell 已实现，但不是完整官方会话系统

`sandbox.py:21 BashProcess` 维护一条 Bash，通过随机结束标记识别命令完成，不为每命令新开 shell。`SandboxSession` 把 workspace 挂为可写，同时对 `model.py/binding.cpp/binding_registry.h/SKILL.md/utils` 加只读覆盖；不能误称 Bash 可以任意改这些基础文件。

CPU 终端无网络、无 GPU、无 Docker socket，以非 root 运行。GPU 作业通过工作区 RPC 请求，只接受固定动作 compile/verify/profile/profile_reference；不接受模型给宿主任意命令或路径。**RPC 只在控制器等待 Bash 命令执行期间被轮询**：把 GPU RPC 丢到后台、随后模型继续生成，不具备论文 BashOutput/KillBash 那种完整后台管理体验。SKILL 中直接在 Bash 里 `import torch` 并 CUDA 调参的例子，在这个无 GPU 终端中也不能照原样执行，需要经 GPU RPC 增加对应流程。

### 7.2 本地 utils 与官方 utils 的准确关系

`run.py` 把 `src/p1/agent_utils` 复制为轨迹的 `utils`。所以模型运行 `python3 -m utils.profiling`，**实际不是直接运行 `upstream/CUDA-Agent/agent_workdir/utils/profiling.py`**。

| 入口 | 官方 | 本地 RPC 客户端 | 判定 |
|---|---|---|---|
| `bash utils/compile.sh` | shell 计墙钟，compile.py 查源文件、清理、load、复制 .so | 发 compile 请求，主控快照/审查/受控编译，成功后发布 .so | 外部用途相近，实现与输出不同 |
| `python3 -m utils.verification` | 同进程参考+候选、5输入、临时禁止 F | 请求 verify，重新编译、隔离双进程、外部比较，默认不测时 | 功能相近；隔离协议不同，也不是同一 F contextmanager |
| `python3 -m utils.profiling` | 默认 iters10，支持 --iters、--single-run | JSON 反馈；新增 --reference-only，未提供相同 --iters/--single-run 参数 | 部分兼容，不能称命令行全部对齐 |
| 无候选时分析参考 | 顶层导入 ModelNew，直接不便使用 | --reference-only 或 profile_reference | 合理工程扩展 |
| 返回码 | 官方脚本执行状态 | 编译成功，或 status=ok 且 correct 不为 False 可返回0 | 本地 timing_valid=False 时仍可能0；退出0不是性能达标证明 |

### 7.3 当前P1是否需要补齐所有工具

当前工具足以支撑P1的同步闭环：分析参考性能、写多文件CUDA、编译、验证、profile、修改和提交。它没有达到论文所列动作空间的完整等价，但不意味着缺少NotebookEdit就不能复现核心开发流程。官方只公开部分技能脚本，没有完整工具服务；本地以同步、受控RPC实现替代。用户此前要求尽量对齐后，持久Shell和原始性能工具已经补上，后台输出/取消能力未继续补齐，属于实现范围上的未完成部分。

优先级应按实验收益判断：NotebookEdit对当前无notebook任务没有必要；ripgrep和官方CLI参数可择需兼容；BashOutput/KillBash在异步长任务、多会话和大规模训练时更有用，当前同步作业已有超时机制，不是P1启动前提。更值得关注的是GPU参数调优通路和充分的性能反馈：CPU终端不能直接运行SKILL的CUDA调优例，现有RPC是否能高效支持同一已编译kernel的config搜索，比仅补工具名称更重要。当前“可运行”也不构成已经实测证明工具裁剪不影响最优性能的结论。

## 8. 编译流程、链接库与硬件适配

官方 `utils/compile.py:11 find_sources` 枚举根目录及 kernels 的 `.cpp/.cu`；清理 build/forced_compile 和根 `.so`；用 `torch.utils.cpp_extension.load`，`-O3`、C++17、`--use_fast_math`；把生成扩展复制到工作区。公开脚本没有训练器或 GPU 池实现。

本地 `Evaluator.snapshot → worker.compile_candidate`：

1. 复制允许源码和头文件，核对固定绑定，记录候选 SHA；拒绝符号链接和越界路径。
2. 静态审查，独立 build 目录重新编译。评分不信任模型自行提供的 `.so`，模型手动 compile 成功也不等于最终通过。
3. 固定 `binding.cpp` 加 kernels 下 `.cu/.cpp`，显式补充 cuDNN/cuBLAS include 与 `libcudnn.so.9/libcublas.so.12` 链接，保留 O3/C++17/fast_math。
4. 使用宿主 CUDA 12.9 Toolkit、kernel Conda 的 torch/libraries；容器提供隔离，不把镜像内 CUDA 环境变量当运行栈证据。
5. 当前 A100 的架构设为 8.0；不是照搬 SKILL 的 9.0。

`compile_seconds=300` 限整个编译容器作业。MAX_JOBS=4、CPU线程4和容器 CPU/内存配额控制资源。超时是本地执行预算，论文没有披露同一数值。头文件虽然被复制和编译引用，却没有被当前 `contract.py` 的 C++ 扫描覆盖，见第 11 节。

## 9. 正确性验证：反馈、隐藏、结构、状态与缓存

### 9.1 官方公开验证到底做了什么

`upstream/CUDA-Agent/agent_workdir/utils/verification.py`：同一进程创建参考和候选，`.eval().cuda()`，把参考 `state_dict` 加到候选，生成输入并克隆，运行 5 组输入。参考先运行，候选在临时 monkeypatch 的 `torch.nn.functional` 阻断上下文中运行；用 atol=rtol=1e-2 比较，最后同步。没有独立隐藏集、宿主比较进程、前后状态监测或公开的缓存投机测试。

该阻断**只针对 F 中公开可调用对象**，不等于禁止 `torch.matmul`、Tensor 运算、cuDNN 全局状态修改或全部 C++ ATen。官方脚本同样不自动证明 SKILL 的全部规则得到执行。

### 9.2 本地验证分成两个层次

| 维度 | 本地代码与具体做法 | 判定 |
|---|---|---|
| 同初始化参数 | `worker.load_model:138` 从可信 state 文件重建，strict=True | 与 state_dict 兼容原则对齐，并显式固定参数 |
| 反馈输入 | init=71001，input_seed=72000…72004 | 数量 5 对齐；固定输入、非每次重采样是本地协议 |
| 隔离参考与候选 | `Evaluator.streamed_checks:580` 两个 Docker 进程 | 工程扩展，不是官方同进程调用 |
| 外部比较 | 宿主 CPU `compare`，不调用候选给出的比较函数 | 有效降低候选篡改 verifier 风险，不是绝对安全证明 |
| 输出流 | `item.pt` 原子发布，宿主比较后确认删除，避免保存全批输出 | 工程扩展，节省磁盘；容量上限字段尚未执行 |
| 类型/shape/dtype | Python 类型、Tensor shape/dtype 检查 | 更明确；tuple 经转换为普通 tuple，不保证 namedtuple 类信息完整保留 |
| 字典/序列 | 序列长度、字典 key 集合完全一致 | 比官方只遍历预期字典键更严格 |
| Tensor 数值 | `torch.testing.assert_close(...atol=.01,rtol=.01,equal_nan=False)` | 核心容差对齐；最大相对误差只是诊断，不是另一个 pass 门槛 |
| 标量 | 类型先比，再用精确相等 | 与官方标量精确比较目标相近 |
| 编译基线正确性 | `worker.bench:487` 对 5 输入逐一比较 compiled 与 eager | 本地新增确认，不能说 compile 正确性完全未查 |
| F 阻断机制 | 本地没有原样使用官方 contextmanager | 改为静态审查+测时阶段 profiler；不是所有验证路径都检查运行期回退 |
| 全局精度状态 | 每次输出后/测时后检查 matmul 与 cuDNN TF32 | 工程扩展；只覆盖两个 flag 的最终状态 |

两容器在**同一张租赁的评测 GPU 上并发做正确性**，不是彼此独占两张卡。这样能让宿主逐例对照，但会增加同时驻留 HBM 的需求；不能声称所有时刻都只有一个评测进程使用 GPU。正式性能测量则不是把两个正确性容器的并发延迟作为结果。

### 9.3 124 次隐藏比较怎样算出来

`hidden_init_seeds=[73001,73002]`，每初始化 20 输入，共 40 基础输入。每个基础输入依次检查普通调用、原输入再次调用、原地址扰动后的调用，得到 40×3=120 次；再做首尾两个样本的逆序调用、在一个模型实例重新加载首尾参数各一次，总计 **124 次**。`evaluation.py:724` 按此公式核对实际输出数，不能只看到 worker 返回 ok 就通过。

隐藏验收不把详细结果返回给模型继续改代码，按反馈 speedup 排序后验证不同源码候选，某候选隐藏失败可检查下一个。这是本地对缓存、状态和泛化的加强，**不是论文公开的“5 随机输入”协议**，会使 acceptance 和 pass_rate 更严格，也增加成本。它还不是未见过任务上的泛化测试：题目、shape、get_inputs 分布不变，增加的是同题输入/参数与调用序列。

隐藏结果虽然不回传模型，控制器仍用它从多个候选中择取可接受结果。因此它也是候选筛选的一部分，不是“完全未被选择过程利用的一次性测试”。这不妨碍把它作为严格验收，但不能额外宣称由此得到无选择偏差的泛化估计。

`worker.perturb:511` 仅对浮点 Tensor 做第 0 维翻转（标量取负）；第 0 维长度为 1、数据对称时翻转可能无效，整数/布尔输入不变。因此“124 次”不是“124 个保证不同的样本”，更不代表已经覆盖所有缓存投机。种子固定，隐藏仅表示不向交互反馈公开结果，不是每次随机刷新、不可推知的挑战集。

### 9.4 错误归属与样本拒绝

搜索中参考流失败会保存证据，重试一次后抛 `InfrastructureError`，而不直接奖励模型 -1；候选编译/执行/数值失败则作为候选失败。这是合理的责任区分（`evaluation.py:676`）。

但**初始预检**把 reference_prepare/reference_bench 失败统一归为 `task_invalid`。其中可能是参考程序真有问题，也可能是容器 24GB 主存、120/300s 超时、显存占用、torch/CUDA 兼容性、输入序列化、compile 失败或严格确定性检查造成。`baseline.rejection_summary` 会细分原因，但仍不应把所有被拒样本解释为“数据坏了”。将这些题从有效题分母排除，会改变样本集；正式报告须同时保留 selected 和 reference_valid 分母及拒绝原因。

## 10. 测时与性能分析：工作流相似，数值协议不相同

### 10.1 三种来源中的测时不能混用

| 项目 | 论文正文/技能示例 | 官方公开 profiling.py | 当前 worker/evaluation |
|---|---|---|---|
| 核心叙述 | 同步、预热、多次重复取平均（p.6） | torch.profiler CUDA 事件总 `device_time` / iters | CUDA Event 逐次间隔，轮内中位数再轮间中位数 |
| 默认重复 | 正文未完整给定；SKILL 调参例 10 warmup+100 calls | warmup5、iters10 | 5 组输入各预热一次，3轮×30次 |
| 输入 | 正文 5 随机例做正确性 | 单组固定输入用于 profiling | 5 反馈输入循环用于稳态测时 |
| 同步位置 | 概括性描述 | warmup后；profiler整批前后同步 | 每个 measured call 前后同步，clone在计时外 |
| 统计单位 | 未公开全部细节 | 微秒，GPU事件累加 | cuda_ms + wall_ms，轮次原始中位数 |
| eager/compile/candidate 顺序 | 声称隔离与可靠反馈 | 同一进程先候选，再 eager，再 compile | 独立容器：reference before → candidate → reference after |
| profiler 是否进入计时 | 不完整披露 | profiler 事件本身就是时间来源 | 另做 profiler，正式时间不从 profiler 取 |
| compile 模式 | torch.compile baseline | `torch.compile(model)` 默认 | `torch.compile(model,mode="default")` |
| 编译成本 | 未声明并入推理时延 | 不在 profiling 数值内，首次 warmup可触发 | `compile_first_call_seconds` 单列，不混入稳态 |
| 漂移门槛 | 未披露相同规则 | 无 | eager/compile before-after 漂移≤3%，失败重试一次 |
| 最终性能重测 | 轨迹最佳结果 | 示例无完整候选选优控制器 | 隐藏验收不再测时；沿用反馈阶段最佳测量 |

CUDA Event 的 stream elapsed 与 profiler 各 GPU 事件时间之和本就不是同一量，特别是在 launch gap、多 stream 或重叠场景。中位数与平均数对异常值的处理也不同。不能把时间单位统一后就声称对齐。当前设计可作为清楚定义的本地稳态协议，但原论文表格不是这种协议的真值。

### 10.2 本地 speedup 的准确计算

`worker.timing:313`：clone 输入和计时前同步在 CUDA Event 外；记录 start event，执行 forward，记录 end event，再同步；wall time 包含调用/提交/结束同步等开销，CUDA elapsed 是对应 stream 的设备时间间隔。每轮 30 个值取中位数，再对 3 轮取中位数。

`Evaluator.evaluate:758` 分别得到前后参考时间和候选时间。若均有限且>0，计算两参考的相对漂移 `abs(after/before-1)`，两个 baseline 都≤.03 才认可。参考分母取 before/after 的中位数；两元素的中位数即二者平均。最终：

```text
speedup_eager   = median(eager_before, eager_after) / candidate_cuda_ms
speedup_compile = median(compile_before, compile_after) / candidate_cuda_ms
```

候选耗时不包含输入克隆、首次扩展编译和隐藏验收，但包含 forward 内工作。参数预处理是否可以移到构造函数、一次性缓存如何计摊，是需要另行声明的优化边界；只看快照和计时不能证明这些成本在所有使用情境都合理。

重复测量并从轨迹中取最好结果，仍可能存在选择偏差/偶然最好值；漂移约束降低噪声，不等于消除“挑最佳测量”的偏差。论文也按轨迹最佳结果选取，当前因此在选优方向上对齐，但没有独立最终性能确认。

### 10.3 TF32：本地统一设置与论文示例污染

`worker.settings:36` 在每个 worker 把 matmul/cudnn `allow_tf32` 都设为 True，并将 cudnn.benchmark=False、CPU threads=4。当前代码没有设置 `torch.set_float32_matmul_precision`，不能把未写的设置补进环境说明。不同 torch 版本对其他精度控制、kernel 选择的影响仍需单独辨别。

官方 `profiling.py:110` 先执行候选，随后在**同一 Python 进程**执行 eager/compile；它没有在三者之间恢复 TF32 等全局设置。用户截图批评的“候选打开 TF32、结束关闭，影响后跑 baseline”，因此具有公开脚本层面的可行机制。

更直接的证据在原始 PDF **p.31 图 18**：forward 开头设置两个 allow_tf32=True；forward 内用 `torch.rsqrt` 及 Tensor 乘除加折叠 BN；结尾把两个 allow_tf32=False。它既与禁用 PyTorch 计算回退的 SKILL 原则冲突，也具备污染后续 baseline 的状态写入。p.29 图 15 还使用允许 Tensor Core 转换的 cuDNN math type。**可以确认代码行为和协议风险，不能仅据此量化作者全部主实验被影响的比例或判定主观动机。**

本地会静态拒绝 `torch.backends`，参考与候选分容器，且检查 TF32 最终状态，所以不会原样接纳图 18。这是应当保留的公平性方向，不能为了“对齐案例”把污染方式放行。不过只监测两个 flag 的调用后状态，不会检测所有中间状态改后恢复、所有 cuDNN handle 设置、全局 backend/精度选项或 Python monkeypatch。

## 11. 静态检查、运行期回退与反 reward hacking 的边界

通俗地说，这一节检查的是“模型是否按规则完成计算”，与“输出数值是否正确”是两件事。例如一个结果数值正确，也可能偷偷调用被禁止的PyTorch计算。当前5组反馈都做数值比较，但计算来源的profiler检查只对第一组输入运行一次；`.cu/.cpp`被扫描，能够被它们include的`.h/.cuh`未同等扫描；写在配置里的8GB限制尚未真正执行。这些是自动验收覆盖或资源保护的缺口，不是已经证实某条现有结果作弊，也不要求禁止所有必要的内存分配/复制操作。

### 11.1 真正已经实现的约束

`contract.py:32 audit_contract` 对 Python AST 的 import、危险函数、forward 及其可达辅助函数做检查，允许无害类型声明和参数初始化；注释去除后扫描 `.cu/.cpp`，区分 torch/at 类型与计算调用，对未解析符号给 warning，减少把类型名、注释、无用 import 一概判作弊的误拒。`snapshot` 只接受工作区源码，固定绑定逐字核对；容器只读挂载脚本和参考，禁止网络、提权和任意宿主执行。

`worker.runtime_fallbacks:385` 在候选正式测时后另外 profile **第一组反馈输入的一次调用**，把未列入白名单的 `aten::` 事件列为回退；`Evaluator.evaluate` 据此拒绝。这比仅禁止 F 更广，但仍是一次执行路径的事件审计。

### 11.2 runtime_fallbacks 白名单的真实含义

| 类别 | 当前允许的事件名（去除 aten:: 前缀） | 与官方原则关系 |
|---|---|---|
| 空张量/内存 | empty、empty_like、empty_strided、empty_permuted、new_empty | 张量创建/分配与 SKILL 方向一致 |
| 填充值创建 | zeros、zeros_like、zero_、ones、ones_like、fill_ | 支持创建输出/初始化，但其中会发生实际写内存；不都是“纯元数据” |
| 视图/别名 | view、_unsafe_view、reshape、as_strided、detach、alias | 常见包装需求；reshape 在不同条件下可能引发复制 |
| 数据移动/转换 | to、_to_copy、copy_、resolve_conj、resolve_neg | 有些能复制、转换或计算；原始论文没有公开同一白名单，不能称逐项对齐 |

这份白名单是本地工程选择。它放行必要包装的同时，放宽了部分实际处理，因此要按调用语境审查，不能说“都不计算，所以与论文完全一致”。原始 SKILL 的“only tensor creation/custom ops”与公开 verification 的“只封 F”本来也不是同一严格程度。

### 11.3 未封闭的审查边界与误拒

| 问题 | 代码证据 | 后果/边界 |
|---|---|---|
| `.h/.cuh` 没进入 C++ 静态扫描 | 快照允许头文件，`contract.py:162` 只扫 cu/cpp | 可被编译器 include 的内容不在同一静态规则覆盖；不能称源码全扫描 |
| Python Tensor 二元运算不由 Call 白名单直接拦截 | forward 遍历以 ast.Call 为主 | 算术可能靠运行期 profiler 才发现；未执行分支可能漏检 |
| 元数据方法按属性名放行 | `contract.py:155` | 是启发式检查，不是类型证明；reshape/view 的语义依赖对象与布局 |
| 供应商 API 用途没做语义验证 | 未检查 cuBLAS 只 GEMM、cuDNN 只 Conv，以及 Conv 必须 cuDNN | 提示合同有规则，自动验收不保证全部遵守 |
| 运行期只 profile 第一输入一次 | `worker.py:469` | 数据依赖分支、仅隐藏路径上的计算回退可能不被捕捉 |
| verify-only/hidden 提前返回 | `evaluation.py:753`，final 或 measure=False 不走测时/profile | 这些阶段本身不提供完整运行期回退审计；候选入选时虽曾通过反馈 profile，覆盖仍有限 |
| 只检查 `aten::` 事件 | `worker.py:440` | 非该命名空间、C++直接行为和 profiler 盲区不因此自动合法 |
| TF32 只查最终两个 flag | `worker.flags`、stream/bench | 修改后恢复或其他全局状态不一定被发现 |
| 写出与 build 目录可写 | container 对 /outputs、/build 为 rw | 必须保留给编译/通信；隔离降低权限，不构成执行任意候选时的形式化安全证明 |
| host 输出体积无执行上限 | `case_storage_limit_bytes` 无引用 | 8GB 配置并没有阻止超大输出文件/host torch.load 资源消耗 |
| 至少一份 `.cu` | `contract.py:165` | 可能误拒纯合法 vendor `.cpp` 候选 |
| 固定输入分布 | case_list/perturb | 不能证明所有形状、dtype、别名、整数输入及缓存场景正确 |

以上并不表示已实测构造了成功攻击，而是说明当前检查能证明什么、不能证明什么。静态允许、profile 没发现回退、5/124 例通过是不同层次证据，应分别保存。对有 warning 的通过候选仍需要读源码，尤其是头文件、cuBLAS/cuDNN 调用、状态缓存与预处理逻辑。

## 12. 奖励、候选选择、Pass Rate 与汇总指标

### 12.1 式 (1) 里程碑奖励确实对齐

`evaluation.py:138 reward` 实现：

```text
不正确                                     → -1
正确且 t < .95*t_eager 且 t < .95*t_compile →  3
正确且 t < .95*t_eager                     →  2
其余正确                                   →  1
```

与论文 PDF p.5 式 (1) 的分支一致。只快于 compile 但不快于 eager 仍为1，不能擅自改成2。这里“5%”是**耗时降低 5%**（speedup >1/.95≈1.05263），不是恰好 speedup≥1.05。式 (1) 用严格 `<`，SKILL 末尾示例写 `≤.95×`，边界措辞本身不一致；本地函数按公式严格比较。

额外规则：无效计时 reward=None；仅验证或隐藏正确时记 reward=1，不重新计算性能等级。当前 reward 是搜索反馈/记录，不被送入 PPO 更新。因此“奖励函数已写”不等于“强化学习已复现”。

### 12.2 轨迹最佳候选：方向对齐，验收额外变严

论文 §4.1：从轨迹里挑相对 torch.compile speedup 最大的解。当前 `select_candidate:489` 对反馈正确且 timing_valid 的解按 speedup_compile 降序，按源码 hash 去重，隐藏验收失败就查下一份。不是简单采用最后一次文件，也不是把每次尝试当成独立题；最终 workspace 还会额外评估一次。

但 `run.py:470` 的最终 `correct=bool(hidden and hidden.correct)` 与入选条件一起形成：

```text
本地最终 correct = 至少一个候选满足“反馈正确 + 有效测时”，且该候选通过额外隐藏检查
论文 Pass         = 编译成功 + 功能正确（论文评测检查定义下）
```

如果功能正确却因为噪声/漂移没有有效计时，本地最终 correct 可能为 False；verify-only 成功也不够。这会把测时质量问题混入正确率。建议将 functional_pass、policy_pass、timing_valid、hidden_pass 和最后用于 performance 的 accepted 分开，保留现行严格验收，但不要给其复合通过率直接冠以“论文 Pass Rate”。本次只提出调整，不改代码或旧结果。

### 12.3 汇总层面对照

| 指标 | 论文定义/表 1 | 当前 `finish_summary:567` / progress | 结论 |
|---|---|---|---|
| Pass Rate | 全评测任务中编译+功能通过比例 | reference_valid 和 selected 两种分母，但分子是复合 correct | 分母透明是优点；分子语义未对齐 |
| Faster Rate eager/compile | 正确且快于 baseline 的任务百分比 | `faster_eager/faster_compile` 保存数量 | 缺明确百分率字段与分母；不能把 count 当 rate |
| Faster 判断 | >1，即比基线快 | timed 中 speedup>1 | 阈值对齐，和 reward 的 5% 不要混淆 |
| Speed-up | 正确解的几何均值 | correct 且 timing_valid 的 speedup 几何均值 | 计算式相近，纳入集合受额外门槛影响 |
| Overall | 表 1 按 L1=100/L2=100/L3=50 加权 | 当前所有有效样本直接 pooled GM，无 level 分层 | 当前不是 KernelBench，不能直接复用为表 1 overall |
| 中途 pass_rate | 论文是完整评测集 | 进度条=已完成 correct / 已完成轨迹 | 中途分母不同，受任务完成顺序影响 |
| 中途 speedup | 无对应在线定义 | 每模式已完成且有效结果 GM，valid=N | 显示工程扩展，不是单题值；RESULT 行另含单题值 |
| 成本 | 未给一套可完全比对的逐项预算表 | generated_tokens、prefill总和、工具数、搜索评估数、容器/墙钟/生成卡保留时间 | 有价值的扩展，但统计口径需要读说明 |

Overall 几何均值需要特别谨慎：按各级全部题数给级内 GM 加权（在 log 空间）和把各级**正确样本**全部合并取 GM，在各级 pass 数量不同的情况下不等价。论文表头只说明按题数加权；本地尚无按级报表，不能暗中选择一种后宣称复现。失败不补零参与 log/GM 是合理的，但必须同时报告正确数量及失败覆盖。

### 12.4 baseline 的 MEAN 不属于论文 Speed-up 指标

`baseline.py:70/140` 的 MEAN 是用户要求的各列**算术均值**，只纳入成功行里的有限值，缺失/拒绝不补零，无有效值记 null。`compile_over_eager_speedup=eager_cuda_ms/compile_cuda_ms` 的均值是**逐题比值的算术平均**，不是两列平均耗时相除，更不是 CUDA 候选相对 compile 的几何均值。

REPORT.md/CSV/JSON 的首数据行包含 MEAN，是报告工程功能；不能拿其数值对照论文表 1 的 GM。JSON 根摘要只保留标量和详情路径是避免重复的工程改进，不影响底层测时定义。

## 13. CPU/GPU 解耦、Docker、Conda、硬件和多卡

### 13.1 论文与本地资源拓扑

论文 PDF p.8：CPU Docker terminal 处理含编译在内的 CPU 任务，预设技能脚本派发验证/profile 到 **128 张 H20** 的隔离 GPU 池，强调独占和稳定 HBM。

本地有两个隔离面：CPU Bash 容器处理文件和命令；RPC 到宿主主控后，GPU worker 容器执行受控作业。**编译也经 `ParallelEvaluator` 租评测卡并在带 GPU 的 worker 容器里完成**，所以不是完全复刻论文的 CPU-only 编译与 GPU-only 验证分工。A100 算力/带宽/指令集不同，也不是把 H20 数量缩小就保持相同实验。

当前配置：generation_gpu=`[4,5]`、evaluation_gpu=`[6,7]`。按物理编号选择，UUID 仅作日志溯源，不用来调用 GPU。生成卡每卡一份模型，整题按队列分配、同题 single 后 agent；评测卡租约按整个 prepare/compile/evaluate 操作持有，嵌套调用沿用同卡。多个工具调用或不同候选可能在不同评测卡完成，虽是同型号，也不能假定频率/热状态完全相同。

锁能防止**遵守本项目锁的进程**共享评测资源，不会阻止其他用户绕过锁占 GPU。pilot 的启动检查约100MiB占用阈值，独立 baseline 的 `gpu_available` 使用 60000MiB 阈值，后者并不意味着卡为空；两入口的占用保护标准不一致。容器 CPU24GB 主存限制不是 GPU HBM 配额。不能把 Docker 与文件锁解释为硬件级绝对独占。

### 13.2 Docker 镜像到底提供什么

当前镜像固定为 `sha256:d58d23b33e24c351dffc6b1bda93dcd7925e3832e2770da6b65e10abfdd05fa0`，本地已有标签 `rpent-vla-env:libero-pro`。它提供 Linux 用户空间、shell、基础系统库/工具和容器隔离起点，不是官方发布的 CUDA-Agent 训练镜像。不是把镜像当代码目录复制进项目；Docker 管理自己的镜像存储。

实际运行 Python 指向只读挂载的 `.runtime/conda/kernel/bin/python`；Toolkit 来自只读挂载 `/usr/local/cuda-12.9`，GPU驱动来自宿主。learner Conda 用于控制/生成，kernel Conda 用于 worker；CPU终端也挂载 kernel 环境。二者与镜像形成组合运行栈。

所以“利用已有镜像”可以减少重复安装，但仍依赖镜像的 glibc/系统动态库、可用编译工具、路径、权限和宿主挂载兼容性。固定 image ID 比可变 tag 稳定；若换服务器而缺该 image，`--pull=never` 会直接失败，配置也不能自动重建该镜像。完整可移植复现尚缺从 Dockerfile/锁定基础镜像重建该镜像及系统包的配方。不能凭镜像里的 CUDA_VERSION 判断运行的是哪个 CUDA。

### 13.3 本次读取到的环境，不等于原论文披露环境

| 部件 | 当前值 | 原论文可对齐程度 |
|---|---|---|
| GPU | A100-SXM4-80GB，8卡，driver575.57.08 | 原论文H20；硬件不同 |
| learner torch | 2.11.0+cu129 | 原论文没有完整版本锁可核对 |
| kernel torch | 2.11.0+cu129 | 本地两环境核心torch版本一致 |
| vLLM | 0.21.0+cu129 | 用于本地生成，原论文引擎细节未公开 |
| transformers/peft | 5.9.0 / 0.21.2 | P1原生加载，不因安装peft就使用adapter |
| kernel Triton/NumPy | 3.6.0 / 2.3.5 | baseline compile可依赖Triton；候选合同禁止Triton是另一层规则 |
| Toolkit | CUDA12.9，nvcc12.9.86 | 本地兼容与A100选择；非原论文相同版本证据 |
| kernel cuDNN/cuBLAS | 9.17.1.4 / 12.9.1.4 | 本地链接的库版本，影响Conv/GEMM baseline和候选 |
| verl | v0.9.1，项目third_party源码可编辑安装 | 当前P1没有调用其训练器；存在不代表训练已完成 |

软件版本来自只读包元信息，未在本次导入 CUDA 执行模型。P0 的既有验通结论不能替代当前代码在所有任务/长上下文/训练条件下的验证。

## 14. 训练与算法：逐阶段、逐公式对照

| 论文阶段/公式 | 原始内容 | 当前 P1 | 判定与原因 |
|---|---|---|---|
| 数据合成 | 图1/§3.1，LLM构造6K | 消费已发布数据 | 未复现生成器；公开代码没有完整流水线 |
| 单轮PPO warm-up | 图3，先提高CUDA基础能力，缓解长Agent训练失配 | raw Qwen single只做推理 | 未实现训练，不可称warm-up完成 |
| Agent轨迹采样 | 经过单轮训练模型采样，供后续初始化 | 可产生本地原生Qwen轨迹 | 功能素材相近，但模型分布、筛选和训练用途不同 |
| Actor RFT | 正奖励+模式过滤轨迹，式(2)负对数似然 | 没有轨迹训练集导出/筛选器与更新 | 未实现 |
| Critic VP | terminal token奖励、GAE、value target与MSE，式(3)(4) | 无value head、无critic推理/训练 | 未实现；P0 smoke不算本阶段 |
| Agentic PPO | 式(5)，γ=1，λ=.95，clip下.2/上.28 | 无policy update/old policy logprob/advantage | 未实现 |
| Actor/Critic LR | 3e-6/6e-6 | P1无对应优化器 | 未实现，不应添加伪“对齐参数” |
| global batch/mini-batch | 都1024 | 每生成卡seq1 | 推理批处理与PPO batch不是同概念 |
| rollout turns | 训练150、评测200 | 只有评测max200 | 评测数字部分对齐，训练缺失 |
| 训练步数 | 150 | 0 | 明确尚未做 |
| robust milestone reward | 式(1)，对比 raw speedup reward | 公式实现，未做两reward的训练消融 | 计算规则对齐，训练效果未验证 |
| 训练稳定性/importance ratio | PDF p.6，BF16/FP16不一致、长轨迹崩溃 | 未建立训练/rollout logprob比对链路 | 不能因两Conda同版本就宣称问题已解决 |
| RFT/VP崩溃曲线 | 图4/图5，entropy、clipping、explained variation等 | 无训练曲线或同指标日志 | 未复现 |

论文完整训练链是其主要贡献。当前 P1 暂不实现是阶段安排，合理；但对外写作应称“推理/环境基线复现”，不能用“只差调参”描述到完整论文的距离。更换为 Qwen 后，原论文的学习率、batch、context 等最多是待验证起点，不能保证相同稳定性或数据效率。

## 15. 实验、消融、案例与结论边界

| 论文内容 | 当前状态 | 可得与不可得结论 |
|---|---|---|
| 表1 KernelBench250，L1/L2/L3=100/100/50 | 代码核验文件数量、做相似性审计；当前P1选题入口还是Ops train/test | 不能称KernelBench主结果已复现 |
| Seed1.6 base、GLM4.6、KimiK2、Gemini3Pro、Opus4.5同Agent Loop | 当前只有Qwen单模型 | 没有多模型公平比较 |
| w/o Agent Loop | 本地single未做原训练背景 | 不能对照表2那行评价训练消融 |
| w/o Robust Reward | 只有里程碑函数，无rawreward训练分支 | 没有消融 |
| w/o RFT、w/o VP | 没有这些训练阶段 | 没有消融；未训练不等于跑了消融 |
| 图4/5稳定训练分析 | 无对应训练 | 无法证明150步稳定收益 |
| Appendix D2，固定4096² diagonal matmul，73.31×compile | 可表达类似CUDA开发；未在本次运行该题 | 算法化简O(N²M)→O(NM)是案例解释，不是本地实测结果 |
| Appendix D3，1024×8192输入/8192²权重，24.04× | 未复现该案例测量 | 把weight先求和再dot、向量化和shared reduction可作为实现思路；不能照搬速度 |
| Appendix D4，ResNet BasicBlock，3.59× | 未复现；其Torch计算/TF32改写将被当前合同拒绝 | 应先修正公平性再独立做案例，不能为通过而放开污染 |
| 与TVM比较 | 论文Appendix E自己未做 | 当前同样没做，不能宣传超越所有编译器 |
| 工业可用性 | 论文有成本与泛化局限 | 本地更小模型/不同数据且无完整测试 | 能跑闭环不等于已达工业专家水平 |

论文主结果（表1）是作者报告值：整体 Pass98.8%、Faster eager98.4%/compile96.8%、GM2.60×/2.11×。本次没有重测这些数值，也不使用它们给本地结果背书。

## 16. 落盘、续跑、多卡进度和缓存清理

这些主要是本地工程扩展，官方公开示例没有对应完整控制器。它们可以提高可用性，但要说明恢复与统计的边界。

### 16.1 当前结果文件的职责

| 路径（相对于某次 run） | 内容 | 读取方式/注意 |
|---|---|---|
| `manifest.json` | 配置、题目总数与ID、代码/硬件快照、状态；baseline有协议/任务指纹 | 是运行清单，不是每候选数值详情 |
| `worker-config.json` | 一份共享公共配置 | 实际评测GPU由宿主Evaluator单独传入Docker；不需要改成逐卡文件名 |
| `tasks.json` | pilot固定题单 | 多卡/续跑沿用，不按临时队列完成顺序改变题意 |
| `references/<id>/model.py` | 原始参考程序 | 即使清理参数缓存仍保留 |
| `references/<id>/cases/prepare_result.json` | 确定性/近常数等准备详情、controller_seconds | 不再额外生成重复prepare.json |
| `references/<id>/initial-bench/bench_result.json` | 初始eager/compile/profile、首调成本、controller_seconds | 不再额外生成重复baseline.json |
| `references/<id>/preflight.json` | 状态和上述详情相对路径 | 索引，不复制完整详情 |
| 后续reference-profile目录 | 再次profile_reference的独立结果 | 不是无意义重复：对应不同时间的测量；初始预检缓存不代表随后不重测 |
| `trajectories/<id>/<single或agent>/generation-*.json` | 每轮原文、token、生成耗时/结束原因 | 把生成文本视为待审查数据 |
| 同轨迹 `tool-*.json/messages.json/workspace/` | 工具参数/反馈、完整消息、开发文件 | workspace最后内容不一定是被选中的最佳候选 |
| `evaluations/<序号>/candidate/` | 每次候选只读快照与源码hash | 追溯实际选优候选应看evaluation_id和candidate_hash |
| 同evaluation编译/流/测时结果与日志 | 编译错误、输出检查、baseline before/after、timing、回退、hidden | 是判断“为何失败/加速是否有效”的底层证据 |
| `trajectories/.../result.json` | 完整最终选优、隐藏验收与成本 | 轨迹完成标志；不等于每一步都成功 |
| 根`results.json` | 各题/模式标量摘要和detail_path | baseline有MEAN首行；pilot不是相同表格式 |
| 根`summary.json` | pilot每模式计数、GM、成本及分母 | 不要把faster数量看成百分率 |
| 根`REPORT.md/results.csv` | 纯baseline的人读/表格汇总 | 不是论文CUDA模型主评测报告 |
| `interrupted/` 等归档 | 残缺轨迹/参考目录 | 留证据，不自动纳入正式完成结果 |

`run.py/baseline.py/evaluation.py` 中的 `dump` 是 `atomic_json` 别名，使用单文件原子写；临时文件刷盘后替换，可避免半截JSON。`data.py:24 dump` 仍是直接write_text，数据准备和引用它的状态收尾并非都具有同样保证。多个汇总文件**不是跨文件事务**，崩溃在两次替换之间可能短时不一致，仍需从阶段详情重建。worker的阶段JSON在容器收尾前还会由controller添加成本；不能把文件刚出现就视为完整终态。

### 16.2 baseline 与 pilot 的恢复可靠性并不相同

| 方面 | 独立baseline | p1-run pilot |
|---|---|---|
| 题单固定 | manifest.task_ids、当前数据匹配、任务指纹 | run/tasks.json固定；有基本ID检查 |
| 配置允许变化 | 主要允许调度卡号，不允许评测定义变化 | validate_resume剔除generation/evaluation_gpu后比较配置 |
| 源码身份 | worker/evaluation/contract指纹；少量已审查兼容映射 | manifest记录源码SHA，但resume没有据此强制比较 |
| 完成识别 | 阶段JSON+controller_seconds+路径/状态重建，拒绝也可复用 | 并行路径校验轨迹task_id/mode/correct存在；参考preflight主要按文件存在读取 |
| 残缺归档 | 归档参考残缺目录再重做 | 并行会归档不完整轨迹；串行恢复路径较弱，行为不完全一致 |
| 摘要损坏 | 可从正式阶段重建 | 依赖各轨迹详情，但无同等严格协议身份校验 |
| 中途停止 | 停止派新题，等待在途题完成并落盘 | 停止派发/下一轮，保留中间文件；长生成/作业结束前未必即时返回 |

`--resume` 不是任意改代码后继续混跑的许可。当前 pilot 即使源码变化但配置没变，也可能接受旧轨迹；随后新 manifest又记录当前源码，追溯可能模糊。要对齐可靠实验身份，应在恢复前比较任务、源码、模型/数据实物指纹，并保留原manifest与各次invocation，而不只在新清单重写当前状态。

### 16.3 多卡与进度条

`helper/pilot_parallel.py` 负责卡号规范化、角色不重叠、GPU评测租约、全局evaluation序号、spawn模型子进程、消息事件、异常收尾。生成以**整题**并行，不把同一Agent会话跨多个模型进程乱序推进；父进程唯一写根摘要，按固定题单顺序展示。独立baseline以线程派发整题，每卡一次一个任务，同样由父控制流程写汇总。

`helper/progress.py` 只显示参考预检和single/agent完成数，不控制评分。进度条的 speedup_eager/compile 是已完成、正确且测时有效的**几何均值**；RESULT日志里是本题值。pass_rate 的分母是当前已完成轨迹数，中途高低可能因先完成容易题而偏移；参考被拒题不作为该进度条的已完成模型题。非TTY保留逐题文本，不强制动态刷新。

### 16.4 自动清理的收益与限制

`helper/baseline_state.py:261 cleanup_reference_artifacts` 仅删除已完成题的 `cases/input-spec/` 与 `compile-cache/`；先核对路径在本run、本题内，拒绝符号链接/逃逸。baseline先保存该题汇总再清理，pilot待同题single和agent均完成、最终验收落盘后再清理；prepare-only保留后续运行需要的参数。中断的未完题不按完成题清理。

这符合用户节省空间的要求，不影响已经保存的结论/日志，但删掉了初始化参数快照和编译缓存；精确复查可能需要重新准备/编译，不能直接把“续跑跳过已完成题”误解为“已完成题所有中间产物都在”。它也**不是 SKILL 要求的 kernels/ 只留最终实现**，不会自动清除所有evaluation构建、候选快照或轨迹工作区历史代码。

成本方面，已完成轨迹的token、工具与总时长会求和；`*_this_invocation` 只统计当前控制器执行段。中断但未完成轨迹的生成成本保存在归档日志，却不会自动全并入这些已完成轨迹总和。多卡reserved_seconds是各模型副本保留时间之和，不能当整次run墙钟或GPU核函数实际忙碌时间。应另列中断成本及各次调用，不能报告为“全实验总成本已经完整”。

## 17. 当前 p1-pilot.json 每一字段的实际用途

以下对应冻结配置，不能把默认脚本示例中的GPU2/3或P0 GPU4当成当前配置。带“元信息/未使用”的字段，写在JSON中不会自动得到其名称暗示的保证。

| 字段 | 当前值 | 实际消费位置/作用 | 与原论文关系或问题 |
|---|---|---|---|
| seed | 42 | data分组/pilot排序、模型引擎、逐轮种子 | 本地可重复性约定，论文未披露相同seed |
| pilot_tasks | 12 | data.py:130准备pilot时使用 | 不是运行时自动截断现有pilot；本地小样本工程 |
| model | models/Qwen3.5-9B | run.py:147本地加载 | 用户替换基座 |
| model_revision | c202236235762e1c871ad0ccb60c8ee5ba337b9a | 当前P1无读取分支，仅随config保存 | 不自动校验模型目录/权重版本 |
| dataset_revision | 44a734c78c947bfcba5189cbfd13f57a6d29a698 | data.py:164写审计 | 元信息；原Parquet另有硬编码SHA校验 |
| kernelbench_revision | 423217d9fda91e0c2d67e4a43bf62f96f6d104f1 | 当前P1无执行校验 | 元信息，不等于实际上游目录字节匹配 |
| generation_gpu | [4,5] | gpu_roles、模型副本与调度 | 多卡本地扩展，按编号 |
| evaluation_gpu | [6,7] | 评测池与具体Evaluator.gpu | 多卡本地扩展；独立baseline新运行有自己的CLI/default规则 |
| docker_image | sha256:d58d23b33e24c351dffc6b1bda93dcd7925e3832e2770da6b65e10abfdd05fa0 | CPU终端与worker容器 | 固定本地已有镜像，不是论文官方镜像 |
| context_tokens | 131072 | agent完整窗口、LLM max_model_len | 论文Agent窗口数字对齐，不等于其完整采样协议 |
| generation_tokens | null | generation_budget | 不额外限4096，使用窗口余量；论文单次上限未披露 |
| tool_feedback_tokens | null | run.py:431不设token截断 | 仍有结构化反馈/日志尾部截断，不是无限原始反馈 |
| temperature | 0.7 | SamplingParams | 本地选择，无法确认原论文相同 |
| top_p | 0.95 | SamplingParams | 同上 |
| max_turns | 200 | Agent循环上限 | 论文评测数字对齐；不是训练150 |
| trajectory_seconds | null | 轨迹时间限制关闭 | 不代表工具无超时 |
| compile_seconds | 300 | 编译容器作业超时 | 本地预算，论文未给同值 |
| evaluation_seconds | 120 | 默认worker/stream控制超时 | 本地预算，可能影响大任务拒绝 |
| atol | 0.01 | 外部比较、compile检查、近常数 | 正确性容差对齐；近常数使用是本地扩展 |
| rtol | 0.01 | 同上 | 同上 |
| timing_iterations | 30 | worker每轮测量次数 | 不同于公开脚本默认10 |
| timing_rounds | 3 | 轮间中位数 | 论文未披露此规则 |
| baseline_drift_limit | 0.03 | 前后参考漂移门槛 | 本地扩展，改变有效测时集合 |
| feedback_cases | 5 | case_list/预期输出数 | 论文数量对齐，种子协议不同 |
| hidden_cases_per_init | 20 | 隐藏基础输入数 | 本地额外协议 |
| hidden_init_seeds | [73001,73002] | 固定参数快照、隐藏输入与重载 | 本地额外协议 |
| case_storage_limit_bytes | 8589934592 | 当前src无任何实际引用 | **未执行的8GB上限，不能作为保护或容量承诺** |
| protocol_version | p1-paper-aligned-v3 | 记录到manifest.protocol | 标签不保证语义对齐；本报告已列明确差异 |
| train_ratio | 0.8 | data.py分组划分 | 用户要求，自划分非论文 |
| pilot_split | train | data.py仅允许train | 用户要求，避免使用test调流程 |
| single_context_tokens | 32768 | single窗口 | 只与论文单轮训练窗口数字相同 |
| shell_seconds | 600 | Bash调用超时 | 本地预算 |
| candidate_selection | best_correct_speedup_compile | 当前无按该字段分派的读取逻辑 | 代码硬编码相同选择规则；改JSON不会切换策略 |

配置还没有穷尽执行协议。TF32=True、cudnn.benchmark=False、5个输入预热、sm80、fast_math、CPU4、容器内存24g、tmpfs8g、vLLM seq1/.85/enforce_eager等在代码或脚本里固定。冻结实验必须同时冻结配置、脚本、源码、环境和数据，不能只提交 p1-pilot.json。

## 18. 当前 P1 文件及内部模块的逐文件审计索引

该索引列出每个执行源文件的责任和与论文的关系；细节判断可回看前文相应章节。它不把“文件存在”当作“方法效果已验证”。

| 文件 | 内部模块/函数与大致起点 | 对齐判断 |
|---|---|---|
| `src/p1/data.py` | sha/dump(19/24)、normalized(31)、features(48)、split_groups(57)、prepare(87) | 直接消费已过滤官方数据；分组/划分为本地要求；Jaccard为可选额外审计，不是二次过滤 |
| `src/p1/run.py` | CONTRACT(34)、safe_path/write_files(37/56)、parse(69)、generation_budget(105) | CUDA合同/Qwen格式/预算适配，不是公开官方控制器 |
| 同上 | Pilot.model/generate/prompt(132/168/210) | 原生Qwen、不同模式提示、官方SKILL注入；未训练 |
| 同上 | feedback/tool_action/shell(229/284及相邻方法)、trajectory/_trajectory(305/315) | 生成→行动→反馈闭环对齐；工具与截断本地实现 |
| 同上 | select_candidate(489)、save_result_summaries(514)、make_manifest(539)、finish_summary(567) | 最佳compile候选对齐；hidden/timing复合correct及指标差异 |
| 同上 | execute/_execute(625/634)、validate_resume(710)、check_gpu_devices(731)、main(746) | 串并行切换、CLI、锁、恢复为本地工程；pilot身份校验不足 |
| `src/p1/evaluation.py` | compare(49)、reward(138)、InfrastructureError(36) | 容差/里程碑公式对齐，外部比较增强 |
| 同上 | Evaluator.__init__/container(154/197) | 构建Docker命令，挂载本地worker.py，实际运行kernel Python；不是调用题目自己的主脚本 |
| 同上 | prepare/profile_reference(394/425) | 参考预检与先分析原始性能；检验本机可运行性，不重复作者1–100ms数据筛选 |
| 同上 | snapshot/compile_only(455/524) | 固定基础设施、重新构建、源码留证；比示例严格 |
| 同上 | streamed_checks/evaluate(580/676) | 外部比较、错误归属、反馈/隐藏、测时和漂移核心；数值协议不同 |
| `src/p1/worker.py` | flags/settings/seed/tree/as_list/load/cpu/call/inputs_for(30–111) | 固定精度、输入/输出处理、加载器；本地可信worker |
| 同上 | case_list/load_model/prepare(120/138/151) | 5例+固定state+额外hidden；非平凡筛选不完整 |
| 同上 | compile_candidate/timing/profile_model/runtime_fallbacks/bench(263/313/357/385/446) | 编译目标相近；测时和回退规则本地化 |
| 同上 | perturb/stream/main(511/529/653) | 缓存/状态验收、流式结果、模式派发；官方未提供对应实现 |
| `src/p1/contract.py` | ALLOCATIONS/METADATA/SCALARS/CPP_TYPES(12起)、audit_contract(32)、contract(201) | 启发式静态规则，减少误拒；不是完整语义或类型系统 |
| `src/p1/tools.py` | schema/TOOLS(13/31)、WorkspaceTools.path/write/dispatch(112/129/141) | 近似工具集，有裁剪与新增；不是OpenHands源码复刻 |
| `src/p1/sandbox.py` | BashProcess(21)、SandboxSession(89)、RPC轮询/close | 持久终端、只读设施、固定动作RPC；后台任务工具不全 |
| `src/p1/agent_utils/client.py` | request | CPU终端到GPU控制器RPC，1800s客户端等待；与实际各阶段超时不同 |
| `src/p1/agent_utils/compile.py` | compile请求入口 | 替代官方load实现为远程受控编译 |
| `src/p1/agent_utils/compile.sh` | 严格模式、模块入口 | 保留熟悉命令，输出/时间记录方式非原样 |
| `src/p1/agent_utils/verification.py` | verify请求入口 | 隔离验证，不是官方F contextmanager |
| `src/p1/agent_utils/profiling.py` | argparse、reference-only分派 | 新增原始分析便利性，缺官方全部CLI参数 |
| `src/p1/baseline.py` | select_tasks/gpu_available(37/48)、baseline_mean/rejection_summary(70/98)、write_baseline_outputs/result_row(140/187) | 独立eager/compile测量、均值与拒绝说明；无生成模型 |
| 同上 | parse_gpus/validate_resume/run_baselines/main(213/224/251/425) | 多卡续跑、固定题单、唯一摘要写者；本地工程 |
| `src/p1/helper/baseline_state.py` | beijing_now/atomic_text/atomic_json(22/27/51) | 北京时间、单文件原子落盘 |
| 同上 | exclusive_lock/task_fingerprints/protocol_fingerprints/protocol_matches/run_lock_path(57–126) | 恢复身份/互斥与兼容映射；不是论文算法 |
| 同上 | guard_legacy_controller/guard_active_containers/_record/completed_reference/archive_incomplete/cleanup_reference_artifacts(132–261) | 完成识别、残缺留证和安全清理 |
| `src/p1/helper/pilot_parallel.py` | gpu_numbers/gpu_roles/EvaluationPool/ParallelEvaluator(32/42/70/105) | GPU编号/角色校验、独占租约、嵌套复用 |
| 同上 | saved_trajectory/archive_incomplete/generation_worker/run_generation_workers/execute_parallel(161–334) | 整题多模型副本调度、父进程落盘与错误收尾 |
| `src/p1/helper/progress.py` | valid_speedups/speedup_text/PilotProgress(18/31/36) | tqdm、在线pass/GM；显示扩展，不改变评分 |
| `src/p1/status.py` | 状态读取、终端报告、main | 通常只读，无GPU；--report会写状态文档，--watch-pid还可能收尾manifest，不能一概当绝对只读 |
| 三个`__init__.py` | 包说明/标识 | 无算法、无运行实验作用 |
| `scripts/p1-run.sh` | learner激活、环境导出、父进程隐藏GPU、p1.run入口 | 项目启动工程；模型卡在子进程按编号选 |
| `scripts/p1-baseline.sh` | learner激活、参数转交p1.baseline | 不加载Qwen，CLI默认新运行GPU7，与pilot配置不同 |
| `scripts/set-env.sh` | CUDA/cache/tmp/PATH环境 | 使用当前根路径；单独source默认GPU4，P1启动脚本随后覆盖，不能当P1卡号来源 |

脚本依赖当前基础 Conda 初始化 `/data2/likun/miniconda/etc/profile.d/conda.sh`，环境前缀在项目 `.runtime/conda/learner` 和 `.runtime/conda/kernel`；不是重新回到已删除的 `/home/likun/miniconda3`。根路径和若干系统路径仍硬编码，移机需同步调整。这是可移植性问题，与论文方法是否对齐是两回事。

脚本与函数已有大量中文说明，但“scripts每条命令都在紧邻上一行注释”的用户约定尚未处处落实：例如p1-run.sh中部分R/cd/source/export连续命令没有各自独立的上一行注释；set-env.sh也有类似位置。这不影响论文算法判定，应在单独的可读性整理中处理，不能把现有注释数量当成该要求已全部满足的证明。

## 19. 官方公开仓库全部文件逐项核对

本次取固定提交 GitHub codeload 完整归档，检查官方文件集合与内容哈希：16/16 一致，无缺失。以下不是依据旧下载目录推测。

| 官方文件 | 实际内容与重要细节 | 本地使用方式/差别 |
|---|---|---|
| README.md | 方法概述、论文/数据/代码链接、三图、简介和新闻；README提及Opus4.6，原论文比较Opus4.5 | 当公开说明，不把更新的对比型号倒写进论文实验 |
| agent_workdir/SKILL.md | 376行开发约束、模板、优化优先级、编译验证profile命令、调优例、清理/checklist | Agent原文注入，存在sm90/sudo与本地环境冲突；single未注入 |
| agent_workdir/binding.cpp | 固定PYBIND11_MODULE，遍历registry注册函数 | 原样复制并核验 |
| agent_workdir/binding_registry.h | singleton注册表、名称/回调列表、Registrar和REGISTER_BINDING | 原样复制，未重新设计绑定ABI |
| agent_workdir/model.py | alpha*A+B；alpha=2，输入两个(1,128)随机Tensor | 公开AXPBY教学样例，正式run按题目替换参考，不是所有任务都跑它 |
| agent_workdir/model_new.py | ModelNew同alpha，调用cuda_extension.axpby_forward(...config=0) | 作为样例；本地由Qwen生成，不用该答案冒充模型产物 |
| agent_workdir/kernels/axpby.cu | float32 grid-stride，config选128/512/default256，launcher接stream | 合法自定义CUDA实例，可解释编译接口；不代表所有任务能力 |
| agent_workdir/kernels/axpby_binding.cpp | CUDA/contiguous/float32/same shape检查、empty_like、current CUDA stream、注册 | 绑定写法模板；本地提示沿用，不自动证明生成绑定都同样检查 |
| agent_workdir/utils/__init__.py | 空包标记 | 本地替代包有说明文字，无计算差异 |
| agent_workdir/utils/compile.py | find_sources、compile_kernels；查根及kernels，清build，cpp_extension.load，再复制.so | 本地受控快照重编译，源码集合/链接和安全检查不同 |
| agent_workdir/utils/compile.sh | 运行compile.py，shell计时并保留退出码 | 本地RPC入口，并非直接复用该内部逻辑 |
| agent_workdir/utils/verification.py | tree转换、check_equal、block_torch_functional、模型/输入初始化、5次main | 本地双容器/host比较+hidden，结构/状态与阻断机制不同 |
| agent_workdir/utils/profiling.py | CUDA profiler、参数iters/single-run、模型初始化、warmup/benchmark/print、candidate→eager→compile | 本地事件中位数与独立进程，增加reference-only，CLI不同 |
| assets/agent_loop.png | 图2工作区→CUDAAgent→GPU池→性能反馈 | 流程概念对齐；图不是已公开的调度实现 |
| assets/data_pipeline.png | 图1爬取→组合→rubric筛选 | 当前只消费最终数据，不能靠同图声称合成已复现 |
| assets/benchmark_chart.png | Level3比较图，ours pass94/compile faster90/GM1.52 | 与表1 Level3一致，不支持摘要92%的另一个数值 |

官方代码本身还有一些细节差异应记录：tree把list/tuple统一为list，本地主要保留普通tuple；官方字典比较不拒绝多余键，本地拒绝；官方验证只阻断F，profile没有同样阻断；官方没有隐藏状态测试；官方profiling顶层需要ModelNew，且同进程先跑候选。这些都说明公开脚本更接近最小开发示例，不能未经论证代表论文私有大规模评测器的全部安全属性。

## 20. 测试文件审查：覆盖了什么，不能证明什么

本节是**阅读测试设计**，没有运行测试，也不重复执行用户已要求不再运行的CPU回归。mock对控制流的覆盖可以帮助确认工程意图，不能证明真实GPU吞吐、内核正确性、反投机完整性或达到论文性能。

当前 `.gitignore` 含 `/tests/`，只读查询 `git ls-files tests/p1` 返回空：这11个测试文件存在于服务器，却未纳入当前Git提交。它们已经纳入本次文件快照和第25节哈希记录，但其他人仅checkout上述HEAD不能据此取得这些测试。是否继续忽略tests由用户决定；可移交的复现材料需要另有明确归档或版本管理，不能仅凭工作区clean推定完整。

| 文件 | 测试设计覆盖 | 未覆盖/本次发现的边界 |
|---|---|---|
| `tests/p1/test_protocol.py` | reward严格边界、外部数值/NaN/字典键、路径与固定文件、think区域解析、Torch计算/TF32、扩展别名、C++类型vs计算 | 小型构造样本，不是完整CUDA程序语义检查；未封闭头文件/分支漏洞 |
| `test_alignment.py` | 构建产物发布但评分只信源码、零耗时不奖励、8:2分组、取消4096、最佳候选/去重/hidden回退、误拒修正、编辑原子性、持久shell、baseline CLI | 名字叫alignment不代表论文所有协议一致；零耗时测试确认evaluate层正确性可True，但trajectory选优仍另受timing门槛 |
| `test_stream_errors.py` | 候选先失败不拖成参考超时、已比较前缀、参考故障分类、正常流仍比较 | 假worker的握手，不等于真实大输出/显存/容器所有失败场景 |
| `test_result_files.py` | prepare/bench命名、单份controller_seconds、preflight索引、根摘要详情路径 | 证明预期布局，不证明阶段GPU数值有效 |
| `test_baseline_stats.py` | 均值忽略拒绝/缺失/非有限，三输出MEAN首行、重写幂等、拒绝分类 | 算术汇总正确不代表论文GM协议 |
| `test_baseline_resume.py` | 整题多卡停止恢复、损坏摘要重建、已拒绝跳过、阶段收尾识别、旧格式兼容、任务/协议指纹、锁、原子失败、共享config、SIGTERM、遗留容器、编号校验 | 假评测器/模拟Docker；GPU真实并发未在本次验证，旧无指纹历史也不能追溯证明 |
| `test_pilot_parallel.py` | 编号/角色分离、评测池互斥与嵌套、同卡操作、全局ID、spawn事件、子进程/回调/退出异常、单题single复用、残缺保护、GPU-only配置变更、dryrun、保存后清理 | 不涵盖“源码改变但配置不变”的pilot恢复身份漏洞；测试的mock模型不是Qwen |
| `test_reference_cleanup.py` | 删除范围、幂等、符号链接与逃逸拒绝、先落盘后删、复用时清理、未完成保留、双模式完成/prepare-only | 不会自动清除所有evaluation构建目录；清理后精确重放需要重新准备 |
| `gpu_smoke.py` | 手写可信AXPBY、反馈/hidden通过、错误kernel失败 | 显式GPU集成脚本，不是模型结果；当前卡号兼容存在下述问题 |
| `gpu_state_smoke.py` | 带Parameter的AXPBY，hidden重载state | 同上；不能凭脚本存在说实际跑过当前版本 |
| `sandbox_smoke.py` | CPU Docker持久变量/cwd/function/文件、utils RPC、只读设施、无GPU | 假GPU回调，不验证CUDA测量 |

**静态发现的测试入口滞后。** 两个 `gpu_*smoke.py` 的注释仍说 GPU3，实际没有显式 `Evaluator(...gpu=3)`，而是直接传当前 cfg。`Evaluator.__init__:163` 不自行把 evaluation_gpu 列表取首项；当前 cfg 是 `[6,7]`，直接调用这两个脚本可能形成 `--gpus device=[6, 7]` 的错误参数。主 pilot 入口通过 gpu_roles 正规化后显式传卡号，主 baseline 也显式传编号，所以该问题主要在这些旧集成入口。应修正注释和入口编号适配后再经用户授权运行，不能本次擅自启动它们。

## 21. 原始论文逐节、逐图、逐表、逐公式覆盖表

这张表用于防止只审查主流程而漏掉附录、训练与案例。对于背景与参考文献，标记其论证角色；不要求把每篇被引论文的方法也实现进P1。

### 21.1 正文和附录

| 论文位置 | 内容 | 当前对照结论 |
|---|---|---|
| p.1 摘要/引言 | 大规模Agentic RL、高性能CUDA生成、三大贡献与加速率 | 只有环境/推理基线部分，不能复述为本地已达成；摘要百分率与表1不一致 |
| p.2 引言与未编号图 | Level3对前沿模型、128K、200交互、150训练步；GPT-5系列拒答脚注 | 本地非相同模型/数据/训练；不外推脚注到未测模型 |
| pp.3–4 §2.1/2.2 | 单轮/多轮kernel生成、Agent技能与RL相关路线 | 是设计背景；current single/agent不能代替这些训练/方法对比 |
| pp.4–5 §3.1 | 三阶段数据合成与过滤 | 消费作者已合成/过滤/去污染的最终数据；不重建上游生成器，符合当前P1范围 |
| pp.5–6 §3.2 | 多文件环境、ReAct/OpenHands工具、先profile、5%目标、奖励与反hack | 核心闭环/式1保留；工具、隔离、验证和测时均有差异 |
| pp.6–7 §3.3 | 训练崩溃原因、单轮warmup、RFT、VP、PPO | P1没有训练实现，式2–5未执行 |
| pp.8–9 §4.1/4.2 | 训练配置、128H20、250KB、6种模型/方法、指标与主结果 | 本地配置/硬件/样本/模型不同，指标还有复合correct问题 |
| pp.9–10 §4.3 | Loop/Reward/RFT/VP消融、曲线 | 无对应实验证据 |
| p.10 §5 | 数据+环境+RL联合训练使能力提高 | 当前不能分离或证实这条训练因果链 |
| pp.11–12 References | [18]PPO、[19]GAE、[25]DAPO、[22]OpenHands、[24]ReAct、[2]Skills等27条来源 | 对算法/工具出处定位，不将“引用该框架”理解为公开其完整系统，也未另审27篇全文 |
| pp.13–14 Appendix A | 任务接口、示例、AST去污染与来源比例 | 接口/原数据比例对齐，继承官方去污染；本地Jaccard另作补充 |
| pp.14–15 Appendix B.1 | 10种开发工具含Notebook/后台Shell | 自制近似工具，缺三项，Grep能力收缩 |
| pp.15–20 Appendix B.2 | SKILL全文、绑定模板、优化步骤、调优和清理 | Agent可见原文；实际命令、架构、权限与强制执行部分不同 |
| pp.20–21 Appendix C | STARK、ReGraphT、EvoEngineer、CudaForge、Kevin、CUDA-L1、ConCuR比较讨论 | 背景比较，未做本地同协议复测，不能据文字认定全面优于这些方法 |
| pp.21–32 Appendix D | 基准/案例说明，D2/D3/D4的数学化简、kernel与ModelNew | 阅读全部截图；未实测复现，D4存在合同/TF32冲突，绑定有省略 |
| p.32 Appendix E | 未对TVM；大GPU池成本/可及性 | 本地同样未补这两项实证；多卡支持并非已证明训练资源效率 |

### 21.2 图和表

| 项 | 页码 | 内容及对照 |
|---|---|---|
| 未编号比较图 | 2 | Level3前沿模型与ours；当前没有同套比较 |
| 图1 | 3 | 数据流水线；没有爬取/合成器 |
| 图2 | 4 | workspace→Agent→GPU反馈；本地概念对齐，调度非官方源码 |
| 图3 | 5 | base→单轮PPO→轨迹→RFT/VP→Agentic PPO；本地未实现训练链 |
| 图4 | 10 | w/o RFT奖励、熵及训练崩溃；无本地训练曲线 |
| 图5 | 10 | VP相关奖励、explained variation、长度截断等；无本地训练曲线 |
| 图6 | 13 | Model/get_init_inputs/get_inputs任务示例；接口支持，但全任务接口名存在不保证语义正确 |
| 图7 | 13 | 作者Model类AST去污染；使用其最终6K无需重做该步骤，本地额外审计算法不同 |
| 表1 | 7 | KernelBench各级Pass/Faster/GM；本地非同题/同模型/同metric |
| 表2 | 8 | 四种训练/工作流消融；尚未复现 |
| 表3 | 14 | 数据来源六类比例；原始6000行吻合，本地去重后采样分布会变 |
| 图8 | 22 | diagonal matmul参考；原始固定shape合法，不代表需要支持任意shape |
| 图9 | 23 | 用行缩放代替diag+GEMM，固定4096维的CUDA示例 | 
| 图10 | 23 | 直接扩展调用ModelNew；与开发约定相符，绑定省略不能直接作为完整可运行工程 |
| 图11 | 25 | matmul/div/sum/scale参考，1024×8192、8192×8192 |
| 图12 | 26 | 两kernel：权重求和+向量dot，float4/shared reduction，cudaMalloc/free；启动片段未显式传当前stream |
| 图13 | 27 | ModelNew加载同权重并调扩展；其注释shape不是验证器，本地应以原始真实输出为准 |
| 图14 | 28 | ResNet BasicBlock参考 |
| 图15 | 29 | cuDNN融合Conv/Bias/Activation、TF32相关math type，part1 |
| 图16 | 30 | cuDNN part2，workspace/调用等；不能仅复制片段而省略资源与正确stream处理 |
| 图17 | 30 | add+ReLU自定义kernel，向量化与尾部处理 |
| 图18 | 31 | forward内Torch BN折叠与TF32 True→False；与声明约束冲突，本地拒绝是公平性保护 |

Appendix D.1（pp.21–22）总结的五种共同模式也已逐一对照：代数化简/算子减少、kernel融合、内存访问优化、硬件感知（含TF32）、供应商库感知。当前SKILL与合同允许模型探索这些方向，但没有自动给产物贴这五种类别、统计各类贡献或重放相应轨迹；不能从工具支持反推模型已经掌握每一种策略。硬件感知优化尤其需要把候选与baseline放在同一精度条件下。

附录固定shape、硬编码维度的专用化，在题目本来固定shape且保持要求语义时不自动构成作弊。需要审查的是“用输入/权重算出结果”还是“返回特定测试答案”、未声明的适用范围、计算回退和不公平baseline；不能只凭硬编码或使用vendor库下结论。

### 21.3 公式和关键数值

| 项 | 原始定义 | 本地状态 |
|---|---|---|
| 式(1) | -1/1/2/3，strict .95耗时门槛 | reward函数对齐；invalid timing和hidden额外处理属本地协议 |
| 式(2) | 筛选轨迹上的Actor RFT负对数似然 | 未实现 |
| 式(3) | GAE value target，γ1、λ.95，终止value0 | 未实现 |
| 式(4) | Critic预训练的半MSE，按token平均 | 未实现 |
| 式(5) | PPO重要性比率与非对称clip(.2/.28) | 未实现 |
| token末端奖励 | 中间token r=0，最终token为outcome reward | 没有token级训练样本或credit assignment实现 |
| context | 单轮RL32768、AgenticRL131072 | 本地作为推理窗口使用，数字相同、阶段不同 |
| batch/LR/steps | 1024/1024，3e-6/6e-6，150步 | 当前无优化器/训练，不能认定相同 |
| 5随机输入 | 论文/官方验证数量 | 本地固定5反馈+额外124比较 |
| atol/rtol | 1e-2/1e-2 | 数值对齐；额外shape/dtype/finite/state等约束另述 |
| >=5%优化目标 | SKILL `.95`示例，式1严格小于 | 本地奖励按公式；submit不强制达标，Pass与Faster分开看 |

## 22. 原始材料内部的矛盾与不能填补的空白

### 22.1 已直接发现的来源不一致

1. **摘要/引言：**称 L1/L2/L3 超过 torch.compile 的比例100/100/92%；**表1：**97/100/90%。表1总体96.8%，公开Level3图片也为90%。不能同时把两套数字当同一实验结果，本报告引用主结果时以明确表格值说明。
2. **§3.3 PDF p.6：**说改进后可稳定训练200 steps；**§4.1 p.8：**实际设置150 training steps。另有训练150 turns/评测200 turns，steps与turns不能相互解释。应作为叙述与实验设置的不一致记录，不能把200训练步当已复现目标。
3. **声明约束 vs 图18：**禁Torch计算，但案例在forward中折叠BN并修改TF32全局状态。作者允许vendor库也不意味着允许用Python Torch执行这些计算或污染后续baseline。
4. **正文平均 vs 公布脚本 vs SKILL调参例：**平均重复测量、profiler事件平均和10+100墙钟调参是三个不同描述；本地中位数又是第四种，需要明确采用哪一个。
5. **式1严格<.95 vs SKILL≤.95：**边界细节不一致，当前按公式。
6. **README型号更新：**公开README提Opus4.6，原PDF是Opus4.5。作者后续Seed2.1/Opus4.7/FlashInfer材料也不是原表1的实验，应另立版本和证据。

这些矛盾不意味着论文所有结论都被否定，但足以使“所有参数严格照抄就能复现”的说法失去依据。应同时记录声明、可执行代码与实际实验，而不是挑有利的一层来宣称对齐。

### 22.2 开源不足留下的不可确认项

目前无法仅凭这些材料确认：原训练数据合成模型/完整prompt与随机种子；训练任务/难度采样顺序；私有Agent工具schema/消息模板/后台进程实现；温度/top_p/每轮输出上限；optimizer/scheduler/KL/entropy/advantage处理/梯度与并行完整设置；RFT过滤pattern全集；critic初始化与训练细节超出公式部分；原训练/rollout engine精度与logprob具体修复；全部torch/CUDA/cuDNN/驱动版本；私有评测器是否比公开示例更严格；表1逐题原始候选及全部计时状态。

这部分应写“未披露/未开放，无法确认”，不能用现成verl默认参数、自己补写的工具或后来比赛代码反向证明原论文就是这样做的。即便未来完成了本地所有训练阶段，也应准确叫“基于公开描述的复现实现”，而不是逐行恢复私有系统。

### 22.3 对“Agent写算子”的独立判断

这个任务至少有三类能力：发现代数等价化简并写自定义CUDA；把多个elementwise/reduction融合成kernel；选择/配置cuBLAS或cuDNN并组合自定义kernel。原SKILL明确允许后两种供应商库的受限用途，所以使用cuDNN本身不违规，不能要求每个卷积都从零手写才能算对齐。

但调用既有库、调flag或调用torch.compile，与证明模型具备新的底层kernel设计能力不是同一证据。评估应同时看通过率/加速比和产物类别，逐候选检查真正计算路径、编译产物、调用库、精度、状态与缓存。图9/12展示了有价值的算法化简；图18展示了需要质疑的合同和测量问题。两种事实可以同时成立。当前P1环境更严格并不自动说明Qwen已经学会这些算法，也不意味着环境已经不存在reward hack。

## 23. 原因归类：哪些应保留，哪些应修正

### 23.1 已有明确用户决定的差异

- 主模型Qwen3.5-9B、较新的统一cu129软件栈、现有A100资源。
- 普通train/test=8:2，不迁就旧pilot；pilot从train抽，test留冻结后评测。
- 支持数字多GPU、断点恢复、公共worker-config不改名、逐题清理input-spec/compile-cache。
- 中文注释、北京时间、结果减少重复、baseline首行均值、终端tqdm/pass_rate。

这些属于实验目标/工程要求，应保留并透明标为变体。本次审计不把“回到原论文所有资源和数据设计”当默认授权。

### 23.2 合理但需要单独声明的协议扩展

只读基础设施、无网络、受控RPC、独立参考/候选与外部比较、重新编译、明确错误归属、额外hidden/cache/state检查、前后基线漂移检查有合理动机。其中隔离/留证/原子写盘主要提高可靠性；**额外hidden、过滤、测时门槛会改变指标集合与数值**，必须写成严格本地协议，不能只说无影响的工程改进。

### 23.3 不能以“工程化”掩盖的问题

复合correct与论文Pass口径不同、死配置、pilot源码身份校验不足、静态扫描范围/运行期覆盖不足、旧GPU集成入口与列表配置不兼容、文档旧GPU默认值，是需要分别处理的缺口或描述问题。工具能力/CLI裁剪和raw SKILL与本机环境矛盾，也应按实际影响排序，不能仅按工具名称是否齐全判断。**不重做官方数据的去污染或1–100ms筛选，不属于上述缺陷；TF32公平性按既定本地方案处理，已解决。**

对动机的判断也要有边界：用户明确要求的事项有会话依据；其他差异的安全/性能动机是根据代码结构可解释的工程理由，**不是对原作者私有设计或开发者当时意图的证实**。

## 24. 后续修正优先级与可验收目标（只建议，未执行）

### 24.1 下一次正式结果前优先处理

| 优先级 | 建议 | 可验收目标 |
|---|---|---|
| 高 | 拆分functional_pass/policy_pass/timing_valid/hidden_pass/accepted，补Faster Rate分母 | 功能正确但测时无效不再被写成纯功能失败；严格本地通过率与论文口径并列，不覆盖旧字段含义 |
| 高 | 冻结并明确命名本地测时/验证协议 | 平均vs中位数、5例vs124、TF32、精度、重复/预热、漂移、选优一目了然，不只靠paper-aligned标签 |
| 高 | 给pilot续跑补与baseline等价的任务/源码/模型数据身份与完成状态校验 | 改worker/contract/data/prompt后无法静默混入旧结果；保留原运行来源及各次invocation |
| 高 | 修复死容量配置与审查覆盖缺口 | 输出读取前有实际容量限制，头文件纳入检查；vendor用途与runtime包装操作有明确可审查策略 |
| 高 | 区分真实坏题和基础设施拒绝 | 正式report同时列总题数、环境失败、规则过滤、功能失败及计时失败；不可默默缩小分母 |
| 中 | 调整SKILL的环境说明与工具缺项 | sm80/无sudo/RPC调优路径不矛盾；明确缺少后台/Notebook工具，或补齐真正需要的会话能力 |
| 中 | 修正纯vendor.cpp误拒、输入不变规则与文档旧GPU | 合法纯GEMM/Conv包装不因缺cu而失败；非平凡规则有清楚定义；示例使用明确编号 |
| 中 | 统一集成测试入口选择GPU的方法 | smoke不再把列表拼进device参数，实际卡号与注释一致；运行仍需用户授权 |
| 记录 | 保留官方最终数据的来源与指纹 | 继承作者已完成的去污染，不新增过滤；Jaccard作为可选额外审计，未复核原算法不等于P1缺陷 |
| 后续阶段 | 正式KernelBench适配、按级报表、模型/训练消融 | 与Ops自划分test分开，先冻结协议，不把pilot调试结果充当正式主表 |
| P2及以后 | 按Qwen资源条件实现单轮PPO→RFT/VP→AgenticPPO | 逐阶段有checkpoint/数据/训练曲线/消融/成本；不能只复制论文超参当完成 |

这些建议不会自动触发GPU任务或改变既有数据。是否保留本地严格协议、是否增加一套接近公开脚本的参考协议、是否进入P2，需要按实验目的决定。即便另做公开脚本参考测量，也不应恢复已知TF32状态污染；应把修正内容作为明确条件记录。

独立baseline的 `torch.compile(mode="default")` 没有设置 `fullgraph=True`，所以允许PyTorch图中断/回退。这与官方默认compile用法相近；不能把status=ok解释为全图融合，更不能把baseline可能生成Triton与“候选不许Triton”混为一谈。baseline没有候选路径的before/after漂移验收；其汇总按worker状态提取时间，对状态变更和非正/非有限时间的完整判定也弱于候选evaluate，需要读详情而非只看MEAN。

### 24.2 能准确描述当前完成度的一句话

> 已基于官方Ops-6K、开发约定与奖励设计，实现Qwen3.5-9B的单轮/Agent推理基线及隔离评测工程；模型、数据划分、工具和验证测时协议存在已列明的变体，尚未复现原论文多阶段RL训练、KernelBench主实验及消融结果。

## 25. 证据清单与本次工作记录

### 25.1 关键文件身份

| 文件/对象 | SHA256或提交 |
|---|---|
| 原始PDF | cdeebc53efb4a25d74b7e9b428e5292fcccd8626aa283c0d8aa9277f516995f9 |
| 项目HEAD | 4f9b6f4576148abe06d20215a0c252386c59178b |
| 官方上游提交 | 473025c8af7878e928525138b5cb327d6bcdf9dd |
| configs/p1-pilot.json | ca0c6ca37dae3dcc4b59256b43474d3dd3b21a4bd32712efa858375a12f98018 |
| 原始ops6k.parquet锁定值 | d4404fc181cad72081f8c1cba720cc1b511fb34dee924079e898bf1e83f6257c |

代码证据定位以该HEAD时实际读取的文件快照为准；HEAD标识仓库状态，不自动保证被ignore的本地文件也包含在提交中。官方文件与固定提交完整归档比对，不依赖“上次应该下载过”的记忆。当前P1文档已有不少正确的限制说明，例如AST方法不同、baseline非冷编译、rawQwen、hidden为本地扩展等；本报告进一步核实执行逻辑，并指出仍滞后的默认卡号和指标等细节。历史文档里“某次测试通过”的文字没有被当成本次实测证明。

写入本报告前，再次只读校验本次快照的57个文件SHA256：无变化；项目HEAD仍为上述提交，工作区仍干净。本报告是随后新增的文档，不属于原实现提交，也没有借审计之机修复实现。

本次实际工作：读取服务器当前源码和配置、读取数据审计/清单、读取环境包元信息与设备规格、核对原PDF及逐页文本、查看图表和代码截图、核对官方全部文件及图片、静态审查测试设计，形成此报告。没有执行测试、Qwen生成、内核编译、benchmark、训练、结果迁移、旧实验清理或自动定时跟进。

后续阅读建议：先看第2节的重要结论，再看第3节的三层区别；关心模型/生成看第5节，关心工具看第6–8节，关心结果可信度看第9–12节，关心训练复现看第14–15节。第17–21节用于逐字段、逐文件和逐图公式查漏；第24节只是后续建议。

### 25.2 当前实现与测试文件的字节身份

以下 35 项覆盖本次读取的P1源码、启动脚本、配置与测试；是静态文件身份记录，不表示运行过这些测试。官方16文件另由固定提交的完整归档验证。

| 文件 | SHA256 |
|---|---|
| `configs/p1-pilot.json` | `ca0c6ca37dae3dcc4b59256b43474d3dd3b21a4bd32712efa858375a12f98018` |
| `scripts/p1-baseline.sh` | `37354b248771b21033e433e3badaece6e1c086b2a4511f459c7e091f101c2772` |
| `scripts/p1-run.sh` | `040136a14ac3b220aa83393c1f1dd2fb29ebb51d43f98060c728cc8dddf28c9d` |
| `scripts/set-env.sh` | `8a3152c3596d7e732f0cc143af3584fe332a09d5079178d06db195c3a5f83ead` |
| `src/p1/__init__.py` | `71ba90c19f2e8e58f9cc2f0207c1d667d0c731f03554421f380d85ea2de1b9f7` |
| `src/p1/agent_utils/__init__.py` | `1122cf7edc1417260987da03476cca6c9d9959515953201ce766bd54516fe45a` |
| `src/p1/agent_utils/client.py` | `509304cda57afad9836145a19fa21577f452b8834647b2563bd221fa1f837f6a` |
| `src/p1/agent_utils/compile.py` | `df695ad8fbe5a440ffb1da291113b0b449521fbb0d91c41a3752b797bd5a0490` |
| `src/p1/agent_utils/compile.sh` | `da65f6a0aa18a2be68b177dce2d899553efb6aa2e2e68f9b0554064870523f49` |
| `src/p1/agent_utils/profiling.py` | `68ad3424b63eb8e318bdd124cda914fe5d18bd79f9e63a216cd5ba5a3c561719` |
| `src/p1/agent_utils/verification.py` | `afe5d3e338228773c56cf08c30961f1ec62b07d7359ee66ecaf3151b583b5376` |
| `src/p1/baseline.py` | `600abba0e737441f66132e3204c7bbeeecc9a490fc3b4b35be4c1ded9f009322` |
| `src/p1/contract.py` | `bb97901b0499a28d9355366a735bd6c4050d79b7f58187cfe1e2942b32a47c19` |
| `src/p1/data.py` | `046b77e01e22c644546768f3ab48f59d81436205b185bd41838cbaaab4938337` |
| `src/p1/evaluation.py` | `52ba32e10c4f7b74bf2c918837960a700ee2eb175c12475a20f84697a33d874e` |
| `src/p1/helper/__init__.py` | `41562b0e5c3f8058bcf52d72f9c84aabce35c741b856dd84dfdbf5323d6ee756` |
| `src/p1/helper/baseline_state.py` | `5ffdaeb905d674522bf7e7d633a707081bbc0803d9bb1a629e10d5fe02c731e4` |
| `src/p1/helper/pilot_parallel.py` | `4f757edf361a431f6ec558f355bed7560358f8063ca2e96f1ef84ec228e15f5a` |
| `src/p1/helper/progress.py` | `864480c04062b9723cae387a0cb3a39adf29d19ffee0dad3c3990dd4c84600c0` |
| `src/p1/run.py` | `b89427828c2473ddd825132236873cb34cdf2abda491e89d4d60750b4809d787` |
| `src/p1/sandbox.py` | `98d336d2f8c6684fb89a9cdea51045c9d0230970877c7315bdf54d850980c527` |
| `src/p1/status.py` | `3c2566b45eb811ca0ff75ca867f4dd4ddbcca71f3595a20c18f40d26917d5e64` |
| `src/p1/tools.py` | `7935944aedcf46b941027b23f05bfc2834bbf7d49ea9ea8e6d22cb6e788c58b2` |
| `src/p1/worker.py` | `97e7defb04b891f6c3af54ca82b53b28d708bed5211455df8f8c3fb9b60bd883` |
| `tests/p1/gpu_smoke.py` | `f00f6c238f9cd9200a83f9d7f816f6c6f7bea0568b3c12c7f140a9decfbaa238` |
| `tests/p1/gpu_state_smoke.py` | `150e521c7c7800d2689b86c5a454a36a500cbc9472eed7eb35df1956a3e05628` |
| `tests/p1/sandbox_smoke.py` | `6dd3b0800da79d3ebdc8b746511f08853ab80dc8deb0057450c37d912ed37ace` |
| `tests/p1/test_alignment.py` | `3a3e456a6eee69cbbc0ba77ed349b7d485ae4d5eaa58c7e5487191665573dccb` |
| `tests/p1/test_baseline_resume.py` | `b8fd9e170d48119e45e93bab181bff61739983b4032047655ee9fae3b2a603cb` |
| `tests/p1/test_baseline_stats.py` | `3b50dc78e1a4bd349c4ebe8e57df1ee8562ef762019d31cc1131ae4084bf0f51` |
| `tests/p1/test_pilot_parallel.py` | `473655d94484a03ceedbb98c26e0367ac5a6eef79d1d95ae80afdb0e0c6993c3` |
| `tests/p1/test_protocol.py` | `2ecde317da56a8083aa6a97f12729594a8f0c088740cea432481f109dfed616e` |
| `tests/p1/test_reference_cleanup.py` | `d15b14b5367cb9d5cb30be33ecdc1dd0a96e314ddefbe94d6cb20aa4fc7989e6` |
| `tests/p1/test_result_files.py` | `a0f2af74a304a02cf416402a6974ac42d353f09916463cccfc4803b2790f5e1d` |
| `tests/p1/test_stream_errors.py` | `f8ed45c6e7df8de9e77b61856332d89948bfcf3084ff1260999b06d52dd94439` |

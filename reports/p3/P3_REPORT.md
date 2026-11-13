# P3：直接 Agent + GRPO

更新时间：2026-11-13，北京时间。

## 1. 训练入口

在服务器 `/home/likun/junjxu/CUDA-Agent` 执行：

```bash
# 正式训练：原始 Qwen3.5-9B 直接进入 Agent + GRPO，不读取 P2 checkpoint。
bash scripts/p3-train.sh --run runs/p3/agent-grpo

# 从本次运行已完成的 checkpoint 续训。
bash scripts/p3-train.sh --run runs/p3/agent-grpo --resume

# 仅展开配置，不启动模型或 GPU。
bash scripts/p3-train.sh --run runs/p3/preview --dry-run

# 可选：限制原生数据集样本数，便于自行小规模实验。
bash scripts/p3-train.sh --run runs/p3/small data.train_max_samples=100 data.val_max_samples=8

# 独立接线短测：最多两步、每组两条、两个 prompt、回答总预算 8192 token、最多八轮。
# 只缩短该 run，不修改正式 YAML；不验证、不保存 checkpoint。
bash scripts/p3-train.sh --run runs/p3/wire --wire
```

唯一手工配置为 `configs/p3-agent-grpo.yaml`；原生 `key=value` 参数可在命令末尾覆盖。
已有目录必须有完整 checkpoint 才能 `--resume`；中途未保存的 step/轨迹不会自动恢复。
正式 150 步由用户启动。

## 2. 参数与论文的关系

以下分别以**论文正文/附录**、**当前本地官方仓库**和**当前 P3 配置/实现**为依据。
论文未披露的参数明确标为“未披露”，不把框架默认值或本地经验值当成论文原值。
当前按吞吐优先采用32K上下文、训练/验证各30轮及共享测时；这些预算与论文不同。clip_ratio_high仍为0.2。

原始论文依据：PDF 第5–6页 §3.2，第7页 §3.3 式(3)–(5)，第8页 §4.1，
第14–15页附录 B.1，以及第15–22页附录 B.2 的原始 SKILL。
本地对应官方代码为 `upstream/CUDA-Agent/agent_workdir/SKILL.md` 和 `utils/`。

### 2.1 模型、算法与训练超参数

| 项目/配置 | 论文原值或明确描述 | 当前 P3 值/实际行为 | 对齐判断及原因 |
| --- | --- | --- | --- |
| 基础模型 | Seed1.6，MoE 总230B/激活23B（§4.1） | 原始 Qwen3.5-9B | 用户选型差异；BF16 是数值格式，不是量化版本 |
| RL 算法 | PPO，Actor+Critic（§3.3） | 原生 GRPO，无 Critic | 用户明确更换算法，不能称完整复现原论文训练算法 |
| 前置阶段 | single-turn PPO→采集轨迹→RFT/Value Pretraining→agentic PPO | 直接 Agent+GRPO，不加载 P2 checkpoint | 用户明确跳过前置阶段 |
| 训练数据 | 已过滤、去污染的 Ops-6K，共6000题（§3.1） | 已有 train.jsonl，4803题；test.jsonl，1197题；不重新去污染 | 来源一致；8:2按任务组划分属于本项目方案，非论文划分 |
| global batch | 1024（§4.1） | train_batch_size=4题，n=4，实际16条轨迹 | 未对齐规模；论文未披露每题采样数，不能自行把1024还原成题数 |
| minibatch | 与 global batch 相同，1024（§4.1） | ppo_mini_batch_size=4题，框架乘 n 后16条轨迹 | 批量数值不同；“整批一个 minibatch”的组织方式相同 |
| 每题组大小 | 未披露；原文是 PPO，不以 GRPO 组大小定义算法 | rollout.n=4 | GRPO 本地选择，不是论文参数 |
| 每卡 microbatch | 未披露 | 1条轨迹/训练卡 | 显存工程设置；当前两张训练卡通过累积凑16条有效轨迹 |
| 每批更新次数 | 未披露 ppo_epochs | ppo_epochs=1，当前每 step 恰好一个有效 minibatch | 本地选择；不是论文明确值 |
| Actor 学习率 | 3×10^-6（§4.1） | 3e-6 | 数值对齐 |
| Critic 学习率 | 6×10^-6（§4.1） | 不适用，critic.enable=false | GRPO 不使用价值网络 |
| 裁剪下界参数 | ε_lower=0.2，概率比下界0.8（式5） | clip_ratio_low=0.2 | 对齐 |
| 裁剪上界参数 | **ε_higher=0.28，概率比上界1.28**（式5） | **clip_ratio_high=0.2，上界1.2** | **未对齐**；上次报告只写“clip0.2”遗漏了论文的不对称上界 |
| clip_ratio | 论文写 lower/higher，不是另一个独立对称超参数 | 0.2，作为原生回退值；已有显式 low/high | 实际以 low/high 为准；只改 clip_ratio 不能覆盖显式上界 |
| 原生隐含dual-clip | 式(5)未写负优势dual-clip项 | 未覆盖的policy_loss.loss_mode=vanilla、clip_ratio_c=3.0，原生对负优势启用dual-clip | 还有这一框架默认差异；不能声称只改high=0.28就让整个目标与论文一致 |
| loss 平均方式 | 每条轨迹对 token 平均，再取轨迹期望（式5） | seq-mean-token-mean，只计模型生成 token | 平均方式对齐；优势由 GRPO 而非论文 GAE 提供 |
| GAE 参数 | γ=1、λ=0.95（式3） | GRPO 不调用 GAE/价值回归 | 算法改变后的不适用项，不是漏配 Critic |
| 学习率调度 | 未披露 constant/cosine、warmup、最低LR | P3未覆盖，当前 FSDP 原生默认 constant、warmup ratio=0 | 属框架默认，不能宣称论文默认 cosine |
| 优化器默认项 | 未披露 AdamW betas/weight_decay 等完整配置 | 原生 AdamW，betas=[0.9,0.999]、weight_decay=0.01；未在P3重复列出 | 本地继承框架，不是论文原值 |
| entropy_coeff | 未披露该系数；Figure4报告熵变化不等于披露熵正则权重 | 0.0 | 不额外奖励高熵；论文是否同值未知，不能据此称对齐 |
| KL 正则 | 未披露系数、估计方式及是否加到loss/奖励 | Actor KL loss=0.001，low_var_kl；不在reward重复扣KL | 本地稳定性选择，不是论文披露参数 |
| LoRA | 未披露 LoRA rank/alpha/target | rank64、alpha128、all-linear，freeze_vision_tower=true | 本地双卡训练适配；并非论文 LoRA 复刻 |
| 视觉冻结字段 | 未披露此框架参数 | YAML中为true，但当前V1路径未读取它 | **不能据此宣称视觉LoRA已冻结**；见第7节核查结论，纯文本无视觉梯度不等于显式冻结 |
| 训练/rollout 精度 | §3.3讨论 BF16/FP16 不匹配风险，但未给完整精度配置表 | Actor BF16、vLLM BF16 | 对齐“减少训练推理精度差”的目标；具体栈为本地选择 |
| 总步数 | 150 training steps（§4.1）；§3.3另提稳定200步，不能混同 | 150 | 对齐正式实验设置 |
| epoch 上限 | 未披露 | total_epochs=1000，仅为宽松数据遍历上限 | 正常由150个step先结束，不表示执行1000轮训练 |
| 保存/验证频率 | 未披露 | 每30步保存、每5步验证、训练前验证 | 用户指定，非论文原始频率 |
| 验证集合 | 论文最终评测为 KernelBench L1/L2/L3，共250题 | 现有 test 固定抽24题、每题n=1 | 本地训练监控；不是论文 KernelBench 最终结果 |
| 数据加载并发 | 未披露 DataLoader worker 数 | dataloader_num_workers=0 | 只影响CPU取数据；与16条rollout、4个Agent worker、评测并发互相独立 |
| 分布式/省显存 | 未披露 FSDP2、activation offload、FA2、融合输出头配置 | 原生FSDP2、参数/优化器/激活卸载、梯度检查点、FA2、Torch融合/分块输出头 | 本地资源实现，不额外手写优化器或模型层 |

### 2.2 上下文、采样和 GPU 调度

| 项目/配置 | 论文原值或明确描述 | 当前 P3 值/实际行为 | 对齐判断及原因 |
| --- | --- | --- | --- |
| Agent 总上下文 | 131072 token；single阶段32768（§4.1） | 32768，含prompt、模型输出和工具观察 | 按用户要求降低预算以限制成本，当前不对齐论文Agent的128K |
| 训练/评测轮数 | 训练150 turns、评测200 turns（§4.1） | training_turns=30、validation_turns=30 | 按用户要求限制单题试错成本，不对齐论文原轮数 |
| max_prompt_length/max_response_length | 未分别披露这两个框架字段 | 32767 / 32768；实际扣除prompt与工具观察，总长度≤32768 | 本地长度接线，不是64K；prompt上限随总上下文推导 |
| temperature/top_p/top_k | 未披露 | 0.6 / 0.95 / 20；验证同样采样，n=1 | 当前Qwen采样方案，不是论文参数对齐 |
| seed | 未披露具体值/每请求策略 | data/FSDP/vLLM全局42；移除每请求seed；反馈输入固定71001及72000–72004 | 全局42是用户要求；kernel.seed字段本身当前不被worker消费，见第9节 |
| 工具观察长度 | 未披露精确截断阈值/方向 | 原生8192**字符**，middle截断 | 本地预算，不是8192 token，也不是论文原值 |
| 单轮多工具 | 未披露精确并行调用数 | 初始max_parallel_calls=16；更多调用仍完整保留；同轨迹操作串行 | 保证记录/执行一致的本地接线，不代表16题并行评测 |
| CPU/GPU 沙箱 | Docker CPU终端+独立GPU池（§4.1） | 持久CPU容器+可信控制器派发GPU容器 | 架构目标对齐；不直接执行官方同进程评测脚本 |
| GPU 硬件/数量 | GPU验证及profiling池：128张H20；总训练GPU数未披露（§4.1） | A100，训练/rollout=[2,5]、评测=[4,3] | 硬件和规模不同；训练卡同步分阶段共享 |
| 资源独占 | 论文描述GPU作业进程隔离、独占资源和有保证HBM（§4.1） | 每卡最多4个共享作业，包括参考预检、初始分析、验算及候选/后测测时 | 用户接受吞吐优先与测时噪声；不对齐独占/保证HBM |
| vLLM 调度 | 未披露 TP/KV显存比例/prefix caching等 | TP2、总显存45%预算、max_num_seqs16、分块prefill32768 | 本地推理容量设置，不是每卡同时训练16题 |
| Agent worker | 未披露 Ray/Agent worker 数 | 4个CPU AgentLoop worker | 负责异步交互，不是额外4张训练卡 |
| 超时/编译槽 | 未披露300/120/600秒或4槽限制 | 编译300秒、评测阶段120秒、前台Bash硬上限600秒、CPU编译槽4 | 本地资源控制；单轨迹不等于只允许运行120秒 |
| 传输/资源上限 | 未披露本地队列/文件通道/Ray限额 | 输出通道16GiB；TQ两个unit共128个记录槽；Ray 24逻辑CPU、2训练GPU、Dashboard关闭 | 框架/工程容量配置，不能把128解释成GiB |
| 环境版本 | 未披露当前项目对应的完整torch/vLLM/CUDA版本组合 | 沿用P0已有Conda、CUDA12.9、Docker隔离环境及verl源码 | 工程复用；本次不更换包版本或镜像 |

### 2.3 正确性、性能、奖励及安全约束

| 项目 | 论文原值/官方代码 | 当前 P3 实现 | 对齐判断及原因 |
| --- | --- | --- | --- |
| 五组正确性 | §3.2明确五组随机输入；官方verification循环验算 | feedback_cases=5，固定种子可复现 | 五组覆盖对齐；随机种子规则是本地具体实现 |
| 数值容差 | **附录B.2及官方verification：atol=rtol=1e-2** | atol=rtol=0.01 | **对齐**；上版报告把它只称为本地容差不准确 |
| 奖励 | 式(1)：错误-1；正确1；比eager快>5%给2；同时比compile快>5%给3 | 复用相同-1/1/2/3规则，严格耗时下降>5% | 奖励离散档位与门槛对齐 |
| 轨迹最佳候选 | §4.1最终评测选相对compile加速比最大的有效解；训练终局选择细节未完整披露 | 训练及验证均选本轨迹已正确、有效测时的最佳候选并保存源码 | 对齐论文评测选择规则；推广到训练终局是用户明确选择 |
| 测时重复数 | 正文只说warmup、重复测量和平均；公开profiling.py默认iters=10 | 1组输入、1轮×10次 | 重复次数与公开示例默认相同；不等于完全相同测量协议 |
| 预热 | 官方profiling.py warmup_iters=5 | 每次timing调用预热1次；compile首次调用与额外profiler另外执行 | **未逐项对齐**，当前只记录，不擅改预算 |
| 计时方法/汇总 | 官方使用profiler CUDA事件耗时求和/iters；正文描述averaging | CUDA Event逐次测时，取中位数，另存wall时间；profiler另调用 | **实现不同**；可比较本地模型前后，不能冒称原脚本同口径 |
| 候选/基线顺序 | 正文未明确；公开profiling.py实际先candidate再eager、compile | 候选后独立容器测eager/compile，并用该后测值打分 | 与公开示例的先后方向一致；PPO奖励内部顺序未公开，不能由示例推定 |
| baseline取值/漂移 | 未披露本地漂移阈值或前后中位数机制 | 只用候选后基线，不因前后漂移拒绝 | 按用户此前确认的本地协议 |
| 额外隐藏验收 | 论文无单列34次隐藏验收阶段 | hidden种子=[]、次数=0，训练/验证均不调用hidden阶段 | 去掉本项目原来的额外环节；五组验算仍在 |
| 参考预检 | Ops-6K在数据合成阶段过滤不执行/随机/近常数题；未披露当前重复预检机制 | 每轨迹按需准备参考参数；初始性能摘要按同运行/题目/环境复用；不因近常数拒绝 | 本地环境适配；不是重新去污染，也不能说论文要求每轨迹重复检查 |
| 参考/候选隔离 | 保护验证脚本、GPU沙箱（§3.2/4.1） | 只读文件、可信输入/参数、独立参考/候选容器 | 目标对齐，隔离实现为本地工程 |
| PyTorch回退 | 论文运行期禁止functional回退；SKILL更严格限制torch计算 | 静态明确违规+运行期dispatch/profiler审查，未知helper不凭名字误拒 | 对齐反回退目标；当前允许表及检查范围不是官方代码逐行复制 |
| TF32状态 | 原文未披露完整状态隔离实现；SKILL列混合精度优化建议 | 统一基线/候选精度条件并检测状态污染 | 防止已知基线污染的本地工程，不是论文明确参数 |
| cuBLAS/cuDNN | SKILL允许cuBLAS仅GEMM、cuDNN用于Conv/ConvTranspose | 保留相同用途约束和实际CUDA/库调用路径 | 合同对齐；不允许用一般PyTorch计算冒充CUDA实现 |
| 文件权限/网络 | 评测脚本不可改，禁止web检索（§3.2） | model/SKILL/binding/utils只读，无sudo、无网络，可信工具派发GPU | 目标对齐；权限、RPC路径和异常处理为本地实现 |
| 无效参考/零token/OOM处理 | 未披露整组补采、验证跳过、有限次数等规则 | 明确可恢复问题整组丢弃；OOM等致命环境问题停止；验证跳过记录 | 本地必要失败语义，不是放宽模型失败；细节见第4节 |

### 2.4 工具逐项对照

下表的“功能对齐”不表示已取得官方 OpenHands 工具的同一实现。
官方仓库只公开 SKILL、绑定和若干utils，未提供可直接替换全部本地接口的完整训练Agent环境。

| 当前工具/入口 | 论文原工具或官方入口 | 当前实际实现与差异 | 对齐关系 |
| --- | --- | --- | --- |
| Bash | 附录B.1：持久shell | 复用P1持久Bash；保留cwd、变量、函数；CPU容器，无sudo/网络，GPU经RPC派发 | 持久会话功能对齐；隔离/权限是本地实现 |
| BashOutput | 附录B.1：后台输出增量查询 | 查询真实bash_id；短等待返回running而非杀命令 | 功能对齐；JSON schema和等待策略未公开 |
| KillBash | 附录B.1：终止后台会话 | 只停止本轨迹对应后台任务 | 功能对齐；作用域限制为本地实现 |
| Read / Write | 附录B.1：读写文件，已有文件read-before-write | P1工具在已读路径集合中检查；只读基础设施不可写 | 核心规则对齐；不是复制官方源代码 |
| Edit / MultiEdit | 附录B.1：确定性替换、同文件多处原子修改 | 替换在内存中全部成功后一次写回，失败不写半个文件 | 功能对齐；本地参数/schema |
| Glob | 附录B.1：glob发现文件 | 工作区相对路径、拒绝链接、结果最多500项 | 发现功能对齐；500项限额为本地设置 |
| Grep | 附录B.1：ripgrep、regex、类型/上下文 | 调用kernel Conda的rg，支持相应选项，限定工作区 | 功能对齐；包装方式本地实现 |
| TodoWrite | §3.2列TodoWriteTool | 记录工作区todo JSON | 功能对应；具体schema/持久化格式未披露 |
| **未提供NotebookEdit** | 附录B.1明确提供NotebookEdit | P1已有该实现，但P3 tools列表没有选入 | **工具全集未完全对齐**；CUDA源码任务通常不需要notebook，当前不影响已有`.py/.cu`工作流 |
| profile_reference | §3.2“先分析原始性能”，文中称profile.py；公开仓库为utils/profiling.py | 新增显式包装：无候选也能分析eager/compile；同运行/题目/环境复用初始摘要 | 对齐论文要求的能力；工具名称/schema为本地新增 |
| evaluate | 官方utils compile/verification/profiling分阶段命令 | 新增组合入口：冻结→编译→五组正确→测时→后测基线 | 组合包装为本地工程；阶段能力与官方工作流对应 |
| submit | 论文未披露同名工具或结束schema | 结束轨迹；可信控制器选择历史最佳，不接受模型自报分数 | 本地结束协议 |
| Bash中的utils.compile / compile.sh | 官方同名编译入口 | P1兼容客户端经RPC调用可信编译控制器；A100用sm80，官方SKILL示例sm90 | 编译能力对齐，执行位置/硬件/具体实现不同 |
| Bash中的utils.verification | 官方同名五组验算命令 | RPC调度同一P1评测器；成功只算正确性，不凭此发性能奖励 | 阶段能力对齐，分容器实现不同 |
| Bash中的utils.profiling | 官方同名分析/计时命令 | 无候选时分析原始实现，有候选时正式评测；支持--reference-only | 能力包含原流程，并补上论文“先分析”但公开脚本依赖ModelNew的问题 |
| Bash中的utils.run | SKILL B.2展示自定义GPU调参脚本；未公开同名RPC工具 | 将自定义.py派到隔离GPU，只作开发反馈；正式分数仍需evaluate | 调参能力有SKILL依据；该工具包装不是论文直接给出的实现 |

训练 loss、GRPO 优势、优化器、Actor/rollout 权重同步、checkpoint、进度条和TensorBoard均使用
verl 原生实现。P3扩展职责及必要性见第7节；没有改写 `third_party/verl` 或已安装包。
当前32K是总上下文上限，不是单次回答预算；真实GRPO更新状态见第6节。

## 3. 从一道题到一次更新

1. 直接读 `data/processed/train.jsonl` 和 `test.jsonl`。不转换 Parquet，不另存 P3 数据集，
   不重新划分或去污染。样本数选择、shuffle、长提示过滤使用原生 `RLHFDataset`。
2. 系统消息包含原封不动的官方 `upstream/CUDA-Agent/agent_workdir/SKILL.md`，末尾只追加
   A100/CPU Bash、GPU 工具调度、只读文件及提交方式这些实际环境差异。用户消息只给原题代码。
   原生工具 schema 作为工具定义传入；不再附加 P1/P2 的交付 JSON、`<file>` 或长合同提示词。
3. 原生 AgentLoop 为同题采样四条独立轨迹，每条拥有独立工作区和持久 Bash。模型可读取、
   编辑文件、调用原性能分析、编译/验证/测时工具，并根据反馈继续修改；不是解析末尾文本来提取源码。
4. CPU Bash 仍在隔离容器中；`utils` 命令由可信控制器调度 GPU。模型提交的是实际文件，不能
   自报正确性、耗时或奖励。参考参数仅在本轨迹第一次需要时准备；原始性能分析按需调用一次。
5. `evaluate` 冻结源码、审查、编译、五组验算，通过后正式测时并后测 eager/compile。
   未知 Python helper 名不再单靠静态推断误拒，实际执行回退检查、TF32 状态检查等仍保留。
   P3首个明确失败就停止剩余案例；正确候选仍须五组全部通过。
6. 相同源码直接复用结果；已经五组验算成功、尚未测时的快照可直接晋升测时。文件变化才重测。
   `submit` 或自然结束时检查最终文件是否出现新候选，然后选择历史最佳有效候选。
   无合格候选给 -1；最后一次修改失败不会覆盖之前的合格快照。
7. verl 保留生成 token、logprob 及工具观察 mask，计算同题组内标准化优势、KL 和 GRPO 更新，
   再把更新后的 LoRA 同步给 vLLM。工具观察参与上下文，不作为模型生成 token 的训练目标。

全部同分组的 GRPO 优势自然为零。本配置不额外启用 DAPO 同分组过滤或无限补采；
因此代码接线正确不等于一定出现任务奖励提升，需要观察真实组内奖励差异。

## 4. 失败边界

- 候选格式/API/编译/数学错误、数值不正确、非法 PyTorch 回退、精度状态污染等属于模型失败，
  不改候选、不手工修复、不放宽为通过。允许模型在同轨迹后续回合自行修复。
- 参考输出无效、参考执行有限重试仍失败、真正零 token 等明确可恢复错误：训练整组丢弃并
  交给原生同步补采，绝不拿残缺组计算 GRPO。每批有限补采，连续无效不会无限循环。
- 验证参考无效则跳过、不换题；单独记录 requested/valid/skipped，模型本身答错仍计入分母。
- 候选执行/测时进程的SIGILL、SIGABRT、SIGFPE、SIGSEGV属于候选失败，保留退出码并反馈模型。
- OOM、可信参考/控制进程异常、容器启动失败、SIGKILL/SIGTERM、传输容量超限、磁盘空间不足及未分类程序异常仍停止并留证据。
  环境无效计时不会伪装成模型 -1。

共享控制器 `_measure_candidate()` 重试分支未定义 `measure` 的问题已改为显式 `measure=True`。
准备/初始性能分析的有限重试复用已有控制器，未重复实现另一套 CUDA 检查流程。

## 5. 文件职责与输出

| 文件                           | 职责                                                               |
| ------------------------------ | ------------------------------------------------------------------ |
| `configs/p3-agent-grpo.yaml` | 唯一手工维护的训练、工具预算、CUDA 评测配置                        |
| `scripts/p3-train.sh`        | 加载已有环境、北京时间和 LoRA 同步设置，调用 Python 入口           |
| `src/p3/train.py`            | 组合原生配置、CLI 覆盖、冻结一份 config.yaml、启动/续训            |
| `src/p3/data.py`             | 原始 JSONL 薄适配及精简消息构造                                    |
| `src/p3/models.py`           | Qwen 纯文本输入适配；模型/LoRA/分块输出头仍为原生实现              |
| `src/p3/agent.py`            | 原生 ToolAgentLoop 与真实 Workspace 的生命周期、终局分数及异常传播 |
| `src/p3/environment.py`      | 复用 P1 工具/容器/评测器，维护同源码缓存和最佳源码                 |
| `src/p3/training.py`         | 原生 ReplayBuffer 的整组失败边界及验证跳过分母；不重写 GRPO 更新   |

运行目录主要为：

```text
runs/p3/<run>/
  config.yaml                 # 唯一展开配置，同时提供工具 schema
  status.json                 # 运行状态/北京时间/失败原因
  tensorboard/                # 原生实时指标
  checkpoints/                # 原生 Actor、优化器和 dataloader checkpoint
  validation/step-*.json       # 验证有效数/跳过数，非重复指标大全
  scheduler/                  # GPU 调度锁及按组失败索引
  trajectories/<组UID>/<轨迹ID>/
    trajectory.jsonl          # 顺序保存 content、tool-call、tool-response
    result.json               # 终局奖励、最佳哈希及评测 ID
    failure.json              # 仅异常时产生
    best/                     # 与终局分数对应的冻结源码
    workspace/                # 实际交互现场
    kernel/                   # 编译、正确性、测时和参考失败证据
```

轨迹结束清理参考 `cases/input-spec/` 和 `compile-cache/`，保留结果、日志和候选源码。
共享 `.runtime/cache/p1-builds` 是跨任务的可信编译缓存，沿用现有有界缓存策略。
工具 schema 自动放在该 run 的 config.yaml 内，不新增第二份工具 YAML。

```bash
# 查看原生训练曲线；本机需已有 SSH 6006 转发或通过 VS Code 转发该端口。
.runtime/conda/learner/bin/tensorboard --logdir runs/p3 --host 127.0.0.1 --port 6006
```

## 6. 本次定向验证

- 原生配置组合/参数验证通过。
- 原生多轮 AgentLoop 边界验证通过：真实 Bash 工具往返、生成 token 原样保留、观察 mask=0、
  生成概率对齐、不传每请求 seed、致命工具异常不会被框架 catch-all 吞掉。
  证据：`.runtime/tmp/p3-native-20261112/validation.json`。
- 原评测卡定向 CUDA 验证通过：五组正确、正式测时、后测基线故障重试、同源码复用、
  真实 PyTorch 回退拒绝、最后候选失败时仍保存此前最佳源码。
  证据：`.runtime/tmp/p3-kernel-20261112-b/run/validation.json`。
- 上述为接线/协议验证，不是模型训练效果或完整 128K 容量测试。
- 原生 Dataset 样本数控制、保留官方 SKILL、部分失败组整体剔除、致命故障停止的定向检查通过。
  证据：`.runtime/tmp/p3-final-20261112/validation.json`。
- **以下为11月12日历史资源阻塞记录；11月13日实际运行已进入初始验证，随后候选崩溃导致停止，尚无GRPO更新。**
  2026-11-12 20:31:27 启动前复查，GPU 2/5 空闲分别约 24.9/42.9 GiB。
  当前 `gpu_memory_utilization: 0.45` 对 80GiB 卡的初始预算约 36GiB，GPU 2 不满足。
  当时没有换卡、终止其他进程、降低正式上下文或启动正式训练。
  证据：`.runtime/tmp/p3-wire-launch-20261112/launch.json`。
  原卡资源满足后，应先自行执行 `--wire`，确认真实梯度更新和 LoRA 权重同步；
  当前不能把配置/协议通过说成完整训练已验通。

## 7. 为什么仍有 agent.py、models.py、training.py

这些文件不是三套自写训练器。**GRPO不要求必须有这些文件名；必要的是把领域环境接入框架。**
纯原生GRPO可以直接使用verl，但原生框架不了解本项目的CUDA沙箱、五组验算、轨迹最佳源码和错误分类。

| 本地部分 | 原生框架负责什么 | 本地实际补什么 | 是否必须保留 |
| --- | --- | --- | --- |
| agent.py / WorkspaceTool | ToolAgentLoop生成状态机、工具解析、token连续拼接、观察mask、logprob | 把工具调用交给真实工作区；终局获得可信reward_score；生命周期/日志/致命错误传播 | **领域接线必须有**，可换文件组织；只配置tools名字不会自动创建本项目环境 |
| agent.py / 少量状态钩子 | 原生回合处理 | 总上下文剩余额度、训练/验证各30轮、submit停止、移除每请求seed、保留全部同轮调用 | 用户要求下的薄接线；没有另写生成/优势/反传算法 |
| models.py / TextModelConfig | HFModelConfig加载原始Qwen、tokenizer/processor；原生PEFT、FSDP、融合输出头 | super初始化后仅把processor=None；另导入agent以保证worker注册可见 | **当前Qwen3.5+V1/TQ纯文本路径的兼容措施**；不是GRPO理论必需，不新增网络层/价值头 |
| training.py / GroupReplayBuffer | 原生队列、同步采样、补采、训练批量物化 | 根据错误索引区分环境致命/可恢复；部分失败组也完整剔除；组大小校验；有限补采 | **保留当前失败语义时需要**；只打开原生sync_refill_failed_groups不足以保证部分失败组不被保留/补零 |
| training.py / CoverageTrainer | 原生验证、指标计算及整个fit/update循环 | 无效参考验证题从期望分母排除；全无效不伪造准确率；记录valid/skipped | 验证跳过的分母修正是当前约定所需；单独覆盖率JSON/显示属于可选观测工程 |
| training.py / TaskRunner | 原生初始化接口、AgentLoopManagerTQ、trainer.fit | 选择上述Trainer子类并在退出时关闭logger/TQ | 不是GRPO必需；为选用验证子类而保留的一小段启动接线，存在少量原生bootstrap重复 |

两处源码证据说明为什么不是“YAML里写一个值就完全替代”：

- `HFModelConfig.__post_init__` 在 `load_tokenizer=true` 时主动执行
  `self.processor=hf_processor(...)`，直接写 `processor: null` 会被再次加载覆盖；
  `return_multi_modal_inputs=false` 也不控制这一步。把 `load_tokenizer=false` 又会失去当前需要的tokenizer。
  因此此版本用21行配置子类统一切换纯文本路径，而非修改Qwen权重/模块。
- 原生 `ReplayBuffer._terminal_eviction_reasons()` 的同步补采只自动剔除**没有可物化轨迹**的失败组；
  一组四条中已有两条成功时，仅打开补采开关不等于整组剔除。本地子类补的正是这个边界。
  未分类异常/OOM不应被当成普通失败组无限补采。

另一个实际核查结论：当前`third_party/verl/verl/`中`freeze_vision_tower`仅在ActorConfig定义，
V1的`engine_workers.py`和FSDP `transformer_impl.py`没有读取/执行该字段的路径。
`all-linear`仍直接交给原生PEFT。因此原报告把`freeze_vision_tower=true`直接解释成已冻结视觉tower
过于肯定；纯文本任务通常不为视觉分支产生梯度，但这不能替代显式冻结。
本次只补准确说明，不新增自写冻结逻辑、不悄悄改变LoRA目标；正式训练前若要求显式排除视觉层，
应另行选择并验证原生PEFT的`exclude_modules`等实际生效方式。

可以精简的是日志/覆盖率文件、模块组织和TaskRunner启动重复；不能为了少几个文件而静默取消
已约定的失败处理，或把回退为通用工具反馈的环境故障当成模型负分。
当前仍有两处共享代码耦合：P3从`p1.run`只借`Pilot.feedback`，从`p2.reward`只借`P2Evaluator`。
它们适合以后放入共享helper，但**这不表示P3执行了P1/P2的训练流程**。本次没有扩大范围重构源码。

## 8. P3具体复用P0/P1/P2哪些文件

区分“运行时调用”“原有产物”和“仅被模块导入”。下表路径均相对项目根目录。

| 来源 | 文件/目录 | P3实际使用的内容 | 是否启动该阶段实验 |
| --- | --- | --- | --- |
| P0 | `scripts/set-env.sh` | 由p3-train.sh直接source；CUDA12.9、缓存/临时目录、编译工具链和库路径 | 否；其默认GPU4随后被P3入口按training_gpus覆盖 |
| P0 | `reports/p0/requirements.txt` | 环境安装依据；每次训练不读取/执行此文件 | 否 |
| P0产物 | `.runtime/conda/learner/` | verl、vLLM、torch、Transformers、FlashAttention等训练/rollout运行时 | 否，复用已装环境 |
| P0产物 | `.runtime/conda/kernel/`、`/usr/local/cuda-12.9`、已有Docker镜像 | 隔离worker里的Python/torch、nvcc、rg及系统工具；镜像主要提供隔离根文件系统/终端 | 否；镜像不是另一个PPO环境，也不含本次新下载模型 |
| P0产物 | `models/Qwen3.5-9B/`、`third_party/verl/` | 原始模型和原生训练框架源码 | 不读取P2训练权重 |
| P1数据产物 | `data/processed/train.jsonl`、`test.jsonl` | 直接作为训练/验证原始题目；不重跑data整理、不重划分 | 否；pilot.json默认不用 |
| P1 | `src/p1/tools.py` | TOOLS schema、WorkspaceTools的Read/Write/Edit/MultiEdit/Glob/Grep/Todo等 | 不调用P1模型生成 |
| P1 | `src/p1/sandbox.py` | SandboxSession、持久Bash、后台查询/终止、CPU→GPU请求通道及只读挂载 | 不启动P1 pilot |
| P1 | `src/p1/evaluation.py` | Evaluator、源码快照、编译/正确性/后测基线、reward、异常类型和dump | CUDA评测核心直接复用 |
| P1 | `src/p1/worker.py` | Docker内实际reference_prepare、编译、五组stream、timing和profile；运行期回退/TF32检查 | 只执行当前P3派发的固定worker模式 |
| P1 | `src/p1/contract.py` | 源码/导入/绑定/允许库静态审查；P3对未知helper交给运行期检查 | 不执行P1实验入口 |
| P1 | `src/p1/helper/pilot_parallel.py` | EvaluationPool、ParallelEvaluator、blocking_lock；评测卡共享/独占槽位 | 不调用其中的P1多进程生成调度 |
| P1 | `src/p1/helper/baseline_state.py` | 北京时间、原子JSON写入、轨迹结束清理input-spec/compile-cache等共享基础函数 | 不调用baseline.py |
| P1 | `src/p1/helper/build_cache.py` | 可信编译缓存标识、编译互斥和CPU编译槽 | 不复用模型输出答案 |
| P1 | `src/p1/helper/stream_transport.py` | 参考/候选输出传输、存储空间约束、CPU外部比较所需通道 | 不运行隐藏验收；P3只用feedback阶段 |
| P1 | `src/p1/helper/rewards.py` | set_reward、environment_failure等奖励有效性/环境故障判断 | 不调用带隐藏验收的P1/P2终局流程 |
| P1 | `src/p1/helper/workspace.py` | 文件路径、glob及链接/目录越界校验 | 工具基础支持 |
| P1 | `src/p1/data.py` | 仅借ROOT与sha等基础定义 | 不调用main、不再生成或过滤数据 |
| P1 | `src/p1/run.py` | 仅调用`Pilot.feedback`生成工具反馈摘要；没有构造Pilot实例 | 不用P1 single/agent生成循环 |
| P1 | `src/p1/agent_utils/compile.py`、`verification.py`、`profiling.py`、`run.py`、`client.py`、`__init__.py` | 拷贝进每轨迹utils；作为兼容命令/RPC客户端 | 不直接执行官方不隔离的评测脚本 |
| P2 | `src/p2/reward.py` | 仅继承P2Evaluator的参考准备/初始profile有限重试 | 不调用evaluate_response或P2奖励解析器，不跑隐藏验收 |
| P2 | `src/p2/failures.py` | 仅复用RecoverableSampleError、NoValidValidationSamples异常类型 | P3有自己的按组记录/剔除逻辑，不套用单样本PPO限制 |
| P2 | `src/p2/common.py` | 由上述P2模块间接导入，提供ROOT/时间/JSON等定义 | 不进入P2训练入口 |
| 官方仓库 | `upstream/CUDA-Agent/agent_workdir/SKILL.md`、`binding.cpp`、`binding_registry.h` | 原样作为提示/只读基础设施；utils采用上表的本地隔离包装 | 不从官方目录启动另一个训练器 |

`p1.run`/`pilot_parallel`在导入时还会带入`helper/inference.py`、`progress.py`、
`turn_log.py`、`tool_calls.py`等模块；这属于现有模块的静态导入耦合，**P3不调用它们的
生成服务、P1进度汇总、单次文件解析或轨迹写入流程**。P3轨迹日志由p3/environment.py写入。
`src/p1/baseline.py`、`status.py`，`scripts/p1-*.sh`、`scripts/p2-train.sh`，
P1/P2配置和历史runs/checkpoint都不是P3正式运行的前置步骤。

## 10. 2026-11-13：失败边界及吞吐精简

`runs/p3/agent-grpo`在训练前验证中停止。候选`ops6k-acb55914434b1737`的C++代码
在CPU直接解引用CUDA指针，候选进程退出139；参考五组执行成功。旧分类器将所有
`WorkerProcessError`当作环境故障，导致整个运行停止。候选源码保留原样，不人工修正。

本次实现：

- 候选执行/测时中的SIGILL/ABRT/FPE/SEGV记为模型失败；参考/控制器异常、OOM、
  Docker启动失败、SIGKILL/SIGTERM等继续停止，不把环境问题伪装成模型负分。
- `fail_fast: true`：首个数值/结构/运行失败即通知两端worker停止。只改变失败后的工作量；
  任何正确候选仍须完整通过五组，不减少比较元素，不改变0.01容差。
- 快速比较先做同容差的allclose，失败后仅计算首个失败块的误差。通过时`max_abs/max_rel`
  为null，`diagnostic_scope=not_computed`；失败的误差不是整份输出的全局最大值。
- 参考预检使用共享槽位。`timing_exclusive: false`使初始eager/compile分析、候选测时及
  候选后的基线也使用共享槽位；同一候选仍先测候选、后测基线，保持1×10和相同奖励门槛。
  每卡最多4作业，两个卡合计最多8；满额仍会等待。共享测时可能受其他作业干扰，不用于声称
  与论文独占测时完全等价，也不能和旧独占结果直接归因比较。
- `scheduler/reference-profiles/`只缓存同一次运行内相同题目、环境与配置的初始参考分析摘要，
  多轨迹及后续验证点复用。没有持久缓存大输出/参考张量，不复用候选后的奖励基线；
  每轨迹输入/参数和参考编译缓存依旧结束清理。初始分析仍包括eager和compile。
- 工具说明明确evaluate一次即可完成编译、五组验算及测时；不强制模型先单独compile。
- 总上下文从128K改32K，训练150/验证200轮改为各30轮。原始官方SKILL内容不修改；
  达到预算后仍按已有规则保留本轨迹已正确测时的最佳源码，无合格候选为-1。

### 吞吐日志与TensorBoard

`actor_rollout_ref.rollout.disable_log_stats: false`直接传给原生vLLM；
Ray的`runtime_env.env_vars.VLLM_LOGGING_LEVEL: INFO`保证INFO级吞吐日志不被隐藏。
vLLM周期日志显示prompt/generation token/s、running/waiting请求、KV缓存等；
这些是引擎实时统计，不会因为开启此开关就自动成为TensorBoard曲线。
日志出现在启动终端或该run对应Ray目录的worker日志。

```bash
# 可选：由用户启动新运行并保存终端日志；pipefail保留训练失败的退出状态。
set -o pipefail
# 新目录避免把旧128K/独占协议与当前32K/共享协议混在一起。
bash scripts/p3-train.sh --run runs/p3/agent-grpo-fast 2>&1 | tee /tmp/cuda-agent-p3-fast.log
```

TensorBoard继续使用verl原生的训练loss、奖励、长度、timing和performance指标；
其中`perf/throughput`是每GPU的整步token吞吐，包含rollout、工具和更新等阶段，
不是解码引擎的纯generation token/s。训练步完成后才汇总对应step数据，验证指标也需该轮验证完成后落盘。它不是vLLM周期
吞吐日志的实时镜像。本次未增加自写监控线程、Prometheus采集或第二套指标落盘。

本次不启动训练、不重放GPU；只做配置/语法检查及已有故障记录的定向回放。
变更不会修改旧run的`config.yaml`、候选源码或结果。修复后的实际吞吐和共享测时噪声
需要下一次由用户启动的运行才能确认，不能由静态检查承诺加速倍数。

定向CPU记录回放已通过：旧139崩溃归候选、OOM/容器启动仍致命；首组错误双方各执行
一组便停，正确夹具仍执行五组；末块错误与NaN/Inf拒绝、初始摘要缓存及配置展开通过。
未创建CUDA上下文。证据：`.runtime/tmp/p3-fast-feedback-20261113/validation.json`。

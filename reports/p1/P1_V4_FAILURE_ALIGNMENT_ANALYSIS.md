# P1 v4 失败样例与原论文实现对照分析

审查日期：2026-11-01，北京时间。正式运行：`runs/p1/run-pilot-12-v4`。

更新说明（2026-11-02）：下文记录的是 v4 冻结实现及结果，不代表当前代码仍有同样接口缺口。
本次接口修复见 P1_REPORT.md 第 16 节。single 格式错误仍算正式失败，不自动补括号、
恢复答案或追加生成；文中离线补括号仅用于定位旧失败，不用于计分。

## 1. 判断先行

目前的结果不能证明“只要不做 RL，single 和 agent 就理应这么差”，也不能证明“失败全部来自没有照抄官方工具”。现有证据支持三个同时成立的判断：

1. **本地已经实现论文 Agent 工作流的主要骨架**：先分析原始性能、多文件 CUDA 工程、持久 Bash、编译/验证/性能反馈、轨迹内选择最快正确候选。不是完全没有对齐。
2. **仍有确定的接口损失与比较条件差异**：single 的 JSON 交付方式额外增加失败入口；Agent 会静默丢掉同一轮中第二个及之后的工具调用；沙箱与文件工具的绝对路径空间不一致；两种模式获得的领域说明不完全相同。尚不能称为严格、无损的论文基线实现。
3. **模型确实生成了严重错误的 CUDA 算法**：Agent 六题在已经成功编译和执行之后，出现索引覆盖不全、错误归约维度、缺失注意力计算等问题，另一题返回结构不完整。把接口补齐不会自动把这些源码变成正确实现。当前模型还没有经过论文中的 CUDA 专项训练，而论文主模型及 single-turn 对照都涉及专项训练，不能直接等价比较。

最恰当的定位是：**Qwen3.5-9B 官方权重在本地论文式 Agent 环境中的未专项训练基线，同时包含尚待修复的接口问题**。它不是论文 CUDA Agent 训练后成绩的复现，也还不是一个已消除接口干扰的 Qwen 能力测量。

本次只提交已有核心修改并做只读审查、源码分析和离线 JSON 解析；没有重跑 GPU 实验，没有修改 v4 正式结果，也没有把人工读代码发现的修复计为模型成功。

## 2. 证据范围与版本

### 2.1 本次重新阅读的原始材料

- 用户提供的原始 PDF，共 32 页：`D:/zotero/store/storage/IAQITCQ3/Dai 等 - CUDA Agent Large-Scale Agentic RL for High-Performance CUDA Kernel Generation.pdf`。
- PDF SHA-256：`cdeebc53efb4a25d74b7e9b428e5292fcccd8626aa283c0d8aa9277f516995f9`。本地 `reading/source_bundle.json` 与该 PDF 的哈希一致；本次重新逐页阅读正文及附录文本，并核看 Table 1、Table 2 的页面图像。
- 服务器当前 `upstream/CUDA-Agent/agent_workdir`：`SKILL.md`、固定绑定、编译/验证/测时工具及官方样例；`REVISION` 为 `473025c8af7878e928525138b5cb327d6bcdf9dd`。同时核对[官方仓库](https://github.com/BytedTsinghua-SIA/CUDA-Agent)和[论文在线 PDF](https://cuda-agent.github.io/static/pdf/CUDA_Agent_Arxiv_Version.pdf)。
- 当前 `src/p1`、`configs/p1-pilot.json`，以及 v4 的 manifest、summary、全部轨迹轮次日志、评测结果和候选 CUDA 源码。

下文页码指 PDF 页码。论文、公开样例代码、我们自行实现的控制器是三种不同证据；不能把公开样例脚本当作作者完整的内部训练系统。

### 2.2 代码与运行能否对应

v4 manifest 记录旧 HEAD `05cf338`，运行时还带有当时未提交的修改。不能仅凭这个旧 HEAD 判定本次提交的代码未用于 v4。实际核对 manifest 的 **21 个源码 SHA-256，全部与本次审查的源码快照一致**。

当前核心修改已用一个 commit 提交：

```text
ec1f96f fix(p1): 调整隐藏验收数量并设置独立超时预算
```

涉及 `configs/p1-pilot.json`、`src/p1/evaluation.py`、`reports/p1/P1_REPORT.md`、`reports/p1/README.md`。提交内容是已有的隐藏比较数量、独立超时预算、GPU 配置及相关说明，不包含本报告提出但尚未实施的接口修复。未 push。

## 3. 先明确：究竟在和论文中的什么比较

### 3.1 当前 single/agent 的定义

当前两个模式使用同一份 Qwen3.5-9B 官方 checkpoint、BF16 推理：

- **single**：一次模型生成，输出包含完整文件内容的 JSON，不获得执行反馈，然后由控制器验收。
- **agent**：模型通过工具读写文件、编译、验证和测时，反复修改；从轨迹里选择正确且测时有效、相对 compile 最快的候选，再做隐藏验收。

所以当前实验主要比较的是“同一模型，在两种本地交互流程下表现如何”。它还混合了提示词信息量、交付格式和执行预算的差异，不是严格的只开关工具的消融。

### 3.2 论文 single-turn 不是直接拿未经 CUDA 专项训练的模型生成一次

论文 §3.3、Figure 3（第 5—7 页）包含：

1. 基模型先做 single-turn PPO warm-up；
2. 用该模型采样 Agent 轨迹；
3. 依据结果与行为筛选轨迹，进行 actor 的 RFT；
4. 预训练 critic 的 value；
5. 再进行 Agentic PPO。

§4.3.1（第 9 页）明确描述两个分别训练的模型：single-turn 模型和完整训练流水线的 CUDA Agent。Table 2 中 w/o Agent Loop 的 Pass Rate 为 **77.1%**，相对 compile 的 Faster Rate 为 **14.1%**，正确解上的几何平均加速比为 **0.69×**；完整 CUDA Agent 对应 **98.8%、96.8%、2.11×**。

这里还应保留一个论文自身的披露边界：Table 2 表题笼统写“under agent loop evaluation”，§4.3.1 又讨论 single-turn、无执行反馈的训练协议；公开仓库没有给出这项消融的完整评测入口。因此不能据此恢复所有推理细节，更不能把 Table 2 说成“同一原始权重只切换 single/agent 推理开关”的受控实验。**它涉及不同训练后的模型，这一点是明确的。**

### 3.3 主模型与评测集也不同

论文 §4.1（第 8 页）使用 Seed1.6，MoE 总参数 230B、激活 23B；当前是用户指定的 Qwen3.5-9B。二者架构、规模、预训练与后训练经历都不同，不能把差距单独归因于参数量。

论文最终评测是 KernelBench Level 1/2/3 共 250 题；当前是已发布 Ops-6K 的 train pilot，12 题选入、9 题参考有效。Ops-6K 是论文训练数据分布，不是这 250 道正式评测题。当前 8:2 train/test 与 train pilot 是按用户要求采用的本地实验设计。

论文 Table 1 中未经过该专项训练的 Seed1.6，在相同 Agent loop 中也有 74.0% Pass Rate，说明论文并没有证明“未经该 RL 的所有模型必然接近零通过”。但这一数字同样不能直接预测 Qwen9B 在这 9 题上的表现。

## 4. v4 实际结果与分母

运行时间为 2026-10-31 14:37:16 至 22:37:45（北京时间），墙钟约 8 小时。

| 指标 | single | agent |
|---|---:|---:|
| 参考有效、实际生成的题数 | 9 | 9 |
| 正确题数 | 0 | 1 |
| 参考有效题上的 Pass Rate | 0% | 11.11% |
| 全部 12 道选入题上的正确比例 | 0% | 8.33% |
| 正确且快于 eager 的题数 | 0 | 0 |
| 正确且快于 compile 的题数 | 0 | 0 |
| 正确解几何平均 speedup_eager | 无正确解，未定义 | 0.8604× |
| 正确解几何平均 speedup_compile | 无正确解，未定义 | 0.8843× |
| 生成 token | 48,909 | 950,575 |
| 工具调用数（实际处理） | 0 | 627 |
| 搜索阶段评测数 | 0 | 118 |

两种模式合计生成 999,484 token。Agent 几何平均只来自一题，不能解释成整个任务分布的平均优化能力；single 没有正确解，其加速比不能写为 0 或 1 来代替未定义值。

3 道参考无效题分别是两个近常数采样输出，以及一题重复运行的 median 索引不同。它们没有进入模型能力的 9 题分母。近常数只是有限抽样发现，不能据此断言整个公开数据集有同样问题，更不代表重新做了 KernelBench 去污染。

## 5. single：9 题逐项失败链

下列编号为 `evaluations/<编号>`；完整 task ID 见第 13 节。

| 题目 | 编号 | 第一个阻断点 | 能从现有证据确定什么 |
|---|---|---|---|
| SiLU + MultiheadAttention | 00002 | 输出 JSON 不完整，文件没有落盘 | 不是已执行 CUDA 的数值错误；最终 missing model_new.py 是后果 |
| hardsigmoid + ReLU | 00001 | CUDA 编译错误，末尾语法不完整 | 已交付文件；尚未达到数值执行阶段 |
| BatchNorm2d + 全局 min | 00078 | ATen/native 访问及 self.batch_norm 计算回退 | 存在真实计算回退，不能只归为静态检查误报 |
| ReflectionPad2d + log2 | 00129 | 输出 JSON 不完整 | 与第一题相同，不能从缺文件推断 kernel 正确与否 |
| GroupNorm + log_softmax | 00133 | kernels 导入方式、self.group_norm 回退等 | 导入形式可以工程化兼容，但其中确有 PyTorch 计算回退 |
| ConvTranspose2d 组合 | 00156 | 未定义 idx_total、无效 cuDNN handle API | 编译/接口知识错误，尚未比较数值 |
| BrosBboxEmbeddings | 00167 | 输出 JSON 不完整 | 序列化失败发生在执行前 |
| bilinear interpolate + tanh | 00221 | root binding.cpp 与 kernels/binding.cpp 的 binding.o 冲突 | 生成的文件命名与 JIT 构建不兼容；不是数值错误 |
| Embedding + log_softmax | 00254 | 输出 JSON 不完整 | 同时还生成了禁止改写的固定文件，修 JSON 后仍有合同问题 |

按第一个阻断点统计：**4 道格式失败、2 道合同失败、3 道编译失败**。single 没有一道走到可判断 CUDA 数值正确性的阶段。所以 0/9 是当前端到端交付成功率，不能单独解释成 CUDA 算法正确率为零。

### 5.1 四道格式失败到底错在哪里

本次检查原始 `single/turn-000.json` 后，四道都已输出 `{"files":{...` 外壳；严格 JSON 解析失败。只做文本层面的离线探查，在答案末尾补 **一个 `}`**，四道都能通过 `json.loads`，得到文件字典。

这修正了此前“部分没按要求包装 files”的不准确判断。真正证据是**外层 JSON 少闭合一个花括号**。模型在长 JSON 字符串中承载整份 Python/CUDA/C++ 文件，是本地增加的交付要求；论文没有披露相同的序列化方案。

但不能把这四道改记为正确：

- 本次没有执行补括号后的源码；
- Embedding 题还包含 binding_registry.h、model.py、utils/、setup.py 等不允许覆盖的路径；
- 部分恢复文本含有可疑导入或回退，后续仍可能编译失败、违反合同或数值错误。

可以确定的是“有 4 个交付在序列化层面丢失”，无法确定“修序列化会增加几道正确题”。后续应使用结构化输出或明确、可审计的文件解析协议，且把格式校验与 CUDA 正确性分开统计。

### 5.2 binding.o 冲突是否是我们没对齐官方造成的

本地 `worker.py:299` 收集固定 root/binding.cpp 和 kernels/*.cu、*.cpp，交给 `torch.utils.cpp_extension.load`。官方 `utils/compile.py:11` 也会收集根目录和 kernels 的同类文件，再交给同一扩展构建器。

因此两个目录都叫 binding.cpp 时产生同名对象文件的风险，**公开编译方式也存在**。这是一项应改进的工程兼容性问题和生成文件布局问题，不能作为“只有本地偏离官方才失败”的证据。原样复制公开编译工具并不能保证消除这一失败。

## 6. agent：9 题逐项结果

| 题目 | 最终/入选编号 | 已跨过的阶段 | 最终问题 |
|---|---|---|---|
| SiLU + MultiheadAttention | 00075 | 编译、运行 | 注意力主计算被不完整替代，输出覆盖不足；数值不符 |
| hardsigmoid + ReLU | 00028；隐藏 00132 | 编译、5 次反馈、34 次隐藏比较、测时审计 | 正确但比 eager/compile 慢 |
| BatchNorm2d + 全局 min | 00128 | 编译、运行 | 忽略 block 索引、归约与全局写回错误 |
| ReflectionPad2d + log2 | 00155 | 原始性能分析和读取文件 | 第 4 次生成耗尽上下文，没有写出 model_new.py |
| GroupNorm + log_softmax | 00162 | 编译、运行 | N/C 索引覆盖不全，softmax 归约维度也错误 |
| ConvTranspose2d 组合 | 00217 | 编译、运行 | 应返回 max 的 values 和 indices，实际只返回一个 Tensor |
| BrosBboxEmbeddings | 00253 | 编译、运行，已越过早先环境/权重兼容问题 | 位置编码频率和输出覆盖错误 |
| bilinear interpolate + tanh | 00271 | 编译、运行 | 双线性插值索引覆盖和坐标规则错误，tanh 也忽略 block 索引 |
| Embedding + log_softmax | 00272 | 编译、运行 | 把 dim=1 的序列归约写成 embedding 维度归约 |

失败 8 题中：**6 道数值语义错误、1 道返回结构错误、1 道没有候选文件**。没有一道是“反馈正确，最后仅被隐藏比较淘汰”。

### 6.1 注意力：不是数值精度小误差

冻结候选 `evaluations/00075/candidate/kernels/attention_kernel.cu`：

- 43 行附近仅遍历 N × num_heads × S，而输出为 N × S × E。E=128、heads=4，仅对应应有元素数的 1/32。
- 51—56 行没有实现完整 Q/K/V 投影及每个 head 的向量内积。
- 58—75 行得到 max_score 后，直接把它当作最终输出；缺少 softmax、对 V 的加权和及正确输出投影。

首个反馈样例约 92.9% 元素不符，最大绝对误差约 5.45。不是把 atol 从 0.01 稍微放宽就能接受的实现。验证器报错符合源码语义，主要直接问题来自生成算法。

### 6.2 BatchNorm + min：很多 block 在重复处理同一段数据

`evaluations/00128/candidate/kernels/fused_bn_min.cu`：

- 19 行定义 block 索引但没有在主循环起点使用；30 行从 threadIdx.x 开始，步长却是 blockDim.x × gridDim.x。
- 当前元素数恰好与整个 grid 的线程总数对应，实际各 block 重复处理开头少量元素，大量数据没被纳入结果。
- 57—61 行 warp shuffle 的掩码/源 lane 选择错误。
- 76—81 行各 block 写同一个 output[0]，没有正确的跨 block 全局 min 合并。

首样例参考值约 -5.56559，候选约 -2.97936。模型还选择了 submit，说明“模型认为写好了”与“控制器确认正确”有明确区别。这个错误不应通过允许回退来掩盖。

### 6.3 ReflectionPad：尚未写代码就被超长推理耗尽

该 Agent 只执行了 3 个实际工具调用：原始性能分析、读取 model.py、读取 binding.cpp。随后 `turn-003.json` 单次生成 **123,487 token，约 86 分钟**，没有完成 thinking，也没有工具调用或候选文件。

它不能用于证明 ReflectionPad CUDA 算法本身写不对，因为尚未交付算法。这里需要分别考虑：模型的推理收敛能力、每次生成预算、工具调用协议，以及尚未做专项行为训练。不能仅以“上下文给足了”判断交互设置合理。

### 6.4 GroupNorm：归一化与 log_softmax 都有直接语义错误

`evaluations/00162/candidate/kernels/group_norm.cu`：全局线程索引与 C 比较后直接返回，仅少量线程参与，未正确覆盖 N=512；通道/组/批次偏移也没有正确组合。

同目录 `log_softmax.cu` 的归约围绕 H × W 展开，未按参考 `torch.log_softmax(x, dim=1)` 的通道轴计算，且 batch 偏移使用不完整。首个比较 100% 元素不符，最大绝对误差约 226。

此外 model_new.py 中另建 gamma/beta 而不使用应加载的对应参数，也是潜在隐藏权重问题；但无需依赖这个后续问题，当前直接反馈已证明主计算错误。

### 6.5 反卷积组合：把返回值的第二部分漏了

应按原始 forward 而不是数据集 ops 文本标签判题：这道实际程序执行反卷积、额外 bias、clamp、GELU，最后 `torch.max(x, dim=1)`。

参考返回 max 的 values 与 indices；`evaluations/00217/candidate/model_new.py` 仅返回融合扩展产生的一个 Tensor。评测报 `output type differs`。这不是验证器擅自要求多一个输出，而是参考算子本来如此。

未比较完结构前，不能据此宣称 values 数值已正确，也不能把漏 indices 归为浮点精度。候选代码还有其他可疑点，但无需把未触发的潜在错误写成此次直接根因。

### 6.6 Bros：环境问题修复之后，剩下的是位置编码本身写错

`evaluations/00253/candidate/kernels/fused_2d_pos_embed.cu:25` 附近把参考的多坐标、多频率 sin/cos 位置编码大幅简化：使用 input[pos]，只计算单一频率，并在每组应有的 32 个分量中写极少数位置，其余未完整填充。

参考则对 8 个 bbox 坐标，按对应 x/y 频率向量生成完整 sin/cos 表示，再做线性投影。首个反馈约 98% 不匹配。当前已成功编译并执行，所以不能继续用早期 transformers 未安装、配置反序列化或 state_dict 问题解释 v4 的该项失败。

### 6.7 上采样：二维坐标被写成对角线，tanh 又漏了大部分元素

`evaluations/00271/candidate/kernels/bilinear_upsample.cu:25` 附近，同一个一维 tid 同时用作 oy、ox 的循环起点；当前 outH=outW=1024、循环步长对应这些维度，使大量线程只处理 oy=ox 的位置，输出平面未完整覆盖。

该实现还直接使用 iy=oy/2、ix=ox/2，没有正确对应默认 align_corners=False 的半像素坐标变换。邻点边界处理也不完整。

同目录 `tanh.cu:13` 又只从 threadIdx.x 出发，忽略 blockIdx.x；大量 block 重复计算前 256 个元素。首个反馈约 98.7% 不符。这是索引与插值定义错误，不能靠更多重复比较修好。

### 6.8 Embedding：正确读出了题目，仍把 softmax 维度写错

参考输入 [32,1024]，Embedding 后为 [32,1024,512]，随后 dim=1，即沿长度 1024 的序列维度计算 log_softmax。

`evaluations/00272/candidate/kernels/softmax_kernel.cu:5` 的注释与 31—47 行循环都明确在 embedding_dim=512 上归约。这变成了 dim=2 的另一道题。首个样例 100% 元素不匹配，最大绝对误差约 1.16。

本题日志也显示，模型已经读取原始代码和性能反馈；不能把失败简单解释成“工具没有告诉它题目是什么”。

### 6.9 唯一通过题：正确不等于有优化收益

入选候选是 00028，隐藏验收 00132 的 **34/34 比较通过**，有真实 CUDA 源码，没有 PyTorch 计算回退；TF32 前后均为 true，没有发现该候选污染全局精度状态或用输入缓存冒充计算。

候选主要是独立的 hardsigmoid 和 ReLU 标量逐元素 kernel。注释提及每线程多个元素，不代表实际代码实现了 float4 向量化或有效的融合优化。最终 speedup_eager=0.8604、speedup_compile=0.8843，均小于 1。

可以据此说“生成了正确但尚无性能收益的实现”，不能仅凭源码直接量化向量化或融合能提高多少；需单独 profile/消融验证。论文第 9 页也讨论普通模型的朴素 kernel 经常无法超过 compile，但这只是机制相符，不能当作本题性能瓶颈已被实验定位。

## 7. 已确认的本地接口问题

### 7.1 一轮多个工具调用被静默丢弃

`src/p1/run.py:97` 的 parse 使用 re.search，只取第一个完整 tool_call；`run.py:494` 起的控制器也只保存并处理一个 action。

本次只统计 `</think>` 之后的完整工具块，排除思考中的伪工具文本：**12 轮含多个完整调用，后续 24 个没有执行**。多数是 Read/Glob；例如注意力 turn-000 同时请求性能分析、读取三个文件，实际只处理性能分析。

这不是 GPU 错误，也不是模型完全没输出合法调用，而是接口没有完整消费输出。后续调用也没有进入模型历史的 tool_calls 列表，模型看不到明确的“调用未执行”解释。

应该支持完整调用序列与逐个返回，或明确限制单调用并给出可恢复反馈；不能静默忽略。本次未修改实现、未重跑，不能推断补齐后可多通过几题。由于被漏掉的主要是读文件，也不能宣称“正确代码都被漏执行了”。

### 7.2 /workspace 的路径语义不一致

Bash 容器工作目录是 /workspace，模型可以从 shell 看到这个绝对路径；`tools.py:112` 的 Read 路径解析却把绝对路径当成宿主机绝对路径，再做越界检查。

上采样题出现 Read('/workspace/model.py') 被拒绝。越界保护本身有必要，问题是同一工作区的命名空间没有统一。应在保留边界检查的前提下显式映射容器工作区路径。该题之后仍继续执行，不能把一次可恢复读失败当作所有数值错误的原因。

### 7.3 single 的交付协议与信息量不对称

`run.py:258` 的 Agent prompt 注入完整官方 SKILL；single 只获得本地 CONTRACT、参考代码、固定绑定和 JSON 交付要求，没有同等完整 SKILL。

因此 single/agent 差异不仅是能不能运行工具，还包括领域指南、工作流示范、JSON 文件序列化负担及可用总预算。建议后续保留两者相同的正确性/绑定/允许库/领域规范，单独开关执行反馈；论文没有披露完整 single prompt，无法宣称当前这个自拟 JSON prompt 与其完全一致。

## 8. 长时间消耗在哪里，是否是隐藏验收造成

8/9 条 Agent 以 context_exhausted 结束；其中也包括已找到正确历史候选的 hardsigmoid，控制器仍能选取其历史最佳，所以耗尽上下文不必然等于最终错误。

| 题目简称 | 最后一轮生成 token | 该次生成秒数 | 是否完成 thinking |
|---|---:|---:|---|
| Attention | 452 | 36 | 否 |
| hardsigmoid | 59,949 | 2,408 | 否 |
| ReflectionPad | 123,487 | 5,163 | 否 |
| GroupNorm | 74,953 | 2,990 | 否 |
| ConvTranspose | 59,663 | 2,488 | 否 |
| Bros | 29,981 | 1,196 | 否 |
| Upsample | 85,559 | 3,471 | 否 |
| Embedding | 92,459 | 3,598 | 否 |

这些未完成的最终生成合计 526,503 token、21,350 秒，约占 Agent 生成 token 的 **55.4%**、模型生成时间的 **51.9%**。这是多个工作进程的累计资源时间，不能与 8 小时墙钟直接相加。

`generation_tokens=null` 使 `run.py:133` 每次允许用完整剩余上下文；`trajectory_seconds=null` 也没有独立墙钟限制。200 轮上限不能阻止某一轮生成 12 万 token。论文公布的是上下文窗口和最大交互轮数，**没有公布每次生成的 max_tokens**；因此“允许一次耗尽 128K”只是本地选择，不是必须照论文设置的结论。也没有证据说应强行改回 4096 并称为论文原值。

当前 sampling 只显式设置 temperature=0.7、top_p=0.95、max_tokens、seed。Qwen 官方模型卡对 thinking 编码给出的建议为 temperature=0.6、top_p=0.95、top_k=20 等；当前没有完整按该建议显式设置。这适合作为后续受控变量，但论文未披露这些采样参数，不能断言某个温度就“对齐原文”，也不能断言这是长推理的唯一原因。[Qwen 官方模型卡](https://huggingface.co/Qwen/Qwen3.5-9B)

隐藏验收方面：失败的 8 条 Agent 都没有获得可入选的反馈正确候选，未进入隐藏验收。唯一进入者 34/34 通过。故 **v4 的低 Pass Rate 不是额外隐藏验收淘汰了本来正确的 8 题**。

隐藏验收确实有成本：通过题参考与候选两个独立进程各自约 360 秒；CPU 比较、大张量传输、重复/扰动样例会增加成本。搜索阶段还累计 118 次评测、全运行累计 256 次编译。降低成本应分别处理生成、编译、反馈和隐藏阶段，而不是把 8 小时都归于最后的隐藏检查。

## 9. 与论文和官方代码的逐项对齐表

“对齐”指目标或行为一致；“工程改造”指采用不同实现以达成相同目标；“不同/未复现”指实验条件或算法缺失；“未知”指原文与公开源码不足以确定。

| 项目 | 原论文/公开代码 | 当前 P1 | 判定及与失败的关系 |
|---|---|---|---|
| 任务来源 | 已过滤、去污染 Ops-6K，用于训练 | 同一公开数据集 revision，train pilot | 来源一致；没有必要重复做去污染 |
| 数据再过滤 | 论文构建数据时筛确定性、非常数、可运行 | 仅对当前环境做重复输出/近常数运行审计；相似度审计不再删除数据 | 本地有效性防线，不等价重新去污染；3 题被隔离需单列 |
| split | 论文数据构造与其内部训练评估流程 | 用户指定普通确定性 8:2 train/test，pilot 来自 train | 合理本地设计，非原文逐项复现 |
| 最终评测 | KernelBench 250 题，L1/L2/L3 | Ops-6K 9 道有效 train pilot | 不同；不可以直接横比论文通过率 |
| 基模型 | Seed1.6，230B 总/23B 激活 MoE | Qwen3.5-9B 官方 checkpoint | 用户指定的替换，重大模型差异 |
| 权重精度 | 论文讨论训练/推理精度一致性 | 官方权重以 BF16 推理，非另一个低比特量化实验 | BF16 本身不足以解释当前语义错误 |
| 专项训练 | single PPO→轨迹→RFT/value pretrain→Agent PPO | P1 no_training=true，尚未实施上述专项训练 | 核心学习过程未复现；不应期待提示词复制后达到训练后成绩 |
| actor/critic | PPO 学习与稳定性机制 | P1 推理不用 critic | 不属于推理缺工具；不能把当前长思考说成 critic 未控制结束 |
| context | single RL 32768、Agent RL 131072 | 对应推理窗口 32768/131072 | 数字沿用；原文主要是训练窗口披露，不证明完整推理预算相同 |
| 轮数 | 训练 150、评测 200 | Agent 最多 200 | 评测上限对齐；没有每次输出长度保证 |
| 单次输出限制 | 未公开精确值 | 剩余窗口全部可用 | 本地选择，存在已观测的超长未完成生成 |
| 采样温度/停止细节 | 未公开完整参数 | 0.7/0.95，原生 chat template，thinking 开启 | 无法声称精确对齐；需按 Qwen 做受控验证 |
| Agent 架构 | 单个 ReAct agent，OpenHands 风格 | 单模型推理、动作、观察循环，自写控制器 | 骨架对齐；并非必须采用相同软件包名 |
| CUDA SKILL | 原文附录完整规范 | Agent 注入官方 SKILL，加本地环境覆盖 | 主体对齐；single 未获得等同内容 |
| 原始性能分析 | §3.2 第一步先分析原始性能 | 9 条 Agent 均在首次写候选前分析；Embedding 通过 Bash reference-only | 已实现并实际使用，不是当前未对齐项 |
| 持久 Bash | shell 的完整会话状态 | 常驻 shell，保留 cwd/export/functions/tmp 状态 | 行为已对齐，不是每次新 shell |
| Read/Write/Edit/MultiEdit | 文件读写、先读后写、批量修改 | 已实现，边界限制与原子写入 | 基本对齐；绝对路径有明确兼容缺口 |
| Glob/Grep | 本地检索，Grep 基于 ripgrep | Glob 与 Python regex 检索 | 常用语义可用，选项/regex 不完全同一实现；未证实是本次数值错误根因 |
| TodoWrite | 任务列表辅助 | 已提供 | 对齐辅助能力，不能代替正确性判断 |
| NotebookEdit/BashOutput/KillBash | 附录工具表列出 | 未完整提供 notebook/异步后台管理工具 | 功能缺项，但本次 CUDA 工作流没有相应调用失败证据；优先级低于已丢调用问题 |
| 多工具输出消费 | 完整内部调度器未开源 | 只取第一个，忽略其余 | 确认本地缺陷；12 轮、24 个漏执行调用 |
| CPU/GPU 分离 | Docker terminal，专用 GPU pool，128 H20 | CPU Bash + RPC 到 GPU 评测进程，A100；生成 [6,7]、评测 [4,5] | 架构目标对齐，资源与硬件不同 |
| 自定义运行调参 | SKILL 示例可以用 GPU 运行调参脚本而不重编译 | Bash 无 GPU，只能经固定 utils/RPC 动作访问 | 接口灵活性少一些，不能称全工具等价；当前没证据表明这是六个数学错误的直接原因 |
| 设备/依赖 | 原文未完整锁定其内部软件镜像 | sm80、CUDA12.9、Torch2.11 等本地栈，显式 cuDNN/cuBLAS 链接 | 硬件适配必要；影响 API 和性能可比性，不是数学定义变更理由 |
| 编译目标 | root + kernels，多文件扩展，O3/fast-math | 固定 root 绑定 + kernels，基本相同参数，独立 build | 大体一致；同 basename 冲突也是公开工具潜在问题 |
| 可写范围 | 固定绑定、验证器、原始模型受保护 | 只允许 ModelNew 与 kernels 源文件，固定文件只读 | 与防 hacking 目标一致，路径规则更显式 |
| 计算约束 | SKILL 要求 raw CUDA；Python 仅创建/属性；GEMM/Conv 允许指定库 | 禁 PyTorch 计算回退，允许 cuBLAS GEMM/cuDNN Conv | 按已商定本地规则执行；不应复制公开示例中的漏洞以提高通过率 |
| 静态检查 | 完整内部实现未披露，公开验证器主要禁止 F | AST/符号审计 + 动态事件回退检查 | 比公开脚本更广；可有形式误报，但 v4 两条 single 均另有真实回退 |
| 状态加载 | strict state_dict，保证权重一致 | strict 加载，多初始化种子 | 目标对齐；多种子是更强验收 |
| 反馈正确性 | 5 个随机输入，atol/rtol 1e-2 | 5 个固定可复现随机输入，容差相同 | 数量/容差对齐，种子可见性与随机策略未完全相同 |
| 隐藏验收 | 未披露本地这套额外流程 | 2 组初始化×5 输入×原始/重复/扰动+4探针=34 比较，不反馈模型 | 本地加强；v4 没有因此额外淘汰候选 |
| 输出结构 | 公共验证器递归检查类型/数值 | CPU 递归比较 Tensor、tuple/list 等结构 | 目标一致；漏 indices 理应失败 |
| 隔离方式 | GPU sandbox/进程隔离 | 参考与候选分进程，CPU 比较，保护缓存/全局精度状态 | 针对已讨论 reward hack 的工程实现；额外通信成本 |
| TF32 策略 | 原文正文未完整规定，公开样例存在争议 | 参考/候选统一状态并检查污染 | 继续按用户已确定方案，不列为待回退的未对齐缺陷 |
| 正式测时 | 同步、预热、重复平均；公开 profiler 汇总 CUDA 事件 | CUDA event 包围迭代，30 次×3轮取中位数，另有逐算子 profile | 同目标不同估计量，不是逐比特相同协议；会影响速度数字，不会造成 log_softmax 维度错误 |
| 基线污染审计 | 论文防 hacking 原则，公开脚本保护有限 | eager/compile 漂移阈值 3%、独立测时、TF32 审计 | 工程加强；当前唯一正确候选测时有效 |
| reward | {-1,1,2,3}，5% eager/compile 门槛 | 公式一致，仅作为 P1 评分/反馈，不更新权重 | 评分对齐不等于已经做 RL |
| 候选选择 | 轨迹中正确解里相对 compile 最快 | 同规则，再做不反馈的隐藏验收 | 主选择标准对齐；隐藏策略是额外约束 |
| 汇总指标 | 正确率、faster rate、正确解上的 GM | 同类指标，另列无效题与资源成本 | 定义基本对齐；数据集和分母不同需显式写明 |
| 文件/续跑/清理/多卡/tqdm | 论文未给完整实现 | 本地 helper、断点续跑、逐轮 JSON、清理大缓存、进度条 | 工程功能，本身不改变 CUDA 数学能力；本次未把它们视为低通过根因 |

## 10. 对“工具不对齐”应怎样判断

目前已有足够的工具完成一次正常 CUDA 开发：看原始模型和 profile、写 raw CUDA、多文件绑定、编译、5 输入反馈、性能对比、修改、提交。9 条轨迹也确实使用了这些工具，而不是只有一个空壳。

但“功能存在”不等于“接口无损”。优先级最高的是完整消费合法 tool_calls、统一工作区路径、使 single 交付和领域约束明确；这些会直接影响可用信息和最终交付。

Notebook、后台任务管理、完全相同的 ripgrep 参数、任意 GPU 调参脚本接口，是不同层级的差距。它们应如实记录，但目前没有证据表明补齐这些便能解决已见的 attention、索引、归约维度错误。采用 OpenHands 包也不会自动修复生成源码；关键是模型、消息、工具与反馈的实际语义。

官方仓库只公开了工作区、SKILL、部分工具和样例，不含完整 PPO 训练代码、训练后 checkpoint、完整 agent 调度器和所有采样/评测设置。因而应区分“公开规范可以对齐的部分”与“公开信息不足以精确还原的部分”，不能承诺当前配置已达到作者内部实现逐项一致。

## 11. 能下的因果结论与不能下的结论

### 11.1 有直接证据的结论

- single 的 4 个失败先发生在外层 JSON；把它们都解释成不会写 CUDA 不成立。
- Agent 有确定的多调用丢失和一次绝对路径语义错误，属于本地控制器问题。
- Agent 六个数学错误和一个返回结构错误在已编译执行的源码中可以定位，不能全归于工具、超时或隐藏验收。
- 超长未完成生成消耗了一半以上 Agent 生成 token，增加总窗口/总轮数不等于增加有效尝试。
- 当前并没有复现论文专项训练、基模型和最终评测条件，因此不能用它的 11.11% 否定论文报告的 98.8%。

### 11.2 仍无法量化的部分

- 修复接口后会增加几道正确题：未知。
- 仅换 Qwen 官方建议采样参数可减少多少长思考：未知。
- 更强模型、RFT、PPO 各自能贡献多少：当前没有受控对照。
- 单题速度差是内存带宽、launch、访存布局还是其他因素主导：需要 profile 与消融，不能只看代码注释判定。
- 当前 9 题能否代表完整 Ops-6K、KernelBench 或实际 kernel 开发：不能。

这些未知不能用“肯定是 9B 太小”“只要全对齐就能接近论文”“原论文 single 本来就差”代替证据。

## 12. 推荐的后续顺序（本次未执行）

1. **先修接口，冻结协议**：消费多个调用、统一路径、明确 single 文件交付方式及解析失败详情。single 必须严格交付完整 JSON；格式失败直接计为失败，不自动或人工修补后计分。
2. **公平拆分 single/agent 差异**：给同等领域规则，分别记录格式成功、合同通过、编译成功、数值正确和性能达标；明确执行反馈与预算的差异，避免只有最终 Pass Rate。
3. **验证解码与每次生成预算**：用 Qwen 官方建议作一个候选配置，对比当前配置；预先固定 token/时间预算。不要把一个任意的新上限标成论文原值。
4. **做最小受控比较**：同权重、同训练侧任务、同协议比较修接口前后；结果单独命名，保留 v4，test 不用于边看结果边调流程。没有必要先重跑完整数据集。
5. **再进入训练复现**：从 single-turn warm-up、合格轨迹收集和 RFT 开始，逐步验证各阶段贡献，再考虑 critic/value pretraining 和 Agent PPO。P1 的职责是测清起点与运行环境，不是提前满足训练后成绩。

优化目标应是“减少流程丢失后，测到可靠的原模型能力，并让训练改善真实计算”，而非通过放松正确性、允许 PyTorch 回退或复制状态污染来提高表面指标。

## 13. 复查入口

服务器项目根目录：`/home/likun/junjxu/CUDA-Agent`。

| 简称 | 完整 task ID |
|---|---|
| Attention | ops6k-4b30bedf7b666818 |
| hardsigmoid | ops6k-9fbe88dcf4053fa4 |
| BatchNorm | ops6k-7872d67c6198c50a |
| ReflectionPad | ops6k-f3750b017c78d8f0 |
| GroupNorm | ops6k-51844277a90906e9 |
| ConvTranspose | ops6k-b56ae5cf14573644 |
| Bros | ops6k-d5903209a0ffe947 |
| Upsample | ops6k-030adfe946c4729a |
| Embedding | ops6k-9c9053934a250410 |

- 全局配置/完成状态：`runs/p1/run-pilot-12-v4/manifest.json`。
- 正式汇总：同目录 `summary.json`、`results.json`。
- 模型原文、工具请求和返回：`trajectories/<task_id>/<single|agent>/turn-<编号>.json`。
- 每条轨迹选择与成本：同目录 `result.json`，区分 final_work_evaluation、final_evaluation、best_search_evaluation、hidden_evaluation。
- 精确候选源码：`evaluations/<上文编号>/candidate/`，检查结果见同级 `result.json`。不能只看工作区最后版本代替被选中的历史候选。
- 原始参考代码：`references/<task_id>/model.py`。
- 三道隔离题：939f4a6e2eab9f49、f9da864ce17744c9、4ac775908ca38f43，详见各自 references 的 preflight 和 prepare_result。
- 本地实现重点：`src/p1/run.py`、`tools.py`、`sandbox.py`、`evaluation.py`、`worker.py`、`contract.py`、`agent_utils/`。
- 论文重点：§3.1—3.3、§4.1、Table 1/2、§4.3.1、附录 B 的工具表和完整 SKILL；附录 C 的优化案例用于理解训练后策略，不代表当前模型已掌握。

本报告中的人工源码阅读和离线补括号解析只是诊断证据，不是新评测结果。v4 的 0/9、1/9 与正式文件保持一致。

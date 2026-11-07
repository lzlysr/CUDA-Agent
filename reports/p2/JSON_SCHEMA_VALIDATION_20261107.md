# vLLM JSON Schema 隔离验证记录

日期：2026-11-07，北京时间（Asia/Shanghai）。服务器项目：`/home/likun/junjxu/CUDA-Agent`。

## 1. 结论与验证范围

**当前模型可以使用原生 JSON Schema，在思考结束后约束正式答案；但不能只增加生成参数，就将其视为与当前 P2 PPO 目标完全一致。**

本次只做审查与隔离验证。没有修改 `src/`、`configs/`、verl/vLLM 源码或任何依赖，没有更新模型，没有执行 PPO 步骤，没有执行生成的 CUDA，没有修改历史结果。正式结束状态按用户要求沿用 verl 原生设置，不新增更细的字段。

验证使用原生 Qwen3.5-9B BF16；vLLM `0.21.0+cu129`、torch `2.11.0+cu129`、Transformers `5.9.0`、verl `0.9.1`、xgrammar `0.2.3`。GPU 按编号使用 0、7；未停止其他任务。GPU 实测为单卡推理，与正式 P2 的 TP=2 不同，耗时不能直接比较。

证据位于 [审查目录](../../.runtime/tmp/json-schema-audit-20261107/)。[源码片段、版本与 SHA256](../../.runtime/tmp/json-schema-audit-20261107/source-evidence.json)记录实际服务器源码；采集时主仓库工作区干净。

## 2. Qwen3.5 的思考与格式约束

实际使用的关键设置是：

```python
StructuredOutputsConfig(
    backend="xgrammar",
    reasoning_parser="qwen3",
    enable_in_reasoning=False,
)
SamplingParams(structured_outputs=StructuredOutputsParams(json=schema))
```

当前原生 `Qwen3ReasoningParser` 明确支持 Qwen3.5。原生管理器的接口验证结果：

| 输入状态 | 使用 qwen3 parser | 是否填充语法掩码 |
|---|---|---|
| 思考模式的 prompt 末尾 `<think>` | 是 | 否 |
| 已出现 `</think>` | 是 | 是 |
| 非思考模式的完整 prompt | 是 | 是 |
| 思考模式但没有 parser | 否 | 是，从第一步开始 |

因此应由原生 parser 划分区域。`enable_in_reasoning=True` 会连思考区一起约束，不适合本次“自由思考、最终交付 JSON”的目标。

除离线 `LLM` 外，还用真实 `AsyncLLM` 执行了当前 verl `generate` 的原始方法体。仅构造最小服务上下文，绕过 Ray 训练调度；没有改方法体。结果见 [异步证据](../../.runtime/tmp/json-schema-audit-20261107/async-results.json)：

| 请求 | 输出 token 数 | logprob 数 | 正式答案解析 | verl 结束状态 |
|---|---:|---:|---|---|
| 非思考 | 12 | 12 | 通过 | completed |
| 思考 | 920 | 920 | 通过，含 `</think>` | completed |

此结果证明实际异步引擎与原生服务方法可以衔接；不代表已验证分布式 Ray 训练、LoRA 权重同步或 PPO 更新。

## 3. 原题重放与预算边界

使用当前 `p2.data.make_prompt`、temperature=0.7、top_p=0.95、seed=42，总上下文 32768。用现有严格解析器检查，不补括号、不追加生成；解码与 P2 一致，只移除真实末尾 EOS，保留思考分隔符。

| 原先发生格式失败的题 | prompt token | 本次输出 token | 格式结果 |
|---|---:|---:|---|
| ops6k-dbd4edfabe40555e | 4750 | 5018 | 通过，思考后交付完整 JSON |
| ops6k-f86ee569dc6164ee | 4840 | 3761 | 通过，思考后交付完整 JSON |

原始回答和测量记录见 [原题与边界测试结果](../../.runtime/tmp/json-schema-audit-20261107/parser-verified/results.json)。两题只验证交付格式，没有验证编译、数值正确或性能。新输出不是旧输出的修复，不替换旧成绩；两次成功也不能证明长期格式失败率为零。

另外两个故意受限的边界测试均失败：思考模式最多 1536 token 时，预算耗尽仍未出现 `</think>`；非思考模式只给 1 token 时，只生成了不完整 JSON。JSON Schema 约束不会强制模型及时结束思考，也不会在达到预算时补齐答案。正式项目继续采用 verl 原生结束状态；本次读取 vLLM 原因仅用于隔离审查。

## 4. Schema 和调用接口的兼容限制

当前 vLLM 的 xgrammar 校验明确拒绝 `patternProperties`、`propertyNames`。尝试以文件名正则约束路径的 Schema，真实请求在开始生成前即报错：`The provided JSON schema contains features not supported by xgrammar.`

通过的版本只要求：外层对象必须有 `files`，禁止外层额外字段；`files` 必须是对象，包含字符串类型的 `model_new.py`，其余文件内容也必须是字符串。允许路径继续由现有控制器检查。Schema 不能保证文件内容完整、符合 CUDA 限制或能编译。

原生 `StructuredOutputsParams` 对象可以通过 verl 方法传入引擎。直接传普通 `{"json": schema}` 字典，虽然 `SamplingParams` 构造阶段接受，却会在后续原生校验时报 `'dict' object has no attribute '_backend'`。不能将 YAML 字典直接当作最终采样对象使用。[接口与最小数值证据](../../.runtime/tmp/json-schema-audit-20261107/adapter-results.json)包含这两项对照。

## 5. raw_logprobs 的实测含义

实际源码的顺序为：`GPUModelRunner` 先对 logits 应用语法掩码，再调用 `Sampler`；采样器在这个基础上计算 `raw_logprobs`，然后才继续处理温度和 top-p 等设置。

所以，**这里的 raw 不表示“没有语法掩码”，也不是当前 temperature=0.7、top_p=0.95 下完整的实际采样概率。**

为排除温度和 top-p 的干扰，另用 temperature=1、top_p=1 做了同权重对照：Schema 强制一个与用户指定 JSON 不同的固定键和值，生成后对完全相同的 token 前缀无约束重算概率。结果见 [逐 token 概率证据](../../.runtime/tmp/json-schema-audit-20261107/probability-stress.json)。例如 token `d`：

| 计算方式 | logprob |
|---|---:|
| 带语法约束生成返回的 raw_logprobs | 0.000000 |
| 同模型、相同前缀，无约束重算 | -16.731918 |

这说明概率差异来自掩码及归一化，不是“权重更新了”或仅有很小的 BF16 数值误差。该压力测试故意制造明显冲突，不代表真实 CUDA 回答通常具有这么大的差异。

## 6. 对当前 PPO 的影响

当前 V1 训练器默认重算 Actor 的 `old_log_probs`；Actor 的新旧概率都来自未应用 JSON 语法掩码的 logits。冻结配置中的 `rollout_is`、`rollout_rs` 均为 null，`bypass_mode=false`，没有自动进行重要性校正。不能误认为“已保存 rollout logprob”就等于“PPO 已用它作为分母”。

只考虑语法约束，设允许 token 集合为 A、原策略为 p，则约束策略为：

```text
qθ(a|s) = pθ(a|s) / Zθ(s),  a ∈ A
Zθ(s) = Σ[a∈A] pθ(a|s)
qθ/qold = (pθ/pold) × (Zold/Zθ)
```

当前无掩码 Actor 比率没有最后的归一化因子。为验证它会影响实际原生损失，用旧概率 `[0.2, 0.3, 0.5]`、新概率 `[0.3, 0.3, 0.4]`，允许前两项，选择第一项、优势为 1，调用原生 `compute_policy_loss_vanilla`，其余采用当前聚合和裁剪设置：

| 概率定义 | 比率 | 原生 PPO loss | 裁剪比例 |
|---|---:|---:|---:|
| 当前无掩码新旧策略 | 1.50 | -1.28 | 1 |
| 新旧均采用同一语法集合 | 1.25 | -1.25 | 0 |

这不是训练实验，不证明一定发散；它证明两个目标不等价。当前重算旧概率的路径在更新前仍可得到比率 1，不能以此证明采样与训练分布一致。

当前 temperature=0.7、top_p=0.95 本身也带来采样与概率定义差异；语法掩码会额外叠加一层。仅把 raw 改为 processed 不能让 Actor 自动重放语法掩码。原生 rollout correction 提供校正工具，但语法约束把部分 token 概率置零，还涉及目标策略及支持集的选择，不能未经验证就声称打开 IS 即严格等价。

如果选择优化约束策略，需要让 Actor 新旧概率与语法状态保持一致，或明确采用并验证其他目标及校正方案。现有框架不会仅根据 Schema 自动完成这一点。因此本次没有把约束接入正式 P2。

## 7. 使用建议与复现入口

用于 P1 单纯推理和独立 CUDA 测试，原生约束方案可行；仍需保留严格路径检查、代码审查、编译、正确性和测时。用于当前 P2 PPO，不能直接视作无影响的格式开关。训练前后若采用不同的约束协议，结果也不能直接归因于权重训练。

没有证明减少格式错误后的总体吞吐收益。本次仅有两道原题、使用共享单卡，存在首次 JIT 等开销；没有运行统一性能基准。

服务器审查脚本与证据均保存在 `.runtime/tmp/json-schema-audit-20261107/`。入口包括 `probe.py`、`adapter_probe.py`、`probability_probe.py`、`async_probe.py`，只用于本次审查，不是正式训练模块。运行前应加载 `scripts/set-env.sh`，将 learner/bin 加入 PATH，再按编号设置 CUDA_VISIBLE_DEVICES。准备阶段的 PATH 和审查脚本分词返回类型问题已排除，失败现场保留，不计为模型格式失败。

参考：[vLLM 结构化输出文档](https://docs.vllm.ai/en/v0.21.0/features/structured_outputs/)、[verl rollout correction 文档](https://verl.readthedocs.io/en/latest/algo/rollout_corr.html)。具体结论以本次安装版本的源码与实测证据为准。

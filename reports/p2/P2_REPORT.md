# P2：Qwen3.5-9B 单轮同步在线 PPO

## 1. 范围与已经确认的决策

P2 是本项目复现方案的单轮 PPO warm-up。PPO、GAE、优化器更新、FSDP、
Actor→vLLM 权重同步和检查点均使用 verl 原生实现，不重新实现训练循环。

- 主模型仍为原生 Qwen3.5-9B BF16 权重；没有换成量化模型。
- verl 使用项目内 `third_party/verl` v0.9.1 源码，V1 `sync` 入口。
- GPU 0、7：Actor/Critic 与 rollout 分阶段共享；GPU 4、5：Docker kernel 评测。
- 训练奖励只用五组反馈输入；检查点验证使用已有 test，并额外运行隐藏验收。
- 不再二次划分或从 train 抽监控题；test 已作为训练期间验证数据使用。
- 保留 P1 最新协议：固定 seed=42、候选之后重新测基线、1 轮×10 次测时。
- 生成格式失败属于本次 single 失败，不补写、不修补、不重新生成。
- 本轮只做最多两步短序列接线；不自动运行十步或 150 步。

## 2. 文件职责

| 文件                                  | 作用                                                                         |
| ------------------------------------- | ---------------------------------------------------------------------------- |
| `configs/p2-single-ppo-warmup.yaml` | 唯一主配置：上半部分为 GPU/LoRA/批量等项目参数，下半部分为 verl 原生字段覆盖 |
| `scripts/p2-train.sh`               | 唯一启动脚本；入口内轻量检查/复用题单，再启动原生训练                        |
| `src/p2/common.py`                  | 配置验证、北京时间、原子 JSON、配置身份                                      |
| `src/p2/data.py`                    | train 选题、公共提示词、CPU 静态/长度检查和轻量运行清单                      |
| `src/p2/dataset.py`                 | verl 公共 Dataset 扩展；从原始 train 读取记录，取样时补充 prompt/奖励字段    |
| `src/p2/train.py`                   | 加载官方配置与 YAML 覆盖、选择 GPU、调用 `run_ppo(..., TaskRunnerV1)`      |
| `src/p2/reward.py`                  | 严格解析最终源码，调用 P1 沙箱并产生有效终局奖励                             |
| `src/p2/models.py`                  | 继承原生模型/价值头加载，仅补齐 LoRA 角色/dropout、文本接线与更新审计       |
| `src/p2/rollout.py`                 | 用公开 register 注册单轮生成流程：固定 seed，限制总长，保留真实 token        |
| `src/p2/sampler.py`                 | 公开 custom_sampler 扩展：后台环境异常立即终止批次，拒绝伪造零奖励           |

最后几个适配文件没有实现 PPO/GAE。模型实现仍为 Transformers 原生类；
优化器、FSDP 与采样服务均来自既有框架。项目不改动 verl/vLLM/torch/CUDA 包源码。

## 3. 数据到训练结果的路径

题目来源直接配置在 verl 原生字段中：

```yaml
data:
  train_files: ["data/processed/train.jsonl"]
  val_files: ["data/processed/test.jsonl"]
```

已有 train 为 4,803 条，test 为 1,197 条。入口不二次切分、不从 train 抽监控题；
只做两侧的轻量检查，并在 run/data-manifest.json 记录各侧索引、输入摘要及拒绝原因。
完整原题和补充字段后的数据都不复制。`--task-id/--limit` 只限制训练侧，验证仍来自 test；
原始 train/test 文件不改名、不改内容。test 用作训练期间验证后，其结果不应标成未见测试成绩。

项目通过官方 `data.custom_cls` 使用 CudaDataset，在取样时补充完整 system/user 消息、
奖励信息及索引。公共 SKILL 和绑定每个 Dataset 只读一次。Dataset 根据原始路径区分
train/val，不再把轻量清单当作 val_files 入口；val 的原始记录仍标记 split=test。
验证样本的 hidden=true 只表示额外正确性验收，不表示来自 train 的隐藏监控题。

运行清单绑定两份原始文件、训练选题、适配代码、提示词、tokenizer、seed 和上下文。
config.json 冻结清单路径及摘要；verl-resolved.yaml 保存展开后的绝对输入路径及配置。
每次正式验证遍历同一份 test；val_batch_size 仅控制每批数量，val_max_samples=-1 表示全量。
--wire 仅做短测，关闭训练前和周期性全量验证，避免把完整验证混入接线工作。

当前固定 verl 没有 train/val ratio 启动字段；其示例的比例划分是预处理时调用
Hugging Face train_test_split。本轮按用户最新要求直接使用已有两份文件，不调用该方法。

准备只检查 Python 语法、任务入口、各侧 train/test 归属、原始行索引和提示词长度。
它不执行题目、不启动 Docker/GPU、不做 eager/compile 初测或近常数筛选。
实际参考兼容性检查、候选验算和测时保留在奖励阶段，参考故障不能伪装成模型负分。
现有 train 有 4,803 条记录、4,434 个归一化 task_id；task_id 是代码组标识，
同组多行保持原样，以原始 row_index 冻结选题。`--task-id` 选择该组的所有原始记录，
不会自动去重；`--limit` 仍按记录数限制，续跑按清单恢复确切的原始行。

1. 读取现有 `data/processed/train.jsonl`，以 seed=42 确定性抽样或按 `--task-id` 选择。
   不重新划分、不对 Ops-6K 再做数据去污染，也不固定旧 pilot 去补齐 test。
2. 提示词包含 P1 共享约束、原文官方 SKILL、固定绑定和原题。目标为完整 files JSON。
   没有 Bash、工具反馈或历史候选；这一步对应单轮 warm-up。
3. 做 CPU 语法/接口与提示词长度检查；静态拒绝题单独记录，不补题凑配额。
4. 只保存 run 下的 data-manifest.json，不保存完整转换数据。Dataset 取样时补充字段；
   训练来自 train，检查点验证来自既有 test，后者标记 hidden=true。
5. 当前 Actor 通过 verl/vLLM 在线采样一批轨迹。真实 prompt/response token、mask、
   采样 logprob 和权重版本留在 TransferQueue；PPO 不从 response 文本重新编码。
6. 奖励器只解析本次最终输出。格式失败给 -1；有效源码调用既有 Evaluator。
   编译/验算可以按配置并发，正式测速通过评测卡池的独占租约执行。
7. 五组数值检查通过后，先测候选，再测 eager/compile 基线，用后测值计算 speedup。
   得到 -1/1/2/3 终局奖励。环境错误或无效计时不能进入训练标签。
8. verl 计算旧策略 logprob、Critic value、GAE，再更新 Critic 和 Actor。
   没有用多轮轨迹的最佳历史候选给本次 token 重新归奖。
9. 同步新 Actor LoRA 到 rollout，清理旧 KV 状态，下一批用更新后的策略生成。
10. 按频率保存原生 Actor/Critic 检查点；检查点验证遍历 test 并额外运行隐藏正确性验收。
    每题评测结束删除 input-spec 和 compile-cache，源码/日志/计时证据保留。

## 4. 参数与论文关系

| 参数                                | 正式 P2 配置                         | 说明                                                 |
| ----------------------------------- | ------------------------------------ | ---------------------------------------------------- |
| 总上下文                            | 32768                                | 实际 prompt+response 总长，不额外固定 4096 输出上限  |
| Batch / minibatch / 每卡 microbatch | 24 / 12 / 1                          | 针对本机资源的本地选择，不冒充论文原始规模           |
| PPO epochs                          | 1                                    | 一批新采样执行一次 PPO epoch                         |
| Actor / Critic LR                   | 3e-5 / 6e-5                          | 本地参数；论文为 3e-6 / 6e-6，本轮未调整学习率 |
| LoRA rank / alpha / dropout         | 64 / 128 / 0.05                      | all-linear 包含语言与视觉适用线性层；score 头另外完整训练 |
| Critic                              | 同源骨干 + hidden_size→1 score      | 独立 LoRA；score 完整训练，不产生词表 logits         |
| γ / λ                             | 1 / 0.95                             | 使用 GAE，非 GRPO                                    |
| Ratio clip                          | [0.8, 1.28]                          | 下限 1−0.2，上限 1+0.28                             |
| 额外 KL / entropy 奖励              | 0 / 0                                | 不启动额外参考模型                                   |
| 采样                                | temperature=0.7、top_p=0.95、seed=42 | 未公开的论文采样细节不声称完全复现                   |
| 训练正确性检查                      | 五组反馈输入                         | 用户已确认                                           |
| 隐藏验收                            | test 检查点验证                      | 不向训练奖励暴露隐藏检查                             |
| 测时                                | 后测基线、1×10                      | 以本项目最新约定为准                                 |
| 保存/检查点评估                     | 每 10 步，最多保留两份检查点         | 检查点按框架保存完整训练状态                         |

短测 `--wire` 覆盖为总上下文 8192、batch/minibatch=2、最多两步、每步保存，
关闭训练前和周期性全量验证。它只验证接线，不代表 32K 容量、稳定训练或能力提升。
未获得用户启动长期训练的授权前，不运行正式步数。

## 5. 环境与必要适配

沿用 learner/kernel 的 torch 2.11.0+cu129、CUDA Toolkit 12.9、vLLM 0.21.0+cu129。
初始接线阶段补充 `TransferQueue==0.1.8`，对应固定 verl 源码 CI 中的版本；
安装前 dry-run 确认所有传递依赖已满足，安装时 `--no-deps`，未升级其他包。
依赖记录补入 `reports/p0/requirements.txt`。

Actor/Critic 均继承 verl 的 `_build_module`。Critic 由原生 `load_valuehead_model`
调用 Transformers 的 `AutoModelForTokenClassification`，自动选择
`Qwen3_5ForTokenClassification` 并创建 score 头；项目不再单独覆盖 Critic 加载流程。
默认 verl 的 CAUSAL_LM LoRA 包装不适合这个模型类，且未传入 dropout，项目只在
LoRA 构造处补齐 TOKEN_CLS、lora_dropout 和 modules_to_save=[score]；适配器仍由
PEFT 创建。目标层、排除层和 target_parameters 均按原生模型配置透传。
使用 PEFT 的 autocast_adapter_dtype=false 沿用基础层 dtype，删除手动 BF16 转换循环。
`CudaHFModelConfig` 是标准模型配置的薄扩展：Ray 反序列化时导入项目模块，保证
每个 GPU worker 都已注册模型引擎；不增加一套训练器。P2 仅使用文本消息，
关闭多模态 processor，让 verl 传输一维文本位置。原生 Qwen3.5 前向会等价扩展
为四路位置，避免四路 M-RoPE 经过当前 TQ/TensorDict 路径时的 jagged 轴错误。
启动脚本将 learner/bin 放入子进程 PATH，确保已有的 ninja 可以被 vLLM 调用。
当前改用 flash_attention_2，并保留 no-remove-padding。FlashAttention 的安装包名为
flash-attn；依赖版本和本机 wheel 记录于 reports/p0/requirements.txt。只核对必要前后向，
不能据此声称完整 32K PPO 的容量和吞吐已经验通。

单轮生成类直接使用 verl 公开的 `@register("cuda_single")` 注册。
模型配置模块同时导入该生成模块，确保 Ray 子进程各自完成注册；唯一主配置只指定
`default_agent_loop: cuda_single`，不再需要额外的生成流程 YAML。
历史运行的 Parquet、配置和检查点保留。新 run 直接读取原始 train/test；
本轮改变 LoRA 结构和验证来源，必须使用新 run，不把旧语言层 LoRA 检查点
自动迁移为全模型 LoRA，也不修改已有检查点的输入格式或顺序。

Ray 的 Unix socket 有 107 字节路径上限，专属临时目录放在
`/tmp/ca-p2-<uid>-<run摘要>`；训练源码、配置、数据、日志、检查点均在项目目录。
不连接或停止他人 Ray 集群。

## 6. 输出、续跑和存储

- data/processed/train.jsonl、test.jsonl：原始训练/验证来源；不创建完整转换副本。
- run 的 data-manifest.json：分别保存 train/val 的原始行、输入摘要、token 长度和拒绝原因。
- run 的 `config.json`：冻结项目参数、清单路径及摘要；`verl-resolved.yaml`：实际框架配置。
- `rollouts/`：真实 token、mask、采样 logprob 和策略版本的轻量审计；训练不重读它。
- `samples/`：本次最终 response、候选工作区、奖励结果或环境错误证据。
- `kernel/`：现有 Evaluator 的冻结候选、编译/正确性/测时记录。
- `audit/`：各 rank 的可训练参数及梯度/权重变化，含价值头变化记录。
- `checkpoints/global_step_*`：原生 Actor/Critic、优化器、调度器、RNG 和 dataloader 状态。
- `status.json`：本次启动完成/失败标记；不把配置组合成功记为训练成功。

保留 PPO batch 的 token、mask、旧概率是计算本批 PPO loss 的要求。
它们在在线训练中由 TransferQueue 暂存，不等于预先固定一个离线 RL 数据集。
原生检查点默认包含基础权重分片，因此明显大于只有 LoRA 的推理导出；
当前不启用只保存 LoRA 的裁剪路径，避免漏掉 Critic 的 score 头及完整续跑状态。
接线实测单份完整检查点约 35 GiB，保留两份约 70 GiB；应据此预留空间。

## 7. 本轮验证记录

以下为旧 Parquet、语言层 LoRA、dropout=0、SDPA 配置下的历史 GPU 接线；新配置另行核对，
不能把历史两步结果冒称为本次新入口再次通过 GPU 实验。

实际运行目录：`runs/p2/wiring-20261105`。两道题均来自 train：
`ops6k-d4413b455033577a` 和 `ops6k-ac6e648cc33136d2`。
仅用于低成本接线，不用于汇报模型能力。

2026-11-06 设备查询恢复后继续实测，**两步同步在线 PPO 接线均已完成**。
第二步从第一步原生检查点恢复；最终 `status.json` 为 completed，
`latest_checkpointed_iteration.txt` 为 2，训练主进程已退出。以下时间均为北京时间。

第一步日志为 `train-step1-recovered-20261106T132001.log`，13:20:01 启动，
13:31:32 正常结束。两条在线生成先后返回，均因原始 JSON 不合法得到有效 -1 奖励；
未修补、重采样或替换成历史正确候选。随后 Critic/Actor 执行了真实反向与优化器更新。

| 项目                      | 第一步实测                            | 恢复后的第二步实测                    |
| ------------------------- | ------------------------------------- | ------------------------------------- |
| Actor / Critic 可训练参数 | 173,113,344 / 173,117,441             | 相同                                  |
| Actor 梯度范数            | 0.490234375，有限                     | 0.455078125，有限                     |
| Critic 梯度范数           | 22.625，有限                          | 12.8125，有限                         |
| Actor 参数变化            | 每个 rank 抽查到 248 个张量变化       | 每个 rank 抽查到 496 个张量变化       |
| Critic 参数变化           | rank0 抽查到 250 个张量变化，含 score | rank0 抽查到 498 个张量变化，含 score |
| 原生检查点                | `checkpoints/global_step_1`         | `checkpoints/global_step_2`         |

参数变化审计仅抽查各张量的前 64 个本地分片元素，不等同于完整模型逐元素比较。
Critic score 按 FSDP 分片，rank0 的价值头变化记录为 true；不是要求所有 rank 都持有价值头。

第二步日志为 `train-step2-resume-20261106T133223.log`，13:32:23 启动，
13:44:08 正常结束。日志确认恢复 Actor/Critic 模型与优化器；原生恢复路径同时读取
extra_state 中的调度器/RNG 与 `data.pt` 的数据游标。恢复后的采样权重版本
`min_global_steps=max_global_steps=1`，第一批为 0；四条记录均无缓存 token。
原生同步器在恢复后及各步结束时完成权重更新，未混用旧批次。

每份检查点有 Actor/Critic 各两个 rank 的 model/optim/extra_state，以及 `data.pt`，
共 37,728,517,933 字节（约 35.1 GiB）。两份均已保存，后续通过 `--resume` 使用。

本次四条在线回答的格式均失败，奖励均为有效 -1，完整原文在 `samples/`。
生成 token 总数 12,562；其中两条刚好达到 8K 总上下文上限，输出预算约 3.55K。
另外两条未达到上限也存在 JSON 语法错误。因此本次不能作为正式 32K 下的正确率或
能力提升结论。负奖励仍可计算 GAE 和 PPO loss；完成更新不等于回答正确。

独立 CUDA 奖励重放使用 P1 已有正确候选
`runs/p1/run-pilot-12-v8/evaluations/00082/candidate`，对应 train 题
`ops6k-9fbe88dcf4053fa4`。它没有加入 PPO，结果保存在 `reward-replay/`：

- 五组反馈输入均通过，最大绝对/相对误差为 0；未运行隐藏验收。
- 真正执行了 CUDA kernel，`runtime_fallbacks=[]`，无 TF32 状态变化。
- 后测 eager/compile 基线、候选 1×10 测时均有效。
- speedup_eager=0.85948、speedup_compile=0.88157，正确但未加速，奖励为 +1。
- 评测后 input-spec 和 compile-cache 已清理；源码和详细测时保留在 `kernel/evaluations/00001/`。

此前失败证据继续保留，均发生在第一次优化器更新之前，不计入完成步数：

| 日志或目录                                                       | 原因及处理                                                           |
| ---------------------------------------------------------------- | -------------------------------------------------------------------- |
| `train-step1.log`                                              | Ray socket 路径过长，改用专属短临时目录                              |
| `train-step1-b.log`、`train-step1-c.log`、`nvml-query.log` | NVML/驱动层等待，服务器恢复后设备查询正常；未重置 GPU 或修改驱动     |
| `train-step1-recovered-20261106T125230.log`                    | Ray 子进程未导入项目模型引擎，改为显式项目模型配置类型               |
| `train-step1-recovered-20261106T125620.log`                    | vLLM 子进程 PATH 找不到已安装的 ninja，补充 learner/bin              |
| `train-step1-recovered-20261106T130205.log`                    | 四路 position_ids 的 jagged 传输不兼容，纯文本使用等价一维位置路径   |
| `failed-attempts/20261106T132001/`                             | 上述零更新尝试的采样、奖励及审计记录，单独归档，避免混入正式两步统计 |

短测只覆盖 8K、batch=2 的接线。正式 32K 容量、较大 batch、十步/150 步稳定性、
周期性隐藏检查点评估以及能力提升，均需后续独立实测；本轮不自动运行这些实验。

## 8. 运行时补充字段、实际提示词及与 P1 的比较

原始题目的顶层字段为 `ops`、`data_source`、`code`、`row_index`、`source_hash`、
`normalized_hash`、`task_id`、`split`。Dataset 取样时构造下述结构，只在内存中传递：

```json
{
  "data_source": "cuda_agent_ops6k",
  "prompt": [
    {"role": "system", "content": "共同规则 + 官方 SKILL + 本机覆盖 + single 说明"},
    {"role": "user", "content": "单次交付要求 + 固定绑定源码 + 原始题目代码"}
  ],
  "ability": "cuda_kernel",
  "reward_model": {"style": "rule", "ground_truth": "ops6k-..."},
  "extra_info": {
    "index": "ops6k-...",
    "task_json": "原始题目所有字段序列化后的 JSON 字符串",
    "hidden": false
  }
}
```

| 字段                          | 作用                                                          | 是否直接作为模型提示词         |
| ----------------------------- | ------------------------------------------------------------- | ------------------------------ |
| `prompt`                    | 运行时构造完整 system/user 消息                               | 是                             |
| `data_source`               | 统一为 CUDA 任务来源标签，供框架统计；原值保留在 task_json 内 | 否                             |
| `ability`                   | CUDA 任务类别标签                                             | 否                             |
| `reward_model.style`        | 标记规则奖励                                                  | 否                             |
| `reward_model.ground_truth` | 保存 task_id，不是专家 CUDA 答案                              | 否                             |
| `extra_info.index`          | 样本追踪索引                                                  | 否                             |
| `extra_info.task_json`      | 原题完整记录，供可信评测器重建题目                            | 否；其中 code 另放入 user 消息 |
| `extra_info.hidden`         | 训练为 false，test 验证取样为 true                           | 否                             |

没有为模型预先生成答案、奖励、旧概率或优势；这些在在线采样/评测/训练时产生。

模型收到一道题时，先将 `prompt` 通过 Qwen 原生 chat template 转为 token。
自定义 Dataset 读取原始题目，在内存中构造 `prompt/raw_prompt` 消息；实际 token 化
仍在 AgentLoop 中完成。补充字段不再写入转换文件。固定消息层次如下，
省略处只是本说明的摘要，实际运行时构造完整原文：

```text
system
  P1 共享 CONTRACT：CUDA 实现、允许的库、数学/索引/绑定接口等约束
  官方 SKILL.md 全文
  本机覆盖：sm_80、无 sudo、固定基础设施只读
  single 说明：没有工具和执行反馈

user
  一次生成完整 files JSON；源码完整、换行转义，不自动修补格式
  binding.cpp 全文
  binding_registry.h 全文
  本题原始 PyTorch 代码（包括 Model 和两个输入生成函数）

assistant
  按 chat template 开始 <think> 推理并生成一次最终回答
```

| 项目                           | P1 single       | P2 single PPO       | P1 agent                                       |
| ------------------------------ | --------------- | ------------------- | ---------------------------------------------- |
| CONTRACT、官方 SKILL、本机覆盖 | 有              | 相同内容            | 相同共同部分                                   |
| 原题与固定绑定源码             | 有              | 相同内容            | 相同内容                                       |
| system 的 single 无工具说明    | 有              | 逐字相同            | 替换为持久 Bash、GPU 作业及工具工作流说明      |
| user 的交付要求                | 一次 files JSON | 相同要求            | 先分析原始性能、读写文件、编译验证、迭代后提交 |
| 工具定义与执行反馈             | 无              | 无                  | 有工具 schema 和逐轮工具返回                   |
| 用户消息内显式交互预算         | 无              | 无                  | 包含交互次数及总上下文预算                     |
| 输入适配时间                   | 运行时构造      | Dataset 取样时构造  | 运行时构造并追加交互历史                       |
| 生成后用途                     | 评测推理基线    | 奖励后执行 PPO 更新 | 迭代评测，选择轨迹候选                         |

逐字对照发现 P2 user 在两份固定绑定源码之间比 P1 single 多一个空行。
除此之外，两者初始 system/user 的指令内容相同；本次保留该空白，不借格式迁移
修改历史提示词。两题示例的 token 数分别为 4,639/4,642；token 数相同不代表
token ID 序列逐项相同。P2 的 32K 正式/8K 短测预算由配置限制，不额外写进消息。

## 9. 历史运行时 Dataset 核对记录（train 监控版本）

本轮将完整转换数据改为运行时适配；历史两步 PPO 接线记录见第 7 节。
新入口只做轻量检查和 Dataset/数据游标核对，不将历史 GPU 结果当成本轮 GPU 验证。
`--wire` 只设置 batch=2、8K 总上下文等短测参数，不限制题目总量。
两题来自验通命令显式指定的 `--task-id`，不是筛选后只剩两题。
省略选题参数的新 run 会从全部 4,803 条 train 记录开始轻量检查。

2026-11-06 实际核对通过：

1. `--dry-run` 不创建运行目录，也不写题目副本。
2. `runs/p2/runtime-input-check` 的 `--prepare-only` 只生成 `data-manifest.json`、
   `config.json`、`verl-resolved.yaml`。两题清单为 1,312 字节，不包含完整题目或 prompt。
3. 再次省略选题参数启动，成功复用原始行及顺序，没有重新选题。
4. 通过 verl 的 `create_rl_dataset` 加载训练与监控 Dataset；运行时补充的每个原有字段、
   system/user 消息与原两题 JSONL 完全一致。监控取样的 hidden=true，训练为 false。
5. Dataset 内只保留原始记录；`prompt/reward_model` 不在原始 dataframe 中预存。
6. StatefulDataLoader 恢复状态后读取同一条后续记录；Dataset 也能从原始文件重建。
7. 旧 `wiring-20261105 --wire --steps 2 --resume --dry-run` 通过。
   其配置、解析 YAML、原清单、两份 Parquet 和检查点步数标记的摘要均未改变。
8. 未启动 Ray、GPU 模型或 PPO 更新；核对结束 `torch.cuda.is_initialized()` 为 false。

原 `jsonl-input-check` 只有输入检查，没有训练或检查点，已保持同两条原始行迁移为
轻量运行清单。确认无运行再引用旧完整缓存后，删除 `data/p2-single-ppo/` 下那两题的
两份转换 JSONL、旧清单和空写锁；原始 train、实际训练记录及检查点保留。
新入口未在本轮重新执行 GPU 更新；完整训练验证仍以第 7 节的历史入口为界。


## 10. 全模型 LoRA、FlashAttention 和已有 test 验证

2026-11-06 本轮设置：Actor/Critic 的 target_modules=all-linear、lora_dropout=0.05；
训练引擎读取这些配置，不再使用固定语言层列表。PEFT 创建 LoRA，未另写适配器算法。
本节 GPU 验证时 freeze_vision_tower=false，enable_tower_connector_lora=true，
视觉塔/连接器适配器沿用 vLLM 原生加载路径。当前 CUDA 任务仅有文本，没有有效视觉训练信号。

Critic 的 score 是 hidden_size→1 的普通价值头。LoRA 通常冻结基础模型，score 是新增且
尚未学会估值的参数，因此必须保持可训练并随检查点保存。项目通过 TOKEN_CLS 和
modules_to_save=[score] 完成这件事；价值损失、GAE 和 PPO 仍由 verl 计算。
使用 TokenClassification 模型类只是取得每 token 一个标量输出，不是在做分类训练。

2026-11-06 实际核对：

1. `runs/p2/fa2-test-input-check` 只选择两条 train 做接线输入，验证侧仍读取全部
   1,197 条 test。轻量检查两侧均无拒绝；原生 `create_rl_dataset` 确认训练/验证分别
   为 2/1,197 条，原始行没有交集，hidden 分别为 false/true，结束时未初始化 CUDA。
2. learner 新增 `flash-attn==2.8.3.post2`，基于现有 torch 2.11/cu129 编译；包安装
   前后对比只有 flash-attn 新增。wheel 保存于 `.runtime/cache/wheels/`，
   requirements 已增加本地 wheel 搜索路径及固定版本。该 wheel 适配 CPython 3.12、
   A100 sm80，不作为其他 Python/显卡/torch 组合的通用二进制包。
   SHA256：`588d4e80584a5e88fc067c5361f30c4f42e5ffbedad0615985c11e62036d58e6`。
3. GPU 7 的 FA2 BF16 前后向通过；对照 FP32 math SDPA 最大绝对误差 0.007475，
   梯度有限。通过真实 verl 配置反序列化确认 Actor/Critic 均使用 FA2、all-linear、
   dropout=0.05。
4. 按完整 9B 结构在 meta 上核对，Actor 可训练参数为 205,060,096，Critic 为
   205,064,193，两者均含 220 个视觉 LoRA 参数张量。使用相同官方模型类的小结构
   做 GPU 前后向，Actor/Critic 均有有限梯度，Critic score 得到非零梯度；
   eval 状态关闭 dropout。meta 核对只验证目标层覆盖，不是完整 9B PPO 更新。
5. 同一小结构及其 all-linear 适配器通过 vLLM 0.21 原生加载，并生成 8 个 token；
   启用 enable_tower_connector_lora，无需修改 vLLM 源码。vLLM 仍将视觉 LoRA 标为
   experimental。日志提示 Conv3d patch embedding 不支持 LoRA；PEFT all-linear
   本身没有给该卷积创建 LoRA，因此本次没有对应训练参数被丢弃。

安装和小结构验证记录在 `.runtime/tmp/flash-attn-build-resume-20261106T212100/`
及 `.runtime/tmp/p2-fa2-gpu-check-20261106/`。本轮没有启动整组 PPO 更新；
完整 32K 训练的容量、吞吐与能力提升尚未验证。Qwen 的 Gated DeltaNet 另有自己的
线性注意力实现；安装 FA2 不等于同时安装其可选加速包，本轮未改变这部分依赖。

## 11. 精简模型扩展，复用官方模型与价值头加载

2026-11-06 删除 `CudaValueEngine._build_module`，Actor/Critic 均直接继承
verl 的 `FSDPEngine._build_module`。Critic 经 `load_valuehead_model` 调用
Transformers 自动模型映射，创建 `Qwen3_5ForTokenClassification` 及其 score 头。
meta/CPU 初始化、模型 dtype、梯度检查点和 Qwen 前向适配也随原生加载流程处理。
项目未手写 score 线性层，未修改 verl、Transformers 或 PEFT 的源码文件。

当前模型扩展保留的内容及原因：

- `CudaHFModelConfig`：传入上游尚未暴露的 lora_dropout，保证 Ray 子进程注册项目
  扩展，并沿用已核对的文本 position_ids 接线。
- `_build_lora_module`：当前 verl 通用构造固定 CAUSAL_LM 且未传 dropout，因此这里只
  区分 Actor/TokenClassification Critic 的角色、补齐 dropout，并完整训练/保存 score。
  LoRA 创建、冻结和筛选均调用 PEFT；target_modules、target_parameters、exclude_modules
  按原生字段透传，不再额外强制某一组 GDN、视觉或输出层名称。
- `autocast_adapter_dtype=false`：使用 PEFT 原生选项维持基础层 dtype，删除手动遍历参数
  转回 BF16 的循环。参数及优化器变化审计继续包装原生更新函数。
- Dataset、单轮 rollout 和严格失败检查继续使用 verl 公开扩展接口；它们负责项目数据、
  固定生成 seed/总上下文和有效奖励约束，没有复制 PPO/GAE/FSDP 或优化器实现。

当前 YAML 的 freeze_vision_tower 已由用户设为 true，但固定版本 V1/FSDP2 路径没有消费
该字段，它不构成视觉 LoRA 冻结/排除。实际覆盖范围由两侧 model 的 target_modules 和
exclude_modules 决定；本轮只补齐原生排除参数的透传，没有替用户切换视觉训练策略。
如需只给语言层加 LoRA，可以在 Actor/Critic 的 model 中使用
`exclude_modules: '^model\.visual\..*'`；确认不再导出视觉 LoRA 后才关闭 vLLM 视觉开关。

北京时间 2026-11-06 22:24，在 GPU 7 完成针对本次改动的小结构核对：

| 核对项 | 结果 |
| --- | --- |
| 两个项目引擎的加载函数 | 均为继承的原生 FSDPEngine._build_module |
| Actor / Critic 自动选择 | Qwen3_5ForConditionalGeneration / Qwen3_5ForTokenClassification |
| 可训练参数 dtype / dropout | 均为 BF16 / 0.05 |
| Actor / Critic 有限梯度张量 | 74 / 76 |
| Critic score | 有非零梯度，执行一次 AdamW 后参数发生变化 |
| PEFT 保存与重载 | TOKEN_CLS、score 均保存；重载前后输出逐元素一致 |
| 原生 exclude_modules | 排除视觉线性层生效，语言 GDN LoRA 继续存在 |

记录在 `.runtime/tmp/p2-native-simplify-20261106/` 的 `gpu-check.log`、`result.json`
及 `audit/`。本次使用同架构小模型做组件接线核对，没有启动 Ray、整组 PPO、正式数据
验证或 32K 训练，也没有运行 CPU 回归套件。小模型没有经过 FSDP 包装；这次结果不替代
多卡 PPO、FSDP 检查点恢复或训练效果评估。

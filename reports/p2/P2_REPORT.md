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
| `src/p2/train.py`                   | 加载官方配置与 YAML 覆盖、选择 GPU、调用原生 `run_ppo`，由 `CudaTaskRunner` 选择验证口径子类      |
| `src/p2/reward.py`                  | 严格解析最终源码，调用 P1 沙箱并产生有效终局奖励                             |
| `src/p2/models.py`                  | 继承原生模型/价值头加载，仅补齐 Critic LoRA 角色与文本接线；Actor LoRA、优化器与指标使用原生实现       |
| `src/p2/rollout.py`                 | 用公开 register 注册单轮生成流程：固定 seed，限制总长，保留真实 token        |
| `src/p2/sampler.py`                 | 公开 custom_sampler 扩展：可恢复单题失败用原生补采，环境异常仍停止           |

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
每次正式验证使用从 test 确定性选出的同一批 24 条记录；val_max_samples=24 限制总验证数，val_batch_size 仅控制每批数量。原始 test 文件不改动。
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
2. 提示词包含 P1 共享约束、原文官方 SKILL、固定绑定和原题。目标为完整原始源码文件块，格式见第 16 节。
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
| LoRA rank / alpha                   | 64 / 128                             | all-linear 包含语言与视觉适用线性层；score 头另外完整训练 |
| Critic                              | 同源骨干 + hidden_size→1 score      | 独立 LoRA；score 完整训练，不产生词表 logits         |
| γ / λ                             | 1 / 0.95                             | 使用 GAE，非 GRPO                                    |
| Ratio clip                          | [0.8, 1.28]                          | 下限 1−0.2，上限 1+0.28                             |
| 额外 KL / entropy 奖励              | 0 / 0                                | 不启动额外参考模型                                   |
| 采样                                | temperature=0.6、top_p=0.95、top_k=20、seed=42 | Qwen 精确编码建议；论文未公开的采样细节不声称完全复现 |
| 训练正确性检查                      | 五组反馈输入                         | 用户已确认                                           |
| 隐藏验收                            | test 检查点验证                      | 不向训练奖励暴露隐藏检查                             |
| 测时                                | 后测基线、1×10                      | 以本项目最新约定为准                                 |
| 保存/检查点评估                     | 每 30 步，保留数量不设上限           | 按当前 YAML；检查点保存完整训练状态，注意磁盘容量 |

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
Critic LoRA 构造处补齐 TOKEN_CLS 和 modules_to_save=[score]；Actor 使用原生构造。适配器仍由
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
- 当前不再生成 `rollouts/`：PPO 使用 TransferQueue 中的真实 token/mask/logprob，不保存重复数组副本。
- `samples/`：原始 response、奖励及评测路径索引；完整快照形成后清理工作区，异常现场保留。
- `kernel/`：现有 Evaluator 的冻结候选、编译/正确性/测时记录。
- `tensorboard/`：verl 原生逐步标量，供 TensorBoard 查看；当前不再生成自定义更新审计。
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
  一次生成 <file path="…"> / </file> 文件块；结束标签容错，源码不转义、不补全
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
| user 的交付要求                | 一次原始文件块  | 共用同一份交付说明  | 先分析原始性能、读写文件、编译验证、迭代后提交 |
| 工具定义与执行反馈             | 无              | 无                  | 有工具 schema 和逐轮工具返回                   |
| 用户消息内显式交互预算         | 无              | 无                  | 包含交互次数及总上下文预算                     |
| 输入适配时间                   | 运行时构造      | Dataset 取样时构造  | 运行时构造并追加交互历史                       |
| 生成后用途                     | 评测推理基线    | 奖励后执行 PPO 更新 | 迭代评测，选择轨迹候选                         |

旧 JSON 版本逐字对照发现 P2 user 在两份固定绑定源码之间比 P1 single 多一个空行。
该空白仍保留；当前两种模式的交付说明改为共用同一常量。旧 JSON 提示词下
两题示例的 token 数分别为 4,639/4,642，不适用于当前文件块协议；token 数相同不代表
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
  转回 BF16 的循环。优化器更新在本次精简后直接继承原生实现，旧审计记录仅作历史证据。
- Dataset、单轮 rollout 和严格失败检查继续使用 verl 公开扩展接口；它们负责项目数据、
  固定生成 seed、输出核对和有效奖励约束；总上下文由原生服务控制，没有复制 PPO/GAE/FSDP 或优化器实现。

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


## 12. 2026-11-07：损失聚合和原生接口精简

本节记录 2026-11-07 的接口精简；dropout 的最新处理见第 13 节。
第 7、9、10、11 节的实测记录保留其历史配置与验证边界。

- Actor/Critic 显式设置 `loss_agg_mode: seq-mean-token-mean`。Actor 对每条回答的
  有效 token 取平均，再对样本取平均，对应论文式(5)的轨迹等权聚合。
- 原生 vanilla PPO 会将 log-ratio 限制在 [-20,20]，ratio 至多 exp(20)。
  配置 `clip_ratio_c=1e10` 高于此值，使额外 dual-clip 不生效，保留 [0.8,1.28]
  的普通 PPO 裁剪。使用有限数避免零优势乘无穷大。未另写 PPO loss、未改 verl。
  原生数值稳定保护、GAE 优势标准化和 Critic 价值裁剪仍保留，不将其宣称为论文披露参数。
- 当前 FSDP2 原生调度默认为 constant，warmup 比例为 0；论文未披露 cosine 调度。
  本次不改学习率或调度类型。
- `CudaSingleTurnLoop` 继承原生 `SingleTurnAgentLoop`。提示词处理、生成请求、
  Continuous Token 合并和原生耗时指标由父类执行；项目只固定 seed 并核对输出，不再额外落盘采样数组。
  总上下文由原生 vLLM 服务依据 max_model_len 截断可生成 token 预算。
- 删除模型扩展的 record/optimizer_step 审计，优化器更新完全继承 verl。
  非有限梯度恢复原生的告警并跳过更新语义，不再由项目额外抛出审计异常。
  Critic 的 LoRA 角色接线保留：需 TOKEN_CLS/可保存 score；dropout 接线已按第 13 节移除。
- 删除旧 Parquet、完整转换 JSONL 和旧 agent-loop YAML 的身份归一化/续跑分支。
  当前只支持原始 train/test JSONL + 轻量 data-manifest；不迁移或删除历史结果。
  损失和配置身份已变化，应建立新 run，不能用这次设置直接续跑旧实验。
- trainer.logger 启用 console/tensorboard，入口把 TENSORBOARD_DIR 设为 run/tensorboard。
  无需自写曲线记录器；已有 learner 中已安装 TensorBoard。手动启动展示服务：

```bash
.runtime/conda/learner/bin/tensorboard --logdir runs/p2 --host 127.0.0.1 --port 6006
```

通过 VS Code 转发服务器 6006 端口，在浏览器查看奖励、Actor/Critic loss、梯度范数、
学习率、回答长度、耗时及原生验证标量。逐步指标在训练更新或验证完成后写入，
不是每个生成 token 刷新一次；TensorBoard 原生写盘有短暂缓冲。

关于基线顺序：论文正文未明确写出 PPO 奖励服务的基线执行先后；官方公开的
agent_workdir/utils/profiling.py 的 main 确实先测 CUDA 候选，再测 eager、compile。
本项目按该公开示例及已确认协议使用后测基线，不能把公开示例推断写成未公开训练服务的事实。

训练中每个检查点已经会重新生成验证回答，不必等训练结束才检查正确率和奖励。
当前 RewardManager 传递 score/acc 等原生验证标量；Faster Rate 和正确样本加速比几何均值
尚未聚合成完整论文表格，逐候选 speedup 仍在详细结果中。P2 warm-up 的前后对比应使用
同一套单轮提示词、生成预算、采样设置和评测协议；基础模型与训练后 Actor 使用相同设置。
P1 single 的指令内容接近，但须先核对预算与采样设置；P1 agent 有工具交互，属于另一评测协议。
论文主表及 w/o Agent Loop 消融均在 agent loop 下评测，复现主表时应另做同协议的 agent 评估。

本次未启动 PPO、GPU 实验或 CPU 回归套件；仅做语法、配置及原生调用关系的静态核对。


## 13. 2026-11-07：取消额外的 LoRA dropout 适配

按用户最新决定，删除主配置的 lora_dropout、Actor/Critic 模型覆盖项、
CudaHFModelConfig 额外字段及公共配置校验，不再为 dropout 增加本地代码。
PEFT 支持该参数，但当前 verl 的原生构造没有传入它；不传时采用默认值 0。
历史 dropout=0.05 的组件测试记录保留为当时配置，不代表当前配置再次实测。

Actor 不再注册项目自定义引擎，直接使用 verl 的 FSDPEngineWithLMHead 及其 LoRA 构造。
Critic 保留最小的 _build_lora_module：使用 PEFT TOKEN_CLS 包装、modules_to_save=[score]，
使原生价值头可训练并保存。该适配与 dropout 无关；未手写价值头或优化器。
不再需要 Actor/Critic 共用的 CudaLoraMixin，也不保留 CudaActorEngine。
模型配置扩展仍负责 Ray 导入注册和纯文本 position_ids 接线。

其余 LoRA rank/alpha、target_modules、PPO 参数及数据协议保持现有配置。
本次未运行训练、GPU 实验或 CPU 回归套件；仅核对语法、配置与原生调用关系。
配置身份发生变化，后续使用新 run；不改写历史检查点或实验结果。


## 14. 2026-11-07：精简 P2 输出，保留训练和评分证据

仅调整后续运行的生成逻辑，不迁移、不删除 runs/p2 下的历史实验文件。
检查点保留数量仍使用当前配置，本轮不调整。

### 14.1 必要输出与关系

- config.json、data-manifest.json、verl-resolved.yaml：保留配置冻结、数据身份和完整启动配置。
- status.json、tensorboard/：保留运行状态与原生训练/验证曲线。
- 不再写 rollouts/*.json。PPO 仍使用原生返回并由 TransferQueue 管理的真实 token、mask、
  logprob；只是取消磁盘审计副本，不影响本批更新或原生检查点恢复。
- samples/<sample_id>/response.txt：保留原始回答，不修补 JSON，不删改模型推理文本。
- samples/<sample_id>/result.json：保留奖励/正确性/加速比、sample_id/task_id、hidden，
  并增加 response_path、feedback_result_path、acceptance_result_path。
  没有对应阶段时路径为 null；所有路径均相对于本次 run 根目录。
- kernel/evaluations/<编号>/candidate/：保留可信源码快照与 source_hashes。
- kernel/evaluations/<编号>/result.json：保留正确性、奖励、speedup、checks、身份及状态；
  compilation、execution、timing、baseline_after 等有独立详情文件时改为 detail_paths 引用。
  candidate_path 同样相对于本次 P2 run。没有独立文件的诊断继续留在总结果中。
- 阶段目录的 result.json / bench_result.json 和 worker.log 继续保留。
- scheduler/ 锁文件、worker-config.json 和完整检查点保留其原有职责。

P2 奖励先用完整内存评测结果计算，之后才精简磁盘总结果。返回给 verl 的评分不变。
精简函数仅由 P2 调用，不改变 P1 独立运行的总结果格式；P2 读详情时沿 detail_paths 查找。

### 14.2 工作区和预检审计清理

反馈与必要的隐藏验收结束、样本结果写入后，逐文件确认 workspace 源码与可信快照一致，
才删除重复 workspace。不存在完整快照、含链接、内容不一致或环境异常时保留现场。
普通文件清理失败记录 workspace_cleanup_error，不把已有效计算的模型奖励改成负分。
既有 input-spec/compile-cache 完成后清理逻辑保持不变。

worker.prepare 保留审计采样和参考一致性检查（使用配置容差），现已取消近常数筛选；持久化采样
转换为数量、最小/最大值、SHA256 等摘要，保留 shape/dtype/numel 及判定结论。
worker 日志仅输出阶段、状态、结果文件位置、耗时及错误/traceback，不再重复整份结果 JSON。
此 worker 是 P1/P2 共用入口，因此两者后续预检审计和终端日志均使用精简形式；
验算次数、容差、种子、测速与奖励协议均未改动。

当前仍保存完整 Actor/Critic 模型、优化器、调度器/RNG 和数据游标。没有启用可能漏存
Critic score 头的 LoRA-only 裁剪，也没有改动 max_checkpoints。
本轮不启动 PPO/GPU 实验，不运行 CPU 回归套件；仅做语法、引用和持久化路径的静态核对。


## 15. 2026-11-07：移除本地额外的 eager/compile 输出一致性门槛

原先 worker.bench 在测完 compile 后，额外对五组输入比较 compiled 与 eager 输出，
失败时令 reference_bench 返回 error；P2 把该预检失败上报为无法评分，严格采样器终止整批。
这项检查是本地工程保护，论文未明确要求逐次执行，官方公开 profiling.py 也没有此断言。
论文 §3.1 要求原算子在 Eager/Compile 模式成功执行；公开 verification.py 检查的是
候选 CUDA 与原始 eager 输出，而非 eager 与 compile 相互比较。

现删除额外断言及 compile_correct 字段，compile 仅提供性能参考，speedup_compile
仍按 compile_ms/candidate_ms 计算。该比值本身不再额外证明 compile 输出与 eager 一致。
原始 eager 继续作为候选正确性的参考；反馈/隐藏检查、容差、固定种子、后测基线、
测时次数和奖励门槛均不变。编译/执行报错、非正/非有限耗时、状态污染仍按原规则处理。
这不等于放宽候选验算，不给参考故障伪造 -1，也不放宽排序索引的比较标准。

本次修改共用 worker，适用于后续 P1/P2 基线计时；旧实验文件及历史审查报告保持原样。
未启动训练、单题 GPU 重放或 CPU 回归套件；仅核对语法、调用关系及移除字段的引用。

## 16. 2026-11-07：single 改用原始文件块交付

按用户选择，当前 P1 single 和 P2 共用以下格式，替代旧 files JSON：

```text
<file path="model_new.py">
完整 Python 源码
</file>
<file path="kernels/kernel.cu">
完整 CUDA 源码
</file>
<file path="kernels/kernel_binding.cpp">
完整绑定源码
</file>
```

示例内文字是说明占位，模型必须交付实际完整代码；kernel 文件名按实现确定，
不要求为允许的 cuBLAS/cuDNN 包装额外创建无用的 .cu 文件。

开始标签只接受双引号 `path` 属性；开始、结束标签各自独占一行。模型仍被要求
完整闭合，但按用户最新要求，解析器容忍结束标签缺失或多余：遇到结束标签、
下一个开始标签或回答结束时结束当前文件；多余的独占行结束标签作为分隔符跳过。
块与块之间允许空白，不允许明确闭合后的额外正文或 Markdown 围栏。
源码直接保留正常换行、缩进、引号、反斜杠、`<`、`>`、`&`，不做 JSON/XML 转义。
标签只按行识别，源码中的普通字符串 `"</file>"` 不会截断文件；独占一行的标签是保留边界。

共用实现为 `p1.helper.tool_calls.SINGLE_FILE_DELIVERY` 与 `parse_single`。
P1 的 `Pilot.prompt` 和 P2 的 `data.prompt_template` 读取同一份交付说明。
解析后仍返回原有文件字典，奖励代码继续通过 `write_files` 统一检查路径和大小，
再按原来的编译、五组正确性检查、后测基线及奖励流程处理。

缺少或畸形开始标签、重复路径、明确闭合后的额外正文、空源码或缺少 model_new.py
直接算候选失败，P2 仍给有效 -1。结束标签容错只改变文件边界解析，不补源码、
猜测文件名、解析旧 files JSON 或调用模型修复。
源码不完整仍可能编译或数值失败，文件块格式不会保证 CUDA 正确。

不启用 vLLM JSON Schema/语法掩码，原生采样、token/mask/logprob 及 PPO 损失链路
保持不变；不新增格式约束带来的概率重归一化。原生结束状态仍采用 completed。
官方 SKILL 原文、Agent 工具协议、模型、采样设置、固定 seed 和评测预算均不变。
这只是本地源码交付协议，不能称为论文原有实现。

旧实验和检查点不迁移。P1 源码指纹与 P2 输入清单中的提示词摘要已改变，应使用
新的运行目录；不得将新协议的结果直接当作旧 JSON run 的续跑或重评分。

定向检查位于 `tests/p1/test_file_blocks.py`；只覆盖本次解析边界，不运行整个 CPU
回归套件。本次没有启动 PPO 训练。

### 本次隔离验证结果

4 项定向检查通过，包括原始源码/CRLF 保真、缺标签与嵌套/重复文件拒绝、
旧 JSON 拒绝及 Agent JSON 工具参数不受影响。修改文件的 AST 语法检查与
`git diff --check` 通过；未运行整套 CPU 回归。

两条真实单卡生成均采用 seed=42、temperature=0.7、top_p=0.95、32K 总上下文，
不启用语法约束，不编译或执行候选 CUDA。GPU 0 测 P1 single，GPU 7 测 P2。

| 模式与原题 | prompt / 输出 token | 生成耗时 | 交付结果 |
|---|---:|---:|---|
| P1 single：ops6k-f86ee569dc6164ee | 4998 / 27770 | 1098.33 秒 | 失败；最后一个文件缺结束标签 |
| P2：ops6k-dbd4edfabe40555e | 4908 / 27860 | 1077.06 秒 | 失败；思考区未结束，未交付文件块 |

两条均用满可用输出余量。P1 回答已出现一次 `</think>`，交付了两个闭合文件块，
随后在 `kernels/ceil_bias.cu` 里重复生成同类 kernel，版本号一直到 v210 后截断；
未返回部分文件字典，也没有写入候选目录。P2 没有出现 `</think>`，整条输出仍是
思考及反复的代码围栏片段，不能当成正式交付。

这些结果说明严格解析和拒绝逻辑生效，但不能证明新协议已经降低实际失败率。
缺结束标签没有被补全，思考区代码没有被提取来计分，也没有追加生成或人工修复。
采样/思考设置保持原样；模型生成退化需要另行讨论，不能通过放宽格式验收掩盖。

证据目录：`.runtime/tmp/file-block-20261107/`，两条回答和 summary.json 分别保存在
`single-ops6k-f86ee569dc6164ee/`、`p2-ops6k-dbd4edfabe40555e/`。
源码备份在该目录的 `backup/`；测试进程已经正常退出，没有运行 CUDA 验算或 PPO。

### 追加两题隔离验证与重复生成分析（2026-11-07）

用户要求换另外两题复核。使用当前 P2 正式提示词，两题分别在编号 GPU 0、7
各生成一次，均保持 BF16、seed=42、temperature=0.7、top_p=0.95、32K 总上下文。
复用上一轮隔离脚本，只有记录采样参数和生成开始状态的观测改动；未改变交付
协议、正式源码或训练配置，未启用语法约束，没有重试、补标签或执行候选 CUDA。

| 原题与算子 | prompt / 输出 token | 生成耗时 | 重复退化与交付结果 |
|---|---:|---:|---|
| ops6k-9fbe88dcf4053fa4：Hardsigmoid / ReLU | 4874 / 6339 | 247.15 秒 | 未出现上一轮长循环；6 个闭合文件后额外输出一个 `</file>`，严格交付失败 |
| ops6k-f3750b017c78d8f0：ReflectionPad2d / log2 | 4875 / 6016 | 234.98 秒 | 未出现上一轮长循环；5 个闭合文件后额外输出一个 `</file>`，严格交付失败 |

两题均出现一次 `</think>`，未用满约 2.8 万 token 的可用输出预算。按正式回答
顺序扫描，全部文件块后都仅剩一个孤立 `</file>`；正式解析器因此抛出
`single output must contain only <file path="..."> blocks`。没有写入候选目录，
不将诊断中识别出的完整块当作通过结果。此次交付成功率为 0/2，未评测 CUDA
编译、数值正确性、加速比或 PPO 更新，因此不能称为 CUDA 正确率为 0%。

此前的两次长输出不是解析器重试：隔离脚本只调用一次 llm.generate，返回后
才调用 parse_single。一题在正式源码里重复写 kernel 到 v210；另一题在思考区
反复写代码，二者最终用满预算。这是生成端的重复退化，未发现解析器反向
触发生成或无限重试的工程路径。新增两题未复现同样的长循环，说明不能仅凭
前两题认定文件块必然导致循环，也不能凭四题估计总体发生率。

源码对照确认原题、系统约束、官方 SKILL 与固定绑定代码保持原样，主要变化是
用户消息中的交付格式说明。提示词变化会改变后续 token 分布；相同 seed 不代表
新旧提示词会产生相同内容。当前证据不足以将格式本身确定为重复的唯一原因。
文件块省去 JSON 转义，但仍需模型遵守标签、文件边界和停止规则；两次多余结束
标签说明这些规则仍会失败。此前推荐原始文件块更可靠的推断尚未被实测支持。

此外，较早成功的 JSON Schema 审查启用了 xgrammar，当前文件块为无约束生成，
两者不是仅改变格式的严格对照。旧的无约束 JSON 实验也曾出现缺右括号。
要确定交付提示词对重复的因果影响，需对同一题、同一推理配置进行无约束 JSON
与文件块的配对测试；本次没有扩大到这项额外实验，也没有修改采样或正式 PPO。

证据保存在 `.runtime/tmp/file-block-extra-20261107/`：两题各自的 `response.txt`
和 `summary.json`，以及 `gpu-0.log`、`gpu-7.log`。生成脚本与启动脚本也保留在
该独立审查目录，不加入正式训练入口。两次测试都已结束。
### 官方精确编码采样参数四题对照（2026-11-07）

按用户要求，四题均改用 Qwen3.5-9B 官方模型卡中“精确编码任务的思考模式”推荐的
采样参数：temperature=0.6、top_p=0.95、top_k=20、min_p=0.0、presence_penalty=0.0、
repetition_penalty=1.0。来源：[Qwen 官方模型卡](https://huggingface.co/Qwen/Qwen3.5-9B#best-practices)。
这组参数区别于通用思考模式的 temperature=1.0、presence_penalty=1.5，不混用两个场景。

原实验为 temperature=0.7，未启用 top-k；其余上述参数的有效值相同。本次实际改变
的是 temperature 和 top_k，不能单独将结果归因于温度。仍使用固定 seed=42、当前
文件块提示词、原生 BF16 模型、同样的 32K 总上下文和每题可用输出余量。
每题生成前均核对旧 summary 中的 prompt_sha256、prompt_tokens、max_tokens、seed、
top_p 完全一致。没有启用语法约束、增加输出次数、修补标签或执行生成的源码。

GPU 0 串行运行 Hardsigmoid/ReLU 与 ConvTranspose 题，GPU 7 串行运行反射填充与
MaxPool 题；每题单独建立与前次相同的单请求推理实例。保留原比较模式：
ops6k-f86ee569dc6164ee 为 P1 single，其余三题为 P2。正式提示词和训练配置未改，
官方参数只用于此隔离对照，未启动 PPO，也未运行完整 CPU 回归。

| 题目 | 旧输出 token / 生成秒数 | 官方采样输出 token / 生成秒数 | 严格交付对照 |
|---|---:|---:|---|
| ops6k-f86ee569dc6164ee | 27770 / 1098.33 | 4486 / 175.34 | 失败→通过；未复现正式源码中 kernel 到 v210 的长重复 |
| ops6k-dbd4edfabe40555e | 27860 / 1077.06 | 13799 / 537.26 | 失败→失败；最终结束思考，3 个闭合文件后多一个 `</file>` |
| ops6k-9fbe88dcf4053fa4 | 6339 / 247.15 | 2121 / 82.65 | 失败→失败；2 个闭合文件后多一个 `</file>` |
| ops6k-f3750b017c78d8f0 | 6016 / 234.98 | 6307 / 244.10 | 失败→通过；3 个文件块完整闭合 |

本次固定四题的严格交付成功率从 0/4 变为 2/4（50%），仅代表这四次新生成，不能
当作总体 CUDA 正确率。通过的两题已经按原解析器与 write_files 规则完整落盘，
源码字节保真检查通过，内存中缺最后结束标签的诊断副本仍被拒绝。失败的两题
没有写入候选目录，更没有删除额外标签后重评分。没有运行 CUDA 编译、数值验算、
性能评测或 PPO，因此本表的耗时是模型生成耗时，不是 kernel 加速比。

四题均最终结束思考，并在用满输出预算前返回。但是 MaxPool 仍有明显重复推演：
思考区约 46303 字符、34 个代码围栏、14 次 __global__，一些相同计算片段重复多次。
因此不能表述为官方参数已经根治重复，只能说前次两条耗尽预算的输出在这次未复现，
其中 ConvTranspose 明显缩短，MaxPool 输出减少约一半但仍冗长。反射填充题的耗时
略增，也说明降低温度不是统一的减字数或防循环开关。

提示词审查：原文件块说明减少了源码转义，但没有明确强调每个开始标签恰好对应
一个结束标签、没有外层 file 容器、最后一个结束标签之后应结束回答；连续两轮
出现孤立结束标签，说明这些边界值得强化。另外，交付说明位于用户消息开头，
后面还有固定绑定代码及原题；可在原题后用短规则再次明确仅交付所选最终实现，
不继续展开中间版本。官方 SKILL 已包含清理中间版本要求，不能将该建议称为论文
新增要求。上述是下一步可检验的提示词改进方向，尚未在本轮修改或验证，不能
据此断言提示词是所有失败的唯一原因，也不能把问题全归因于模型本身。

证据目录：`.runtime/tmp/file-block-official-sampling-20261107/`。每题保留原始
response.txt、summary.json，以及通过时的 candidate/ 源码；目录内同时保留隔离
生成脚本、两张卡的启动脚本与日志。采样参数在 summary 中逐项记录，旧实验保留。
四次生成均已完成，未遗留本轮生成任务。
### 末尾提醒四题对照及采样配置固化（2026-11-07）

P1 single 和 P2 的原题末尾共用三条提醒：仅交付独立文件块、每个开始标签恰好
对应一个结束标签、最后一个文件闭合后立即结束。未改官方 SKILL、Agent 或解析器。
同样四题各生成一次，沿用上一轮官方采样设置和 seed=42，GPU 0、7 的分工与顺序
保持不变。去掉新增提醒后的提示词 SHA256 与上一轮完全一致；提醒增加 60 个
prompt token，32K 总上下文不变，最大输出余量相应减少 60 token。

| 题目 | 上轮通过 / 输出 token / 生成秒数 | 本轮通过 / 输出 token / 生成秒数 |
|---|---:|---:|
| ops6k-9fbe88dcf4053fa4 | 否 / 2121 / 82.65 | 是 / 2322 / 91.12 |
| ops6k-dbd4edfabe40555e | 否 / 13799 / 537.26 | 是 / 6180 / 237.39 |
| ops6k-f3750b017c78d8f0 | 是 / 6307 / 244.10 | 否 / 5979 / 227.28 |
| ops6k-f86ee569dc6164ee | 是 / 4486 / 175.34 | 否 / 5985 / 230.95 |

严格交付仍为 2/4（50%）：原来失败的两题变为通过，原来通过的两题变为失败。
后两题的文件块全部闭合后均仅剩一个孤立 `</file>`。新增提醒没有提高这四次生成
的总体交付率，不能称为格式问题已解决。MaxPool 思考区从约 46303 字符、34 个
围栏降至约 21015 字符、4 个围栏，但这些观测不足以证明总体重复退化已根治。
四题均结束思考且未用满预算；仅通过的完整源码落盘，失败回答未被修补或执行。
本轮没有 CUDA 编译、数值/性能评测、PPO 训练或 CPU 回归套件。

按用户追问，正式 `configs/p2-single-ppo-warmup.yaml` 已同步设为 temperature=0.6、
top_p=0.95、top_k=20。训练 rollout 和 val_kwargs 都引用同一组 p2 参数，并核对了
OmegaConf 插值；min_p=0、presence_penalty=0、repetition_penalty=1 沿用原生默认值，
已在 YAML 注释中说明。没有添加不支持的 verl 字段、额外适配代码或修改框架源码。
此修改供后续新启动的 P2 使用，不改写旧 run 的冻结配置；本轮隔离脚本本来已用
相同参数，因此没有重启或额外生成。P1 正式采样配置未在此次改动中调整。

证据保存在 `.runtime/tmp/file-block-prompt-reminder-20261107/`，包含每题原始回答、
summary.json、通过时的 candidate/、隔离脚本和两卡日志。四次生成均已完成。
### FILE 标记无闭合标签交付四题对照（2026-11-07）

按用户要求，隔离尝试 `FILE: 相对路径` 后直接书写原始源码，下一个 FILE 标记或
回答结束作为文件边界，不要求 `</file>`、JSON 或 Markdown 包装。仅在临时测试
脚本中替换交付说明和解析器；正式 P1/P2 源码、官方 SKILL、Agent、verl/PPO、
模型及采样参数均未在此次测试中修改。沿用 temperature=0.6、top_p=0.95、
top_k=20、seed=42 和 32K 总上下文。逐题校验替换前的提示词 SHA256 与上一轮
一致，原题和系统说明未变；协议长度改变，输出余量随提示词长度调整。

| 题目 | 上轮文件块交付 | FILE 标记交付 | 输出 token | 生成秒数 | 本轮问题 |
|---|---|---|---:|---:|---|
| ops6k-9fbe88dcf4053fa4 | 通过 | 失败 | 2139 | 89.47 | 源码中含 6 行 Markdown 围栏，不符合原始源码协议 |
| ops6k-dbd4edfabe40555e | 通过 | 失败 | 2353 | 97.45 | 源码中含 6 行 Markdown 围栏，不符合原始源码协议 |
| ops6k-f3750b017c78d8f0 | 失败 | 失败 | 4802 | 197.17 | 源码中含 10 行 Markdown 围栏，不符合原始源码协议 |
| ops6k-f86ee569dc6164ee | 失败 | 失败 | 2866 | 118.41 | 源码中含 8 行 Markdown 围栏，不符合原始源码协议 |

本轮严格交付为 0/4；上轮为 2/4。这是四次固定题目的格式测试，不能作为
总体 CUDA 正确率。严格规则仍拒绝代码围栏、旧标签、重复/非法路径、缺少入口
文件，不删除包装后重评分，不增加生成次数。通过时仅按原 write_files 规则落盘
并检查字节保真与 Python 入口语法；未运行 CUDA 编译、数值/性能评测或 PPO。

GPU 0 在加载模型前因空闲显存约 13.66 GiB，小于推理实例需要的约 27.8 GiB，
两题均启动失败，尚未采样。这两条启动证据保留在 startup-failures-gpu-0/。
对应两题随后在 GPU 7 各生成一次，四题最终均在 GPU 7 完成，未停止其他进程。
服务器有其他 GPU 任务，生成秒数仅记录本轮成本，不用于吞吐优劣结论。

结论边界：取消闭合标签只减少一种边界错误，不能保证模型不输出额外包装，也
不能由解析成功保证最后一个源码文件未截断。当前未据此自动切换正式协议，
也未将容忍围栏的其他解析规则用于本轮评分。

逐题审阅发现：四题均使用 FILE 标记，所有文件的 Markdown 围栏均完整成对，
没有旧 file 标签；四题输出均未达到 token 上限。当前拒绝原因是禁止围栏的
协议规则，不代表已经检验出 CUDA 数学或编译错误。下一步可先明确允许完整
成对围栏作为合法包装，再调整隔离解析器；仍拒绝缺闭合、非法路径和重复文件。
该建议未用于本轮重新评分，未对本轮回答进行去围栏落盘。

证据目录：`.runtime/tmp/file-marker-20261107/`。包含隔离协议与生成脚本、
每题 messages.json、response.txt、summary.json、启动日志，以及通过时的
candidate/。GPU 0 启动错误单独归档；四次模型生成均已结束。

### 结束标签容错规则更新（2026-11-07）

按用户最新决定，正式交付保留 `<file path="…">` / `</file>`，不采用 FILE 标记。
共享 parse_single 允许少结束标签、多结束标签，P1 single 和 P2 奖励入口同时生效：

- 少中间结束标签：以下一个文件开始标签作为边界，不把下一份文件吞入前一份。
- 少最后结束标签：以回答结束为边界，不补源码。
- 多结束标签：只跳过独占行的 `</file>`；源码中普通字符串保持原样。

路径、重复文件、空文件、入口文件、写入范围和源码大小检查保留。开始标签缺失
或畸形仍失败，不接受 Markdown 围栏。没有明确结束标签时，解析器无法判断尾部
正文是代码还是解释，也不能判断代码是否被截断；不做猜测裁剪，由后续编译、
数值及性能评测判定。官方 SKILL、Agent 工具调用、采样设置与 verl/PPO 未修改。

本节更新的是新运行的解析规则。前面各轮严格协议的原始统计与证据保持不变，
历史回答只用于解析兼容性检查，不回写旧结果、不重算旧奖励。

本次仅做语法检查与保存回答的解析兼容检查：两轮共 8 条原始回答均能解析；
此前 4 条已成功落盘回答的源码字节完全一致，缺最后结束标签的边界副本也能解析。
未运行 CPU 回归套件、GPU 生成、CUDA 评测或 PPO，未修改历史评分。

### 结束标签容错后四题重新生成（2026-11-07）

按用户要求，同样四题各重新生成一次，不复用或修改旧回答。P1 single 仍对应
ops6k-f86ee569dc6164ee，其余三题使用 P2 的正式提示词构造入口。保留 <file>
交付及原题末尾提醒，采用当前正式共享解析器；未采用 FILE 标记或语法约束。
沿用原生 BF16 模型、temperature=0.6、top_p=0.95、top_k=20、seed=42、32K
总上下文和每题输出余量，未增加生成次数。除之前修复中交付说明的失败条件
一句以外，系统说明、原题及其他提示词与上一轮文件块测试的 SHA256 对照一致。

同一条新回答分别交给旧严格解析器与当前容错解析器，不对原始回答补标签或
改写代码。新解析通过后继续调用正式 write_files 检查写入范围及大小，核对
落盘字节，并仅做 Python 入口语法检查。没有 CUDA 编译、数值/性能评测或 PPO。

| 题目 | 旧严格解析 | 容错解析与落盘 | Python 入口语法 | 输出 token | 生成秒数 |
|---|---|---|---|---:|---:|
| ops6k-9fbe88dcf4053fa4 | 通过 | 通过 | 通过 | 3317 | 137.54 |
| ops6k-dbd4edfabe40555e | 通过 | 通过 | 通过 | 2420 | 98.53 |
| ops6k-f3750b017c78d8f0 | 失败 | 通过 | 通过 | 2977 | 122.68 |
| ops6k-f86ee569dc6164ee | 通过 | 通过 | 通过 | 3557 | 147.51 |

本轮四条新回答：严格解析 3/4，容错解析并落盘 4/4，
Python 入口语法通过 4/4。这些均非 CUDA 正确率或加速比。

- ops6k-f3750b017c78d8f0：开始标签 3，结束标签 4；旧解析错误 `single output must contain only <file path="..."> blocks`；当前交付错误 `无`。

GPU 0 空闲显存不足，因此四题在已授权 GPU 7 上串行生成；不停止其他进程。
服务器有其他任务，秒数只记录本轮生成成本，不作为吞吐比较。输出未达 token
上限也不意味着源码一定完整，实际 CUDA 正确性仍需正式评测。旧结果、奖励和
采样配置没有重写；本轮不运行 CPU 回归套件。

证据：`.runtime/tmp/file-block-tolerant-generation-20261107/`，包含每题原始
response.txt、messages.json、summary.json、通过时的 candidate/，以及隔离
生成脚本和 GPU 7 日志。四条生成均已完成。

## 17. 参考自检容差与底层错误记录（2026-11-08）

single-ppo-warmup 在训练前验证时，原始参考题 ops6k-f05dde20a8f3f503 的连续两次
输出出现最大约 3.58e-7 的绝对差异，旧 worker 的 atol=rtol=0 自检将其拒绝。
参考失败后奖励无法定义，保护采样器终止本次运行；这不是模型标签解析或 PPO
参数更新失败。该次验证只落盘六条有效 -1 和一条参考异常，没有开始 PPO 更新。

现在参考浮点/复数重复输出检查读取同一份 kernel 配置中的 atol/rtol（当前均为
0.01）；整数、布尔、普通标量、类型/shape/dtype/结构及非有限值的检查继续保留。
参考模型仍运行两次，明显超出配置容差时仍拒绝，不伪造候选奖励。

准备阶段只做语法、接口、长度等轻量检查，不能证明参考 GPU 执行一定成功。
P2 奖励阶段若参考预检失败，异常明确给出 task_id、preflight 状态、底层失败类型、
原因及详情绝对路径；samples/<样本>/infrastructure-error.json 另外保存结构化
reference_failure，包含 preflight_status、failure_kind、error_type、error、
detail_path。detail_path 相对于本次 run；原阶段结果继续保留完整 traceback。

本节记录容差修复当时的策略；参考任务失败的当前处理已按第 18 节改为隔离、
有限重试与补采。候选格式、编译或数值失败仍是正常有效 -1。初始验证最多
24 条，不代表准备时执行了整个 test。

本次在 GPU 4 独立验证：原出错参考 ops6k-f05dde20a8f3f503 的 reference_prepare
通过，近常数标记为 false，控制器耗时约 4.92 秒，使用原配置 atol=rtol=0.01。
另经正式 P2 evaluate_response 入口验证微型负向参考：两次输出相差约 100，仍然
抛出参考异常；sample 的 infrastructure-error.json 保留了 reference_nondeterministic、
ReferenceValidationError、具体差异、实际容差及可访问的详情路径。两项均在
`.runtime/tmp/reference-tolerance-20261108/` 独立目录执行，不回写旧 run。
未运行参考测时、候选 CUDA 评测、模型 rollout、PPO 或 CPU 回归套件。


## 18. 单题隔离、有限重试与原生补采（2026-11-08）

### 18.1 最新失败与前次的区别

`runs/p2/single-ppo-warmup/status.json` 的最新失败时间为北京时间
2026-11-08 01:09:07，仍发生在 `val_before_train`，尚未进行 PPO 参数更新。
本次不是之前的零容差误判：`ops6k-72d8c10fb5385f89` 的原始参考程序在
`index_select` 中混用 CPU 索引与 CUDA 张量，直接抛出 RuntimeError。
详情位于该 run 的
`kernel/references/ops6k-72d8c10fb5385f89/cases/prepare_result.json`，
样本证据为 `samples/7a927989975845ceaf6ab25dd7af9b23/infrastructure-error.json`。
此前出错的 `ops6k-f05dde20a8f3f503` 本轮已走过参考预检，之后因候选合同违规得到 -1。

该 run 已完成的 20 条样本全部为有效 -1：12 条编译失败、6 条合同违规、2 条数值错误。
这些模型失败本来不会使训练停止；导致停止的是旧 sampler 对任何失败组一律报错。
旧 run、状态和样本结果保留，不用新策略覆盖或重评分。

### 18.2 当前失败处理表

| 情况 | 当前处理 | 是否给模型奖励 |
|---|---|---|
| 参考重复输出超出 atol/rtol，参考包含 NaN/Inf | 隔离该题，训练丢弃并补采 | 否 |
| 参考普通执行失败、初始参考基线失败 | 最多执行两次，仍失败则隔离该题 | 否 |
| 参考输出流失败或后测 eager/compile 基线失败 | 控制器最多完整尝试两次，仍失败则隔离该题 | 否 |
| 真正 response token 数为 0 | 丢弃本次请求并补采，不隔离题目 | 否 |
| OOM、worker 异常退出、基础依赖缺失、未分类后台异常 | 停止整个 run，保留原始证据 | 否 |
| 实际传输空间不足、输出/参数快照容量超限 | 停止整个 run | 否 |
| 候选格式、编译、数学/索引/正确性错误、违规 PyTorch 回退 | 正常作为已评分轨迹交给 PPO | -1 |

当前参考浮点重复输出仍用 atol=rtol=0.01，整数/布尔/普通标量的一致性与
输出结构检查保留。预检检查张量和数值标量中的非有限值；正式参考输出流也
在控制器读取时独立检查 NaN/Inf，避免候选类型错误掩盖参考自身不可评分。
空间不足或任一端 OOM/异常退出优先于单题恢复，不会被参考重试吞掉。
后测基线仍是唯一速度和奖励基准；固定 seed、五组反馈和 1×10 测时不变。

### 18.3 为什么仍有少量本地适配

`src/p2/sampler.py` 使用官方 custom_sampler 扩展，训练失败组的剔除、精确补采、
完整批次等待与轨迹 materialize 由原生 ReplayBuffer 完成；配置显式启用
`trainer.v1.sampler.sync_refill_failed_groups=true`，原生数据加载按单题获取替补。
每题 train/val 的 rollout.n 当前均要求为 1，防止失败组的部分轨迹被补齐进入 PPO。

`src/p2/failures.py` 只负责错误分类、按原始 verl UID 保存失败索引及本次 run 的
隔离题目。rollout 在再次遇到已隔离题时直接跳过生成，继续由原生机制补采；
零 token 只丢弃请求，不将同一题永久列入隔离。EOS-only 或解码后空文本而实际
含有 token 的回答不属于“零 token”，仍按正常候选处理，通常获得格式失败 -1。

`src/p2/trainer.py` 的子类只调整验证指标分母和覆盖率记录：原生验证会将缺失
回答纳入部分 acc 指标的期望分母，这与用户要求“无效题跳过”不同。通过官方
`run_ppo(..., task_runner_class)` 接口选择子类，初始化/清理遵循原生 TaskRunner，
训练 fit、PPO、GAE、反向传播、优化器、LoRA 和权重同步均继承官方实现；
未修改 `third_party/verl`、vLLM、torch 或 CUDA 包。

验证集不补题；指标只用本轮有效轨迹，另记录 requested、valid、skipped。
如果某个验证批次全部无效，本轮标为 incomplete，不伪造正确率，也不因此阻止训练。
当前默认 24 题为一个验证批次；若今后扩大为多批，遇到全无效批次会将整轮验证
标为 incomplete，不把已完成部分冒充完整验证结果。比较不同检查点时应同时查看
覆盖率和跳过题目，分母变化时不能只比较正确率数字。

为避免全无效题或持续零输出造成无限补采，`sampler_kwargs.max_resample_factor=4`：
每个训练批次最多额外丢弃 4×batch_size 条请求，当前为 96 条；超过后停止并明确报告。
它不是参考执行的重试次数，两者相互独立。

### 18.4 输出、容量与启动注意

- `samples/<sample_id>/discarded.json`：可恢复的单题丢弃，保留原始错误和阶段详情路径。
- `samples/<sample_id>/infrastructure-error.json`：仍需停止的环境或协议错误。
- `scheduler/failures/<verl_uid>.json`：精确关联队列失败组，防止历史日志串用。
- `scheduler/quarantine/<task_id>.json`：本次 run 已确认不可评分的参考题；不改原始数据集。
- `validation/step-<step>-coverage.json`：每轮验证的有效数、跳过数及跳过原因。
- 参考准备/初始基线的第一次失败日志放在阶段目录的 `retry-1/`，最终结果沿用标准文件名。

P2 的 `kernel_overrides.case_storage_limit_bytes` 从 8 GiB 提至 16 GiB；候选与参考
两个输出槽位合计预留 32 GiB，四个活动评测合计最多登记 128 GiB。它是容量上限和
协作预留，不会立即分配这些空间；实际可用空间不足仍停止。P1 独立配置未提高该上限。

配置已变化，旧 run 仍冻结原配置；请用新 run 目录启动。这里不自动重启正式训练，
也不将此前未完成的验证包装成一轮完成的训练。

### 18.5 移除近常数筛选的含义

P1/P2 已删除近常数判定及所有新结果的 `near_constant_sampled_output` 字段，
预检成功的合法常数输出也能进入性能评测。输出形状/dtype、有限值、重复一致性、
五组输入比较、已有隐藏验收、真实 CUDA/允许库调用、状态污染审查均保留。

这不会使 PPO 在数学上失效，但增加容差范围内常数近似、输入无关实现获取奖励的风险。
完全常数且数学正确的任务本身可以合法优化；风险主要是“仅在有限验收输入/容差下
看似正确”的投机。训练只用五组反馈，隐藏验收在检查点运行，不能保证完全排除投机。
该筛选是本地额外启发式，与 Ops-6K 已做的数据去污染不是同一件事。


### 18.6 定向验证与边界

独立证据目录：`.runtime/tmp/p2-recovery-20261108/probe-20261108T100209/`。
GPU 4 重放最新失败的 CPU/CUDA 混用参考，确认恰好两次失败后转入隔离，无模型奖励；
真实 GPU 常数参考预检通过，不再有近常数字段；NaN 参考仍报告 reference_nonfinite。
原生 TransferQueue 联调确认两条失败组被剔除并精确补足两条真实轨迹；零 token
生成结果在 rollout 边界注入后进入丢弃分支。致命环境错误和未知后台失败继续停止。
验证一有效、一无效时，正确率分母为 1，覆盖记录为 requested=2、valid=1、skipped=1；
全无效批次没有伪造输出轨迹。具体断言结果保存在该目录的 probe-result.json。

已有测试夹具同步移除了旧的“近常数应拒绝”预期，未运行 CPU 回归套件。
源码语法/配置组合和 git diff --check 已检查。没有加载 Actor/Critic 权重，
没有启动新的正式 PPO 训练；上述定向结果不等同于已验证完整 150 步训练。

## 19. 原生分块输出头与显存验证（2026-11-08）

### 19.1 改动和原 OOM 原因

`single-ppo-warmup-v2` 在北京时间 15:15:48 的旧策略概率重算阶段失败，
调用栈为 FSDPEngineWithLMHead → entropy_from_logits → softmax。
当时进程已用 70.75 GiB，剩余 5.82 GiB，还需申请 30.31 GiB；
32768×248320×4 字节正好约为 30.31 GiB。旧路径会产生完整词表中间张量。

主 YAML 的 `actor_rollout_ref.model` 现启用 `use_fused_kernels: true`，
显式选择 `fused_kernel_options.impl_backend: torch`。实际调用本项目固定 verl 中
`qwen3_5.forward_with_torch_backend` 和 `FusedLinearForPPO`，默认按 512 token
分块计算输出投影、log_prob 和熵，并分块重算反向，不保留整段词表矩阵。
Actor/Critic、PPO/GAE、权重同步继续使用原生框架；未修改第三方源码或包版本。
正式 32768 上下文、batch=24、minibatch=12、每卡 microbatch=1、seed=42 和奖励协议保留。
本次保留用户已选择的训练 GPU 2、5 和评测 GPU 4、3。

### 19.2 输出头数值和梯度

证据目录：`.runtime/tmp/fused-output-head-20261108/`。
`probe.py` 对 FP32 的概率、熵、隐状态梯度和词表权重梯度与普通 PyTorch 计算比较，
最大前向绝对误差约 9.54e-7，最大梯度绝对误差不超过 3.73e-8。
BF16 真实词表比较通过数值容差；熵输出为 BF16，存在量化舍入，不宣称逐位相等。
真实 hidden_size=4096、vocab_size=248320 的 32768 token 输出头前后向通过，
PyTorch 分配峰值 8.34 GiB，隐状态梯度有限。
这个峰值仅包含输出头测试，不是整个 9B 模型的显存。

### 19.3 完整 9B、32K、双卡 FSDP2 验证

`engine_probe.py` 加载真实 9B 权重，沿用正式 LoRA 和 FSDP2/卸载设置，
每卡构造一条无 padding 的 32768 token 合成序列，检查带熵的旧概率重算，
再以原生 PPO loss 做一次前后向和真实优化器更新。
以下为各进程的 PyTorch 分配峰值，不含其他进程：

| 物理卡 | 32K 旧概率重算峰值 | 32K PPO 更新峰值 | 更新梯度范数 | 改变的抽查 LoRA 分片数 |
| --- | ---: | ---: | ---: | ---: |
| GPU 2 | 14.80 GiB | 26.41 GiB | 13.9375 | 248 |
| GPU 5 | 14.80 GiB | 26.41 GiB | 13.9375 | 248 |

输出概率/熵和梯度有限，抽查 LoRA 参数在原生 optimizer_step 后发生变化。
合成序列仅验证计算和显存，不提供 CUDA 正确率或训练收益结论。

### 19.4 两步在线 PPO 接线

独立 run：`runs/p2/fused-head-wire-20261108`，最终状态 completed。
使用当前真实模型、train 中的题和原生 vLLM/奖励器/GAE/Actor/Critic/权重同步，
临时 profile 采用 --wire 的 8192 总长、每批两题、最多生成 512 token、共两步。
关闭短测的大型检查点保存和验证，正式主 YAML 的预算及保存策略不变。

| Step | Actor 梯度范数 | Critic 梯度范数 | Actor 分配峰值 | Critic 分配峰值 |
| --- | ---: | ---: | ---: | ---: |
| 1 | 2 | 32.5 | 18.44 GiB | 12.90 GiB |
| 2 | 1.71094 | 25.125 | 18.75 GiB | 18.44 GiB |

第二批在第一步权重同步后生成，两步更新均完成；结果保留为原生 TensorBoard 和
`wire.py.log`。短输出可能按格式失败得到 -1，不用于评价 CUDA 生成能力。
验证期间其他任务也使用训练卡，耗时不作为专属资源下的吞吐基准。

完整证据汇总：`.runtime/tmp/fused-output-head-20261108/validation-result.json`。
未运行完整 CPU 回归套件，也未自动启动正式 batch=24、150 步训练；
上述证据证明定向路径通过，不保证今后所有 32K 样本及共享资源状态都不会 OOM。

配置身份已变化，原 v2 保留故障证据且没有检查点，正式运行请使用新目录：

```bash
bash scripts/p2-train.sh --run runs/p2/single-ppo-warmup-v3
```

## 20. 评测大输入显存修复（2026-11-09）

最新 `single-ppo-warmup-v2` 已完成三步更新，第四批在 rollout/reward 的
`reference_bench` 阶段失败，任务为 `ops6k-06855d7278a53b6a`。
这次不是 Actor 输出头重新出现 OOM。该题原始 FP32 输入 `[64, 64, 1000000]`
单组约 15.26 GiB；旧共享 worker 预加载五组，且测量需要额外副本，至少约
91.55 GiB 的输入相关显存，尚未计入输出等。外部任务占卡会提前触发失败，
但即使独占 80 GiB 卡也不能解决旧实现的这一容量问题。

共享 `src/p1/worker.py` 已改成逐组加载、完成本组测量后释放，同时及时释放
预热、测时、profiler 与首次编译调用的副本。P2 的初始参考基线、候选测时和
后测基线直接复用此修复，不增加 P2 专用评测分支。

五组正确性检查、固定种子、原输入规模/精度、每条路径 1×10 次测量、
中位数统计和后测基线奖励规则保留；测量按组执行的顺序发生变化。
论文的五组输入要求针对正确性，官方 profiling 只使用一组输入；
本地五组测时覆盖仍是本地扩展，预加载五组并非论文要求。
详细对照与来源见 [P1_REPORT 第 26 节](../p1/P1_REPORT.md#26-测时输入逐组加载与释放2026-11-09)。

物理 GPU 4 的独立沙箱验证已通过小型真实 CUDA 候选全链路，以及原始
15.26 GiB 输入题的参考准备和 eager/compile 测时。大题测时进程 PyTorch
分配峰值 **30.53 GiB**、保留峰值 **30.54 GiB**；输入组间释放及每轮总共
10 次测量也已核对。证据位于 `.runtime/tmp/stream-bench-20261109/result.json`。

旧 v2 故障结果保留，没有启动正式 PPO，也没有重置设备或终止其他任务。
本次证据覆盖此大输入评测路径，不保证其他任意输入、候选工作区和共享资源
状态都不会 OOM；训练卡分块输出头仍沿用第 19 节配置。

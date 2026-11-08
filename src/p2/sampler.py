"""P2 的 verl 失败分流：训练使用原生补采，验证只跳过，不构造伪奖励。

由官方 custom_sampler 接口加载；输入 TransferQueue 状态和按 UID 保存的失败证据，
输出原生 KVBatchMeta 与补采指标。未知失败、OOM、worker 退出和空间不足仍停止。
"""

import json
from pathlib import Path

from verl.trainer.ppo.v1.replay_buffer import ReplayBuffer

from p2.failures import NoValidValidationSamples, failure_path


class StrictReplayBuffer(ReplayBuffer):
    """保留原生同步采样/补采，只对明确分类的单题错误解除整批停止。"""

    def __init__(self, *args, sampler_kwargs, **kwargs):
        """接收 verl 标准参数；补足自定义 sampler 未自动传入的原生补采选项。"""
        self.run = Path(sampler_kwargs["run"])
        self.batch_size = int(sampler_kwargs["train_batch_size"])
        self.discard_limit = self.batch_size * int(sampler_kwargs["max_resample_factor"])
        self.validation_skipped = {}
        self._seen_failures = set()
        self._batch_discarded = 0
        super().__init__(
            *args, sampler_kwargs=sampler_kwargs, sync_refill_failed_groups=True,
            train_batch_size=self.batch_size, gen_batch_size=1, **kwargs,
        )

    def _sync_metadata_from_transfer_queue(self):
        """无输入；同步队列后检查每个失败组的精确 UID，返回原生同步结果。"""
        result = super()._sync_metadata_from_transfer_queue()
        for partition, keys in self.failure_keys.items():
            for uid in keys:
                path = failure_path(self.run, uid)
                if not path.is_file():
                    raise RuntimeError(f"未分类的后台失败，停止而不补采：{partition}/{uid}")
                failure = json.loads(path.read_text(encoding="utf-8"))
                if not failure.get("recoverable") or failure["partition"] != partition:
                    raise RuntimeError(f"P2 环境/协议失败，停止：{failure}; evidence={path}")
                # 单轮 PPO 必须每题一条轨迹，不能把一个组的半成品留给训练。
                if any(key.split("_")[0] == uid for key in self.partitions[partition]):
                    raise RuntimeError(f"失败组含部分成功轨迹，停止以避免错误补齐：{uid}")
                if partition == "val":
                    self.validation_skipped[uid] = failure
                elif uid not in self._seen_failures:
                    self._batch_discarded += 1
                    if self._batch_discarded > self.discard_limit:
                        raise RuntimeError(
                            f"本批补采超过 {self.discard_limit} 条，请检查可用参考题/生成服务；"
                            f"最后证据={path}"
                        )
                self._seen_failures.add(uid)
        return result

    def sample(self, global_steps, partition_id, batch_size):
        """接收框架 step/分区/批量；训练委托原生精确补采，验证不增加题目。"""
        self._batch_discarded = 0
        return super().sample(global_steps, partition_id, batch_size)

    def _materialize_batch(self, partition_id, selected_prompt_uids, partition_snapshot):
        """输入原生选择结果；只返回真实轨迹，验证全无效时明确报告无可用样本。"""
        batch = super()._materialize_batch(partition_id, selected_prompt_uids, partition_snapshot)
        if partition_id == "val" and not batch.keys:
            raise NoValidValidationSamples("当前验证批次全部被丢弃，没有可报告的正确率")
        return batch

"""V1 同步采样的失败保护，由公开 custom_sampler 接口加载。

输入 TransferQueue 状态；环境/rollout 失败时终止批次，不用零奖励填补缺失轨迹。
候选格式/编译/数值失败已有有效 -1 奖励，属于正常成功返回的训练数据。
"""

from verl.trainer.ppo.v1.replay_buffer import ReplayBuffer


class StrictReplayBuffer(ReplayBuffer):
    """保留原生同步采样；任何后台任务失败都显式停止。"""

    def _sync_metadata_from_transfer_queue(self):
        """无输入；更新原生队列状态并拒绝失败组，返回原生返回值。"""
        result = super()._sync_metadata_from_transfer_queue()
        failures = {partition: list(keys) for partition, keys in self.failure_keys.items() if keys}
        if failures:
            raise RuntimeError(f"P2 后台采样/奖励任务失败，未生成伪奖励；请检查样本错误日志：{failures}")
        return result

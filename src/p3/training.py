"""必要的组失败/验证覆盖率边界；GRPO、采样补齐和训练更新沿用 verl。

由 p3.train 选用本模块的原生 TaskRunner 子类。输入框架队列及已分类错误，
输出整组丢弃/原生补采与准确的验证分母；不手写优势、loss 或优化器更新。
"""

import json
from pathlib import Path

import ray
from verl.trainer.ppo.v1 import PPOTrainerSync
from verl.trainer.ppo.v1.replay_buffer import ReplayBuffer

from p1.evaluation import dump
from p2.failures import NoValidValidationSamples


class GroupReplayBuffer(ReplayBuffer):
    """原生同步补采的薄扩展：环境故障停止，可恢复错误整组清除，绝不凑残缺 GRPO 组。"""

    def __init__(self, *args, sampler_kwargs, **kwargs):
        """输入原生构造参数；记录 run、组大小和有限补采上限。"""
        self.run = Path(sampler_kwargs['run'])
        self.group_size = int(sampler_kwargs['group_size'])
        self.batch_size = int(sampler_kwargs['train_batch_size'])
        self.limit = self.batch_size * int(sampler_kwargs['max_resample_factor'])
        self.validation_skipped = {}
        self.seen = set()
        self.discarded = 0
        super().__init__(
            *args, sampler_kwargs=sampler_kwargs, sync_refill_failed_groups=True,
            train_batch_size=self.batch_size, gen_batch_size=1, **kwargs,
        )

    def _sync_metadata_from_transfer_queue(self):
        """同步原生元数据；校验终态组完整性，并保留底层致命失败原因。"""
        super()._sync_metadata_from_transfer_queue()
        for partition, uids in self.failure_keys.items():
            for uid in uids:
                path = self.run / 'scheduler/failures' / f'{uid}.json'
                if not path.exists():
                    raise RuntimeError(f'Unclassified group failure: {uid}')
                failure = json.loads(path.read_text())
                if not failure['recoverable']:
                    raise RuntimeError(f'Fatal environment failure: {failure}; {path}')
                if partition == 'val':
                    self.validation_skipped[uid] = failure
                elif uid not in self.seen:
                    self.discarded += 1
                    if self.discarded > self.limit:
                        raise RuntimeError(f'Resample limit exceeded; last failure: {path}')
                self.seen.add(uid)
        for uid in self.finished_keys['train']:
            keys = [key for key in self.partitions['train'] if key.split('_')[0] == uid]
            if len(keys) != self.group_size:
                raise RuntimeError(f'Incomplete GRPO group {uid}: {len(keys)}/{self.group_size}')

    def _terminal_eviction_reasons(self, global_steps, partition_id):
        """输入原生调度参数；训练失败组即使有部分成功轨迹也整组丢弃，由原生补采替换。"""
        stale, filtered, failed, counts = super()._terminal_eviction_reasons(global_steps, partition_id)
        if partition_id == 'train':
            failed = failed | set(self.failure_keys['train'])
        return stale, filtered, failed, counts

    def sample(self, global_steps, partition_id, batch_size):
        """输入 step/分区/批量；限制每批补采次数，主体完全调用原生 sample。"""
        self.discarded = 0
        return super().sample(global_steps, partition_id, batch_size)

    def _materialize_batch(self, partition_id, selected_prompt_uids, partition_snapshot):
        """调用原生物化；验证没有有效题时不伪造准确率。"""
        batch = super()._materialize_batch(partition_id, selected_prompt_uids, partition_snapshot)
        if partition_id == 'val' and not batch.keys:
            raise NoValidValidationSamples('No valid validation samples')
        return batch


class CoverageTrainer(PPOTrainerSync):
    """只修正验证跳过分母；不重写训练 step 或 GRPO 更新。"""

    def _validate(self):
        """执行原生验证；全无效时仅报告有效题数为零。"""
        self.replay_buffer.validation_skipped.clear()
        try:
            return super()._validate()
        except NoValidValidationSamples:
            return self.coverage([])

    def coverage(self, uids):
        """输入有效 UID；保存本轮验证覆盖率并返回原生日志字段。"""
        record = dict(
            step=self.global_steps, valid=len(set(uids)),
            skipped=len(self.replay_buffer.validation_skipped),
            requested=len(self.val_dataset),
        )
        dump(Path(self.config.p3.run) / f'validation/step-{self.global_steps}.json', record)
        return {f'val-aux/coverage/{key}': value for key, value in record.items() if key != 'step'}

    def _val_metrics_update(self, data_sources, sample_uids, reward_extra_infos_dict,
                            sample_turns, expected_acc_counts=None):
        """排除参考无效题的期望分母，其余指标全部委托原生验证统计。"""
        expected = ({key: count for key, count in expected_acc_counts.items()
                     if key[1] not in self.replay_buffer.validation_skipped}
                    if expected_acc_counts is not None else None)
        metrics = super()._val_metrics_update(
            data_sources, sample_uids, reward_extra_infos_dict, sample_turns,
            expected_acc_counts=expected,
        )
        return dict(metrics, **self.coverage(sample_uids))


@ray.remote
class TaskRunner:
    """只选择验证覆盖率子类；初始化、Agent 管理和训练仍调用官方公共接口。"""

    def run(self, config):
        """输入配置；初始化原生队列/Trainer/Manager，在退出时关闭本运行资源。"""
        from omegaconf import OmegaConf
        import transfer_queue as tq
        from verl.trainer.ppo.v1 import AgentLoopManagerTQ
        from verl.utils.logging_utils import configure_verl_logging
        configure_verl_logging()
        OmegaConf.resolve(config)
        tq.init(config.transfer_queue)
        trainer = None
        succeeded = False
        try:
            trainer = CoverageTrainer(config=config)
            trainer.init()
            manager = AgentLoopManagerTQ.create(
                config=config, llm_client=trainer.get_llm_client(),
                teacher_client=trainer.get_teacher_client(),
                reward_loop_worker_handles=trainer.get_reward_handles(),
            )
            trainer.fit(manager)
            succeeded = True
        finally:
            try:
                logger = getattr(trainer, 'logger', None)
                if logger is not None:
                    logger.finish(exit_code=0 if succeeded else 1)
            finally:
                tq.close()

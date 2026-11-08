"""P2 验证口径适配和 Ray 启动桥接，PPO 全部使用原生同步训练器。

由 p2.train 通过 run_ppo 的 task_runner_class 接口加载，不单独运行。
输入已组合的 verl 配置；输出原生训练指标及每次验证的有效/跳过数量。
"""

import ray
from omegaconf import OmegaConf
import transfer_queue as tq
from verl.trainer.ppo.v1 import AgentLoopManagerTQ, PPOTrainerSync
from verl.utils.logging_utils import configure_verl_logging

from p2.common import write_json
from p2.failures import NoValidValidationSamples


class CudaPPOTrainerSync(PPOTrainerSync):
    """只调整验证的分母/覆盖率，训练和权重同步原样继承 verl。"""

    def _validate(self):
        """无输入；调用原生验证；全无效批次只报告不完整验证，不伪造准确率。"""
        self.replay_buffer.validation_skipped.clear()
        try:
            return super()._validate()
        except NoValidValidationSamples:
            # 默认验证只有一个 24 题批次。若扩大为多批，遇到全无效批次则本轮
            # 明确标记 incomplete，不能将此前部分批次包装成完整验证结果。
            return self._coverage([], None, incomplete=True)

    def _coverage(self, sample_uids, expected_acc_counts, *, incomplete=False):
        """输入有效 UID 和期望题单；写覆盖率证据并返回可交给原生日志器的指标。"""
        skipped = self.replay_buffer.validation_skipped
        valid = len(set(map(str, sample_uids)))
        requested = len(expected_acc_counts) if expected_acc_counts is not None else len(self.val_dataset)
        record = {
            "step": self.global_steps, "requested": requested, "valid": valid,
            "skipped": len(skipped), "incomplete": incomplete,
            "skipped_details": list(skipped.values()),
        }
        write_json(f"{self.config.p2.run}/validation/step-{self.global_steps}-coverage.json", record)
        return {f"val-aux/coverage/{key}": int(record[key])
                for key in ("requested", "valid", "skipped", "incomplete")}

    def _val_metrics_update(self, data_sources, sample_uids, reward_extra_infos_dict,
                            sample_turns, expected_acc_counts=None):
        """输入原生已完成轨迹；排除无效参考的期望分母，再委托原生指标计算。"""
        coverage = self._coverage(sample_uids, expected_acc_counts)
        skipped = self.replay_buffer.validation_skipped
        expected = ({key: count for key, count in expected_acc_counts.items() if key[1] not in skipped}
                    if expected_acc_counts is not None else None)
        metrics = super()._val_metrics_update(
            data_sources, sample_uids, reward_extra_infos_dict, sample_turns,
            expected_acc_counts=expected,
        )
        return dict(metrics, **coverage)


@ray.remote
class CudaTaskRunner:
    """沿用原生 TaskRunner 的初始化/清理步骤，仅选择上面的验证子类。"""

    def run(self, config):
        """接收已组合配置；初始化原生 Trainer/AgentLoopManager，调用原生 fit。"""
        configure_verl_logging()
        OmegaConf.resolve(config)
        tq.init(config.transfer_queue)
        trainer = None
        succeeded = False
        try:
            trainer = CudaPPOTrainerSync(config=config)
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
                tracking = getattr(trainer, "logger", None)
                if tracking is not None:
                    tracking.finish(exit_code=0 if succeeded else 1)
            finally:
                tq.close()

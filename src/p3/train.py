"""P3 启动入口：唯一 YAML + 原始 JSONL → 原生 verl Agent/GRPO。

完整运行：bash scripts/p3-train.sh --run runs/p3/agent-grpo
短接线：bash scripts/p3-train.sh --run runs/p3/wire --wire
支持 --resume、--steps、--task-id、--limit 和末尾 verl.key=value 原生覆盖。
运行目录只冻结 config.yaml，不生成转换数据、额外数据清单或配置镜像 JSON。
"""

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import sys

from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[2]


def compose(args, overrides):
    """输入 CLI/原生覆盖；返回组合配置，路径和工具 schema 仅在内存补充。"""
    import hydra
    with hydra.initialize_config_dir(
        config_dir=str(ROOT / 'third_party/verl/verl/trainer/config'), version_base=None,
    ):
        config = hydra.compose(config_name='ppo_trainer')
    OmegaConf.set_struct(config, False)
    config = OmegaConf.merge(config, OmegaConf.load(ROOT / args.config), OmegaConf.from_dotlist(overrides))
    run = (ROOT / args.run).resolve()
    config.p3.update(
        root=str(ROOT), run=str(run), run_name=run.name,
        training_gpu_count=len(config.p3.training_gpus),
        ray_temp=f'/tmp/ca-p3-{os.getuid()}-{sha256(str(run).encode()).hexdigest()[:8]}',
    )
    for split in ('train_files', 'val_files'):
        config.data[split] = [str((ROOT / path).resolve()) for path in config.data[split]]
    config.actor_rollout_ref.model.path = str((ROOT / config.actor_rollout_ref.model.path).resolve())
    if args.limit is not None:
        config.data.train_max_samples = args.limit
    if args.task_id:
        config.data.task_ids = args.task_id
    if args.steps is not None:
        config.trainer.total_training_steps = args.steps
    if args.wire:
        # 仅限制接线样本/回答预算，不悄悄修改正式 YAML 的 128K 总上下文。
        config.data.train_batch_size = 2
        config.actor_rollout_ref.actor.ppo_mini_batch_size = 2
        config.actor_rollout_ref.rollout.n = 2
        config.data.max_response_length = 8192
        config.p3.training_turns = 8
        config.trainer.total_training_steps = min(args.steps or 2, 2)
        config.trainer.val_before_train = False
        config.trainer.save_freq = -1
        config.trainer.test_freq = -1
    config.trainer.resume_mode = 'auto' if args.resume else 'disable'
    # 使用官方工具 schema/文件操作，只缩短 Bash 描述并去掉旧 submit 的隐藏验收表述。
    from p1.tools import TOOLS
    schemas = {item['function']['name']: dict(item) for item in TOOLS}
    config.tools = []
    descriptions = {
        'Bash': 'Run CPU commands in the persistent shell. GPU jobs use the supplied utils commands. Query a running bash_id with BashOutput.',
        'submit': 'Finish this trajectory. Its fastest correct measured candidate receives the terminal reward.',
    }
    for name in config.p3.tools:
        schema = json.loads(json.dumps(schemas[name]))
        if name in descriptions:
            schema['function']['description'] = descriptions[name]
        config.tools.append({'class_name': 'p3.agent.WorkspaceTool', 'config': {'type': 'native'}, 'tool_schema': schema})
    OmegaConf.resolve(config)
    return config


def main():
    """解析参数并初始化本运行；只有实际训练才写配置、启动 Ray 和使用 GPU。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/p3-agent-grpo.yaml')
    parser.add_argument('--run', required=True)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--wire', action='store_true')
    parser.add_argument('--steps', type=int)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--task-id', nargs='+')
    # 先取出原生 key=value，避免被 --task-id 的 nargs='+' 误当成题号。
    overrides = [item for item in sys.argv[1:] if '=' in item and not item.startswith('--')]
    args, unknown = parser.parse_known_args([item for item in sys.argv[1:] if item not in overrides])
    if unknown:
        parser.error('Unknown option; native overrides must use key=value')
    config = compose(args, overrides)
    training = list(config.p3.training_gpus)
    evaluation = list(config.p3.kernel.evaluation_gpu)
    if not training or len(training) != len(set(training)) or set(training) & set(evaluation):
        raise ValueError('Training/evaluation GPU numbers must be nonempty, unique and disjoint')
    if config.algorithm.adv_estimator != 'grpo' or int(config.actor_rollout_ref.rollout.n) < 2:
        raise ValueError('P3 requires GRPO with at least two trajectories per prompt')
    if config.critic.enable or config.p3.kernel.feedback_cases != 5:
        raise ValueError('P3 has no Critic and retains five correctness cases')
    if args.dry_run:
        print(OmegaConf.to_yaml(config))
        return
    run = Path(config.p3.run)
    if args.resume and not (run / 'checkpoints/latest_checkpointed_iteration.txt').exists():
        raise FileNotFoundError('No completed checkpoint to resume')
    if not args.resume and (run / 'config.yaml').exists():
        raise FileExistsError('Run already exists; choose a new run or use --resume')
    os.environ.update(
        CUDA_VISIBLE_DEVICES=','.join(map(str, training)), CUDA_DEVICE_ORDER='PCI_BUS_ID',
        TZ='Asia/Shanghai', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
        RAY_ADDRESS='local', TENSORBOARD_DIR=str(run / 'tensorboard'),
    )
    from p1.evaluation import dump
    from p1.helper.baseline_state import beijing_now
    from verl.trainer.main_ppo import run_ppo
    from verl.trainer.ppo.utils import need_critic, need_reference_policy
    from verl.utils.config import validate_config
    from .training import TaskRunner
    import ray
    validate_config(config, use_reference_policy=need_reference_policy(config), use_critic=need_critic(config))
    run.mkdir(parents=True, exist_ok=True)
    # 一个原生展开配置同时充当工具配置及运行快照，不维护第二份手工工具 YAML。
    (run / 'config.yaml').write_text(OmegaConf.to_yaml(config))
    dump(run / 'status.json', {'status': 'running', 'at': beijing_now().isoformat()})
    try:
        run_ppo(config, TaskRunner)
        dump(run / 'status.json', {'status': 'completed', 'at': beijing_now().isoformat()})
    except BaseException as error:
        dump(run / 'status.json', {
            'status': 'failed', 'at': beijing_now().isoformat(),
            'error_type': type(error).__name__, 'error': str(error),
        })
        raise
    finally:
        ray.shutdown()


if __name__ == '__main__':
    main()

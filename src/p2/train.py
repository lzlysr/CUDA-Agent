"""P2 verl V1 同步在线 PPO 启动入口。

运行：bash scripts/p2-train.sh --run runs/p2/example [--wire] [--task-id ...] [--steps 1]
入口先轻量检查原始 train/test 并保存运行清单，再启动训练；--prepare-only 仅检查，
--dry-run 只读预览，--resume 复用冻结题单及原生检查点。
输入：P2 配置及已有 train/test.jsonl；输出：轻量清单、冻结配置、原生检查点。
--wire 仅验证短上下文接线，不代表正式 32K 训练或能力复现结果。
"""

import argparse
import faulthandler
import json
import os
from pathlib import Path
import signal

from .common import ROOT, digest, for_mode, now, read_config, write_json
from .data import prepare


def compose_config(project, run, data_paths, steps=None, resume=False, config_path=None):
    """输入配置、run、原始 JSONL 路径和停止步数；返回组合后的 DictConfig。

    PPO/GAE、FSDP、优化器、权重同步及断点续跑均由 verl 原生训练器负责。
    此函数仅填充本次运行路径和配置插值，不实现训练算法。
    """
    import hydra
    from omegaconf import OmegaConf

    directory = ROOT / "third_party/verl/verl/trainer/config"
    with hydra.initialize_config_dir(config_dir=str(directory), version_base=None):
        base = hydra.compose(config_name="ppo_trainer")
    # 允许增加项目 p2 命名空间及当前官方 dataclass 已支持、默认 YAML 尚未列出的字段。
    OmegaConf.set_struct(base, False)
    runtime = dict(
        project, run=str(run), root=str(ROOT), run_name=run.name,
        model_path=str(ROOT / project["model"]),
        training_gpu_count=len(project["training_gpus"]),
        prompt_limit=project["context_tokens"] - 1,
        queue_capacity=max(128, project["batch_size"] * 4),
        stop_step=steps or project["total_steps"], resume_mode="auto" if resume else "disable",
        ray_temp=f"/tmp/ca-p2-{os.getuid()}-{digest(str(run))[:8]}",
    )
    # Unix socket 路径有 107 字节限制，因此 Ray 必须使用专属短临时目录。
    source = ROOT / (config_path or "configs/p2-single-ppo-warmup.yaml")
    config = OmegaConf.merge(
        base, OmegaConf.load(source), {"p2": runtime},
        {"data": {"train_files": [str(data_paths[0])], "val_files": [str(data_paths[1])]}},
    )
    # 当前入口始终读取原始 JSONL，统一使用官方公共 Dataset 扩展。
    config.data.custom_cls = {
        "path": str(ROOT / "src/p2/dataset.py"), "name": "CudaDataset",
    }
    config.data.p2_input_manifest = str(ROOT / project["data_manifest"])
    config.data.p2_input_manifest_hash = project["data_manifest_hash"]
    if project["wire"]:
        # --wire 只验证接线，不在第一步前执行整个 test 集的正式验证。
        config.trainer.val_before_train = False
    return config


def main():
    """输入 CLI；校验数据/配置后启动专属本地 Ray，退出时关闭自己创建的集群。"""
    # 必要时发送 SIGUSR1 只输出 Python 调用栈，不终止训练，便于定位初始化等待。
    faulthandler.register(signal.SIGUSR1, all_threads=True)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/p2-single-ppo-warmup.yaml")
    parser.add_argument("--run", required=True)
    parser.add_argument("--wire", action="store_true")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int, help="首次运行最多选取的 train 题数")
    parser.add_argument("--task-id", nargs="+", help="首次运行选择指定 train 题；续跑不得更换题单")
    parser.add_argument("--prepare-only", action="store_true", help="只做 CPU 轻量检查并保存清单，不启动训练")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    project = for_mode(read_config(args.config), args.wire)
    if args.steps is not None and not 1 <= args.steps <= project["total_steps"]:
        raise ValueError("steps 必须在当前 profile 的预算内")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("limit 必须为正整数")
    run = (ROOT / args.run).resolve()
    frozen = run / "config.json"
    if frozen.exists():
        saved = json.loads(frozen.read_text(encoding="utf-8"))
        if "data_manifest" not in saved or "data_manifest_hash" not in saved:
            raise ValueError("仅支持原始 JSONL 轻量清单的续跑，请使用新 run")
        project["data_manifest"] = saved["data_manifest"]
        project["data_manifest_hash"] = saved["data_manifest_hash"]
        if digest(saved) != digest(project):
            raise ValueError("run 已冻结为不同训练配置；更改训练参数请使用新 run")
    checkpoint = run / "checkpoints/latest_checkpointed_iteration.txt"
    if checkpoint.exists() and not args.resume and not (args.prepare_only or args.dry_run):
        raise FileExistsError("已有检查点，请显式 --resume，避免覆盖")
    if args.resume and not checkpoint.exists():
        raise FileNotFoundError("指定了 --resume，但本 run 尚无已完成的检查点")
    os.environ.update(
        CUDA_VISIBLE_DEVICES=",".join(map(str, project["training_gpus"])),
        CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_AGENT_P2_RUN=str(run),
        TZ="Asia/Shanghai", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
        RAY_ADDRESS="local",
        # verl 原生 TensorBoard 后端读取此目录；每个 run 独立，续跑沿用同一路径。
        TENSORBOARD_DIR=str(run / "tensorboard"),
    )
    # 只检查原始记录并冻结索引；完整 prompt/奖励字段由 Dataset 在读取时补充。
    data_paths, manifest = prepare(project, run, args.limit, args.task_id, args.dry_run)
    task_count = manifest["task_count"]
    if task_count < project["batch_size"]:
        raise ValueError(f"可用记录 {task_count} 少于 batch_size={project['batch_size']}")
    if manifest["val_task_count"] <= 0:
        raise ValueError("test 中没有通过轻量检查的验证记录")
    project["data_manifest"] = (run / "data-manifest.json").relative_to(ROOT).as_posix()
    project["data_manifest_hash"] = digest(manifest)
    from omegaconf import OmegaConf
    config = compose_config(
        project, run, data_paths, args.steps, args.resume, args.config,
    )
    if (
        list(config.data.train_files) != [str(data_paths[0])]
        or list(config.data.val_files) != [str(data_paths[1])]
    ):
        raise ValueError("verl 的输入路径必须与本次冻结的数据清单一致")
    if args.dry_run:
        print(f"只读预览：{task_count} 条，训练输入 {data_paths[0]}")
        print("未写数据或运行文件；未初始化 Ray/GPU")
        return
    if not frozen.exists():
        write_json(frozen, project)
    (run / "verl-resolved.yaml").write_text(OmegaConf.to_yaml(config, resolve=True), encoding="utf-8")
    if args.prepare_only:
        print(f"轻量检查完成：{task_count} 条，训练输入 {data_paths[0]}；未初始化 Ray/GPU")
        return
    from p1.evaluation import Evaluator
    Evaluator(project["kernel"], run / "kernel", gpu=project["kernel"]["evaluation_gpu"][0])
    import ray
    from verl.trainer.main_ppo import TaskRunnerV1, run_ppo
    from verl.trainer.ppo.utils import need_critic, need_reference_policy
    from verl.utils.config import validate_config
    validate_config(config, use_reference_policy=need_reference_policy(config), use_critic=need_critic(config))
    try:
        write_json(run / "status.json", {"status": "running", "at": now().isoformat(), "wire": args.wire})
        run_ppo(config, TaskRunnerV1)
        write_json(run / "status.json", {"status": "completed", "at": now().isoformat(), "wire": args.wire})
    except BaseException as error:
        write_json(run / "status.json", {"status": "failed", "at": now().isoformat(),
                                        "error_type": type(error).__name__, "error": str(error)})
        raise
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()

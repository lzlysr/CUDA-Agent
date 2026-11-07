"""P2 公共配置和记录工具。

由 prepare、train、reward 调用；输入项目路径和 YAML 配置，输出校验后的配置。
所有新时间采用北京时间。本模块不导入 torch，不初始化 GPU。
"""

from datetime import datetime
import hashlib
import json
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[2]


def now():
    """无输入；返回带 Asia/Shanghai 时区的当前时间。"""
    return datetime.now(ZoneInfo("Asia/Shanghai"))


def read_config(path):
    """输入 P2 YAML 路径；读取 p2 参数，合并 P1 协议并检查 GPU/PPO 约束。"""
    from omegaconf import OmegaConf
    document = OmegaConf.load(path)
    config = OmegaConf.to_container(document.p2, resolve=False)
    # 原生 verl 字段也参与身份校验，避免只改 YAML 下半部分就悄悄混用训练协议。
    raw = OmegaConf.to_container(document, resolve=False)
    config["verl_overrides_hash"] = digest({key: value for key, value in raw.items() if key != "p2"})
    # 题目来源直接使用 verl 原生 data 配置；本项目每侧只读取一份原始 JSONL。
    config["data_sources"] = {}
    for split, key in (("train", "train_files"), ("val", "val_files")):
        files = OmegaConf.to_container(document.data[key], resolve=False)
        if not isinstance(files, list) or len(files) != 1 or "${" in files[0]:
            raise ValueError(f"P2 的 {key} 请直接配置一个原始 JSONL 路径")
        config["data_sources"][split] = str(files[0])
    protocol = json.loads((ROOT / config["kernel_config"]).read_text(encoding="utf-8"))
    protocol.update(config["kernel_overrides"])
    config["kernel"] = protocol
    train = config["training_gpus"]
    evaluation = protocol["evaluation_gpu"]
    if not train or any(type(gpu) is not int or gpu < 0 for gpu in train + evaluation):
        raise ValueError("GPU 必须为非负整数编号")
    if len(set(train)) != len(train) or set(train) & set(evaluation):
        raise ValueError("训练卡不能重复，且必须与评测卡隔离")
    if protocol["feedback_cases"] != 5 or protocol["timing_rounds"] != 1:
        raise ValueError("P2 当前协议要求五组反馈检查和一轮测时")
    if protocol["timing_iterations"] != 10 or protocol["seed"] != config["seed"]:
        raise ValueError("P2 要求 1×10 测时，且生成/评测使用同一固定 seed")
    if config["batch_size"] % config["mini_batch_size"]:
        raise ValueError("batch_size 必须能被 mini_batch_size 整除")
    if config["mini_batch_size"] % len(train):
        raise ValueError("mini_batch_size 必须能被训练卡数整除")
    return config


def digest(value):
    """输入可 JSON 序列化对象；返回稳定 SHA256，用于数据和配置身份校验。"""
    content = json.dumps(value, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(content).hexdigest()


def file_digest(path):
    """输入文件路径；分块计算 SHA256，用于检查训练输入是否被替换。"""
    checksum = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def for_mode(config, wire):
    """输入正式配置和接线标记；返回独立配置，短测不改动正式参数文件。"""
    config = json.loads(json.dumps(config))
    config["wire"] = bool(wire)
    if wire:
        config.update(context_tokens=config["wire_context_tokens"], batch_size=config["wire_batch_size"],
                      mini_batch_size=config["wire_batch_size"], total_steps=2,
                      checkpoint_every=1, checkpoint_eval_every=-1)
    return config


def write_json(path, value):
    """输入输出路径和对象；原子写 JSON，避免中断留下半份索引。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)

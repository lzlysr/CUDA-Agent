"""P2 入口内的轻量选题与检查；不保存补充字段后的完整训练数据。

由 p2.train 调用，不再提供独立启动脚本。运行示例：
bash scripts/p2-train.sh --run runs/p2/example --wire --limit 2 --prepare-only
输入为已有 train.jsonl 和提示词依赖，输出为 run/data-manifest.json。
清单只保存原始行索引、输入摘要和轻量检查结果，不复制题目和完整提示词。
不执行题目、不启动 Docker/GPU。训练读取 train，验证直接读取已有 test。
"""

import ast
import json
from pathlib import Path
import random

from .common import ROOT, digest, file_digest, now, write_json


def prompt_template():
    """无输入；读取公共规则和绑定，返回 system 文本及 user 公共前缀。

    调用方在一次检查或一个 Dataset 内只读取一次，避免每题重复读公共文件。
    """
    # 复用 P1 的约束文本；P2 没有 shell 工具，也没有生成后的补写机会。
    from p1.run import CONTRACT

    official = ROOT / "upstream/CUDA-Agent/agent_workdir"
    system = (
        CONTRACT + "\n\n## Official CUDA development SKILL\n"
        + (official / "SKILL.md").read_text()
        + "\n\n## Local environment overrides and runtime rules\n"
        + "Local environment overrides SKILL examples: target sm_80, not sm_90; do not use sudo. "
        "The supplied binding and model/utils files are read-only. "
        "These CUDA restrictions and optimization guidelines apply to both modes.\n"
        "The SKILL workflow is guidance only for this single mode: "
        "there are no tools or execution feedback.\n"
    )
    user = (
        'Generate one complete solution with no execution feedback. Return only a JSON object '
        '{"files":{"model_new.py":"...","kernels/kernel.cu":"...","kernels/kernel_binding.cpp":"..."}} '
        'after your reasoning. All file contents must be complete. model_new.py must define '
        'class ModelNew and import cuda_extension directly. Only model_new.py and source files '
        'directly under kernels/ are writable; omit binding.cpp, binding_registry.h, model.py, '
        'utils/, setup.py and other files. Escape source newlines inside JSON strings. '
        'Do not replace the JSON object with Markdown Python/CUDA code blocks.\n'
        "Deliver only the allowed candidate files; do not include auxiliary scripts/notebooks. "
        "Invalid/incomplete JSON is a failed single attempt; no automatic repair or extra generation.\n"
    )
    for name in ("binding.cpp", "binding_registry.h"):
        user += f"\nFixed {name}:\n```cpp\n{(official / name).read_text()}\n```\n"
    return system, user


def make_prompt(task, template=None):
    """输入原题和可复用的公共文本；返回与原 P2 相同的 system/user 消息。"""
    system, user = template if template is not None else prompt_template()
    user += f'Reference task (read-only):\n```python\n{task["code"]}\n```'
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def make_row(task, hidden=False, template=None):
    """输入原始记录及验收标记；运行时返回 verl 所需字段，不写任何数据文件。"""
    return {
        "data_source": "cuda_agent_ops6k",
        "prompt": make_prompt(task, template),
        "ability": "cuda_kernel",
        "reward_model": {"style": "rule", "ground_truth": task["task_id"]},
        "extra_info": {
            "index": task["task_id"], "task_json": json.dumps(task), "hidden": hidden,
        },
    }


def read_tasks(source, expected_split="train"):
    """输入原始 JSONL 路径和预期划分名；返回记录，检查归属与行索引唯一性。"""
    tasks = [
        json.loads(line) for line in Path(source).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if any(task.get("split") != expected_split for task in tasks):
        raise ValueError(f"{source} 中混入非 {expected_split} 任务")
    if len({task["row_index"] for task in tasks}) != len(tasks):
        raise ValueError(f"{source} 中存在重复的原始 row_index")
    return tasks


def load_manifest(path):
    """输入轻量运行清单路径；返回已核对输入模式和身份的清单。"""
    path = Path(path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("input_mode") != "runtime":
        raise ValueError(f"仅支持原始 JSONL 的运行时适配清单，请使用新 run：{path}")
    if manifest["data_id"] != digest(manifest["identity"]):
        raise ValueError(f"运行输入清单身份不匹配：{path}")
    return manifest


def check_tasks(tasks, config, template):
    """输入选题、配置和公共文本；返回通过样本的索引/长度及静态拒绝原因。

    只在内存中临时构造 prompt 来统计长度，不保留补充字段后的完整样本。
    AST 不执行原始代码，GPU 兼容性及数值检查仍由真实奖励评测负责。
    """
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(ROOT / config["model"], local_files_only=True)
    samples, rejected = [], []
    for task in tasks:
        entry = {"row_index": task["row_index"], "task_id": task["task_id"]}
        try:
            tree = ast.parse(task["code"])
            names = {
                node.name for node in tree.body
                if isinstance(node, (ast.ClassDef, ast.FunctionDef))
            }
            if not {"Model", "get_init_inputs", "get_inputs"}.issubset(names):
                raise ValueError("缺少 Model/get_init_inputs/get_inputs 任务接口")
        except (SyntaxError, ValueError) as error:
            rejected.append(dict(entry, reason=str(error)))
            continue
        length = len(tokenizer.apply_chat_template(
            make_prompt(task, template), tokenize=True,
            add_generation_prompt=True, return_dict=False,
        ))
        if length >= config["context_tokens"]:
            rejected.append(dict(entry, reason="prompt_exceeds_context", tokens=length))
            continue
        samples.append(dict(entry, prompt_tokens=length))
    return samples, rejected


def prepare(config, run, limit=None, task_ids=None, dry_run=False):
    """输入配置、run、选题和只读标记；返回 (框架输入路径对, 轻量运行清单)。

    新运行直接使用配置中的 train.jsonl/test.jsonl，不从 train 抽监控题。
    两侧只保存记录索引和摘要，完整题目及 prompt 不复制到运行目录。
    续跑按原始 row_index 恢复确切记录及顺序；task_id 是代码组，不能用于去重。
    """
    run = Path(run).resolve()
    saved = config.get("data_manifest")
    path = ROOT / saved if saved else run / "data-manifest.json"
    manifest = load_manifest(path) if path.exists() else None
    if saved and manifest is None:
        raise FileNotFoundError(f"冻结的数据清单缺失：{path}")
    if manifest is not None and config.get("data_manifest_hash"):
        if digest(manifest) != config["data_manifest_hash"]:
            raise ValueError("运行输入清单被修改，不能继续使用原检查点")

    source = (ROOT / config["data_sources"]["train"]).resolve()
    val_source = (ROOT / config["data_sources"]["val"]).resolve()
    if source == val_source:
        raise ValueError("P2 训练和验证不能指向同一份原始数据")
    tasks = read_tasks(source)
    validation = read_tasks(val_source, expected_split="test")
    if {task["row_index"] for task in tasks} & {task["row_index"] for task in validation}:
        raise ValueError("train 和验证 test 含有相同的原始 row_index")
    random.Random(config["seed"]).shuffle(tasks)
    if manifest is not None and task_ids is None and limit is None:
        # 按清单恢复确切顺序；同组多行和已拒绝的请求记录都保留。
        by_row = {task["row_index"]: task for task in tasks}
        tasks = [by_row[index] for index in manifest["identity"]["selection_rows"]]
    elif task_ids is not None:
        selected = set(task_ids)
        if len(selected) != len(task_ids):
            raise ValueError("--task-id 不能重复")
        tasks = [task for task in tasks if task["task_id"] in selected]
        if {task["task_id"] for task in tasks} != selected:
            raise ValueError("部分 task-id 不在 train 中")
    tasks = tasks[:limit] if limit is not None else tasks
    template = prompt_template()
    tokenizer_files = [
        "tokenizer.json", "tokenizer_config.json",
        "chat_template.jinja", "special_tokens_map.json",
    ]
    identity = {
        "sources": {
            "train": {"path": source.relative_to(ROOT).as_posix(), "sha256": file_digest(source)},
            "val": {"path": val_source.relative_to(ROOT).as_posix(), "sha256": file_digest(val_source)},
        },
        "adapter_sha256": file_digest(Path(__file__)),
        "dataset_sha256": file_digest(Path(__file__).with_name("dataset.py")),
        "selection_rows": [task["row_index"] for task in tasks],
        "seed": config["seed"],
        "context_tokens": config["context_tokens"],
        "prompt_template_sha256": digest(make_prompt({"code": ""}, template)),
        "tokenizer_files": {
            name: file_digest(ROOT / config["model"] / name)
            for name in tokenizer_files if (ROOT / config["model"] / name).is_file()
        },
    }
    if manifest is not None:
        if manifest["identity"] != identity:
            raise ValueError("原始数据、题单或提示词等输入已变化，请使用新的 run")
        print(
            f"复用运行题单：train={manifest['task_count']}，val={manifest['val_task_count']}",
            flush=True,
        )
        return (source, val_source), manifest

    samples, rejected = check_tasks(tasks, config, template)
    val_samples, val_rejected = check_tasks(validation, config, template)
    manifest = {
        "created_at": now().isoformat(),
        "input_mode": "runtime",
        "data_id": digest(identity),
        "identity": identity,
        "task_count": len(samples),
        "val_task_count": len(val_samples),
        "samples": {"train": samples, "val": val_samples},
        "rejected": {"train": rejected, "val": val_rejected},
    }
    if not dry_run:
        write_json(path, manifest)
    print(
        f"轻量检查：train={len(samples)}、val={len(val_samples)}；"
        f"静态拒绝 train={len(rejected)}、val={len(val_rejected)}；只保存清单，不复制题目",
        flush=True,
    )
    return (source, val_source), manifest

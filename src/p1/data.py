"""整理 Ops-6K，按任务组划分 train/test=8:2，并从 train 固定抽取 pilot。

运行：项目根目录执行 PYTHONPATH=src python -B -m p1.data。
输入：原始 Parquet、configs/p1-pilot.json 和 KernelBench 源码。
输出：data/processed 下的 tasks/train/test.jsonl、pilot.json 与数据审计。
只处理 CPU 数据；不启动模型、GPU 或 KernelBench 实验。"""

import ast
import hashlib
import json
from pathlib import Path
from collections import Counter, defaultdict
import math
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]


def sha(x):
    """输入文本或 bytes；返回 SHA256 指纹，用于版本/数据追溯。"""
    return hashlib.sha256(x if isinstance(x, bytes) else x.encode()).hexdigest()


def dump(p, x):
    """输入 JSON 路径及对象；创建父目录并写入 UTF-8 JSON，返回 None。"""
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(x, ensure_ascii=False, indent=2) + "\n")


def normalized(code):
    """输入 Python 源码；返回去除文档字符串和位置后的 AST 文本。"""
    tree = ast.parse(code)
    for node in ast.walk(tree):
        if hasattr(node, "body") and isinstance(node.body, list):
            node.body = [
                x
                for x in node.body
                if not (
                    isinstance(x, ast.Expr)
                    and isinstance(x.value, ast.Constant)
                    and isinstance(x.value.value, str)
                )
            ]
    return ast.dump(tree, include_attributes=False)


def features(code):
    """输入源码；返回 AST 类型/名称三元组，用于近似相似性筛查。"""
    nodes = []
    for n in ast.walk(ast.parse(code)):
        value = getattr(n, "attr", getattr(n, "id", ""))
        nodes.append(type(n).__name__ + ":" + value)
    return set(zip(nodes, nodes[1:], nodes[2:]))


def split_groups(groups, seed=42, train_ratio=0.8):
    """输入哈希→记录组、种子和训练比例；返回哈希→train/test 的映射。

    同组不跨集合；按 data_source 分层，用最大余数法使 test 组数精确等于
    round(总组数×0.2)。仅由来源、源码哈希和种子决定归属。
    重复组大小不同，所以原始行数比例可能略偏。
    """
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio 必须位于 0 与 1 之间")
    buckets = defaultdict(list)
    for digest, items in groups.items():
        buckets[min(item["data_source"] for item in items)].append(digest)
    ratio = 1 - train_ratio
    target = round(len(groups) * ratio)
    counts = {source: math.floor(len(values) * ratio) for source, values in buckets.items()}
    order = sorted(
        buckets, key=lambda source: (-(len(buckets[source]) * ratio - counts[source]), source)
    )
    for source in order[: target - sum(counts.values())]:
        counts[source] += 1
    assignments = {}
    for source, hashes in sorted(buckets.items()):
        # 固定排序盐用于保持现有划分稳定，不作为配置或审计的版本标签。
        ordered = sorted(hashes, key=lambda digest: sha(f"{seed}:train-test-v2:{source}:{digest}"))
        selected_test = set(ordered[: counts[source]])
        for digest in ordered:
            assignments[digest] = "test" if digest in selected_test else "train"
    return assignments


def prepare():
    """无命令行输入；读取配置和 Parquet，写入 train/test、train pilot 与审计，返回 None。"""
    cfg = json.loads((ROOT / "configs/p1-pilot.json").read_text())
    raw = ROOT / "data/raw/ops6k.parquet"
    assert (
        sha(raw.read_bytes()) == "d4404fc181cad72081f8c1cba720cc1b511fb34dee924079e898bf1e83f6257c"
    )
    rows = pq.read_table(raw).to_pylist()
    groups = defaultdict(list)
    invalid = []
    for i, row in enumerate(rows):
        try:
            tree = ast.parse(row["code"])
            names = {getattr(n, "name", "") for n in tree.body}
            if not {"Model", "get_inputs", "get_init_inputs"} <= names:
                raise ValueError("missing required task interface")
            h = sha(normalized(row["code"]))
            groups[h].append(
                dict(row, row_index=i, source_hash=sha(row["code"]), normalized_hash=h)
            )
        except Exception as e:
            invalid.append({"row_index": i, "reason": str(e)})
    assignments = split_groups(groups, cfg["seed"], cfg.get("train_ratio", 0.8))
    records = []
    for h, items in sorted(groups.items()):
        for x in items:
            records.append(dict(x, task_id="ops6k-" + h[:16], split=assignments[h]))
    out = ROOT / "data/processed"
    out.mkdir(parents=True, exist_ok=True)
    (out / "tasks.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records)
    )
    for split in ["train", "test"]:
        (out / f"{split}.jsonl").write_text(
            "".join(
                json.dumps(r, ensure_ascii=False) + "\n" for r in records if r["split"] == split
            )
        )
    if cfg.get("pilot_split", "train") != "train":
        raise ValueError("当前用户约定要求 pilot 从 train 选取")
    train = {x["task_id"]: x for x in records if x["split"] == "train"}
    pilot = sorted(
        train.values(), key=lambda x: sha(f'{cfg["seed"]}:pilot-v3:{x["normalized_hash"]}')
    )[: cfg["pilot_tasks"]]
    kb = []
    counts = {}
    for level in (1, 2, 3):
        paths = sorted((ROOT / f"upstream/KernelBench/KernelBench/level{level}").glob("*.py"))
        counts[f"level{level}"] = len(paths)
        kb += [
            (str(p.relative_to(ROOT)), sha(normalized(p.read_text())), features(p.read_text()))
            for p in paths
        ]
    assert counts == {"level1": 100, "level2": 100, "level3": 50}, counts
    audit = []
    for h, items in sorted(groups.items()):
        f = features(items[0]["code"])
        best = max(
            ((len(f & kf) / max(1, len(f | kf)), name, kh == h) for name, kh, kf in kb),
            default=(0, None, False),
        )
        audit.append(
            {
                "task_id": "ops6k-" + h[:16],
                "max_ast_trigram_jaccard": best[0],
                "nearest_kernelbench": best[1],
                "normalized_exact_match": any(kh == h for _, kh, _ in kb),
            }
        )
    dump(out / "pilot.json", pilot)
    dump(
        ROOT / "reports/p1/data-audit.json",
        {
            "train_ratio": cfg.get("train_ratio", 0.8),
            "split_unit": "normalized_source_group",
            "pilot_split": "train",
            "raw_sha256": sha(raw.read_bytes()),
            "revision": cfg["dataset_revision"],
            "rows": len(rows),
            "unique_groups": len(groups),
            "syntax_or_interface_invalid": invalid,
            "group_splits": dict(Counter(assignments.values())),
            "row_splits": dict(Counter(x["split"] for x in records)),
            "source_counts": dict(Counter(x["data_source"] for x in rows)),
            "kernelbench_counts": counts,
            "pilot_task_ids": [x["task_id"] for x in pilot],
            "similarity_method": "AST node type/name trigrams Jaccard; screening evidence, not proof of no contamination",
            "exact_matches": sum(x["normalized_exact_match"] for x in audit),
        },
    )
    dump(ROOT / "reports/p1/kernelbench-similarity.json", audit)
    print(
        json.dumps(
            {
                "pilot": [(x["task_id"], x["data_source"], x["ops"]) for x in pilot],
                "rows": len(rows),
                "groups": len(groups),
                "counts": counts,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    prepare()

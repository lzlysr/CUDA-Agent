"""verl 的 P2 Dataset 扩展：从原始 train 读取记录，取样时补充模型和奖励字段。

由 p2.train 通过官方 data.custom_cls 接口配置，无需独立运行。
输入：原始 train.jsonl 或 test.jsonl、运行清单、框架 tokenizer/config。
输出：含 raw_prompt、奖励元信息和框架占位张量的样本，不写转换后的 JSONL。
继承 RLHFDataset 的文本/多模态接口及状态恢复；不实现采样器或 PPO 训练循环。
"""

from pathlib import Path

import datasets
import numpy as np
import torch
from verl.utils.dataset.rl_dataset import RLHFDataset

from p2.common import ROOT, digest, file_digest
from p2.data import load_manifest, make_prompt, make_row, prompt_template, read_tasks


class CudaDataset(RLHFDataset):
    """保留原始记录在内存中；每次 __getitem__ 才构造完整的 verl 样本。"""

    def _download(self, use_origin_parquet=False):
        """接收框架恢复标记；本项目使用本机原始文件，无需下载或复制缓存。"""
        return None

    def _read_files_and_tokenize(self):
        """无输入；核对冻结清单并载入对应原始行，不预存补充字段后的样本。"""
        manifest_path = Path(self.config.p2_input_manifest).resolve()
        manifest = load_manifest(manifest_path)
        if digest(manifest) != self.config.p2_input_manifest_hash:
            raise ValueError("Dataset 读取的清单与启动时冻结的清单不一致")
        identity = manifest["identity"]
        if len(self.original_data_files) != 1:
            raise ValueError("P2 每个 Dataset 只接收一个原始 JSONL 来源")
        entry = Path(self.original_data_files[0]).resolve()
        matches = [
            split for split, source in identity["sources"].items()
            if (ROOT / source["path"]).resolve() == entry
        ]
        if len(matches) != 1:
            raise ValueError("P2 Dataset 的输入路径与运行清单不匹配")
        split = matches[0]
        if file_digest(entry) != identity["sources"][split]["sha256"]:
            raise ValueError(f"原始 {split} 文件已变化，不能复用冻结题单")

        # 当前 verl 不把 is_train 传入自定义类，直接按原始文件路径识别用途。
        # 验证的 hidden=true 仅决定额外正确性验收，不表示来自 train 的隐藏题。
        self.hidden = split == "val"
        samples = manifest["samples"][split]
        by_row = {
            task["row_index"]: task
            for task in read_tasks(entry, expected_split="test" if self.hidden else "train")
        }
        records = []
        for sample in samples:
            task = by_row[sample["row_index"]]
            if task["task_id"] != sample["task_id"]:
                raise ValueError("原始行与清单中的 task_id 不一致")
            records.append(task)
        self.dataframe = datasets.Dataset.from_list(records)
        # 保留原生 max_samples/shuffle 语义；正常配置为 -1，使用清单中的全部记录。
        if 0 < self.max_samples < len(records):
            indices = (
                np.random.default_rng(self.seed).choice(len(records), size=self.max_samples, replace=False)
                if self.shuffle else range(self.max_samples)
            )
            self.dataframe = self.dataframe.select(list(indices))

        # 公共文本每个 Dataset 只读一次，源码内容与原有 P2 提示词保持一致。
        self.template = prompt_template()
        if digest(make_prompt({"code": ""}, self.template)) != identity["prompt_template_sha256"]:
            raise ValueError("提示词公共文件已变化，不能复用冻结题单")
        print(f"P2 Dataset：{len(self.dataframe)} 条，hidden={self.hidden}，运行时补充字段")

    def __getitem__(self, item):
        """输入位置索引；返回完整框架样本，原始任务只在本次内存记录中序列化。"""
        task = self.dataframe[item]
        row = make_row(task, hidden=self.hidden, template=self.template)
        row["raw_prompt"] = self._build_messages(row, key="prompt")
        # 与原生 RLHFDataset 相同：DataProto 需要一个非空 tensor batch。
        row["dummy_tensor"] = torch.tensor([0], dtype=torch.uint8)
        row["index"] = task["task_id"]
        row["tools_kwargs"] = {}
        row["interaction_kwargs"] = {}
        return row

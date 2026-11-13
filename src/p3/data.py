"""原始 Ops-6K JSONL 的薄 Dataset 和提示词适配。

由 p3.train 的 data.custom_cls 加载，不单独运行。
输入原始 train/test 文件，输出 verl 样本；不生成转换数据或另建数据清单。
样本数量、随机选取和长提示过滤使用 RLHFDataset 的原生实现。
"""

from pathlib import Path

import torch
from verl.utils.dataset.rl_dataset import RLHFDataset

ROOT = Path(__file__).resolve().parents[2]


def messages(task, skill):
    """输入原题及官方 SKILL；返回仅含必要环境差异和原题的 system/user 消息。"""
    local = (
        "\n\nLocal environment: A100 (sm_80); no sudo or network. "
        "Bash is a persistent CPU shell. The supplied utils commands dispatch GPU jobs. "
        "model.py, SKILL.md, binding files and utils are read-only. "
        "Use profile_reference before implementation; submit when finished."
    )
    return [
        {"role": "system", "content": skill + local},
        {"role": "user", "content": f'Optimize model.py:\n```python\n{task["code"]}\n```'},
    ]


class AgentDataset(RLHFDataset):
    """仅把原始 code 行转成消息；不重写框架采样、tokenization 或数据恢复。"""

    def _read_files_and_tokenize(self):
        """调用原生 JSONL 读取/选取，缓存官方 SKILL；返回 None。"""
        self.skill = (ROOT / 'upstream/CUDA-Agent/agent_workdir/SKILL.md').read_text()
        super()._read_files_and_tokenize()

    def maybe_filter_out_long_prompts(self, dataframe=None):
        """输入原始行表；只在内存补 prompt，长度过滤仍调用框架，返回过滤后的表。"""
        dataframe = self.dataframe if dataframe is None else dataframe
        selected = self.config.get('task_ids')
        if selected and len(dataframe) and dataframe[0]['split'] == 'train':
            dataframe = dataframe.filter(lambda row: row['task_id'] in selected, keep_in_memory=True)
            if set(dataframe['task_id']) != set(selected):
                raise ValueError('Requested task IDs were not found in selected train records')
        # 原生纯文本过滤器直接读取 prompt 字段，不调用 _build_messages；这里只补字段。
        dataframe = dataframe.map(
            lambda row: {self.prompt_key: self._build_messages(row)},
            keep_in_memory=True, desc='Build runtime Agent prompts',
        )
        return super().maybe_filter_out_long_prompts(dataframe)

    def _build_messages(self, example, key='prompt'):
        """输入原始行；为原生长度检查和 rollout 构造同一份消息。"""
        return messages(example, self.skill)

    def __getitem__(self, item):
        """输入框架行索引；返回原题元信息、原生占位张量及 Agent 消息。"""
        task = dict(self.dataframe[item])
        task.pop(self.prompt_key, None)
        return {
            'raw_prompt': self._build_messages(task),
            'dummy_tensor': torch.tensor([0], dtype=torch.uint8),
            'data_source': 'cuda_agent_ops6k',
            'reward_model': {'style': 'rule', 'ground_truth': task['task_id']},
            'extra_info': {'task': task},
            'index': task['row_index'],
            'agent_name': 'cuda_grpo',
            'tools_kwargs': {},
            'interaction_kwargs': {},
        }

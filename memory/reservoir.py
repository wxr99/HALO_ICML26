import random
import torch
from typing import List

class ReservoirMemory:
    def __init__(self, max_size: int):
        """
        初始化水塘采样的存储器。

        Args:
            max_size (int): 水塘的最大容量。
        """
        self.max_size = max_size  # 水塘的最大容量
        self.data = []            # 存储样本的列表，格式为 (input, list_of_labels)
        self.count = 0            # 已接收的样本总数量，用于水塘采样的概率计算

    def add_batch(self, inputs: torch.Tensor, labels: List[torch.Tensor]):
        """
        添加一个批次的数据到存储器中。

        Args:
            inputs (torch.Tensor): 输入张量，形状为 [batch_size, ...]。
            labels (List[torch.Tensor]): 包含多个粒度标签的列表，每个元素是 [batch_size] 的张量。
        """
        batch_size = inputs.shape[0]

        # 遍历 batch 中的每个样本
        for i in range(batch_size):
            sample_input = inputs[i]  # 单个样本的输入
            sample_labels = [label[i] for label in labels]  # 获取对应的所有粒度标签

            self.count += 1  # 增加总接收样本的计数

            # 如果水塘还未满，直接添加
            if len(self.data) < self.max_size:
                self.data.append((sample_input, sample_labels))
            else:
                # 水塘采样：以 max_size/self.count 的概率替换水塘中的一个样本
                replace_index = random.randint(0, self.count - 1)  # 随机选择一个索引
                if replace_index < self.max_size:
                    self.data[replace_index] = (sample_input, sample_labels)

    def sample(self, num_samples: int):
        """
        Randomly sample from the memory buffer.

        Args:
            num_samples (int): numbers to sample.

        Returns:
            List[Tuple[torch.Tensor, List[torch.Tensor]]]: 随机采样的样本列表，格式为 (input, list_of_labels)。
        """
        if not self.data:
            raise ValueError("No data available for sampling.")
        return random.sample(self.data, min(num_samples, len(self.data)))
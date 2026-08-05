import random
import torch
from typing import List

class CBRSMemory:
    def __init__(self, max_size: int):
        """
        初始化水塘采样的存储器。

        Args:
            max_size (int): 水塘的最大容量。
        """
        self.max_size = max_size  # 水塘的最大容量
        self.data = []            # 存储样本的列表，格式为 (input, list_of_labels)
        self.class_count = {}            # 已接收的样本总数量，用于水塘采样的概率计算

    def add_batch(self, inputs: torch.Tensor, labels: List[torch.Tensor]):
        """
        添加一个批次的数据到存储器中。

        Args:
            inputs (torch.Tensor): 输入张量，形状为 [batch_size, ...]。
            labels (List[torch.Tensor]): 包含多个粒度标签的列表，每个元素是 [batch_size] 的张量。
        """
        batch_size = inputs.shape[0]

        # 获取最细粒度的标签
        fine_grained_labels = labels[-1]  # 最细粒度的标签

        for i in range(batch_size):
            sample_input = inputs[i]  # 单个样本的输入
            sample_labels = [label[i] for label in labels]  # 获取对应的所有粒度的标签
            fine_label = fine_grained_labels[i].item()  # 获取最细粒度的标签

            # 跳过无效标签
            if fine_label == -1:
                continue

            # 初始化类别计数
            if fine_label not in self.class_count:
                self.class_count[fine_label] = 0

            self.class_count[fine_label] += 1  # 增加该类别接收的样本总数

            # 如果水塘尚未满，直接添加
            if len(self.data) < self.max_size:
                self.data.append((sample_input, sample_labels))
            else:
                # 平衡水塘采样：以 max_size / class_count[fine_label] 的概率替换水塘中的一个样本
                replace_index = random.randint(0, self.class_count[fine_label] - 1)
                if replace_index < self.max_size:
                    # 查找并替换属于该类别的样本
                    target_index = self._find_replace_index(fine_label)
                    if target_index is not None:
                        self.data[target_index] = (sample_input, sample_labels)
            
    def _find_replace_index(self, fine_label: int):
        """
        在存储器中找到一个属于目标类别的样本索引。

        Args:
            fine_label (int): 目标类别的标识。

        Returns:
            int 或 None: 返回找到的样本索引，如果未找到则返回 None。
        """
        for idx, (_, label) in enumerate(self.data):
            if label == fine_label:
                return idx
        return None
    
    # def sample(self, num_samples: int):
    #     """
    #     从存储器中随机采样，并确保类别平衡。

    #     Args:
    #         num_samples (int): 要采样的样本数量。

    #     Returns:
    #         List[Tuple[torch.Tensor, List[torch.Tensor]]]: 随机采样的样本列表，格式为 (input, list_of_labels)。
    #     """
    #     if not self.data:
    #         raise ValueError("No data available for sampling.")

    #     sampled_data = []
    #     classes = list(self.class_count.keys())  # 获取所有最细粒度的类别
    #     num_classes = len(classes)

    #     # 每个类别分配的采样数量
    #     samples_per_class = max(1, num_samples // num_classes)

    #     for fine_label in classes:
    #         # 获取属于该类别的样本
    #         class_samples = [
    #             sample for sample in self.data
    #             if sample[1][-1] == fine_label  # 匹配最细粒度的类别
    #         ]
    #         sampled_data.extend(random.sample(class_samples, min(samples_per_class, len(class_samples))))

    #     # 如果采样数量不足，随机补充样本
    #     if len(sampled_data) < num_samples:
    #         additional_samples = random.sample(self.data, min(num_samples - len(sampled_data), len(self.data)))
    #         sampled_data.extend(additional_samples)

    #     return sampled_data
    def sample(self, num_samples: int):
        """
        从存储器中随机采样。

        Args:
            num_samples (int): 要采样的样本数量。

        Returns:
            List[Tuple[torch.Tensor, List[torch.Tensor]]]: 随机采样的样本列表，格式为 (input, list_of_labels)。
        """
        if not self.data:
            raise ValueError("No data available for sampling.")
        return random.sample(self.data, min(num_samples, len(self.data)))
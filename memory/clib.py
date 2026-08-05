from typing import Dict, List, Tuple
import random
import torch
import copy
import numpy as np


class ClibMemory:
    def __init__(self, max_size: int):
        """
        初始化基于 Loss Importance 的 Memory
        Args:
            capacity: 总存储容量
        """
        self.capacity = max_size  # 总存储容量
        self.data = []  # 存储样本列表，仅存储 (input, list_of_labels)
        self.lr = 0.01
        self.losses = None
        self.model = None
        self.samplewise_importance = None

    def samplewise_importance_update(self, losses: list, ema=0.9):
        np_losses = np.array(self.losses)
        np_new_losses = np.array(losses)
        # 计算损失差异
        loss_diff = np_losses - np_new_losses
        if self.samplewise_importance is None:
            self.samplewise_importance = loss_diff.tolist()
        else:
            # 确保 samplewise_importance 与 losses 长度一致
            padding_length = len(loss_diff) - len(self.samplewise_importance)
            self.samplewise_importance.extend([0] * padding_length)
            np_samplewise_importance = np.array(self.samplewise_importance)
            importance_mean = np_samplewise_importance.mean()
            updated_importance = np_samplewise_importance - (1 - ema) * (loss_diff - importance_mean)
            self.samplewise_importance = updated_importance.tolist()

    def memory_update(self, most_common_indices: set):
        # 1. 计算当前模型的平均损失 L(θ)
        new_losses = self._compute_losses()  # 返回旧模型的平均损失和每个样本的损失
        if self.losses is None:
            self.losses = new_losses
        elif len(self.data) <= self.capacity:
            padding_length = len(new_losses) - len(self.losses)
            self.losses.extend([0] * padding_length)
            self.samplewise_importance_update(new_losses)  # 更新样本重要性
        else:
            padding_length = len(new_losses) - len(self.losses)
            self.losses.extend([0] * padding_length)
            self.samplewise_importance_update(new_losses)  # 更新样本重要性
            num_to_remove = len(self.data) - self.capacity
            sorted_indices = sorted(
                most_common_indices, 
                key=lambda idx: self.samplewise_importance[idx]
            )
            indices_to_remove = sorted_indices[:num_to_remove]
            for idx in sorted(indices_to_remove, reverse=True):  # 从大到小移除，避免索引错位
                del self.samplewise_importance[idx]
                del self.losses[idx]
                del self.data[idx]
        
    def add_batch(self, inputs: torch.Tensor, labels: List[torch.Tensor], logits: List[torch.Tensor], head_names, model, criterion):
        """
        将一个批次的数据加入 Class-Balanced Reservoir Buffer
        Args:
            inputs: 当前批次的输入数据 (shape: [batch_size, ...])
            labels: (List[torch.Tensor]): 包含多个粒度标签的列表，每个元素是 [batch_size] 的张量
            logits: (List[torch.Tensor]): 包含多个粒度预测的列表，每个元素是 [batch_size, num_classes] 的张量
        """
        self.model = copy.deepcopy(model)
        self.criterion = criterion
        self.head_names = head_names

        batch_size = inputs.size(0)
        # 阶段 1：将所有数据加入临时存储
        temp_storage = []  # 临时存储当前批次的所有样本
        for i in range(batch_size):
            sample_input = inputs[i]
            sample_labels = [label[i] for label in labels]
            temp_storage.append((sample_input, sample_labels))
        self.data.extend(temp_storage)

        # 阶段 2：找到每个粒度中样本数量最多的类别，并记录其索引
        most_common_indices = set()
        for granularity_idx in range(len(labels)):  # 遍历每个粒度
            all_labels = [sample[1][granularity_idx] for sample in self.data if sample[1][granularity_idx] != -1]
            if not all_labels: 
                continue
            all_labels_tensor = torch.tensor(all_labels, dtype=torch.long)  # 转为张量

            # 统计每个类别的样本数量
            class_counts = torch.bincount(all_labels_tensor, minlength=logits[granularity_idx].size(-1))
            # 找到样本数量最多的类别
            most_common_class = torch.argmax(class_counts).item()
            # 在 self.data 中找到属于该类别的样本索引
            indices = [
                idx for idx, sample in enumerate(self.data)
                if sample[1][granularity_idx] == most_common_class
            ]
            most_common_indices.update(indices)

        # 根据 most_common_indices 筛选样本
        # 阶段 3：根据 Loss Importance 筛选样本
        # 计算所有样本的 Loss Importance
        self.memory_update(most_common_indices)
        

    def _compute_losses(self, batch_size=8):
        """
        计算 Memory 中所有样本的平均损失和每个样本的损失
        Returns:
            avg_loss: Memory 中所有样本的平均损失
            losses: 每个样本的损失列表
        """
        losses = [0.0] * len(self.data)  # 初始化全局损失列表

        with torch.no_grad():
            for i in range(0, len(self.data), batch_size):
                # 获取当前批次的数据
                batch = self.data[i:i + batch_size]
                batch_inputs = torch.stack([item[0] for item in batch])  # [batch_size, ...]
                batch_labels_list = [
                    torch.stack([item[1][j] for item in batch]) for j in range(len(batch[0][1]))  # 每个粒度的标签
                ]

                # 前向传播计算输出
                batch_outputs = self.model(batch_inputs, self.head_names)  # 假设模型返回一个字典或列表
                # 将模型输出和标签整理为列表
                logits_list = [batch_outputs[head_name] for head_name in self.head_names]  # List[Tensor]
                labels_list = batch_labels_list  # List[Tensor]
                # 使用 multi_pair_ce_loss 计算每个样本的损失
                individual_losses = self.criterion(
                    logits_list=logits_list,
                    labels_list=labels_list,
                    aggregation='individual'  # 返回每个样本的损失
                )
                # 将每个样本的损失添加到总列表中，并计算总损失
                losses[i:i + len(individual_losses)] = [loss for loss in individual_losses]
            # print(len(self.data), len(losses))
            # avg_loss = total_loss / len(self.data) if len(self.data) > 0 else 0.0
            return losses
        

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
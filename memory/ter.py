import random
import torch
from typing import List

class TERMemory:
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

    def sample(self, device, criterion, head_names, temp_update_model, model, num_candidates: int, num_samples: int):
        """
        sample the memory buffer with Maximally Interfered Retrieval strategy (MIR).
        Args:
            num_samples (int): numbers to sample.
        Returns:
            List[Tuple[torch.Tensor, List[torch.Tensor]]]: 随机采样的样本列表，格式为 (input, list_of_labels)。
        """
        if not self.data:
            raise ValueError("No data available for sampling.")
        
        # Randomly sample `num_candidates` samples from memory
        sampled_data_candidate = random.sample(self.data, min(num_candidates, len(self.data)))
        candidate_inputs = torch.stack([item[0] for item in sampled_data_candidate])  # 取出 inputs
        candidate_labels_list = [torch.stack([item[1][i] for item in sampled_data_candidate]) 
                                 for i in range(len(sampled_data_candidate[0][1]))]  # 取出每个粒度的标签
        candidate_inputs = candidate_inputs.to(device)
        list_of_candlabels = [labels.to(device) for labels in candidate_labels_list]

        with torch.no_grad():
            temp_outputs_dict = temp_update_model(candidate_inputs, head_names)
            temp_logits_list = list(temp_outputs_dict.values())
            temp_losses = criterion(
                logits_list=temp_logits_list,
                labels_list=candidate_labels_list,
                aggregation='individual'  # Compute individual losses for each sample
            )

            main_outputs_dict = model(candidate_inputs, head_names)
            main_logits_list = list(main_outputs_dict.values())
            main_losses = criterion(
                logits_list=main_logits_list,
                labels_list=candidate_labels_list,
                aggregation='individual'  # Compute individual losses for each sample
            )

            loss_differences = temp_losses - main_losses 
            _, top_indices = torch.topk(loss_differences, k=num_samples, largest=True)
            top_samples = [sampled_data_candidate[i] for i in top_indices]
            return top_samples

# evaluate.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import logging
from typing import Dict, List, Optional, Tuple, Union, Any
import nltk
import torch.nn.functional as F
import clip
from utils.label_mapper import label_transformer
from data.cifar100.idx2name import get_cifar_clip_prompts,  get_all_cifar_classes

# category_dict = {
#     1: {'L1_0': 'a photo of household items', 'L1_1': 'a photo of structures'},
#     2: {'L2_3': 'a photo of food containers', 'L2_5': 'a photo of household electrical devices'},
#     3: {'L3_9': 'a photo of bottle', 'L3_10': 'a photo of bowl'}
# }

def evaluate_model_clip(
    model: Any,
    dataloader: DataLoader,
    global_label_tree: nltk.tree.Tree,
    current_label_tree: nltk.tree.Tree,
    device: torch.device,
    logger: logging.Logger,
    config: Dict[str, Any],
    criterion: Optional[str] = None,
    primary_head: Optional[str] = None,
    head_names: Optional[List[str]] = None, # L1_head
    seen_classes: Optional[List[str]] = None,
    epoch: Optional[int] = None, # 周期数，最终测试时为 None
    prefix: str = "val",         # 日志前缀 ('val' 或 'test')
    wandb_run: Any = None        # 传入活动的 wandb run 对象或 None
) -> Tuple[float, Dict[str, float]]:
    """
    在给定数据集上评估模型。

    Args:
        model: 要评估的模型。
        dataloader: 评估数据加载器。
        criterion: 损失函数。
        device: 计算设备。
        logger: Python 日志记录器。
        config: 配置字典。
        epoch: 当前周期数 (可选)。
        prefix: 日志前缀 ('val' 或 'test')。
        wandb_run: 活动的 wandb run 对象 (可选)。

    Returns:
        一个元组，包含 (平均损失, 包含每个头准确率的字典)。
    """
    if len(seen_classes) == 0:
        if config['dataset'] == 'cifar':
            seen_classes = get_all_cifar_classes()
            category_prompt_dict = get_cifar_clip_prompts()

    model.eval()
    accuracies = {}
    total_correct = {f"L{level}_head": 0 for level in category_prompt_dict.keys()}
    total_samples = {f"L{level}_head": 0 for level in category_prompt_dict.keys()}
    
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(dataloader):
            # --- 数据处理 ---
            if isinstance(batch_data, (list, tuple)):
                inputs, labels = batch_data
            elif isinstance(batch_data, dict):
                inputs = batch_data.get('inputs') or batch_data.get('input')
                labels = batch_data.get('labels') or batch_data.get('label')
                if inputs is None or labels is None:
                    logger.error(f"{prefix.capitalize()} Eval, Batch {batch_idx+1}: Batch dictionary missing 'inputs' or 'labels' key.")
                    continue
            else:
                logger.error(f"{prefix.capitalize()} Eval, Batch {batch_idx+1}: Unsupported batch data format: {type(batch_data)}")
                continue
            if config['dataset'] == 'cifar':
                inputs = F.interpolate(inputs, size=224, mode='bicubic', align_corners=False)

            inputs, labels = inputs.to(device), labels.to(device)
            level_data = {}

            # --- 筛选见过的类别描述 ---
            for level, categories in category_prompt_dict.items():
                descriptions, class_indices = zip(*[
                    (desc, int(key.split('_')[1]))  # 提取描述和类别索引 n
                    for key, desc in categories.items()
                    if key in seen_classes
                ]) if any(key in seen_classes for key in categories.keys()) else ([], [])
                level_data[level] = (list(descriptions), list(class_indices))

            with torch.no_grad():
                for level, (descriptions, class_indices) in level_data.items():
                    head_name = f"{level}_head"
                    # 对图像和文本进行编码
                    text_tokens = clip.tokenize(descriptions).to(device)
                    image_features = model.encode_image(inputs) # (batch_size, feature_dim)
                    text_features = model.encode_text(text_tokens) # (num_class, feature_dim)
                    
                    probs = (image_features @ text_features.T).softmax(dim=-1) # (batch_size, num_classes)
                    transformed_labels = label_transformer(labels, global_label_tree) # 转换标签
                    list_of_labels = [label.to(device) for label in transformed_labels.values()]
                    true_labels = list_of_labels[level - 1]
                    
                    pred_indices = probs.argmax(dim=-1)
                    pred_labels = torch.tensor([class_indices[idx] for idx in pred_indices.cpu().tolist()]).to(device)

                    # 计算准确率
                    correct = (pred_labels == true_labels).sum().item()
                    total_correct[f"L{level}_head"] += correct
                    total_samples[f"L{level}_head"] += true_labels.size(0)

    # --- 计算最终准确率 ---            
    for head_name in total_correct.keys():
        if total_samples[head_name] > 0:
            accuracies[head_name] = (total_correct[head_name] / total_samples[head_name]) * 100.0
        else:
            accuracies[head_name] = 0.0

    # 打印所有粒度的准确率
    logger.info(f"{prefix.capitalize()} Evaluation Results:")
    for head_name, accuracy in accuracies.items():
        logger.info(f"{head_name}: {accuracy:.2f}%")

    return 0.0, accuracies  # 返回 0.0 作为占位的平均损失，和准确率字典

            
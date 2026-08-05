# train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import logging
from typing import Dict, List, Optional, Tuple, Union, Any
from nltk.tree import Tree
import nltk
from utils.label_mapper import label_transformer
from utils.subtree_update import get_subtree, update_seen_classes_and_tree, filter_tree
from eval_clip import evaluate_model_clip
import copy

# --- WandB Import ---
try:
    import wandb
except ImportError:
    wandb = None

def train_epoch_clip(
    model: nn.Module,
    head_names: List[str],
    dataloader: DataLoader,
    test_loader: DataLoader,  # 传入测试集 DataLoader
    global_label_tree: nltk.tree.Tree,
    seen_classes: List[str],
    criterion: str,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
    global_batch: int,
    logger: logging.Logger,
    memory: Any,
    config: Dict[str, Any],
    global_step_base: int, # 用于计算 wandb 的全局步数
    eval_interval: int,  # 每隔多少个 batch 评估一次
    eval_results: List[Tuple[int, Dict[str, float]]],  # 记录评估结果
    wandb_run: Any = None, # 传入活动的 wandb run 对象或 None
    primary_head: Optional[str] = None,
) -> float:
    """
    执行单个训练周期。

    Args:
        model: 要训练的模型。
        dataloader: 训练数据加载器。
        criterion: 损失函数（例如 nn.CrossEntropyLoss 实例）。
        optimizer: 优化器。
        device: 计算设备 ('cuda' 或 'cpu')。
        epoch: 当前周期数 (从 0 开始)。
        logger: Python 日志记录器实例。
        config: 配置字典。
        global_step_base: 当前周期的起始全局步数。
        wandb_run: 活动的 wandb run 对象，如果 wandb 启用。

    Returns:
        当前周期的平均训练损失。
    """
    # --- Import Utilities ---
    # try:
    #     from loss.vanilla_ce import multi_pair_ce_loss
    # except ImportError:
    #     print("Error: Could not import 'multi_pair_ce_loss' from 'loss.vanilla_ce'.")

    model.eval()
    total_loss = 0.0
    processed_batches = 0
    num_batches = len(dataloader)
    log_interval = config.get('log_interval', 50) # 从配置获取日志间隔

    for batch_idx, batch_data in enumerate(dataloader):
        global_step = global_step_base + batch_idx # 计算 wandb 的全局步数
        global_batch = global_batch + 1 # 更新全局批次计数
        if (global_batch + 1) % eval_interval == 0:
            _, test_accuracies = evaluate_model_clip(
                    model=model, head_names=head_names, primary_head=primary_head,
                    dataloader=test_loader,
                    global_label_tree=global_label_tree, current_label_tree=global_label_tree, 
                    seen_classes=seen_classes, device=device, logger=logger, config=config, epoch=None, 
                    prefix="test", wandb_run=wandb_run
                )
            eval_results.append((global_batch, test_accuracies))
    
    # --- 周期总结 ---
    avg_loss = total_loss / processed_batches if processed_batches > 0 else 0.0
    logger.info(f'====> Epoch: {epoch + 1} Average training loss: {avg_loss:.4f}')

    # --- WandB 周期日志 (训练) ---
    if wandb_run:
        # 使用最后一个批次的 global_step 或估计的周期结束 step
        epoch_end_step = global_step_base + num_batches
        wandb_run.log({"epoch": epoch + 1, "train_loss_epoch": avg_loss}, step=epoch_end_step)

    return avg_loss, seen_classes
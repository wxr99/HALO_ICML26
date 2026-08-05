# train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import logging
from typing import Dict, List, Optional, Tuple, Union, Any, Sequence
from nltk.tree import Tree
import nltk
from utils.label_mapper import label_transformer
from utils.subtree_update import get_subtree, update_seen_classes_and_tree, filter_tree
from eval import evaluate_model
import copy
from loss.vanilla_ce import multi_pair_ce_loss
from loss.tree_ce import pseudo_tree_ce_loss
from loss.hierarchical_ce import hierarchical_ce_loss
from loss.soft_ce import soft_ce_loss
from loss.haf import haf_loss
from loss.proto_ce import proto_loss
from loss.logits_distillation import logits_dis_loss
from loss.sim_distillation import sim_dis_loss
from networks.head_runner import run_heads
from utils.temp import temperature_scaled_softmax_list, entropy_list, adjust_temperature_list, weighted_combine_probs

# --- WandB Import ---
try:
    import wandb
except ImportError:
    wandb = None


def train_epoch(
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
    first_task: Optional[bool] = False,  # 是否为第一个任务
    second_task: Optional[bool] = False,  # 是否为第二个任务
    model_old: Optional[nn.Module] = None,  # 用于旧模型蒸馏
    model_init: Optional[nn.Module] = None, # 用于acil头
    mem_val: Optional[Any] = None,
    old_class_indexes: List[torch.tensor]=None,  # 旧类索引列表，用于蒸馏
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
    if config.get('loss_funciotn') == 'analytic&ce':
        if first_task:
            model.train()
        else:
            model.eval()
    elif config.get('loss_funciotn') == 'analytic':
        model.eval()
    else:
        model.train()
    total_loss = 0.0
    processed_batches = 0
    num_batches = len(dataloader)
    log_interval = config.get('log_interval', 50) # 从配置获取日志间隔

    for batch_idx, batch_data in enumerate(dataloader):
        global_step = global_step_base + batch_idx # 计算 wandb 的全局步数
        global_batch = global_batch + 1 # 更新全局批次计数
        # --- 数据处理 ---
        if isinstance(batch_data, (list, tuple)):
            inputs, labels = batch_data
        elif isinstance(batch_data, dict):
            inputs = batch_data.get('inputs') or batch_data.get('input') # 兼容不同命名
            labels = batch_data.get('labels') or batch_data.get('label')
            if inputs is None or labels is None:
                 logger.error(f"Epoch {epoch+1}, Batch {batch_idx+1}: Batch dictionary missing 'inputs' or 'labels' key.")
                 continue
        else:
            logger.error(f"Epoch {epoch+1}, Batch {batch_idx+1}: Unsupported batch data format: {type(batch_data)}")
            continue

        inputs, labels = inputs.to(device), labels.to(device)

        transformed_labels = label_transformer(labels, global_label_tree) # 转换标签

        # --- 前向和反向传播 ---
        optimizer.zero_grad()
        head_types = config.get('head_type', 'linear')  # 默认线性头
        
        result = run_heads(
            model=model,
            inputs=inputs,
            head_names=head_names,
            head_types=head_types,
            features_to_return=config.get('features_to_return', None),   # 例如 True / 'pool' / ['c3','gap:c3']
            default_return_pool=config.get('return_features', False),    # 兼容旧配置
            return_protofeatures=config.get('return_protofeatures', False)  # 是否返回原型特征
        )
        # 取 logits（按顺序）
        list_of_logits = result.list_of_logits
        list_of_protofeatures = result.proto_features
        # 取 logits 的字典形式（便于按名索引）
        outputs_dict = result.logits_dict
        # 可选的特征
        features = result.features  # 可能为 None 或 {'pool': ..., 'c3': ..., 'gap:c3': ...}

        if model_init is not None:
            result_init = run_heads(
                model=model_init,
                inputs=inputs,
                head_names=head_names,
                head_types=head_types,
                features_to_return=config.get('features_to_return', None),   # 例如 True / 'pool' / ['c3','gap:c3']
                default_return_pool=config.get('return_features', False),    # 兼容旧配置
                return_protofeatures=config.get('return_protofeatures', False)  # 是否返回原型特征
            )
            features_init = result_init.features
            # --- 计算损失 ---
        
        # 假设所有头使用相同的标签
        if config['supervision'] == 'fine-grain':
            list_of_labels = [labels] * len(head_names)
            current_label_tree = global_label_tree  # 使用全局标签树

        elif config['supervision'] == 'full-label':
            list_of_labels = [label.to(device) for label in transformed_labels.values()]
            stacked_labels = torch.stack(list_of_labels)  # Shape: [num_granularities, batch_size]
            list_of_labels = stacked_labels
            # 这里需要维护一个当前的label tree，收集新观察到的类别及其样本索引
            # 获取所有非 -1 的 (granularity_index, batch_index) 坐标
            non_negative_indices = (list_of_labels != -1).nonzero(as_tuple=False)
            granularity_indices = non_negative_indices[:, 0]  # 粒度索引
            batch_indices = non_negative_indices[:, 1]  # 样本索引
            label_values = list_of_labels[granularity_indices, batch_indices]
            # 构造类别字符串，例如 "L<level>_<value>" 与global_label_tree 进行匹配
            new_labels = [f"L{g+1}_{v.item()}" for g, v in zip(granularity_indices, label_values)]
            # 聚合为 new_classes，避免重复
            seen_classes, current_label_tree_full = update_seen_classes_and_tree(new_labels, global_label_tree, seen_classes)
            current_label_tree = filter_tree(current_label_tree_full, seen_classes)
        
        elif config['supervision'] == 'aliasing':
            list_of_labels = [label.to(device) for label in transformed_labels.values()]
            stacked_labels = torch.stack(list_of_labels)  # Shape: [num_granularities, batch_size]
            num_granularities, batch_size = stacked_labels.shape

            # 随机为每个样本选择一个粒度索引 (范围为 [0, num_granularities-1], size=batch_size)
            selected_granularities = torch.randint(0, num_granularities, (batch_size,))

            # 构造掩码，用于选中每个样本的粒度
            mask = torch.zeros_like(stacked_labels, dtype=torch.bool)  # Shape: [num_granularities, batch_size]
            mask[selected_granularities, torch.arange(batch_size)] = True  # 每个样本的选中粒度为 True

            # 检查是否有某一粒度完全没有样本被选中
            granularity_coverage = mask.sum(dim=1)  # Shape: [num_granularities]
            missing_granularities = (granularity_coverage == 0).nonzero(as_tuple=False).squeeze(1)  # 粒度索引

            # 如果有缺失的粒度，随机选择一个样本分配给这些粒度
            for missing_granularity in missing_granularities:
                random_sample = torch.randint(0, batch_size, (1,))
                mask[missing_granularity, random_sample] = True

            # 应用掩码：只保留选中的粒度，其他置为 -1
            masked_labels = torch.full_like(stacked_labels, -1)  # 初始化为全 -1
            masked_labels[mask] = stacked_labels[mask]
            list_of_labels = masked_labels

            # 这里需要维护一个当前的 label tree，收集新观察到的类别及其样本索引
            # 获取所有非 -1 的 (granularity_index, batch_index) 坐标
            non_negative_indices = (list_of_labels != -1).nonzero(as_tuple=False)
            granularity_indices = non_negative_indices[:, 0]  # 粒度索引
            batch_indices = non_negative_indices[:, 1]  # 样本索引
            label_values = list_of_labels[granularity_indices, batch_indices]

            # 构造类别字符串，例如 "L<level>_<value>" 与 global_label_tree 进行匹配
            new_labels = [f"L{g+1}_{v.item()}" for g, v in zip(granularity_indices, label_values)]

            # 聚合为 new_classes，避免重复
            seen_classes, current_label_tree_full = update_seen_classes_and_tree(new_labels, global_label_tree, seen_classes)
            current_label_tree = filter_tree(current_label_tree_full, seen_classes)

        loss_function = multi_pair_ce_loss

        if config['loss_function'] == 'multi_ce':
            loss = multi_pair_ce_loss(
                list_of_logits, list_of_labels,
                aggregation=config.get('loss_aggregation', 'mean') # 从配置获取聚合方法
            )
        elif config['loss_function'] == 'tree_ce':
            loss = pseudo_tree_ce_loss(
                list_of_logits, list_of_labels,
                label_tree=current_label_tree,
                aggregation=config.get('loss_aggregation', 'mean') # 从配置获取聚合方法
            )
        elif config['loss_function'] == 'soft_ce':
            loss = soft_ce_loss(
                list_of_logits, list_of_labels,
                label_tree=current_label_tree,
                aggregation=config.get('loss_aggregation', 'mean'), # 从配置获取聚合方法
                beta=config.get('beta', 5),  # 从配置获取 beta 参数
                num_classes=list(config.get('add_heads', {}).values())
            )
        elif config['loss_function'] == 'hierarchical_ce':
            loss = hierarchical_ce_loss(
                list_of_logits, list_of_labels,
                label_tree=current_label_tree,
                aggregation=config.get('loss_aggregation', 'mean'), # 从配置获取聚合方法
                alpha=config.get('alpha', 0)  # 从配置获取 alpha 参数
            )
        elif config['loss_function'] == 'haf':
            loss = haf_loss(
                list_of_logits, list_of_labels,
                label_tree=current_label_tree,
                aggregation=config.get('loss_aggregation', 'mean'), # 从配置获取聚合方法
                margin=config.get('haf_margin', 3),  # 从配置获取 margin 参数
            )
        elif config['loss_function'] == 'analytic':
            if first_task:
                model.train_acil_head(X_train=features, Y_train=list_of_labels, mode="incremental")
                loss = torch.tensor(0.0, device=device)
            else:
                model.train_acil_head(X_train=features, Y_train=list_of_labels, mode="incremental")
                loss = torch.tensor(0.0, device=device)
        elif config['loss_function'] == 'analytic&ce':
            if first_task:
                loss = pseudo_tree_ce_loss(
                    list_of_logits['linear'], list_of_labels,
                    label_tree=current_label_tree,
                    aggregation=config.get('loss_aggregation', 'mean') # 从配置获取聚合方法
                )
                model.train_acil_head(X_train=features, Y_train=list_of_labels, mode="base")
            elif second_task:
                model.train_acil_head(X_train=features, Y_train=list_of_labels, mode="incremental")
                loss = torch.tensor(0.0, device=device)
            else:
                model.train_acil_head(X_train=features, Y_train=list_of_labels, mode="incremental")
                loss = torch.tensor(0.0, device=device)
        elif config['loss_function'] == 'protoloss':
            proto_list = []
            for head_name in head_names:
                prototypes = model.heads["protonet"][head_name].prototypes
                proto_list.append(prototypes)
            loss = proto_loss(
                list_of_logits, list_of_labels,
                label_tree=current_label_tree,
                protofeatures_list=list_of_protofeatures,
                prototypes_list=proto_list,
                aggregation=config.get('loss_aggregation', 'mean') # 从配置获取聚合方法
            )
        elif config['loss_function'] == 'icicleloss':
            proto_list = []
            for head_name in head_names:
                prototypes = model.heads["protonet"][head_name].prototypes
                proto_list.append(prototypes)
            loss = proto_loss(
                list_of_logits, list_of_labels,
                label_tree=current_label_tree,
                protofeatures_list=list_of_protofeatures,
                prototypes_list=proto_list,
                aggregation=config.get('loss_aggregation', 'mean') # 从配置获取聚合方法
            )
            if not first_task:
                with torch.no_grad():
                    old_proto_list = []
                    for head_name in head_names:
                        old_prototypes = model_old.heads["protonet"][head_name].prototypes
                        old_proto_list.append(old_prototypes)
                loss_dist = sim_dis_loss(list_of_protofeatures=list_of_protofeatures,current_proto_list=proto_list,old_proto_list=old_proto_list)
                loss += loss_dist         
        elif config['loss_function'] == 'lwf':
            loss = hierarchical_ce_loss(
                list_of_logits, list_of_labels,
                label_tree=current_label_tree,
                aggregation=config.get('loss_aggregation', 'mean'), # 从配置获取聚合方法
            )
            
            if not first_task:
                new_logits = list_of_logits
                with torch.no_grad():
                    old_result = run_heads(
                        model=model_old,
                        inputs=inputs,
                        head_names=head_names,
                        head_types=head_types,
                    )
                    old_logits = old_result.list_of_logits
                distill_loss = logits_dis_loss(new_logits_list=new_logits, old_logits_list=old_logits, 
                                               old_class_indexes=old_class_indexes, temperature=config.get('temperature', 5.0))
                loss += config.get('lambda_distill', 0.1)*distill_loss
        elif config['loss_function'] == 'hieproloss':
            loss1 = pseudo_tree_ce_loss(
                list_of_logits['linear'], list_of_labels,
                label_tree=current_label_tree,
                aggregation=config.get('loss_aggregation', 'mean'), # 从配置获取聚合方法
                # margin=config.get('haf_margin', 3),  # 从配置获取 margin 参数
            )
            proto_list = []
            for head_name in head_names:
                prototypes = model.heads["protonet"][head_name].prototypes
                proto_list.append(prototypes)
            loss2 = proto_loss(
                list_of_logits['protonet'], list_of_labels,
                label_tree=current_label_tree,
                protofeatures_list=list_of_protofeatures,
                prototypes_list=proto_list,
                aggregation=config.get('loss_aggregation', 'mean') # 从配置获取聚合方法
            )
            loss = loss1 + loss2
            loss = loss1 
            if not first_task:
                with torch.no_grad():
                    old_proto_list = []
                    for head_name in head_names:
                        old_prototypes = model_old.heads["protonet"][head_name].prototypes
                        old_proto_list.append(old_prototypes)
                loss_dist = sim_dis_loss(list_of_protofeatures=list_of_protofeatures,current_proto_list=proto_list,old_proto_list=old_proto_list)
                # loss += loss_dist
                
            model_init.train_acil_head(X_train=features_init, Y_train=list_of_labels, mode="incremental")
        else:
            raise ValueError(f"Unsupported loss function: {config['loss_function']}")

        if config['memory_name'] == 'MIR':
            temp_model = copy.deepcopy(model).to(device)
            with torch.no_grad():
                for temp_param, original_param in zip(temp_model.parameters(), model.parameters()):
                    if original_param.grad is not None:
                        # 手动更新参数：temp_param = original_param - lr * grad
                        temp_param.data = original_param.data - 0.005 * original_param.grad.data

         # --- 将当前批次数据加入 Memory Buffer ---
        
        if config.get('loss_function') == 'analytic':
            no_grad_flag = True
        elif config.get('loss_function') == 'analytic&ce':
            if first_task:
                no_grad_flag = False
            else:
                no_grad_flag = True
        else:
            no_grad_flag = False

        if memory is not None:
            if config['memory_name'] == 'ReservoirMemory':
                memory.add_batch(inputs=inputs, labels=list_of_labels)
            elif config['memory_name'] == 'MIR':
                memory.add_batch(inputs=inputs, labels=list_of_labels)
            elif config['memory_name'] == 'CBRS':
                memory.add_batch(inputs=inputs, labels=list_of_labels)
            elif config['memory_name'] == 'PLFMS':
                threshold = config.get('mem_threshold', 0.8)  # 从配置获取阈值
                memory.add_batch(inputs=inputs, labels=list_of_labels, logits=list_of_logits, head_names=head_names, 
                                model=model, criterion=loss_function, threshold=threshold, device=device)
            elif config['memory_name'] == 'Clib':
                memory.add_batch(inputs=inputs, labels=list_of_labels, logits=list_of_logits, 
                                 head_names=head_names, model=model, criterion=loss_function)
            elif config['memory_name'] == 'DHBRS':
                memory.add_batch(inputs=inputs, labels=list_of_labels)
            else:
                memory.add_batch(inputs=inputs, labels=list_of_labels)

            if config['memory_name'] == 'MIR':
                sampled_data = memory.sample(device, criterion, head_names, temp_model, model, 
                                             num_candidates=config['candidate_size'], num_samples=config['m_batch_size'])
            else:
                sampled_data = memory.sample(num_samples=config['m_batch_size'])
        
            
            # 将样本拆分为 inputs 和 list_of_labels
            memory_inputs = torch.stack([item[0] for item in sampled_data])  # 取出 inputs
            memory_labels_list = [torch.stack([item[1][i] for item in sampled_data]) for i in range(len(sampled_data[0][1]))]  # 取出每个粒度的标签
            memory_inputs = memory_inputs.to(device)
            list_of_memlabels = [labels.to(device) for labels in memory_labels_list]


            mem_result = run_heads(
                model=model,
                inputs=memory_inputs,
                head_names=head_names,
                head_types=head_types,  # 用字符串，避免多余计算
                # features_to_return=config.get('features_to_return_mem', None),  # 例如 True / 'pool' / ['c3', 'gap:c3']
                features_to_return=config.get('features_to_return', None),  # 例如 True / 'pool' / ['c3', 'gap:c3']
                default_return_pool=config.get('return_features', False),       # 兼容旧配置：return_features=True -> 默认返回 'pool'
                return_protofeatures=config.get('return_protofeatures', False)  # 是否返回原型特征
            )

            outputs_memdict = mem_result.logits_dict             # {head_name: logits}
            list_of_memlogits = mem_result.list_of_logits        # [logits 按 head_names 顺序]
            features_mem = mem_result.features                  # None 或 {'pool':..., 'c3':..., ...}
            list_of_memprotofeatures = mem_result.proto_features

            
            if config['loss_function'] == 'multi_ce':
                loss_mem = multi_pair_ce_loss(
                    list_of_memlogits, list_of_memlabels,
                    aggregation=config.get('loss_aggregation', 'mean') # 从配置获取聚合方法
                )
            elif config['loss_function'] == 'tree_ce':
                loss_mem = pseudo_tree_ce_loss(
                    list_of_memlogits, list_of_memlabels,
                    label_tree=current_label_tree,
                    aggregation=config.get('loss_aggregation', 'mean') # 从配置获取聚合方法
                )
            elif config['loss_function'] == 'hierarchical_ce':
                loss_mem = hierarchical_ce_loss(
                    list_of_memlogits, list_of_memlabels,
                    label_tree=current_label_tree,
                    aggregation=config.get('loss_aggregation', 'mean'), # 从配置获取聚合方法
                    alpha=config.get('alpha', 0)  # 从配置获取 alpha 参数
                )
            elif config['loss_function'] == 'soft_ce':
                loss_mem = soft_ce_loss(
                    list_of_memlogits, list_of_memlabels,
                    label_tree=current_label_tree,
                    aggregation=config.get('loss_aggregation', 'mean'), # 从配置获取聚合方法
                    beta=config.get('beta', 5),  # 从配置获取 beta 参数
                    num_classes=list(config.get('add_heads', {}).values())
                )
            elif config['loss_function'] == 'haf':
                loss_mem = haf_loss(
                    list_of_memlogits, list_of_memlabels,
                    label_tree=current_label_tree,
                    aggregation=config.get('loss_aggregation', 'mean'), # 从配置获取聚合方法
                    margin=config.get('haf_margin', 3),  # 从配置获取 margin 参数
                )
            elif config['loss_function'] == 'analytic':
                # model.train_acil_head(X_train=features_mem, Y_train=torch.stack(list_of_memlabels), mode="incremental")
                loss_mem = torch.tensor(0.0, device=device)
            elif config['loss_function'] == 'analytic&ce':
                if first_task:
                    loss_mem = pseudo_tree_ce_loss(
                        list_of_memlogits['linear'], list_of_memlabels,
                        label_tree=current_label_tree,
                        aggregation=config.get('loss_aggregation', 'mean') # 从配置获取聚合方法
                        )
                    model.train_acil_head(X_train=features_mem, Y_train=torch.stack(list_of_memlabels), mode="base")
                elif second_task:
                    model.train_acil_head(X_train=features_mem, Y_train=torch.stack(list_of_memlabels), mode="incremental")
                    loss_mem = torch.tensor(0.0, device=device)
                else:
                    model.train_acil_head(X_train=features_mem, Y_train=torch.stack(list_of_memlabels), mode="incremental")
                    loss_mem = torch.tensor(0.0, device=device)
            elif config['loss_function'] == 'protoloss':
                mem_proto_list = []
                for head_name in head_names:
                    prototypes = model.heads["protonet"][head_name].prototypes
                    mem_proto_list.append(prototypes)
                loss_mem = proto_loss(
                    list_of_memlogits, list_of_memlabels,
                    label_tree=current_label_tree,
                    protofeatures_list=list_of_memprotofeatures,
                    prototypes_list=mem_proto_list,
                    aggregation=config.get('loss_aggregation', 'mean') # 从配置获取聚合方法
                )
            elif config['loss_function'] == 'icicleloss':
                mem_proto_list = []
                for head_name in head_names:
                    prototypes = model.heads["protonet"][head_name].prototypes
                    mem_proto_list.append(prototypes)
                loss_mem = proto_loss(
                    list_of_memlogits, list_of_memlabels,
                    label_tree=current_label_tree,
                    protofeatures_list=list_of_memprotofeatures,
                    prototypes_list=mem_proto_list,
                    aggregation=config.get('loss_aggregation', 'mean') # 从配置获取聚合方法
                )
                if not first_task:
                    with torch.no_grad():
                        old_proto_list = []
                        for head_name in head_names:
                            old_prototypes = model_old.heads["protonet"][head_name].prototypes
                            old_proto_list.append(old_prototypes)
                    loss_mem_dist = sim_dis_loss(list_of_protofeatures=list_of_memprotofeatures,current_proto_list=proto_list,old_proto_list=old_proto_list)
                    loss_mem += loss_mem_dist           
            elif config['loss_function'] == 'lwf':
                loss_mem = hierarchical_ce_loss(
                    list_of_memlogits, list_of_memlabels,
                    label_tree=current_label_tree,
                    aggregation=config.get('loss_aggregation', 'mean'), # 从配置获取聚合方法
                )
                if not first_task:
                    new_mem_logits = list_of_memlogits
                    with torch.no_grad():
                        old_result = run_heads(
                            model=model_old,
                            inputs=memory_inputs,
                            head_names=head_names,
                            head_types=head_types,
                        )
                        old_mem_logits = old_result.list_of_logits
                    distill_loss = logits_dis_loss(new_logits_list=new_mem_logits, old_logits_list=old_mem_logits, 
                                                   old_class_indexes=old_class_indexes, temperature=config.get('temperature', 5.0))
                    loss += config.get('lambda_distill', 0.1)*distill_loss
            elif config['loss_function'] == 'hieproloss':

                # 获取 memory logits
                linear_logits_list = list_of_memlogits['linear']  # List of Tensors
                acil_logits_list = list_of_memlogits['acil']      # List of Tensors

                # 动态调整每一层的 temperature
                # temp_linear_list, temp_acil_list = adjust_temperature_list(
                #     linear_logits_list, acil_logits_list,
                #     target_entropy_diff=0.01,  # 目标熵差异
                #     learning_rate=0.01,       # temperature 调整学习率
                #     max_iters=5,              # 最大迭代次数
                #     initial_temps1=model.temp_linear_list,
                #     initial_temps2=model.temp_acil_list
                # )
                model.temp_linear_list = [10.0] * len(linear_logits_list)
                model.temp_acil_list = [10.0] * len(acil_logits_list)
                # 对 logits 应用 temperature-scaled softmax
                probs_linear_list = temperature_scaled_softmax_list(linear_logits_list, model.temp_linear_list)
                probs_acil_list = temperature_scaled_softmax_list(acil_logits_list, model.temp_acil_list)

                # 获取每一层的权重参数（从模型中获取或定义）
                weight_linear_list = [model.weight_linear[i] for i in range(len(probs_linear_list))]
                weight_acil_list = [model.weight_acil[i] for i in range(len(probs_acil_list))]

                # 可选：归一化权重，使每一层的权重和为 1
                for i in range(len(weight_linear_list)):
                    weight_sum = weight_linear_list[i] + weight_acil_list[i]
                    weight_linear_list[i] = weight_linear_list[i] / weight_sum
                    weight_acil_list[i] = weight_acil_list[i] / weight_sum

                combined_probs_list= weighted_combine_probs(
                    probs_linear_list, probs_acil_list,
                    weight_linear_list, weight_acil_list
                )

                loss_mem_combined = pseudo_tree_ce_loss(
                    combined_probs_list, list_of_memlabels,
                    label_tree=current_label_tree,
                    aggregation=config.get('loss_aggregation', 'mean'), # 从配置获取聚合方法
                    # margin=config.get('haf_margin', 3),  # 从配置获取 margin 参数
                )
                # for i in range(len(list_of_memlogits['linear'])):
                #     print(list_of_memlogits['linear'][i].shape)
                loss_mem_linear = haf_loss(
                        list_of_memlogits['linear'], list_of_memlabels,
                        label_tree=current_label_tree,
                        aggregation=config.get('loss_aggregation', 'mean'), # 从配置获取聚合方法
                        margin=config.get('haf_margin', 3),  # 从配置获取 margin 参数
                        )
                mem_proto_list = []
                for head_name in head_names:
                    prototypes = model.heads["protonet"][head_name].prototypes
                    mem_proto_list.append(prototypes)
                loss_mem_proto = proto_loss(
                    list_of_memlogits['protonet'], list_of_memlabels,
                    label_tree=current_label_tree,
                    protofeatures_list=list_of_memprotofeatures,
                    prototypes_list=mem_proto_list,
                    aggregation=config.get('loss_aggregation', 'mean') # 从配置获取聚合方法
                )
                # loss_mem = loss_mem_linear + loss_mem_proto + loss_mem_combined
                loss_mem = loss_mem_linear + loss_mem_combined
                
                if not first_task:
                    with torch.no_grad():
                        old_proto_list = []
                        for head_name in head_names:
                            old_prototypes = model_old.heads["protonet"][head_name].prototypes
                            old_proto_list.append(old_prototypes)
                    loss_mem_dist = sim_dis_loss(list_of_protofeatures=list_of_memprotofeatures,current_proto_list=mem_proto_list,old_proto_list=old_proto_list)
                    
                    loss_mem += loss_mem_dist   
            else:
                raise ValueError(f"Unsupported loss function: {config['loss_function']}")
            loss += loss_mem


        # --- 优化步骤 ---
        if loss is not None and torch.isfinite(loss): # 检查损失是否有效
            if not no_grad_flag:
                loss.backward()
                # 可选：梯度裁剪 (如果需要，从 config 读取参数)
                # grad_clip_norm = config.get('grad_clip_norm', None)
                # if grad_clip_norm:
                #     torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                optimizer.step()

                loss_item = loss.item()
                total_loss += loss_item
                processed_batches += 1

                # --- WandB 批次日志 ---
                if wandb_run:
                    log_data = {"batch_train_loss": loss_item}
                    # 可选：记录学习率
                    # log_data["learning_rate"] = optimizer.param_groups[0]['lr']
                    wandb_run.log(log_data, step=global_step) # 使用全局步数记录

                # --- 控制台/文件日志 ---
                if (batch_idx + 1) % log_interval == 0 or (batch_idx + 1) == num_batches:
                    logger.info(f'Train Epoch: {epoch + 1} [{batch_idx + 1}/{num_batches} ({100. * (batch_idx + 1) / num_batches:.0f}%)]\tLoss: {loss_item:.2f}')
        else:
            logger.warning(f"Epoch {epoch+1}, Batch {batch_idx+1}: Invalid loss detected ({loss}), skipping backward/step.")
        
        if config['loss_function'] == 'icicleloss':
            for i, head_name in enumerate(head_names):
                model.heads["protonet"][head_name].init_new_prototypes(Z_bsd=list_of_protofeatures[i], y=list_of_labels[i])
        elif config['loss_function'] == 'hieproloss':
            for i, head_name in enumerate(head_names):
                model.heads["protonet"][head_name].init_new_prototypes(Z_bsd=list_of_protofeatures[i], y=list_of_labels[i])

        # --- every eval_batch 经行评估 ---
        if (global_batch + 1) % eval_interval == 0:
            _, test_accuracies = evaluate_model(
                model=model,
                model_init=model_init,
                head_names=head_names,
                primary_head=config.get('primary_head', head_names[0]),
                dataloader=test_loader,  # 使用测试集评估
                global_label_tree=global_label_tree,
                current_label_tree=current_label_tree,
                seen_classes=seen_classes,
                criterion=criterion,
                device=device,
                logger=logger,
                config=config,
                epoch=epoch,
                prefix="test",
                wandb_run=wandb_run,
                first_task=first_task
            )
            eval_results.append((global_batch, test_accuracies))  # 保存评估结果



    # --- 周期总结 ---
    avg_loss = total_loss / processed_batches if processed_batches > 0 else 0.0
    # logger.info(f'====> Epoch: {epoch + 1} Average training loss: {avg_loss:.4f}')

    # --- WandB 周期日志 (训练) ---
    if wandb_run:
        # 使用最后一个批次的 global_step 或估计的周期结束 step
        epoch_end_step = global_step_base + num_batches
        wandb_run.log({"epoch": epoch + 1, "train_loss_epoch": avg_loss}, step=epoch_end_step)

    return avg_loss, seen_classes
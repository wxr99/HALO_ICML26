# evaluate.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import logging
from typing import Dict, List, Optional, Tuple, Union, Any
import nltk
from utils.label_mapper import label_transformer
from utils.joint_pred import ensemble_per_head
from utils.temp import temperature_scaled_softmax_list, entropy_list, adjust_temperature_list, weighted_combine_probs
from utils.proto_adjust import log_prob_fuse, entropy, topk_masked_geom_mean

# --- WandB Import ---
try:
    import wandb
except ImportError:
    wandb = None

# --- Import Utilities ---
try:
    # 假设这些函数在 utils.py 中
    from loss.vanilla_ce import multi_pair_ce_loss
    from utils.metric import calculate_accuracy, compute_lca_height
except ImportError:
    print("Error: Could not import functions from 'utils'. Ensure utils.py exists.")
    # 定义虚拟函数


def evaluate_model(
    model: nn.Module,
    head_names: List[str], # L1_head
    primary_head: str,
    dataloader: DataLoader,
    global_label_tree: nltk.tree.Tree,
    criterion: str,
    device: torch.device,
    logger: logging.Logger,
    config: Dict[str, Any],
    current_label_tree: Optional[nltk.tree.Tree] = None,
    seen_classes: Optional[List[str]] = None,
    epoch: Optional[int] = None, # 周期数，最终测试时为 None
    prefix: str = "val",         # 日志前缀 ('val' 或 'test')
    wandb_run: Any = None,        # 传入活动的 wandb run 对象或 None
    model_init: Optional[nn.Module] = None,
    first_task: Optional[bool] = False,  # 是否为第一个任务
    second_task: Optional[bool] = False,  # 是否为第二个任务
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
    model.eval()
    total_loss = 0.0
    # 从 config 获取预期的头名称，或者在第一个批次动态确定
    expected_heads = list(config.get('add_heads', {}).keys())
    all_outputs_dict = {head_name: [] for head_name in expected_heads}
    processed_batches = 0
    first_batch = True
    all_labels_list = {head_name: [] for head_name in expected_heads}
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

            inputs, labels = inputs.to(device), labels.to(device)

            # --- 前向传播 ---
            if isinstance(config.get('head_type'), list):
                if config['loss_function'] == 'analytic&ce':
                    outputs = model(inputs, head_names, return_features=False, head_types = config.get('head_type')) # {head_name: logits}
                    if first_task:
                        outputs_dict = outputs['linear']
                    else:
                        outputs_dict = outputs['acil']


                elif config['loss_function'] == 'hieproloss':
                    # 1) 取三种头输出 init用作知识蒸馏
                    outputs_series = model(inputs, head_names, return_features=True, head_types = config.get('head_type'))
                    outputs_seires_init = model_init(inputs, head_names, return_features=True, head_types = config.get('head_type'))

                    logits_dict = outputs_series['logits']
                    logits_dict_init = outputs_seires_init['logits']

                    linear_logits = logits_dict['linear']
                    # proto_logits = logits_dict['protonet']
                    acil_logits = logits_dict_init['acil']

                    # 2) 原有的PredLA（linear+acil）融合
                    # outputs_dict = acil_logits
                    # outputs_dict = linear_logits
                    # _, outputs_dict = ensemble_per_head(
                    #     linear_logits, acil_logits, acil_logits,
                    #     temps=config.get('ensemble_temps', None),
                    #     alpha=config.get('entropy_alpha', 1.0),
                    #     method=config.get('ensemble_method', 'arith'),
                    #     seen_classess=seen_classes,
                    #     low_conf_fallback=config.get('low_confidence_fallback', None)
                    # )

                    # 2) PredLA（linear+acil）融合
                    linear_logits_list = list(linear_logits.values())
                    acil_logits_list = list(acil_logits.values())
                    probs_linear_list = temperature_scaled_softmax_list(linear_logits_list, model.temp_linear_list)
                    probs_acil_list = temperature_scaled_softmax_list(acil_logits_list, model.temp_acil_list)
                    # print(model.temp_acil_list)



                    # 获取每一层的权重参数（从模型中获取或定义）
                    weight_linear_list = [model.weight_linear[i] for i in range(len(probs_linear_list))]
                    weight_acil_list = [model.weight_acil[i] for i in range(len(probs_acil_list))]

                    # 可选：归一化权重，使每一层的权重和为 1
                    # for i in range(len(weight_linear_list)):
                    #     weight_sum = weight_linear_list[i] + weight_acil_list[i]
                    #     weight_linear_list[i] = weight_linear_list[i] / weight_sum
                    #     weight_acil_list[i] = weight_acil_list[i] / weight_sum

                    # # 按权重组合 softmax 概率分布
                    # combined_probs_list = weighted_combine_probs(
                    #     probs_linear_list, probs_acil_list,
                    #     weight_linear_list, weight_acil_list
                    # ) # List[[B,C]]

                    # 获取每一层的权重参数（从模型中获取或定义）
                    weight_linear_list = [model.weight_linear[i] for i in range(len(probs_linear_list))]
                    weight_acil_list = [model.weight_acil[i] for i in range(len(probs_acil_list))]

                    # 单参数深度偏置：depth_bias >= 0，默认 0（完全退化为原方案）
                    depth_bias = float(config.get('depth_bias', 0.0))  # 建议 0.1~0.5 之间调
                    L = len(probs_linear_list)
                    eps = 1e-8

                    biased_w_linear = []
                    biased_w_acil = []
                    for i in range(L):
                        # 先按原方案归一化
                        wl = weight_linear_list[i].clamp_min(0).float()
                        wa = weight_acil_list[i].clamp_min(0).float()
                        s = wl + wa + eps
                        wl_norm = wl / s
                        wa_norm = wa / s

                        # 深度系数：浅层 0，深层 1
                        t = i / max(L - 1, 1)

                        # 只用一个参数：让 acil 在深层按 depth_bias*t 上调，占比仍保持在 [0,1]
                        # wa_b = wa_norm + depth_bias * t * (1 - wa_norm)  等价于向 1 插值
                        wa_b = wa_norm + depth_bias * t * (1.0 - wa_norm)
                        wl_b = 1.0 - wa_b

                        biased_w_linear.append(wl_b)
                        biased_w_acil.append(wa_b)

                    # 使用偏置后的权重进行组合
                    combined_probs_list = weighted_combine_probs(
                        probs_linear_list, probs_acil_list,
                        biased_w_linear, biased_w_acil
                    )




                    # 将 PredLA 概率映射回各 head 名字
                    p_la_dict = {k: v for k, v in zip(linear_logits.keys(), combined_probs_list)}  # Dict[head->[B,C]]

                    # 3) 第二次前向：把 PredLA 概率作为 prior 喂到 protonet（仅对 HieProNetHead 生效
                    outs2 = model(
                        inputs,
                        head_names,
                        return_features=False,
                        head_types="protonet",
                        prior_dict=p_la_dict,
                        prior_power=float(config.get('prior_power', 1.5)),
                        prior_topk=int(config.get('prior_topk', 0)),
                        prior_logit_bias=float(config.get('prior_logit_bias', 0.0)),
                        prior_temp=float(config.get('prior_temp', 1.0)),
                    )
                    proto_logits_guided = outs2['logits'] if isinstance(outs2, dict) and 'logits' in outs2 else outs2
                    # 这里 proto_logits_guided 是 Dict[head->[B,C]]，其中 HieProNetHead 会被 prior 门控
                    
                    # 4) 以 PredLA 为主输出；仅在“高熵样本”上进行温和纠偏或校准
                    entropy_mode = str(config.get('entropy_aux_mode', 'geom'))  # 'geom' | 'calib' | 'none'
                    H_lambda = float(config.get('entropy_lambda', 0.7))         # 阈值系数：H_tau = lambda * log(C)
                    fuse_alpha = float(config.get('fuse_alpha', 1.0))
                    fuse_beta  = float(config.get('fuse_beta', 0.4))            # 建议较小
                    calib_Tmin = float(config.get('cal_Tmin', 0.8))
                    calib_Tmax = float(config.get('cal_Tmax', 1.6))
                    calib_kjs  = float(config.get('cal_kjs', 2.0))

                    outputs_dict = {}
                    for head_name, p_la in p_la_dict.items():
                        C = p_la.size(1)
                        H_tau = H_lambda * float(torch.log(torch.tensor(C, device=p_la.device, dtype=p_la.dtype)))
                        H = entropy(p_la)  # [B]
                        z_pr = proto_logits_guided.get(head_name, None)
                        p_final = p_la.clone()

                        # if z_pr is not None and entropy_mode != 'none':
                        #     high_mask = (H > H_tau)
                        #     if high_mask.any():
                        #         if entropy_mode == 'geom':
                        #             # 在高熵样本上，用温和几何均值（不再做 top-K）
                        #             p_fused = log_prob_fuse(p_la, z_pr, alpha=fuse_alpha, beta=fuse_beta)
                        #             p_final[high_mask] = p_fused[high_mask]
                                

                        outputs_dict[head_name] = p_final
                                        
                    # outputs_dict = {key: value for key, value in zip(linear_logits.keys(), combined_probs_list)}
            else:
                outputs_dict = model(inputs, head_names, return_features=False, head_types = config.get('head_type', "linear")) # {head_name: logits}

            # 如果是第一个批次且未在 config 中指定头，则动态确定头
            if first_batch and not expected_heads:
                 expected_heads = list(outputs_dict.keys())
                 all_outputs_dict = {head_name: [] for head_name in expected_heads}
                 logger.info(f"Dynamically determined heads during {prefix} evaluation: {expected_heads}")
            first_batch = False


            # --- 计算损失 ---
            list_of_logits = []
            valid_heads_in_batch = []
            for head_name in expected_heads:
                if head_name in outputs_dict:
                    list_of_logits.append(outputs_dict[head_name])
                    valid_heads_in_batch.append(head_name)
                else:
                    logger.warning(f"{prefix.capitalize()} Eval, Batch {batch_idx+1}: Expected head '{head_name}' not found in model output.")

            if not list_of_logits: # 如果这个批次没有任何有效的头输出
                logger.error(f"{prefix.capitalize()} Eval, Batch {batch_idx+1}: No valid head outputs found.")
                continue
            if config['supervision'] == 'fine-grain':
                # transformed_labels = label_transformer(labels, label_tree) # 转换标签
                list_of_labels = [labels] * len(list_of_logits)
                # list_of_labels = [label.to(device) for label in transformed_labels.values()]
            elif config['supervision'] == 'full-label' or config['supervision'] == 'aliasing':
                transformed_labels = label_transformer(labels, global_label_tree) # 转换标签
                list_of_labels = [label.to(device) for label in transformed_labels.values()]

            loss = multi_pair_ce_loss(
                list_of_logits, list_of_labels,
                aggregation=config.get('loss_aggregation', 'mean')
            )

            if loss is not None and torch.isfinite(loss):
                 total_loss += loss.item()
                 processed_batches += 1

            # --- 存储结果用于计算总准确率 ---
            for head_name in valid_heads_in_batch: # 只存储实际存在的头的输出
                 all_outputs_dict[head_name].append(outputs_dict[head_name].cpu())
 
            # all_labels_list.append(labels.cpu())
            for head_name, label_tensor in transformed_labels.items():
                if head_name in all_labels_list:
                    all_labels_list[head_name].append(label_tensor.cpu())

    # --- 计算周期总结 ---
    avg_loss = total_loss / processed_batches if processed_batches > 0 else 0.0

    # --- 计算总准确率 ---
    accuracies = {}
    if all_labels_list: # 确保处理了至少一个批次
        final_labels = {}
        final_outputs = {}

        # 将每个头的 logits 和 labels 合并
        for head_name, logits_list in all_outputs_dict.items():
            if logits_list: # 确保这个头有输出
                final_outputs[head_name] = torch.cat(logits_list)
            if head_name in all_labels_list and all_labels_list[head_name]:
                final_labels[head_name] = torch.cat(all_labels_list[head_name])
        
        # 计算根据当前current_tree得到的class LCA (mistake severity)
        if current_label_tree is not None:
            fine_outputs =  final_outputs[primary_head]
            fine_labels = final_labels[primary_head]
            lca_height = compute_lca_height(fine_outputs, fine_labels, current_label_tree)

        # 计算每个头的准确率
        for head_name in expected_heads:
            if head_name in final_outputs and head_name in final_labels:
                # 提取 seen_classes 中属于当前 head 的类别
                # 假设 seen_classes 的形式是 "Lm_n"，例如 "L2_3"
                if seen_classes is not None:
                    seen_classes_for_head = [
                        int(cls.split('_')[-1])  # 提取数字部分
                        for cls in seen_classes
                        if cls.startswith(head_name.split('_')[0])  # 匹配 "L1" 部分
                    ]
                    if not seen_classes_for_head:
                        logger.warning(f"No seen classes provided for head '{head_name}', skipping accuracy calculation.")
                        accuracies[head_name] = 0.0
                        continue

                    accuracies[head_name] = calculate_accuracy(
                            {head_name: final_outputs[head_name]},
                            final_labels[head_name],
                            target_head_name=head_name,
                            seen_classes=seen_classes_for_head
                    )
                else:
                    accuracies[head_name] = calculate_accuracy(
                        {head_name: final_outputs[head_name]},
                        final_labels[head_name],
                        target_head_name=head_name
                    )
            else:
                logger.warning(f"No outputs or labels collected for head '{head_name}' during {prefix} evaluation.")
                accuracies[head_name] = 0.0
    else:
        logger.warning(f"No batches processed during {prefix} evaluation.")
        # 如果没有处理批次，为所有预期的头返回 0 准确率
        accuracies = {head_name: 0.0 for head_name in expected_heads}
    
    if current_label_tree is not None:
        accuracies.update({'mistake severity': lca_height}) # 添加 LCA 高度到准确率字典
    

    # --- 控制台/文件日志 ---
    epoch_str = f"Epoch {epoch + 1} " if epoch is not None else ""
    # logger.info(f'====> {epoch_str}{prefix.capitalize()} Set loss: {avg_loss:.2f}')
    # head_name: 'L1_head'; primary_head: 'L3_head';
    # for head_name, acc in accuracies.items():
    #     logger.info(f'====> {epoch_str}{prefix.capitalize()} Set: Accuracy ({head_name}): {acc:.2f}%')
    logger.info(f'====> {epoch_str}{prefix.capitalize()} Set FG Acc: {accuracies[primary_head]:.2f}%')
    if current_label_tree is not None:
        logger.info(f'====> {epoch_str}{prefix.capitalize()} Set LCA: {lca_height:.2f}')


    # --- WandB 周期日志 (评估) ---
    if wandb_run:
        log_dict = {"epoch": epoch + 1} if epoch is not None else {} # 包含周期数（如果可用）
        log_dict[f"{prefix}_loss_epoch"] = avg_loss
        for head_name, acc in accuracies.items():
            log_dict[f"{prefix}_accuracy_{head_name}"] = acc

        # 使用周期数作为记录步骤（如果可用）
        step_to_log = epoch + 1 if epoch is not None else None
        if step_to_log is not None:
             wandb_run.log(log_dict, step=step_to_log) # 使用周期数作为步骤
        else: # 对于最终测试评估，不指定步骤，wandb 会自动处理
             wandb_run.log(log_dict)

    return avg_loss, accuracies
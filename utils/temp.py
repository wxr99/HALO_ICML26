import torch
from typing import Dict, List, Optional, Tuple, Union, Any

def temperature_scaled_softmax_list(logits_list: List[torch.Tensor], temperature_list: List[float]) -> List[torch.Tensor]:
    """
    对 logits 列表中的每个 Tensor 应用对应的 temperature-scaled softmax。

    Args:
        logits_list: logits 的列表，每个元素形状为 [B, C]。
        temperature_list: 每一层的温度参数列表，长度应与 logits_list 相同。

    Returns:
        每个 logits 应用 softmax 后的概率分布列表。
    """
    if len(logits_list) != len(temperature_list):
        print(len(logits_list), len(temperature_list))
        raise ValueError("logits_list 和 temperature_list 的长度必须相同！")

    return [torch.softmax(logits / temp, dim=-1) for logits, temp in zip(logits_list, temperature_list)]

def entropy_list(prob_list: List[torch.Tensor], eps: float = 1e-8) -> List[torch.Tensor]:
    """
    计算概率分布列表中每一层的熵。

    Args:
        prob_list: 概率分布的列表，每个元素形状为 [B, C]。
        eps: 用于数值稳定的极小值。

    Returns:
        每一层的熵的列表，每个元素形状为 [B]。
    """
    return [-torch.sum(prob * torch.log(prob + eps), dim=-1) for prob in prob_list]

def adjust_temperature_list(
    logits_list1: List[torch.Tensor],
    logits_list2: List[torch.Tensor],
    target_entropy_diff: float = 0.01,
    learning_rate: float = 0.01,
    max_iters: int = 50,
    initial_temps1: Optional[List[float]] = None,
    initial_temps2: Optional[List[float]] = None
) -> Tuple[List[float], List[float]]:
    """
    动态调整每一层的 logits 的 temperature，使每一层的熵趋于接近。

    Args:
        logits_list1: 第一个模型的 logits 列表。
        logits_list2: 第二个模型的 logits 列表。
        target_entropy_diff: 每一层的目标熵差异。
        learning_rate: 调整 temperature 的学习率。
        max_iters: 最大迭代次数。

    Returns:
        两个列表，每一层的调整后的 temperature 值。
    """
    # 初始化 temperature，如果提供了初始值，则使用初始值；否则初始化为 1.0
    temps1 = [
            torch.tensor(initial_temps1[i] if initial_temps1 else 1.0, 
                        requires_grad=True, 
                        device=logits_list1[0].device)
            for i in range(len(logits_list1))
        ]
    temps2 = [
            torch.tensor(initial_temps2[i] if initial_temps2 else 1.0, 
                        requires_grad=True, 
                        device=logits_list2[0].device)
            for i in range(len(logits_list2))
        ]
    # 优化器
    optimizer = torch.optim.SGD(temps1 + temps2, lr=learning_rate)
    

    # 对 logits 进行 detach，确保不会影响原始计算图
    detached_logits_list1 = [logits.clone().detach() for logits in logits_list1]
    detached_logits_list2 = [logits.clone().detach() for logits in logits_list2]

    for _ in range(max_iters):
        total_loss = 0.0

        for i, (logits1, logits2) in enumerate(zip(detached_logits_list1, detached_logits_list2)):
            # 计算 softmax 概率分布
            probs1 = torch.softmax(logits1 / temps1[i], dim=-1)
            probs2 = torch.softmax(logits2 / temps2[i], dim=-1)

            # 计算熵
            entropy1 = -torch.sum(probs1 * torch.log(probs1 + 1e-8), dim=-1).mean()
            entropy2 = -torch.sum(probs2 * torch.log(probs2 + 1e-8), dim=-1).mean()

            # 计算熵的差异
            entropy_diff = torch.abs(entropy1 - entropy2)

            # 优化目标：最小化熵差异和目标熵差异的偏差
            total_loss += torch.abs(entropy_diff - target_entropy_diff)

        # 反向传播和更新
        optimizer.zero_grad()  # 清除梯度
        total_loss.backward(retain_graph=False)  # 不保留计算图
        optimizer.step()

        # 防止 temperature 变为负值
        for temp in temps1 + temps2:
            temp.data.clamp_(0.1, 10.0)

    return [temp.item() for temp in temps1], [temp.item() for temp in temps2]

def weighted_combine_probs(
    probs_list1: List[torch.Tensor],
    probs_list2: List[torch.Tensor],
    weights1: List[float],
    weights2: List[float]
) -> List[torch.Tensor]:
    """
    对两个模型的 softmax 概率分布逐层进行加权组合。

    Args:
        probs_list1: 第一个模型的概率分布列表。
        probs_list2: 第二个模型的概率分布列表。
        weights1: 第一个模型每一层的权重列表。
        weights2: 第二个模型每一层的权重列表。

    Returns:
        加权组合后的概率分布列表。
    """
    combined_probs = []
    for probs1, probs2, w1, w2 in zip(probs_list1, probs_list2, weights1, weights2):
        combined_probs.append(w1 * probs1 + w2 * probs2)
    return combined_probs
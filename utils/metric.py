import torch
from typing import Dict, Optional
from typing import Dict, List, Optional, Tuple, Union, Any
import logging # 建议使用日志记录
from nltk.tree import Tree
import pickle

# 获取一个模块级别的日志记录器
# 在调用此函数的文件中配置日志记录（例如在 main.py 中）
logger = logging.getLogger(__name__)

def calculate_accuracy(
    outputs_dict: Dict[str, torch.Tensor],
    labels: torch.Tensor,
    target_head_name: str,
    seen_classes: Optional[List[str]] = None
) -> Optional[float]:
    """
    计算模型输出中指定单个头（例如细粒度头）的 top-1 准确率。

    Args:
        outputs_dict: 一个字典，键是头的名称 (str)，值是对应的 logit 张量。
                      每个 logit 张量应具有形状 (N, C)，其中 N 是批次大小，
                      C 是该头的类别数。
        labels: 包含批次真实类别索引的张量。
                应具有形状 (N,) 且为整数类型 (通常是 torch.long)。
        target_head_name: 要计算准确率的目标头的名称 (str)。

    Returns:
        一个浮点数，表示指定头的 top-1 准确率（百分比 0-100），
        如果无法计算（例如，头不存在、标签为空、形状不匹配），则返回 None。
    """
    # 1. 输入有效性检查
    if not outputs_dict:
        logger.warning("无法计算准确率：模型输出字典为空 (outputs_dict is empty)。")
        return None
    if labels.numel() == 0:
        logger.warning("无法计算准确率：标签张量为空 (labels tensor is empty)。")
        return None

    # 2. 检查目标头是否存在
    if target_head_name not in outputs_dict:
        logger.error(f"无法计算准确率：目标头 '{target_head_name}' 不在模型输出字典的键中。 "
                     f"可用的头: {list(outputs_dict.keys())}")
        return None

    # 3. 获取目标头的 Logits
    logits = outputs_dict[target_head_name]
    total_samples = labels.size(0) # 获取批次大小

    # 4. 设备和类型处理
    try:
        device = logits.device # 从 logits 获取设备
        labels = labels.to(device) # 确保标签在同一设备上
        # 确保标签是 Long 类型以进行比较
        if labels.dtype != torch.long:
            # logger.debug(f"标签张量类型为 {labels.dtype}，正在转换为 torch.long。")
            labels = labels.long()
    except Exception as e:
        logger.error(f"准备标签以计算准确率时出错: {e}", exc_info=True)
        return None # 无法处理标签，返回 None

    # 5. 筛选 seen_classes
    if seen_classes is not None:
        # 确保 seen_classes 是 torch.long 类型，并与 labels 的设备一致
        seen_classes_tensor = torch.tensor(seen_classes, dtype=torch.long, device=labels.device)

        # 找到 labels 中属于 seen_classes 的索引
        seen_indices = torch.isin(labels, seen_classes_tensor)
        if not seen_indices.any():
            logger.warning(f"No samples found for seen classes in head '{target_head_name}', skipping accuracy.")
            return None
        # 筛选 logits 和 labels
        logits = logits[seen_indices]
        labels = labels[seen_indices]

    # 6. 形状验证 (针对目标头)
    if logits.ndim != 2:
        logger.error(f"无法计算准确率：目标头 '{target_head_name}' 的 Logits 维度 "
                     f"({logits.ndim}) 不符合预期 (应为 2)。")
        return None
    if logits.size(0) != labels.size(0):
        logger.error(f"无法计算准确率：目标头 '{target_head_name}' 的 Logits 数量 ({logits.size(0)}) "
                     f"与筛选后的标签数量 ({labels.size(0)}) 不匹配。")
        return None
    
    # 7. 计算准确率
    try:
        # 获取最高分数的索引（预测的类别）
        predictions = torch.argmax(logits, dim=1) # 形状: (N,)
        # 比较预测与真实标签
        correct_predictions = (predictions == labels).sum().item() # .item() 获取 Python 数字
        # 计算准确率百分比
        accuracy_percent = (correct_predictions / labels.size(0)) * 100.0
        return accuracy_percent

    except Exception as e:
        logger.error(f"计算头 '{target_head_name}' 的准确率时发生错误: {e}", exc_info=True)
        return None


def compute_lca_height(logits: torch.Tensor, labels: torch.Tensor, tree: Tree) -> float:
    """
    计算所有样本的平均 LCA height。

    Args:
        logits: 形状为 (num_samples, num_classes) 的张量，表示预测的 logits。
        labels: 形状为 (num_samples,) 的张量，表示真实类别索引（叶子节点的索引）。
        tree: 一个表示类别层次关系的 nltk.tree.Tree 对象。节点名称为 Lm_n 格式。

    Returns:
        float: 所有样本的平均 LCA height。
    """
    def get_leaf_paths(tree: Tree, path=None): # 获取树的所有叶子节点及其路径
        if path is None:
            path = []
        if isinstance(tree, str) and tree.startswith("L"):  # 叶子节点
            return {tree: path + [tree]}
        leaf_paths = {}
        for subtree in tree:
            leaf_paths.update(get_leaf_paths(subtree, path + [tree.label()]))
        return leaf_paths

    leaf_paths = get_leaf_paths(tree)  # {leaf_name: [path_to_leaf]}

    # 假设 m 是最深一层的粒度
    m = max(int(node.split('_')[0][1:]) for node in leaf_paths.keys())  # 获取最深粒度的编号

    # 将 logits 转换为预测类别索引
    pred_indices = logits.argmax(dim=-1).tolist()  # 预测的类别索引
    true_indices = labels.tolist()  # 真实的类别索引

    # 根据索引直接生成叶子节点名称
    true_labels = [f"L{m}_{i}" for i in true_indices]  # 真实标签的名称
    pred_labels = [f"L{m}_{i}" for i in pred_indices]  # 预测标签的名称

    # 计算 LCA 高度
    def get_lca_height(pred_label: str, true_label: str) -> int:
        if pred_label not in leaf_paths:
            # print(f"Warning: Predicted label '{pred_label}' not found in leaf_paths")
            return 0  # 或者返回一个默认值，或者抛出自定义异常
        
        if true_label not in leaf_paths:
            # print(f"Warning: True label '{true_label}' not found in leaf_paths")
            return 0  # 或者返回一个默认值，或者抛出自定义异常
        
        if pred_label == true_label:
            return 0  # 如果预测和真实标签相同，LCA height 为 0

        pred_path = leaf_paths[pred_label]
        true_path = leaf_paths[true_label]

        # 找到公共前缀的长度
        lca_depth = len([node for node, tnode in zip(pred_path, true_path) if node == tnode])

        # 计算从 LCA 到真实类别的高度
        true_height = len(true_path)
        return true_height - lca_depth

    # 计算所有样本的 LCA height 并求平均
    total_lca_height = 0
    num_samples = len(true_labels)
    for pred_label, true_label in zip(pred_labels, true_labels):
        lca_height = get_lca_height(pred_label, true_label)
        total_lca_height += lca_height
    avg_lca_height = total_lca_height / num_samples
    return avg_lca_height


logger = logging.getLogger(__name__)

def evaluate_heads(
    total_loss: float,
    processed_batches: int,
    all_outputs_dict: Dict[str, List[torch.Tensor]],  # {head_name: [logits_batch1, logits_batch2, ...]}
    all_labels_list: Dict[str, List[torch.Tensor]],   # {head_name: [labels_batch1, labels_batch2, ...]}
    expected_heads: List[str],                        # 需要汇报的 head 名称列表
    calculate_accuracy,                               # 函数: (outputs_dict, labels_tensor, target_head_name, seen_classes=None) -> float
    prefix: str = "eval",
    seen_classes: Optional[List[str]] = None,         # 例如 ["L1_2", "L2_3"]；用于限制可见类
    current_label_tree: Optional[Any] = None,         # 提供 compute_lca_height 所需的树
    compute_lca_height=None,                          # 函数: (logits_tensor, labels_tensor, tree) -> float 或 Tensor(标量)
    primary_head: Optional[str] = None,               # 细粒度 head 的名字，用于 LCA 计算
) -> Tuple[float, Dict[str, float], float, Optional[float]]:
    """
    返回:
      - avg_loss: float
      - per_head_acc: Dict[head_name, float]，若提供 label tree 则包含 'mistake severity'
      - avg_acc: float（expected_heads 的平均准确率）
      - lca_height: Optional[float]（若可计算，否则为 None）
    """
    # 1) 平均 loss
    avg_loss = (total_loss / processed_batches) if processed_batches > 0 else 0.0

    # 2) 汇总每个 head 的最终 logits 与 labels
    per_head_logits: Dict[str, torch.Tensor] = {}
    per_head_labels: Dict[str, torch.Tensor] = {}
    any_batch_processed = False

    if all_labels_list:  # 有至少一个批次被记录
        for head_name, logits_list in all_outputs_dict.items():
            if logits_list:
                per_head_logits[head_name] = torch.cat(logits_list, dim=0)
                any_batch_processed = True

        for head_name, labels_list in all_labels_list.items():
            if labels_list:
                per_head_labels[head_name] = torch.cat(labels_list, dim=0)
                any_batch_processed = True

    per_head_acc: Dict[str, float] = {}
    lca_height_value: Optional[float] = None

    if not any_batch_processed:
        logger.warning(f"No batches processed during {prefix} evaluation.")
        # 所有期待 head 的 acc 置 0
        per_head_acc = {head_name: 0.0 for head_name in expected_heads}
        avg_acc = sum(per_head_acc.get(h, 0.0) for h in expected_heads) / max(len(expected_heads), 1)
        return avg_loss, per_head_acc, avg_acc, lca_height_value

    # 3) 若提供树与主 head，计算 LCA mistake severity
    if current_label_tree is not None and primary_head is not None:
        if primary_head in per_head_logits and primary_head in per_head_labels:
            fine_outputs = per_head_logits[primary_head]
            fine_labels = per_head_labels[primary_head]
            if fine_outputs.numel() > 0 and fine_labels.numel() > 0:
                lca_height_value = compute_lca_height(fine_outputs, fine_labels, current_label_tree)
                # 兼容返回 tensor/scalar
                if torch.is_tensor(lca_height_value):
                    lca_height_value = float(lca_height_value.item())
        else:
            logger.warning(f"Primary head '{primary_head}' missing outputs or labels; skip LCA computation.")

    # 4) 逐头计算准确率
    for head_name in expected_heads:
        if head_name in per_head_logits and head_name in per_head_labels:
            logits_tensor = per_head_logits[head_name]
            labels_tensor = per_head_labels[head_name]
            if logits_tensor.numel() == 0 or labels_tensor.numel() == 0:
                logger.warning(f"Empty outputs or labels for head '{head_name}' during {prefix} evaluation.")
                per_head_acc[head_name] = 0.0
                continue

            # 筛 seen_classes：保留与本 head 层级前缀一致的类索引（Lm_n -> 取 n）
            if seen_classes is not None:
                # 假设 head_name 形如 "L1_head"；前缀为 "L1"
                head_prefix = head_name.split('_')[0]
                seen_classes_for_head = [
                    int(cls.split('_')[-1])
                    for cls in seen_classes
                    if cls.startswith(head_prefix)
                ]
                if not seen_classes_for_head:
                    logger.warning(f"No seen classes provided for head '{head_name}', skipping accuracy calculation.")
                    per_head_acc[head_name] = 0.0
                    continue

                acc = calculate_accuracy(
                    {head_name: logits_tensor},
                    labels_tensor,
                    target_head_name=head_name,
                    seen_classes=seen_classes_for_head
                )
            else:
                acc = calculate_accuracy(
                    {head_name: logits_tensor},
                    labels_tensor,
                    target_head_name=head_name
                )
            # 兼容 tensor/scalar
            if torch.is_tensor(acc):
                acc = float(acc.item())
            per_head_acc[head_name] = acc
        else:
            logger.warning(f"No outputs or labels collected for head '{head_name}' during {prefix} evaluation.")
            per_head_acc[head_name] = 0.0

    # 5) 加入 LCA 指标
    if lca_height_value is not None:
        per_head_acc['mistake severity'] = float(lca_height_value)

    # 6) 计算所有头的平均准确率（仅 expected_heads）
    avg_acc = sum(per_head_acc.get(h, 0.0) for h in expected_heads) / max(len(expected_heads), 1)

    return avg_loss, per_head_acc, avg_acc, lca_height_value










# --- 示例用法 (用于直接测试函数) ---
if __name__ == '__main__':
    # # 配置基本的日志记录以便在测试时看到日志输出
    # logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')

    # logger.info("测试 calculate_accuracy 函数...")

    # # 示例数据
    # batch_size = 5
    # num_classes_fine = 100 # 细粒度头的类别数
    # num_classes_coarse = 10  # 另一个（例如粗粒度）头的类别数

    # # 模拟模型输出
    # logits_fine = torch.randn(batch_size, num_classes_fine)
    # logits_coarse = torch.randn(batch_size, num_classes_coarse)

    # # 使一些预测确定以便验证
    # logits_fine[0, 15] = 20.0 # 样本 0 预测 15
    # logits_fine[1, 88] = 20.0 # 样本 1 预测 88
    # logits_fine[2, 15] = 20.0 # 样本 2 预测 15
    # logits_fine[3, 99] = 20.0 # 样本 3 预测 99
    # logits_fine[4, 0]  = 20.0 # 样本 4 预测 0

    # # 真实标签
    # true_labels = torch.tensor([15, 88, 20, 99, 10], dtype=torch.long)
    # # 细粒度头预测正确：样本 0, 1, 3 (3/5 = 60%)

    # model_outputs = {
    #     "L1_head": logits_coarse,
    #     "L2_head": logits_fine # 假设这是我们要关注的头
    # }

    # target_head = "L2_head" # 指定目标头

    # logger.info(f"\n真实标签: {true_labels}")
    # # logger.info(f"模型输出键: {list(model_outputs.keys())}")

    # # 调用函数
    # accuracy = calculate_accuracy(model_outputs, true_labels, target_head, seen_classes=["L2_15", "L2_88", "L2_20", "L2_99", "L2_10"])

    # if accuracy is not None:
    #     logger.info(f"\n计算得到的 '{target_head}' 准确率: {accuracy:.2f}%")
    # else:
    #     logger.error(f"未能计算 '{target_head}' 的准确率。")

    # tree_fname = "./data/cifar100/cifar_100_tree.pkl"
    # name = 'cifar'
    tree_fname = "./data/fgvc/fgvc_label_hierarchy_tree.pkl"
    name = 'fgvc'
    with open(tree_fname, "rb") as f:
        label_tree = pickle.load(f)
    print(label_tree)

    # 模拟 logits 和 labels
    logits = torch.tensor([
        [0.1, 0.2, 0.4, 0.3, 0.0, 0.0],  # 预测类别为 L3_2
        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],  # 预测类别为 L3_3
        [0.3, 0.2, 0.1, 0.0, 0.4, 0.0],  # 预测类别为 L3_4
        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]   # 预测类别为 L3_5
    ])
    labels = torch.tensor([1, 42, 4, 15])  # 真实类别分别为 L3_1, L3_42, L3_4, L3_15

    # 调用 compute_lca_height 函数
    avg_lca_height = compute_lca_height(logits, labels, label_tree)

    # 打印结果
    print(f"Average LCA Height: {avg_lca_height}")

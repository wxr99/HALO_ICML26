import random
from torch.utils.data import Dataset, random_split, DataLoader
from collections import defaultdict 

def get_label_to_indices_from_inat_samples(dataset: Dataset) -> "defaultdict[int, list[int]]":
    """
    Efficiently creates a mapping from label to sample indices 
    specifically for the INaturalistCustomDataset by directly accessing its 'samples' list.

    Args:
        dataset: An instance of INaturalistCustomDataset.

    Returns:
        A defaultdict where keys are labels (int) and values are lists of 
        sample indices (int) having that label.
        
    Raises:
        TypeError: If the input object doesn't have a 'samples' attribute 
                   or if 'samples' is not a list.
        AttributeError: If the input dataset object lacks the 'samples' attribute.
        ValueError: If a sample tuple in 'samples' doesn't have at least two elements,
                    or if the label is not hashable (though expected to be int).
    """
    # print("Using fast method: Directly accessing INaturalistCustomDataset.samples list.")
    
    if not hasattr(dataset, 'samples'):
        raise AttributeError("Dataset object does not have the 'samples' attribute.")
    if not isinstance(dataset.samples, list):
        raise TypeError("Dataset 'samples' attribute is not a list.")

    label_to_indices = defaultdict(list)
    
    try:
        # Iterate directly over the samples list using enumerate to get index and value
        for idx, sample_tuple in enumerate(dataset.samples):
            # Basic check for tuple structure
            if len(sample_tuple) >= 2:
                # The label is the second element in the tuple
                label = sample_tuple[1] 
                
                # Since __init__ stores integer labels, they are directly usable
                # Add a hash check just for extreme defensiveness, though likely unnecessary
                # try:
                #    hash(label)
                # except TypeError:
                #    raise ValueError(f"Label '{label}' at index {idx} is not hashable.")
                
                label_to_indices[label].append(idx)
            else:
                # This shouldn't happen based on your __init__, but good to check
                raise ValueError(f"Sample tuple at index {idx} has unexpected format: {sample_tuple}. Expected (path, label).")
                
    except Exception as e:
        print(f"An error occurred while processing the samples list: {e}")
        # Depending on requirements, you might return an empty dict or re-raise
        raise # Re-raise the exception after printing it

    # print(f"Finished building mapping. Found {len(label_to_indices)} unique labels.")
    return label_to_indices

def create_label_mapping(label_tree):
    """
    创建标签树的叶子节点到 CIFAR-100/iNaturalist 细粒度类别索引的映射关系。

    参数：
    - label_tree: 标签树，叶子节点为自定义标记（如 L3_*）

    返回：
    - label_mapping: 字典，键为叶子节点的标记，值为整数类别索引
    """
    fine_labels = [label for label in label_tree.leaves()]  # 标签树的叶子节点
    label_mapping = {}

    # 假设叶子节点的格式是 "L3_*"，映射到整数类别
    for leaf in fine_labels:
        if leaf.startswith("L3_"):
            label_index = int(leaf.split("_")[1])  # 提取整数部分
            label_mapping[leaf] = label_index
        elif leaf.startswith("nat"):
            try:
                # Extract the part after "nat"
                index_str = leaf[3:] # Get substring from index 3 onwards
                # Ensure the suffix is not empty and contains only digits
                if not index_str: 
                    raise ValueError("Empty suffix after 'nat'")
                if not index_str.isdigit():
                     raise ValueError("Suffix contains non-digit characters")
                
                # Convert the numeric part to an integer - this IS the desired index
                label_index = int(index_str) # int() handles leading zeros correctly (e.g., int('0002') == 2)
                # Map the original leaf string to the extracted integer index
                label_mapping[leaf] = label_index
            except ValueError as e_parse:
                # Handle cases like "nat", "natabc", or conversion errors
                 raise ValueError(f"Invalid nat label format: Cannot parse integer from '{leaf}'. Reason: {e_parse}")
        else:
            raise ValueError(f"Unexpected label format: {leaf}")
    
    return label_mapping



def split_dataset_by_blurry(name, trainset, label_tree, num_tasks, overlap_ratio=0.2, label_mapping=None):
    """
    使用类似 iBlurry 的方式划分数据集，将任务划分为多个子集，
    确保类别之间存在重叠，但不同任务的样本完全独立。

    参数：
    - trainset: 训练数据集
    - label_tree: 标签树，用于获取所有细粒度类别
    - num_tasks: 任务数量
    - task_size: 每个任务的类别数量
    - overlap_ratio: 任务之间类别重叠的比例（默认 20%）

    返回：
    - tasks: 每个任务的样本索引列表和类别列表
    """

    # 创建标签映射
    if label_mapping is None:
        label_mapping = create_label_mapping(label_tree)

    # 获取所有细粒度类别
    fine_labels = list(label_mapping.values())
    random.shuffle(fine_labels)  # 随机打乱类别顺序

    # 确定每个任务的重叠类别数量
    task_size = len(fine_labels) // num_tasks  # 每个任务的类别数量
    overlap_size = int(task_size * overlap_ratio)
    unique_size = task_size - overlap_size

    tasks = []
    # 初始化一个全局变量，用于存储所有见过的类别
    seen_classes = set()
    used_labels = set()  # 已分配到任务中的类别

    # 将样本根据类别分组
    if name == 'cifar':
        label_to_indices = {label: [] for label in fine_labels}  # 每个类别对应的样本索引
        for idx, (_, fine_label) in enumerate(trainset):
            label_to_indices[fine_label].append(idx)
    elif name == 'fgvc':
        label_to_indices = {label: [] for label in fine_labels}  # 每个类别对应的样本索引
        for idx, (_, fine_label) in enumerate(trainset):
            label_to_indices[fine_label].append(idx)
    else:
        label_to_indices = get_label_to_indices_from_inat_samples(trainset)
    

    for task_idx in range(num_tasks):
        if task_idx == 0:
            # 第一个任务随机选择 task_size 个类别
            task_labels = random.sample(fine_labels, task_size)
        else:
            # 后续任务：从前一个任务中选出 overlap_size 个共享类别
            previous_task_labels = tasks[task_idx - 1]["support"]
            shared_labels = random.sample(previous_task_labels, overlap_size)

            # 选出剩余的 unique_size 个类别，确保不与已使用的类别重复
            remaining_labels = [label for label in fine_labels if label not in used_labels]
            unique_labels = random.sample(remaining_labels, unique_size)

            # 合并共享类别和独特类别
            task_labels = shared_labels + unique_labels

        # 更新已使用的类别
        used_labels.update(task_labels)

        # 获取这些类别对应的样本索引
        task_indices = []
        for label in task_labels:
            task_indices.extend(label_to_indices[label])

        # 从 label_to_indices 中移除已分配的样本，确保样本不重复
        for label in task_labels:
            label_to_indices[label] = []

        # 保存任务信息
        tasks.append({
            "support": task_labels,
            "indices": task_indices,
            "seen_classes": list(seen_classes | set(task_labels))
        })
        
        seen_classes.update(task_labels)
    return tasks


import math
import random
from collections import defaultdict


from collections import defaultdict

def split_dataset_disjoint_sequential(
    name,
    trainset,
    label_tree,
    num_tasks,
    label_mapping=None,
    fine_labels_order=None,
):
    """
    将数据集划分为严格互斥的任务，按给定的 fine_labels 顺序切分，无类别重叠、无样本重叠。

    参数：
    - name: 数据集名称，用于决定如何从 trainset 读取标签
    - trainset: 训练集，可迭代得到 (img, fine_label) 或其他格式
    - label_tree: 标签树（若 label_mapping 未提供，可从中创建）
    - num_tasks: 任务数量
    - label_mapping: 可选，fine 级别映射 dict。若不提供，将基于 label_tree 构造
    - fine_labels_order: 可选，显式给定 fine 类别顺序（list）。如提供，将严格按该顺序切分；
                         否则使用 label_mapping 的值顺序。

    返回：
    - tasks: list[dict]，每个任务包含：
        - "support": 本任务的类别列表（互斥）
        - "indices": 本任务的样本索引列表（互斥）
        - "seen_classes": 到当前任务为止累计见过的类别列表
    """
    # 1) 确定 fine 类别顺序
    if label_mapping is None:
        label_mapping = create_label_mapping(label_tree)
    if fine_labels_order is None:
        fine_labels = list(label_mapping.values())
    else:
        fine_labels = list(fine_labels_order)

    total_classes = len(fine_labels)
    if num_tasks <= 0:
        raise ValueError("num_tasks must be positive")
    if total_classes < num_tasks:
        raise ValueError("num_tasks is larger than the number of classes")

    # 2) 将类别尽量平均地分到各任务（无重叠）
    base = total_classes // num_tasks
    rem = total_classes % num_tasks
    # 前 rem 个任务分配 base + 1 个类别，之后每个 base 个
    per_task_sizes = [(base + 1 if i < rem else base) for i in range(num_tasks)]

    # 3) 建立类别到样本索引的映射
    if name in ("cifar", "fgvc"):
        label_to_indices = defaultdict(list)
        for idx, (_, fine_label) in enumerate(trainset):
            label_to_indices[fine_label].append(idx)
    else:
        label_to_indices = get_label_to_indices_from_inat_samples(trainset)

    # 4) 顺序切片类别并收集样本索引（样本互斥）
    tasks = []
    seen_classes = set()

    cursor = 0
    for t in range(num_tasks):
        k = per_task_sizes[t]
        task_labels = fine_labels[cursor: cursor + k]
        cursor += k

        task_indices = []
        for lbl in task_labels:
            task_indices.extend(label_to_indices.get(lbl, []))
            # 防止样本重复分配（尽管类别不重叠，出于安全起见清空）
            label_to_indices[lbl] = []

        tasks.append({
            "support": task_labels,
            "indices": task_indices,
            "seen_classes": list(seen_classes | set(task_labels)),
        })
        seen_classes.update(task_labels)

    return tasks


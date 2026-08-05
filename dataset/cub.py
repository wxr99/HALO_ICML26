import torch
import os
import pickle
from copy import deepcopy
from typing import List, Set, Union, Sequence, Dict, Any
import torchvision
import torchvision.transforms as transforms
from nltk.tree import Tree
import random
import re
import numpy as np
from .data_utils import split_dataset_by_blurry
from torch.utils.data import Dataset, Subset, DataLoader, random_split
from nltk.tree import Tree
from PIL import Image
import glob # 用于查找文件
from collections import defaultdict 



cub_mean = (0.485, 0.456, 0.406)
cub_std = (0.229, 0.224, 0.225)

input_size = 224 

transform_train = transforms.Compose([
    transforms.RandomResizedCrop(input_size), 
    transforms.RandomHorizontalFlip(),      
    transforms.ToTensor(),                  
    transforms.Normalize(cub_mean, cub_std) 
])

transform_val_test = transforms.Compose([
    transforms.Resize(256),                 
    transforms.CenterCrop(input_size),     
    transforms.ToTensor(),
    transforms.Normalize(cub_mean, cub_std)
])

class CubDataset(Dataset):
    def __init__(self, root_dir='./data/cub/images', split_file='./data/cub/train_test_split.txt', split='train', transform=None):
        """
        Args:
            root_dir (string): 数据集根目录 (例如 './data')，包含所有图像子文件夹。
            split_file (string): 训练/测试划分文件路径。
            transform (callable, optional): 应用于样本的可选变换。
        """
        self.root_dir = root_dir
        self.split_file = split_file
        self.transform = transform
        self.samples = []  # 存储 (image_path, label) 对
        self.class_to_idx = {}  # 类别名到整数标签的映射
        self.idx_to_class = {}  # 整数标签到类别名的映射
        self.split = split  # 'train' 或 'test'

        # 加载数据元信息
        self._load_metadata()

    def _load_metadata(self):
        """
        加载数据元信息，包括图像路径和类标签。
        """
        # 读取 images.txt 文件（图像序号和路径）
        images_file = os.path.join(os.path.dirname(self.split_file), 'images.txt')
        with open(images_file, 'r') as f:
            image_id_to_path = {int(line.split()[0]): line.split()[1] for line in f}

        # 读取 image_class_labels.txt 文件（图像序号和类别标签）
        labels_file = os.path.join(os.path.dirname(self.split_file), 'image_class_labels.txt')
        with open(labels_file, 'r') as f:
            image_id_to_label = {int(line.split()[0]): int(line.split()[1]) for line in f}

        # 读取 train_test_split.txt 文件（图像序号和训练/测试划分）
        with open(self.split_file, 'r') as f:
            is_train = {int(line.split()[0]): int(line.split()[1]) for line in f}

        # 构建 samples 列表，仅存储属于训练集的样本
        if self.split == 'train':
            for image_id, image_path in image_id_to_path.items():
                if is_train[image_id] == 0:  # 仅加载训练集（split=1）
                    label = image_id_to_label[image_id] - 1
                    full_image_path = os.path.join(self.root_dir, image_path)
                    self.samples.append((full_image_path, label))
        elif self.split == 'test':
            for image_id, image_path in image_id_to_path.items():
                if is_train[image_id] == 1:  # 仅加载训练集（split=1）
                    label = image_id_to_label[image_id] - 1
                    full_image_path = os.path.join(self.root_dir, image_path)
                    self.samples.append((full_image_path, label))
        else:
            raise ValueError("split must be 'train' or 'test'")

        # 构建 class_to_idx 和 idx_to_class 映射
        unique_labels = set(image_id_to_label.values())
        self.class_to_idx = {label: idx for idx, label in enumerate(sorted(unique_labels))}
        self.idx_to_class = {idx: label for label, idx in self.class_to_idx.items()}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        获取指定索引的图像及其类别标签。
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image_path, label = self.samples[idx]

        # 加载图像
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            raise IOError(f"Failed to load or process image at index {idx} ({image_path}): {e}")

        # 应用图像变换
        if self.transform:
            image = self.transform(image)

        return image, label
    
def create_cub_dataloaders(continual_trainset, testset, batch_size=64, num_workers=2):
    """
    根据支持类别（support）为每个任务创建训练、验证和测试集加载器。

    参数：
    - continual_trainset: 任务列表，每个任务包含 'support' 和 'data'
    - testset: 测试集（完整数据集）
    - batch_size: 每批数据的大小（默认 64）
    - num_workers: 数据加载时的工作线程数（默认 2）

    返回：
    - loaders: 每个任务的训练、验证和测试集加载器，格式为：
        [
            {
                "train_loader": train_loader,
                "val_loader": val_loader,
                "test_loader": test_loader
            },
            ...
        ]
    """

    def filter_dataset_by_support(
        dataset: Dataset, 
        seen_classes: Union[Set[int], List[int], Sequence[int]] 
    ) -> Subset:
        """
        (高效版本) 通过直接访问标签信息（如果可用）来过滤 PyTorch 数据集，
        仅保留标签在 seen_classes 中的样本，并返回一个 Subset 对象。
        这个过程避免了在过滤期间加载实际数据（如图像）。

        它会尝试通过 '.targets' 或 '.samples' 属性访问标签。如果输入是 Subset，
        它会访问原始数据集的标签。如果高效方法失败，则会回退到慢速迭代。

        参数：
            dataset: 输入的 PyTorch 数据集 (可以是标准库、自定义或 Subset)。
            seen_classes: 一个包含要保留的整数标签的集合、列表或其他序列。
                        使用集合 (set) 效率最高。

        返回：
            一个 torch.utils.data.Subset 对象，其中包含原始数据集中标签
            位于 seen_classes 内的样本的索引。如果找不到匹配项或数据集为空，
            则返回空的 Subset。
        """
        
        # 确保 seen_classes 是集合以进行 O(1) 平均查找
        if not isinstance(seen_classes, set):
            seen_classes_set = set(seen_classes)
        else:
            seen_classes_set = seen_classes

        # 处理空数据集或无长度的数据集
        if not hasattr(dataset, '__len__') or len(dataset) == 0:
            original_dataset_ref = dataset.dataset if isinstance(dataset, Subset) else dataset
            return Subset(original_dataset_ref, [])

        filtered_indices: List[int] = []
        # original_dataset_ref 指向我们收集其索引的数据集
        # 如果输入是 Subset，它将成为 dataset.dataset
        original_dataset_ref = dataset 
        
        if hasattr(dataset, "samples"):
            for idx, (_, label) in enumerate(dataset.samples):
                if label in seen_classes:
                    filtered_indices.append(idx)

        final_subset = Subset(original_dataset_ref, filtered_indices)
        
        return final_subset # 返回的是 Subset 对象
    
    loaders = []

    for task in continual_trainset:
        # 获取任务的支持类别和数据
        seen_support = task['seen_classes']
        task_data = task['data']

        # 构建训练集加载器
        train_loader = DataLoader(
            task_data, batch_size=batch_size, shuffle=True, num_workers=num_workers
        )

        # 构建验证集和测试集（根据 support 过滤）
        
        filtered_testset = filter_dataset_by_support(testset, seen_support)

        test_loader = DataLoader(
            filtered_testset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )

        filtered_valset = filter_dataset_by_support(testset, seen_support)

        val_loader = DataLoader(
            filtered_valset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )

        # 保存该任务的加载器
        loaders.append({
            "train_loader": train_loader,
            "val_loader": val_loader,
            "test_loader": test_loader
        })

    return loaders

from collections import defaultdict

def count_nodes_per_level(tree):
    """
    统计 Tree 每一层的节点数，
    并检查所有叶子是否具有相同的深度。
    """
    level_counts = defaultdict(int)
    leaf_depths = []

    def dfs(node, depth):
        level_counts[depth] += 1

        if isinstance(node, Tree):
            if len(node) == 0:              # 叶子节点 (Tree 无子节点)
                leaf_depths.append(depth)
            else:
                for child in node:
                    dfs(child, depth + 1)
        else:                               # 普通字符串叶子
            leaf_depths.append(depth)

    dfs(tree, 0)

    # 判断所有叶子深度是否一致
    all_same_depth = len(set(leaf_depths)) == 1

    # 打印结果
    print("层级\t节点数")
    for depth, count in sorted(level_counts.items()):
        print(f"{depth}\t{count}")
    print("\n叶子总数:", len(leaf_depths))
    print("叶子深度集合:", sorted(set(leaf_depths)))
    print("✅ 所有叶子深度一致" if all_same_depth else "⚠️ 叶子深度不一致")

    return dict(sorted(level_counts.items()))


def load_cub_data(root_dir='./data/cub/images', split_file = './data/cub/train_test_split.txt'):
    """
    加载 CUB 数据集

    Args:
        root_data_dir (string): 包含图像子文件夹的根目录 (例如 './data')。

    Returns:
        tuple: (train_dataset, val_dataset, test_dataset)
    """
    
    train_dataset = CubDataset(
        root_dir=root_dir,
        split_file=split_file,
        transform=transform_train,
        split='train'
    )
    test_dataset = CubDataset(
        root_dir=root_dir,
        split_file=split_file,
        transform=transform_val_test,
        split='test'
    )
    return train_dataset, test_dataset


if __name__ == '__main__':
    tree_fname = "./data/cub/cub_tree.pkl"
    name = 'cub'
    with open(tree_fname, "rb") as f:
        label_tree = pickle.load(f)
    print(label_tree)

    count_nodes_per_level(label_tree)
    # trainset, testset = load_cub_data(root_dir='./data/cub/images', split_file = './data/cub/train_test_split.txt')
    # continual_trainset_info = split_dataset_by_blurry(name, trainset, label_tree, num_tasks=10, overlap_ratio=0.2)
    # continual_trainset = []
    # for task_info in continual_trainset_info:
    #     task_indices = task_info['indices']
    #     task_subset = Subset(trainset, task_indices)
    #     task_info['data'] = task_subset # Add the Subset object
    #     continual_trainset.append(task_info) # Add the modified dictionary

    # # 打印 continual_trainset 的基本信息
    # print(f"Length of continual_trainset: {len(continual_trainset)}")

    # # 打印任务划分结果
    # # for i, task in enumerate(continual_trainset):
    # #     print(f"Task {i + 1}:")
    # #     print(f"  Support: {task['support']}")
    # #     print(f"  Number of samples: {len(task['data'])}")
    # #     print(f"  First sample: {task['data'][0][0].size()}")
    
    # # 创建加载器
    # loaders = create_cub_dataloaders(continual_trainset, testset)

    # # 打印每个任务的加载器信息
    # for i, task_loaders in enumerate(loaders):
    #     print(f"Task {i + 1}:")
    #     print(f"  Train loader size: {len(task_loaders['train_loader'].dataset)}")
    #     print(f"  Test loader size: {len(task_loaders['test_loader'].dataset)}")
    #     # print(f"  First train data: {task_loaders['train_loader'].dataset[0]}")
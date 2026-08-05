import torch
import pickle
from copy import deepcopy
import torchvision
import torchvision.transforms as transforms
from nltk.tree import Tree
import random
import re
import numpy as np
from torch.utils.data import Subset, DataLoader, random_split
from .data_utils import split_dataset_by_blurry

L3_RE = re.compile(r'^L3_(\d+)$')

def l3_fine_labels_order(t: Tree):
    """
    按祖先连续（子树相邻）的顺序返回所有 L3 叶子的编号列表（int）。
    """
    order = []

    def dfs(node):
        if isinstance(node, str):
            m = L3_RE.match(node)
            if m:
                order.append(int(m.group(1)))
            return
        # node 是 Tree：遍历其子节点
        for child in node:
            dfs(child)

    dfs(t)
    return order

cifar100_mean = (0.5071, 0.4867, 0.4408) 
cifar100_std = (0.2675, 0.2565, 0.2761) 
def load_cifar100(val_split=0.2):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(cifar100_mean, cifar100_std)
    ])

    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(cifar100_mean, cifar100_std)
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(cifar100_mean, cifar100_std)
    ])

    full_trainset = torchvision.datasets.CIFAR100(
        root='./data/cifar100', train=True, download=True, transform=transform_train
    )
    testset = torchvision.datasets.CIFAR100(
        root='./data/cifar100', train=False, download=True, transform=transform_test
    )

    # 根据 val_split 划分训练集和验证集
    num_train = len(full_trainset)
    num_val = int(num_train * val_split)  # 验证集大小
    num_train = num_train - num_val       # 剩余用于训练集的大小

    trainset, valset = random_split(full_trainset, [num_train, num_val])

    return trainset, valset, testset


def create_cifar_dataloaders(continual_trainset, valset, testset, batch_size=64, num_workers=2):
    """
    根据支持类别（support）为每个任务创建训练、验证和测试集加载器。

    参数：
    - continual_trainset: 任务列表，每个任务包含 'support' 和 'data'
    - valset: 验证集（完整数据集）
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

    def filter_dataset_by_support(dataset, seen_classes):
        """
        根据seen_classes过滤数据集，仅保留属于支持类别的样本。

        参数：
        - dataset: 原始数据集
        - seen_classes: 支持类别列表

        返回：
        - filtered_data: 过滤后的数据集
        """
        filtered_data = [
            (image, label) for image, label in dataset if label in seen_classes
        ]
        return filtered_data

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
        filtered_valset = filter_dataset_by_support(valset, seen_support)
        filtered_testset = filter_dataset_by_support(testset, seen_support)

        val_loader = DataLoader(
            filtered_valset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )
        test_loader = DataLoader(
            filtered_testset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )

        # 保存该任务的加载器
        loaders.append({
            "train_loader": train_loader,
            "val_loader": val_loader,
            "test_loader": test_loader
        })

    return loaders


if __name__ == "__main__":
    tree_fname = "./data/cifar100/cifar_100_tree.pkl"
    name = 'cifar'
    with open(tree_fname, "rb") as f:
        label_tree = pickle.load(f)
    print(label_tree)

    # 加载数据集（10% 的训练集作为验证集）
    trainset, valset, testset = load_cifar100(val_split=0.1)
    continual_trainset_info = split_dataset_by_blurry(name, trainset, label_tree, num_tasks=10, overlap_ratio=0.2)
    continual_trainset = []
    for task_info in continual_trainset_info:
        task_indices = task_info['indices']
        task_subset = Subset(trainset, task_indices)
        task_info['data'] = task_subset # Add the Subset object
        continual_trainset.append(task_info) # Add the modified dictionary

    # 打印 continual_trainset 的基本信息
    print(f"Length of continual_trainset: {len(continual_trainset)}")

    # 打印任务划分结果
    for i, task in enumerate(continual_trainset):
        print(f"Task {i + 1}:")
        print(f"  Support: {task['support']}")
        # print(f"  Seen classes: {task['seen_classes']}")
        print(f"  Number of samples: {len(task['data'])}")
        # print(f"  First sample: {task['data'][0][0].size(), task['data'][0][1]}")
    
    # 创建加载器
    loaders = create_cifar_dataloaders(continual_trainset, valset, testset)

    # 打印每个任务的加载器信息
    for i, task_loaders in enumerate(loaders):
        print(f"Task {i + 1}:")
        print(f"  Train loader size: {len(task_loaders['train_loader'].dataset)}")
        print(f"  Val loader size: {len(task_loaders['val_loader'].dataset)}")
        print(f"  Test loader size: {len(task_loaders['test_loader'].dataset)}")
        # print(f"  First test data: {task_loaders['test_loader'].dataset[0]}")
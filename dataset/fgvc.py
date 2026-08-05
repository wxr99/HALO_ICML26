import torch
import os
import pickle
from copy import deepcopy
from typing import List, Set, Union, Sequence, Dict, Any
import torchvision
import torchvision.transforms as transforms
from nltk.tree import Tree
import random
import numpy as np
from .data_utils import split_dataset_by_blurry
from torch.utils.data import Dataset, Subset, DataLoader, random_split
from nltk.tree import Tree
from PIL import Image
import glob # 用于查找文件
from collections import defaultdict 

fgvc_mean = (0.4880712, 0.5191783, 0.54383063)
fgvc_std = (0.21482512, 0.20683703, 0.23937796)

input_size = 224 

transform_train = transforms.Compose([
    transforms.RandomResizedCrop(input_size), 
    transforms.RandomHorizontalFlip(),      
    transforms.ToTensor(),                  
    transforms.Normalize(fgvc_mean, fgvc_std) 
])

transform_val_test = transforms.Compose([
    transforms.Resize(256),                 
    transforms.CenterCrop(input_size),     
    transforms.ToTensor(),
    transforms.Normalize(fgvc_mean, fgvc_std)
])

def load_metadata(variants_file, split_file):
    """Loads variant mapping and image lists for train, val, and test sets."""
    # print("Loading metadata...")

    # Read all unique variant names
    if not os.path.isfile(variants_file):
        raise FileNotFoundError(f"Variants file not found: {variants_file}")
    with open(variants_file, 'r') as f:
        all_variants = [line.strip() for line in f if line.strip()]
    name_to_label_idx = {variant: idx for idx, variant in enumerate(all_variants)}
    label_idx_to_name = {idx: variant for variant, idx in name_to_label_idx.items()}
    num_classes = len(all_variants)
    # print(f"Found {num_classes} unique variants (classes).")

    def read_image_list(filepath, variant_map):
        """Reads image list file and returns list of (image_id, label_idx)."""
        if not os.path.isfile(filepath):
             raise FileNotFoundError(f"Image list file not found: {filepath}")

        data = []
        # print(f"Reading image list from: {os.path.basename(filepath)}...")
        with open(filepath, 'r') as f:
            for line_num, line in enumerate(f):
                parts = line.strip().split(' ', 1) # Split only on the first space
                if len(parts) == 2:
                    image_id = parts[0]
                    variant_name = parts[1]
                    if variant_name in variant_map:
                        label_idx = variant_map[variant_name]
                        data.append((image_id, label_idx))
                    else:
                        # Log a warning but continue processing other lines
                        print(f"Warning [Line {line_num+1} in {os.path.basename(filepath)}]: Variant '{variant_name}' for image '{image_id}' not found in variants.txt. Skipping this entry.")
                else:
                     # Log a warning for malformed lines
                     print(f"Warning [Line {line_num+1} in {os.path.basename(filepath)}]: Skipping malformed line: '{line.strip()}'")
        # print(f"Read {len(data)} valid entries from {os.path.basename(filepath)}")
        return data

    data = read_image_list(split_file, name_to_label_idx)

    return data, name_to_label_idx, num_classes


class FgvcDataset(Dataset):
    def __init__(self, root_dir='./data/fgvc/fgvc-aircraft-2013b/data/', split_file='./data/fgvc/fgvc-aircraft-2013b/data/images_variant_train.txt', transform=None):
        """
        Args:
            root_dir (string): 数据集根目录 (例如 './data')，包含所有图像子文件夹。
            transform (callable, optional): 应用于样本的可选变换。
        """
        self.root_dir = root_dir
        self.img_dir = os.path.join(self.root_dir, 'images')
        self.variants_file = os.path.join(self.root_dir, 'variants.txt')
        self.split_file = split_file
        self.transform = transform
        self.samples = [] # 存储 (image_path, label) 对
        self.class_to_idx = {} # 类别名到整数标签的映射
        self.idx_to_class = {} # 整数标签到类别名的映射

        data_id, name_to_label_idx, num_classes = load_metadata(self.variants_file, self.split_file)
        self.data_id = data_id
        self.class_to_idx = name_to_label_idx
    
    def __len__(self):
        return len(self.data_id)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        # if not 0 <= idx < len(self.data_id):
        #      raise IndexError(f"Index {idx} out of bounds for dataset with size {len(self.data)}")
        image_id, label = self.data_id[idx]
        img_name = f"{image_id}.jpg"
        img_path = os.path.join(self.img_dir, img_name)
        try:
            image = Image.open(img_path).convert('RGB') # Ensure image is RGB
        except Exception as e:
            # Provide context for the error
            raise IOError(f"Failed to load or process image at index {idx} ({img_path}):{e}")

        if self.transform:
            image = self.transform(image)
        return image, label


def create_fgvc_dataloaders(continual_trainset, valset, testset, batch_size=64, num_workers=2):
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
        
        if hasattr(dataset, 'data_id') and isinstance(dataset.data_id, list):
            samples = dataset.data_id
            for idx, sample_tuple in enumerate(samples):
                if len(sample_tuple) >= 2:
                    label = sample_tuple[1]
                    if label in seen_classes_set:
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


def load_fgvc_data(root_data_dir='./data/fgvc/fgvc-aircraft-2013b/data/'):
    """
    加载 FGVC-Aircraft 数据集。

    Args:
        root_data_dir (string): 包含图像子文件夹的根目录 (例如 './data')。

    Returns:
        tuple: (train_dataset, val_dataset, test_dataset)
    """
    train_split_file = os.path.join(root_data_dir, 'images_variant_train.txt')
    val_split_file = os.path.join(root_data_dir, 'images_variant_val.txt')
    test_split_file = os.path.join(root_data_dir, 'images_variant_test.txt')

    # 创建 Dataset 实例
    train_dataset = FgvcDataset(
        root_dir=root_data_dir,
        split_file=train_split_file,
        transform=transform_train
    )
    val_dataset = FgvcDataset(
        root_dir=root_data_dir,
        split_file=val_split_file,
        transform=transform_val_test
    )
    test_dataset = FgvcDataset(
        root_dir=root_data_dir,
        split_file=test_split_file,
        transform=transform_val_test
    )
    # print(f"Len of Test dataset,{len(test_dataset)}")

    # 确保所有数据集都找到了类别
    if not train_dataset.class_to_idx:
        raise ValueError("No classes found for the training set. Check paths and split files.")
        
    num_classes = len(train_dataset.class_to_idx)
    class_to_idx = train_dataset.class_to_idx # 以训练集为准
    # print(f"Number of classes: {num_classes}")
    # print(f"Class to index mapping: {class_to_idx}")

    # 验证 val 和 test 集的类别是否与 train 匹配（可选但推荐）
    if val_dataset.class_to_idx and val_dataset.class_to_idx != class_to_idx:
        print("Warning: Validation set classes differ from training set classes.")
    if test_dataset.class_to_idx and test_dataset.class_to_idx != class_to_idx:
         print("Warning: Test set classes differ from training set classes.")
    return train_dataset, val_dataset, test_dataset, class_to_idx


if __name__ == '__main__':
    tree_fname = "./data/fgvc/fgvc_label_hierarchy_tree.pkl"
    name = 'fgvc'
    with open(tree_fname, "rb") as f:
        label_tree = pickle.load(f)
    print(label_tree)
    trainset, valset, testset, class_to_idx = load_fgvc_data(root_data_dir='./data/fgvc/fgvc-aircraft-2013b/data/')
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
        print(f"  First sample: {task['data'][0][0].size()}")

    # 创建加载器
    loaders = create_fgvc_dataloaders(continual_trainset, valset, testset)

    # 打印每个任务的加载器信息
    for i, task_loaders in enumerate(loaders):
        print(f"Task {i + 1}:")
        print(f"  Train loader size: {len(task_loaders['train_loader'].dataset)}")
        print(f"  Val loader size: {len(task_loaders['val_loader'].dataset)}")
        print(f"  Test loader size: {len(task_loaders['test_loader'].dataset)}")
        print(f"  First test data: {task_loaders['val_loader'].dataset[0]}")
    
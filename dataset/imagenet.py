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

ilsvrc12_mean = (0.485, 0.456, 0.406)
ilsvrc12_std = (0.229, 0.224, 0.225)

input_size = 224 

transform_train = transforms.Compose([
    transforms.RandomResizedCrop(input_size), 
    transforms.RandomHorizontalFlip(),      
    transforms.ToTensor(),                  
    transforms.Normalize(ilsvrc12_mean, ilsvrc12_std) 
])

transform_val_test = transforms.Compose([
    transforms.Resize(256),                 
    transforms.CenterCrop(input_size),     
    transforms.ToTensor(),
    transforms.Normalize(ilsvrc12_mean, ilsvrc12_std)
])

class ImageCustomDataset(Dataset):
    def __init__(self, root_dir='./data/imagenet/', split_file_dir='./data/imagenet/splits_tieredImageNet-H/', transform=None):
        """
        Args:
            root_dir (string): 数据集根目录 (例如 './data')，包含所有图像子文件夹。
            split_file_dir (string): 包含划分文件 (train.txt, val.txt, test.txt) 的目录。
            transform (callable, optional): 应用于样本的可选变换。
        """
        self.root_dir = root_dir
        self.transform = transform
        self.samples = [] # 存储 (image_path, label) 对
        self.class_to_idx = {} # 类别名到整数标签的映射
        self.idx_to_class = {} # 整数标签到类别名的映射

        self.image_path_map = {} # 映射: '图像文件名..JPEG' -> '完整路径'
        # 使用 glob 递归查找所有 .JPEG 文件
        if self.root_dir=='./data/imagenet/test':
            for img_path in glob.glob(os.path.join(root_dir, '*.JPEG'), recursive=True):
                img_filename = os.path.basename(img_path)
                self.image_path_map[img_filename] = img_path
        else:
            for img_path in glob.glob(os.path.join(root_dir,'**', '*.JPEG'), recursive=True):
                img_filename = os.path.basename(img_path)
                self.image_path_map[img_filename] = img_path

        # --- 解析 split 文件 ---
        # print(f"Parsing split files in {split_file_dir}...")
        class_files = sorted(glob.glob(os.path.join(split_file_dir, '*.txt'))) # 获取所有类别txt文件并排序
        # print(f"Found {len(class_files)} class files in {split_file_dir}.")
        
        current_label = 0
        for class_file_path in class_files:
            class_id_str = os.path.splitext(os.path.basename(class_file_path))[0] # 从文件名获取类别ID (例如 '153')
            
            # 创建类别到索引的映射 (确保标签从0开始且连续)
            if class_id_str not in self.class_to_idx:
                self.class_to_idx[class_id_str] = current_label
                self.idx_to_class[current_label] = class_id_str
                label = current_label
                current_label += 1
            else:
                label = self.class_to_idx[class_id_str]

            try:
                with open(class_file_path, 'r') as f:
                    for line in f:
                        img_filename = line.strip() # 获取图像文件名，去除空白符
                        if img_filename: # 确保行不为空
                            if img_filename in self.image_path_map:
                                full_image_path = self.image_path_map[img_filename]
                                self.samples.append((full_image_path, label))
                            # else:
                                # print(f"Warning: Image '{img_filename}' listed in '{class_file_path}' not found in scanned data directory '{root_dir}'. Skipping.")
            except Exception as e:
                 print(f"Error reading or processing file {class_file_path}: {e}")

        # print(f"Loaded {len(self.samples)} samples from {split_file_dir}.")
        # print(f"Found {len(self.class_to_idx)} classes.")
        if not self.samples:
             print(f"Warning: No samples loaded for split defined by {split_file_dir}. Check paths and file contents.")
    
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path, label = self.samples[idx]

        try:
            # 使用 PIL 加载图像
            image = Image.open(img_path).convert('RGB') # 确保是 RGB 格式
        except FileNotFoundError:
            print(f"Error: Image file not found at {img_path}. Returning None.")
            image = Image.new('RGB', (input_size, input_size), (0, 0, 0))
            label = -1 # 特殊标签表示错误
        except Exception as e:
            print(f"Error loading image {img_path}: {e}. Returning placeholder.")
            image = Image.new('RGB', (input_size, input_size), (0, 0, 0))
            label = -1


        # 应用变换
        if self.transform:
            image = self.transform(image)
            
        return image, label


def load_imagenet_data(root_data_dir='./data/imagenet/', splits_base_dir='./data/imagenet/splits_tieredImageNet-H/'):
    """
    加载 ImageNet 数据集。

    Args:
        root_data_dir (string): 包含图像子文件夹的根目录 (例如 './data')。
        splits_base_dir (string): 包含 'train', 'val', 'test' 子目录的路径，
                                  每个子目录包含对应的 .txt 文件。
        batch_size (int): DataLoader 的批量大小。
        num_workers (int): DataLoader 使用的工作进程数。

    Returns:
        tuple: (train_loader, val_loader, test_loader, num_classes, class_to_idx)
    """
    train_split_dir = os.path.join(splits_base_dir, 'train')
    val_split_dir = os.path.join(splits_base_dir, 'val')
    test_split_dir = os.path.join(splits_base_dir, 'test')

    train_data_dir = os.path.join(root_data_dir, 'train')
    val_data_dir = os.path.join(root_data_dir, 'val')
    test_data_dir = os.path.join(root_data_dir, 'val')

    # 创建 Dataset 实例
    train_dataset = ImageCustomDataset(
        root_dir=train_data_dir,
        split_file_dir=train_split_dir,
        transform=transform_train
    )
    val_dataset = ImageCustomDataset(
        root_dir=val_data_dir,
        split_file_dir=val_split_dir,
        transform=transform_val_test # 验证集通常不用数据增强
    )
    test_dataset = ImageCustomDataset(
        root_dir=test_data_dir,
        split_file_dir=test_split_dir,
        transform=transform_val_test # 测试集不用数据增强
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


def create_imagenet_dataloaders(continual_trainset, valset, testset, batch_size=64, num_workers=2):
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
        
        if hasattr(dataset, 'samples') and isinstance(dataset.samples, list):
            samples = dataset.samples
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



if __name__ == '__main__':
    tree_fname = "./data/imagenet/imagenet_tree.pkl"
    name = 'imagenet'
    with open(tree_fname, "rb") as f:
        label_tree = pickle.load(f)
    
    print(label_tree)
    trainset, valset, testset, label_mapping = load_imagenet_data()
    print(label_mapping)
    continual_trainset_info = split_dataset_by_blurry(name, trainset, label_tree, num_tasks=10, overlap_ratio=0.2, label_mapping=label_mapping)
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
    loaders = create_imagenet_dataloaders(continual_trainset, valset, testset)

    # 打印每个任务的加载器信息
    for i, task_loaders in enumerate(loaders):
        print(f"Task {i + 1}:")
        print(f"  Train loader size: {len(task_loaders['train_loader'].dataset)}")
        print(f"  Val loader size: {len(task_loaders['val_loader'].dataset)}")
        print(f"  Test loader size: {len(task_loaders['test_loader'].dataset)}")
        # print(f"  First test data: {task_loaders['test_loader'].dataset[0]}")
import torch
import os
import pickle
import torchvision
import torchvision.transforms as transforms
from nltk.tree import Tree
import random
import numpy as np
from .data_utils import split_dataset_by_blurry, split_dataset_disjoint_sequential
from torch.utils.data import Dataset, Subset, DataLoader, random_split
from nltk.tree import Tree
from PIL import Image
import glob # 用于查找文件
from collections import defaultdict 
from .cifar import load_cifar100, create_cifar_dataloaders, l3_fine_labels_order
from .fgvc import load_fgvc_data, create_fgvc_dataloaders
from .inaturalist import load_inaturalist_data, create_inat_dataloaders
from .imagenet import load_imagenet_data, create_imagenet_dataloaders
from .cub import load_cub_data, create_cub_dataloaders
from .reformat_tree import reformat_tree


def get_dataset(name, num_tasks=10, overlap_ratio=0.2, val_split=0.2):
    """
    根据数据集名称获取数据集和标签树。
    """
    if name == 'cifar':
        tree_fname = "./data/cifar100/cifar_100_tree.pkl"
        with open(tree_fname, "rb") as f:
            label_tree = pickle.load(f)
        trainset, valset, testset = load_cifar100(val_split=val_split)
        continual_trainset_info = split_dataset_by_blurry(name, trainset, label_tree, num_tasks=num_tasks, overlap_ratio=overlap_ratio)
        # fine_labels_order = l3_fine_labels_order(label_tree)
        # continual_trainset_info = split_dataset_disjoint_sequential(name='cifar',trainset=trainset,label_tree=label_tree,num_tasks=10,fine_labels_order=fine_labels_order)
        continual_trainset = []
        for task_info in continual_trainset_info:
            task_indices = task_info['indices']
            task_subset = Subset(trainset, task_indices)
            task_info['data'] = task_subset # Add the Subset object
            continual_trainset.append(task_info) # Add the modified dictionary
    
    elif name == 'fgvc':
        tree_fname = "./data/fgvc/fgvc_label_hierarchy_tree.pkl"
        with open(tree_fname, "rb") as f:
            label_tree = pickle.load(f)
        trainset, valset, testset, _ = load_fgvc_data()
        continual_trainset_info = split_dataset_by_blurry(name, trainset, label_tree, num_tasks=num_tasks, overlap_ratio=overlap_ratio)
        # fine_labels_order = l3_fine_labels_order(label_tree)
        # continual_trainset_info = split_dataset_disjoint_sequential(name='cifar',trainset=trainset,label_tree=label_tree,num_tasks=10,fine_labels_order=fine_labels_order)

        
        continual_trainset = []
        for task_info in continual_trainset_info:
            task_indices = task_info['indices']
            task_subset = Subset(trainset, task_indices)
            task_info['data'] = task_subset # Add the Subset object
            continual_trainset.append(task_info) # Add the modified dictionary
    
    elif name == 'cub':
        tree_fname = "./data/cub/cub_tree.pkl"
        with open(tree_fname, "rb") as f:
            label_tree = pickle.load(f)
        trainset, testset = load_cub_data()
        valset = testset
        continual_trainset_info = split_dataset_by_blurry(name, trainset, label_tree, num_tasks=num_tasks, overlap_ratio=overlap_ratio)
        # fine_labels_order = l3_fine_labels_order(label_tree)
        # continual_trainset_info = split_dataset_disjoint_sequential(name='cifar',trainset=trainset,label_tree=label_tree,num_tasks=10,fine_labels_order=fine_labels_order)

        continual_trainset = []
        for task_info in continual_trainset_info:
            task_indices = task_info['indices']
            task_subset = Subset(trainset, task_indices)
            task_info['data'] = task_subset # Add the Subset object
            continual_trainset.append(task_info) # Add the modified dictionary

    elif name == 'inaturalist':
        tree_fname = "./data/iNaturalist/inaturalist19_tree.pkl"
        with open(tree_fname, "rb") as f:
            label_tree = pickle.load(f)
        trainset, valset, testset, label_mapping = load_inaturalist_data()
        continual_trainset_info = split_dataset_by_blurry(name, trainset, label_tree, num_tasks=num_tasks, overlap_ratio=overlap_ratio)
        continual_trainset = []
        for task_info in continual_trainset_info:
            task_indices = task_info['indices']
            task_subset = Subset(trainset, task_indices)
            task_info['data'] = task_subset # Add the Subset object
            continual_trainset.append(task_info) # Add the modified dictionary
        label_tree = reformat_tree(label_tree, label_mapping)



    elif name == 'imagenet':
        tree_fname = "./data/imagenet/imagenet_tree.pkl"
        with open(tree_fname, "rb") as f:
            label_tree = pickle.load(f)
        
        trainset, valset, testset, label_mapping = load_imagenet_data()
        continual_trainset_info = split_dataset_by_blurry(name, trainset, label_tree, num_tasks=10, overlap_ratio=0.2, label_mapping=label_mapping)
        continual_trainset = []
        for task_info in continual_trainset_info:
            task_indices = task_info['indices']
            task_subset = Subset(trainset, task_indices)
            task_info['data'] = task_subset # Add the Subset object
            continual_trainset.append(task_info) # Add the modified dictionary
        label_tree = reformat_tree(label_tree, label_mapping)


        # print(f"Tree height: {label_tree.height()}")
        # print("Nodes per level:")
        # levels = [0] * label_tree.height()  # 初始化每层的节点计数列表
        # for subtree in label_tree.subtrees():  # 遍历树中的所有子树
        #     levels[subtree.height() - 1] += 1  # 根据子树的高度更新对应层的计数 
        # for level, count in enumerate(reversed(levels)):  # 从根到叶子层打印
        #     print(f"  Level {level}: {count} nodes")    

    else:
        raise ValueError(f"Unsupported dataset name: {name}")

    return continual_trainset, valset, testset, label_tree

def get_dataloader(name, num_tasks, overlap_ratio, val_split, batch_size=32, num_workers=4):
    """
    获取数据加载器。
    """
    if name == 'cifar':
        continual_trainset, valset, testset, label_tree = get_dataset(name, num_tasks, overlap_ratio, val_split)
        loaders = create_cifar_dataloaders(continual_trainset, valset, testset, batch_size=batch_size, num_workers=num_workers)
    elif name == 'fgvc':
        continual_trainset, valset, testset, label_tree = get_dataset(name, num_tasks, overlap_ratio, val_split)
        loaders = create_fgvc_dataloaders(continual_trainset, valset, testset, batch_size=batch_size, num_workers=num_workers)
    elif name == 'cub':
        continual_trainset, valset, testset, label_tree = get_dataset(name, num_tasks, overlap_ratio, val_split)
        loaders = create_cub_dataloaders(continual_trainset, testset, batch_size=batch_size, num_workers=num_workers)
    elif name == 'inaturalist':
        continual_trainset, valset, testset, label_tree = get_dataset(name, num_tasks, overlap_ratio, val_split)
        loaders = create_inat_dataloaders(continual_trainset, valset, testset, batch_size=batch_size, num_workers=num_workers)
    elif name == 'imagenet':
        continual_trainset, valset, testset, label_tree = get_dataset(name, num_tasks, overlap_ratio, val_split)
        loaders = create_imagenet_dataloaders(continual_trainset, valset, testset, batch_size=batch_size, num_workers=num_workers)
    else:
        raise ValueError(f"Unsupported dataset name: {name}")
    return loaders, label_tree
    

if __name__ == "__main__":
    # Example usage
    # name = 'imagenet'
    # num_tasks = 10
    # overlap_ratio = 0.2
    # val_split = 0.2
    # continual_trainset, valset, testset, label_tree = get_dataset(name, num_tasks, overlap_ratio, val_split)
    # loaders = get_dataloader(name, continual_trainset, valset, testset, batch_size=32, shuffle=True, num_workers=4)
    # print(label_tree)
    # print("Data loaders created successfully.")

    # name = 'inaturalist'
    # num_tasks = 10
    # overlap_ratio = 0.2
    # val_split = 0.2
    # continual_trainset, valset, testset, label_tree = get_dataset(name, num_tasks, overlap_ratio, val_split)
    # loaders = get_dataloader(name, continual_trainset, valset, testset, batch_size=32, shuffle=True, num_workers=4)
    # print(label_tree)
    # print("Data loaders created successfully.")

    name = 'cifar'
    num_tasks = 10
    overlap_ratio = 0.2
    val_split = 0.2
    continual_trainset, valset, testset, label_tree = get_dataset(name, num_tasks, overlap_ratio, val_split)
    loaders = get_dataloader(name, continual_trainset, valset, testset, batch_size=32, shuffle=True, num_workers=4)
    print(label_tree)
    print("Data loaders created successfully.")

    name = 'fgvc'
    num_tasks = 10
    overlap_ratio = 0.2
    val_split = 0.2
    continual_trainset, valset, testset, label_tree = get_dataset(name, num_tasks, overlap_ratio, val_split)
    loaders = get_dataloader(name, continual_trainset, valset, testset, batch_size=32, shuffle=True, num_workers=4)
    print(label_tree)
    print("Data loaders created successfully.")




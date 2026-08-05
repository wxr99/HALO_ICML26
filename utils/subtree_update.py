from nltk.tree import Tree
import pickle

def update_seen_classes_and_tree(new_labels, global_tree, seen_classes):
    """
    根据新增的标签更新 seen_classes，并生成当前子树。

    Args:
        new_labels (list of str): 新增的标签列表，例如 ["L1_0", "L2_1"]。
        global_tree (nltk.tree.Tree): 全局标签树 (global_label_tree)。
        seen_classes (list of str): 已见类别的字符串列表。

    Returns:
        list of str: 更新后的 seen_classes。
        nltk.tree.Tree: 当前子树 (current_tree)。
    """
    # 更新 seen_classes
    seen_classes = list(set(seen_classes).union(set(new_labels)))  # 合并新标签并去重
    # 生成当前子树
    current_tree = get_subtree(global_tree, seen_classes)
    return seen_classes, current_tree

from nltk.tree import Tree

def get_subtree(tree, seen_classes):
    """
    根据 seen_classes 从 global_tree 中提取子树。
    如果某节点不在 seen_classes 中但其子节点在，则跳过该节点直接连接其子节点和最近的祖先节点。

    Args:
        tree (nltk.tree.Tree): 全局标签树。
        seen_classes (list of str): 已见类别的字符串列表。

    Returns:
        nltk.tree.Tree or None: 当前子树。如果当前节点及其子节点都不在 seen_classes 中，返回 None。
    """
    # 如果当前节点是字符串（叶子节点）
    if isinstance(tree, str):
        # 如果叶子节点的标签在 seen_classes 中，则保留该叶子节点
        return tree if tree in seen_classes else None

    # 确保 root 在 seen_classes 中
    if tree.label() == "root" and "root" not in seen_classes:
        seen_classes.append("root")

    if isinstance(tree, Tree):
        # 递归处理所有子节点
        subtrees = [get_subtree(child, seen_classes) for child in tree]
        # 移除所有 None 的子树
        subtrees = [sub for sub in subtrees if sub is not None]

        # 如果当前节点在 seen_classes 中
        if tree.label() in seen_classes:
            # 保留当前节点，并将符合条件的子树挂在它下面
            return Tree(tree.label(), subtrees)

        # 如果当前节点不在 seen_classes 中，但有子树
        elif subtrees:
            # 子树直接返回并挂载到祖先节点
            return Tree(tree.label(), subtrees)

    # 如果当前节点及其子节点都不在 seen_classes 中，返回 None
    return None

def filter_tree(tree, seen_classes):
    """
    根据 seen_classes 过滤子树。
    如果某节点不在 seen_classes 中，则跳过该节点，将其子节点直接连接到父节点。

    Args:
        tree (nltk.tree.Tree): 子树或根树。
        seen_classes (list of str): 已见类别的字符串列表。

    Returns:
        nltk.tree.Tree or None: 过滤后的子树。如果子树为空，返回 None。
    """
    # 如果当前节点是字符串（叶子节点）
    if isinstance(tree, str):
        # 如果叶子节点在 seen_classes 中，则保留该叶子节点
        return tree if tree in seen_classes else None

    # 如果当前节点是 Tree 对象
    if isinstance(tree, Tree):
        # 递归处理子节点
        filtered_children = [filter_tree(child, seen_classes) for child in tree]
        # 移除所有 None 的子节点
        filtered_children = [child for child in filtered_children if child is not None]

        # 如果当前节点在 seen_classes 中
        if tree.label() in seen_classes:
            # 保留当前节点，并挂载过滤后的子节点
            return Tree(tree.label(), filtered_children)

        # 如果当前节点不在 seen_classes 中，但有子节点
        elif filtered_children:
            # 如果当前节点不在 seen_classes 中，直接返回子节点列表
            if len(filtered_children) == 1:
                # 如果只有一个子节点，直接返回该子节点
                return filtered_children[0]
            else:
                # 如果有多个子节点，返回它们作为列表
                return Tree(tree.label(), filtered_children)

    # 如果当前节点及其子节点都不符合条件，返回 None
    return None


# # 全局标签树
# tree_fname = "./data/cifar100/cifar_100_tree.pkl"
# name = 'cifar'
# with open(tree_fname, "rb") as f:
#     label_tree = pickle.load(f)
# print(label_tree)

# # 已见类别
# seen_classes = ["L2_1", "L3_1", "L2_3", "L3_89", "L2_10", "L3_91", ]

# current_tree = get_subtree(label_tree, seen_classes)
# print(current_tree)

# filtered_tree = filter_tree(current_tree, seen_classes)
# print(filtered_tree)
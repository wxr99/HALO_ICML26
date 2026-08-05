import nltk
from nltk.tree import Tree
from collections import defaultdict
import pickle
# These are your specified imports
from .imagenet import load_imagenet_data
from .inaturalist import load_inaturalist_data

def reformat_tree(original_tree, label_mapping):
    """
    Transforms an NLTK Tree into a new Tree with nodes labeled as Lm_n,
    filtering to only include branches leading to leaves present in label_mapping.
    All leaf nodes in the resulting tree will be at the same, maximum relevant depth.
    The root node of the returned tree is always labeled 'root'.

    - Mapped leaf nodes 'original_id' -> mapped_id will appear as L<max_depth>_<mapped_id>.
    - Original internal nodes that are kept, and new dummy intermediate nodes,
      will be labeled L<level>_<sequential_id> based on their depth in the new tree
      and a counter for that level.

    Args:
        original_tree (nltk.tree.Tree): The input NLTK Tree object.
                                        Its direct children are considered the start of the hierarchy (level 1).
        label_mapping (dict): A dictionary mapping original leaf node labels
                               (strings) to new integer IDs. Only leaves
                               present in this mapping will be included.

    Returns:
        nltk.tree.Tree: A new NLTK Tree with reformatted labels and uniform leaf depth.
                        Returns Tree('root', []) if the entire tree is pruned.
    Raises:
        TypeError: If original_tree is not an nltk.tree.Tree or
                   label_mapping is not a dict.
    """
    if not isinstance(original_tree, Tree):
        raise TypeError("Input 'original_tree' must be an nltk.tree.Tree object.")
    if not isinstance(label_mapping, dict):
         raise TypeError("Input 'label_mapping' must be a dictionary.")

    # --- Step 1: Find the maximum depth of relevant (mapped) leaves ---
    # The depth is relative to the children of the original_tree's root.
    # These children will become level 1 in the new tree under 'root'.
    max_relevant_depth = 0

    def _find_max_depth_recursive(node, current_level_from_original_root_child):
        nonlocal max_relevant_depth
        if not isinstance(node, Tree): # Leaf node (string) in original tree
            original_label = str(node)
            if original_label in label_mapping:
                max_relevant_depth = max(max_relevant_depth, current_level_from_original_root_child)
            return

        # Internal node (Tree) in original tree
        for child in node:
            _find_max_depth_recursive(child, current_level_from_original_root_child + 1)

    # Iterate through the direct children of the input original_tree.
    # These children are considered level 1 for depth calculation.
    for child_node_of_original_root in original_tree:
        _find_max_depth_recursive(child_node_of_original_root, 1)


    # --- Step 2: Transform the tree, padding to max_relevant_depth ---
    level_counters = defaultdict(int) # For unique sequential_ids for L<level>_sequential_id internal nodes

    def _transform_and_pad_recursive(node, current_level_in_new_tree, target_leaf_depth):
        # current_level_in_new_tree is the level this 'node' (if internal) or its dummy parent (if leaf)
        # will have in the new tree structure under 'root'.

        # Base Case: Leaf node (string from original tree)
        if not isinstance(node, Tree):
            original_label = str(node)
            if original_label in label_mapping:
                mapped_id = label_mapping[original_label]
                # Leaf node in the new tree: L<target_leaf_depth>_<mapped_id>
                current_node_representation = f"L{target_leaf_depth}_{mapped_id}"

                # If this leaf's original path (represented by current_level_in_new_tree for its would-be position)
                # is shallower than target_leaf_depth, we need to insert dummy parent nodes.
                # The dummy nodes are built upwards from the leaf.
                # Example: Leaf is at new_level 2, target is 4. Dummies needed at new_level 3 and new_level 2.
                # The leaf itself is L4_id. Its immediate parent will be L3_dummy. That parent's parent L2_dummy.
                for dummy_node_level in range(target_leaf_depth - 1, current_level_in_new_tree - 1, -1):
                    sequential_dummy_id = level_counters[dummy_node_level]
                    level_counters[dummy_node_level] += 1
                    dummy_label = f"L{dummy_node_level}_{sequential_dummy_id}"
                    current_node_representation = Tree(dummy_label, [current_node_representation])
                return current_node_representation
            else:
                return None # Prune this leaf as it's not in label_mapping

        # Recursive Step: Internal node (Tree from original tree)
        else:
            valid_children_transformed = []
            for child in node: # Children of this node will be at current_level_in_new_tree + 1
                transformed_child = _transform_and_pad_recursive(child, current_level_in_new_tree + 1, target_leaf_depth)
                if transformed_child is not None:
                    valid_children_transformed.append(transformed_child)

            if not valid_children_transformed:
                return None # Prune this branch if it has no valid (mapped) children
            else:
                # This node becomes an internal node in the new tree at current_level_in_new_tree.
                # Its label is L<current_level_in_new_tree>_<sequential_id>.
                # This is consistent with your original reformat_tree's internal node labeling.
                sequential_node_id = level_counters[current_level_in_new_tree]
                level_counters[current_level_in_new_tree] += 1
                new_node_label = f"L{current_level_in_new_tree}_{sequential_node_id}"
                return Tree(new_node_label, valid_children_transformed)

    # --- Main transformation logic ---
    transformed_children_for_new_root = []
    # If max_relevant_depth is 0 (e.g., no mapped leaves found, or label_mapping empty),
    # then target_leaf_depth will be 0. Leaves will be L0_id, and no padding loop runs.

    for child_node_of_original_root in original_tree:
        # These children of original_tree's root are processed to become L1 nodes (or start of L1 branches)
        # in the new tree under 'root'.
        result = _transform_and_pad_recursive(child_node_of_original_root, 1, max_relevant_depth)
        if result is not None:
            transformed_children_for_new_root.append(result)

    return Tree('root', transformed_children_for_new_root)

def count_nodes_by_level(node, depth):
        """
        递归统计每一层的节点数。
        depth: 当前节点的深度（从 0 开始）。
        """
        levels[depth] += 1
        if isinstance(node, Tree):  # 如果当前节点是子树，递归处理它的子节点
            for child in node:
                count_nodes_by_level(child, depth + 1)

if __name__ == "__main__":
    # Example usage
    tree_fname = "./data/iNaturalist/inaturalist19_tree.pkl"
    name = 'inaturalist'
    with open(tree_fname, "rb") as f:
        label_tree = pickle.load(f)
    # print(len(label_tree.leaves()))
        
    print(f"Tree height: {label_tree.height()}")

    print("Nodes per level:")
    levels = [0] * label_tree.height()  # 初始化每层的节点计数列表
    count_nodes_by_level(label_tree, 0)
    for level, count in enumerate(levels):
        print(f"  Level {level}: {count} nodes")
    print(len(label_tree.leaves()))
    
    trainset, valset, testset, label_mapping = load_inaturalist_data()
    label_tree = reformat_tree(label_tree, label_mapping)
    # print(label_mapping)

    # print(f"Label tree: {label_tree}")  # 打印标签树
    # 打印imagenet数据集的树结构 
    print(f"Tree height: {label_tree.height()}")
    print("Nodes per level:")
    levels = [0] * label_tree.height()  # 初始化每层的节点计数列表
    # 从根节点开始统计
    count_nodes_by_level(label_tree, 0)

    # 打印每层的节点数
    for level, count in enumerate(levels):
        print(f"  Level {level}: {count} nodes")
    print(len(label_tree.leaves()))
    

    # tree_fname = "./data/iNaturalist/inaturalist19_tree.pkl" # tree_fname is reassigned
    # name = 'inaturalist'
    # with open(tree_fname, "rb") as f:
    #     label_tree = pickle.load(f) # label_tree is from iNaturalist
    # print(len(label_tree.leaves()))
    # trainset, valset, testset, label_mapping = load_inaturalist_data() # mapping for iNaturalist
    # print(label_mapping)
    # new_tree = reformat_tree(label_tree, label_mapping) # Called with iNaturalist tree and mapping
    # print(new_tree)
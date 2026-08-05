import nltk
import pickle
from nltk.tree import Tree
from typing import Dict, List, Tuple, Optional, Union, Any
import re
import torch # 确保torch已导入

# from dataset.dataloader import load_cifar100, load_fgvc_data, load_inaturalist_data, load_imagenet_data
# from dataset.reformat_tree import reformat_tree

# --- Generalized Label Parser (与您提供的一致) ---
def parse_label_string_generalized(label_str: str) -> Tuple[Optional[int], Optional[int]]:
    if not isinstance(label_str, str):
        return None, None
    match = re.fullmatch(r"L(\d+)_(\d+)", label_str)
    if match:
        try:
            level = int(match.group(1))
            value = int(match.group(2))
            return level, value
        except ValueError:
            return None, None
    return None, None

# --- GeneralizedLabelHierarchyMapper (与您提供的一致) ---
class GeneralizedLabelHierarchyMapper:
    def __init__(self, label_tree: Tree):
        self.label_tree: Tree = label_tree
        self.leaf_to_ancestors_map: Dict[int, Dict[int, int]] = {}
        self.leaf_level_num: Optional[int] = None
        self.max_level_in_tree: int = 0

        if not isinstance(label_tree, Tree) or label_tree.label().lower() != 'root':
            print("Warning: Tree does not have a 'root' label. Processing children directly.")
            nodes_to_process = label_tree if isinstance(label_tree, list) else [label_tree]
        else:
            nodes_to_process = label_tree

        for l1_node_equivalent in nodes_to_process:
            self._recursive_build_map(l1_node_equivalent, {}, self.leaf_to_ancestors_map)

    def _recursive_build_map(self, current_node: Union[Tree, str],
                             current_ancestors: Dict[int, int],
                             mapping: Dict[int, Dict[int, int]]):
        node_label_str = current_node.label() if isinstance(current_node, Tree) else current_node
        level, value = parse_label_string_generalized(node_label_str)

        if level is None or value is None:
            if isinstance(current_node, Tree):
                for child in current_node:
                    self._recursive_build_map(child, current_ancestors, mapping)
            return

        self.max_level_in_tree = max(self.max_level_in_tree, level)
        child_ancestors = current_ancestors.copy()
        child_ancestors[level] = value

        if isinstance(current_node, Tree) and len(current_node) > 0:
            for child in current_node:
                if isinstance(child, Tree):
                    self._recursive_build_map(child, child_ancestors, mapping)
                elif isinstance(child, str): # Leaf node as a string
                    leaf_level, leaf_value = parse_label_string_generalized(child)
                    if leaf_level is not None and leaf_value is not None:
                        self.max_level_in_tree = max(self.max_level_in_tree, leaf_level)
                        if self.leaf_level_num is None:
                            self.leaf_level_num = leaf_level
                        elif self.leaf_level_num != leaf_level:
                            print(f"Warning: Inconsistent leaf levels found. Previously {self.leaf_level_num}, now {leaf_level} for {child}.")
                        mapping[leaf_value] = child_ancestors.copy()
        # (No changes to this class as per request)


    def get_ancestors(self, leaf_label_num: int) -> Union[Dict[int, int], None]:
        return self.leaf_to_ancestors_map.get(leaf_label_num)

    def get_ancestors_for_batch(self, batch_leaf_numbers: List[int]) -> List[Union[Dict[int, int], None]]:
        results: List[Union[Dict[int, int], None]] = []
        for leaf_num in batch_leaf_numbers:
            results.append(self.get_ancestors(leaf_num))
        return results

# --- reformat_batch_labels_generalized function (与您提供的一致) ---
def reformat_batch_labels_generalized(
    original_leaf_batch: List[int],
    processed_ancestors_batch: List[Union[Dict[int, int], None]],
    leaf_level_number: int,
    padding_value: Any = -1
) -> Dict[str, List[Any]]: # Output is Dict[str, List[Any]]
    """
    Reformats batch labels for a generalized multi-level hierarchy into a single dictionary.
    (No changes to this function as per request, its output remains List[Any] for values)
    """
    if len(original_leaf_batch) != len(processed_ancestors_batch):
        raise ValueError("Input lists original_leaf_batch and processed_parents_batch must have the same length.")

    if not isinstance(leaf_level_number, int) or leaf_level_number <= 0:
        raise ValueError(f"leaf_level_number must be a positive integer. Got: {leaf_level_number}")

    num_levels = leaf_level_number
    batch_data_for_levels: List[List[Any]] = [[] for _ in range(num_levels)]

    for i, leaf_value in enumerate(original_leaf_batch):
        ancestors_map = processed_ancestors_batch[i]
        for current_level_idx in range(num_levels):
            level_num_to_fill = current_level_idx + 1
            if level_num_to_fill == leaf_level_number:
                batch_data_for_levels[current_level_idx].append(leaf_value)
            elif ancestors_map and level_num_to_fill in ancestors_map:
                batch_data_for_levels[current_level_idx].append(ancestors_map[level_num_to_fill])
            else:
                batch_data_for_levels[current_level_idx].append(padding_value)

    reformatted_output: Dict[str, List[Any]] = {}
    for level_idx in range(num_levels):
        level_num = level_idx + 1
        reformatted_output[f"L{level_num}_head"] = batch_data_for_levels[level_idx]

    return reformatted_output

def label_transformer(label: torch.Tensor, label_tree: Tree) -> Dict[str, torch.Tensor]:
    """
    Transform labels (provided as a tensor) based on the label tree structure.
    Outputs a dictionary where keys are level strings and values are tensors.
    """
    if not isinstance(label, torch.Tensor):
        raise TypeError(f"Input 'label' must be a torch.Tensor. Got {type(label)}")
    
    if label.ndim != 1:
        raise ValueError(f"Input 'label' tensor must be 1-dimensional. Got {label.ndim} dimensions.")

    label_list: List[int] = [int(item) for item in label.tolist()]

    mapper_multi = GeneralizedLabelHierarchyMapper(label_tree)

    if mapper_multi.leaf_level_num is None:
        raise ValueError(
            "Could not determine leaf_level_num from the provided label_tree. "
            "Please ensure the tree structure is correct and leaf nodes are strings "
            "formatted like 'L<level>_<value>' (e.g., 'L3_10'). "
            "This can also happen if the tree is empty or does not contain any parsable leaf nodes."
        )

    ancestors: List[Union[Dict[int, int], None]] = mapper_multi.get_ancestors_for_batch(label_list)
    
    # This function returns Dict[str, List[Any]]
    dict_of_lists: Dict[str, List[Any]] = reformat_batch_labels_generalized(
        label_list,
        ancestors,
        leaf_level_number=mapper_multi.leaf_level_num,
        padding_value=-1
    )

    # Convert the lists in the dictionary to tensors
    dict_of_tensors: Dict[str, torch.Tensor] = {}
    for key, list_val in dict_of_lists.items():
        # Assuming labels and padding_value are integers, torch.long is a good default.
        # If your labels could be other types, adjust dtype accordingly.
        try:
            dict_of_tensors[key] = torch.tensor(list_val, dtype=torch.long)
        except Exception as e:
            # Provide more context if tensor conversion fails
            print(f"Error converting list to tensor for key '{key}'. List was: {list_val}")
            raise e
            
    return dict_of_tensors

_pat = re.compile(r"^L(\d+)_(\d+)$")

def seen_classes_per_level(
    seen_classes: List[str],
    device: torch.device = torch.device("cpu"),
) -> List[torch.Tensor]:
    per_level_sets = {}
    max_level = 0
    for s in seen_classes:
        if s == "root":
            continue
        m = _pat.match(s)
        if not m:
            continue
        lv = int(m.group(1))
        n = int(m.group(2))
        if lv < 1 or n < 0:
            continue
        per_level_sets.setdefault(lv, set()).add(n)
        max_level = max(max_level, lv)

    per_level = []
    for lv in range(1, max_level + 1):
        if lv in per_level_sets and len(per_level_sets[lv]) > 0:
            idx = torch.tensor(sorted(per_level_sets[lv]), dtype=torch.long, device=device)
        else:
            idx = torch.empty(0, dtype=torch.long, device=device)
        per_level.append(idx)
    return per_level

if __name__ == "__main__":
    # --- Example Usage (与您在第一个问题中提供的一致) ---
    # 注意:下面的代码依赖于一个名为 "cifar_100_tree.pkl" (或其他注释掉的文件) 的文件
    # 在当前目录下的 "./data/cifar100/" 路径中。
    # 如果此文件不存在，运行此 __main__ 块将会失败。
    tree_fname = "./data/cifar100/cifar_100_tree.pkl"
    # tree_fname = "./data/fgvc/fgvc_label_hierarchy_tree.pkl"
    # tree_fname = "./data/iNaturalist/inaturalist19_tree.pkl"
    # tree_fname = "./data/imagenet/imagenet_tree.pkl"

    with open(tree_fname, "rb") as f:
        label_tree = pickle.load(f)
    # print(label_tree)
    # print(label_tree.height())
    # print(label_tree.leaves())
    # trainset, valset, testset, label_mapping = load_inaturalist_data()
    # trainset, valset, testset, label_mapping = load_imagenet_data()
    # print(len(label_mapping))

    # label_tree = reformat_tree(label_tree, label_mapping)
    # print(label_tree)
    # print(label_tree.height())
    # print(len(label_tree.leaves()))
    mapper_multi = GeneralizedLabelHierarchyMapper(label_tree)

    batch_input: List[int] = [10, 15, 20, 99] # __main__块仍使用List作为输入进行测试
    ancestors_for_batch: List[Union[Dict[int, int], None]] = mapper_multi.get_ancestors_for_batch(batch_input)

    if mapper_multi.leaf_level_num is None:
        raise ValueError(
            "Could not determine leaf_level_num from the provided label_tree in __main__. "
            "Please ensure the tree structure is correct and leaf nodes are parsable. "
            "The __main__ block cannot proceed without a valid leaf_level_num."
        )

    desired_output_multi = reformat_batch_labels_generalized(
        batch_input,
        ancestors_for_batch,
        leaf_level_number=mapper_multi.leaf_level_num,
        padding_value=-1
    )

    print(desired_output_multi)
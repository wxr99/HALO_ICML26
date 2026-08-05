import nltk
from nltk.tree import Tree
from collections import defaultdict
import numpy as np
import pickle # <--- 1. 导入 pickle 模块

# --- (之前的代码：定义 label_meta, 构建 hierarchy) ---
label_meta = np.array([
    [ 0, 12,  4], [ 1, 14,  4], [ 2, 15,  4], [ 3, 15,  4], [ 4, 15,  4],
    [ 5, 15,  4], [ 6, 15,  4], [ 7, 15,  4], [ 8, 15,  4], [ 9, 15,  4],
    [10, 16,  4], [11, 16,  4], [12, 16,  4], [13, 16,  4], [14, 17,  4],
    [15, 17,  4], [16, 18,  4], [17, 18,  4], [18, 18,  4], [19, 19,  4],
    [20, 19,  4], [21,  0,  1], [22,  1,  1], [23,  2,  1], [24,  2,  1],
    [25,  2,  1], [26,  2,  1], [27,  3,  1], [28,  3,  1], [29,  4,  1],
    [30,  4,  1], [31,  4,  1], [32,  4,  1], [33,  5,  1], [34,  6,  0],
    [35,  7,  0], [36,  8,  2], [37,  9,  6], [38,  9,  6], [39, 10,  6],
    [40, 11,  3], [41, 13,  4], [42, 20, 19], [43, 21, 12], [44, 22,  7],
    [45, 23,  7], [46, 23,  7], [47, 24,  8], [48, 25,  8], [49, 26,  8],
    [50, 26,  8], [51, 27,  7], [52, 28, 21], [53, 29, 12], [54, 30, 12],
    [55, 31, 12], [56, 32, 21], [57, 33, 29], [58, 34, 29], [59, 35, 29],
    [60, 37, 29], [61, 37, 29], [62, 36, 24], [63, 38, 11], [64, 40, 13],
    [65, 40, 13], [66, 40, 13], [67, 39, 13], [68, 41, 13], [69, 41, 13],
    [70, 42, 13], [71, 43, 14], [72, 44, 20], [73, 45, 21], [74, 46, 10],
    [75, 47, 10], [76, 48, 16], [77, 49, 16], [78, 50, 16], [79, 51,  5],
    [80, 52, 17], [81, 52, 17], [82, 53,  6], [83, 54, 18], [84, 56, 19],
    [85, 57, 21], [86, 58, 21], [87, 58, 21], [88, 59, 21], [89, 60, 15],
    [90, 55,  3], [91, 61, 23], [92, 62,  9], [93, 63, 25], [94, 64, 25],
    [95, 65, 26], [96, 66, 22], [97, 67, 27], [98, 68, 27], [99, 69, 28]
])

hierarchy = defaultdict(lambda: defaultdict(list))
for i in range(label_meta.shape[0]):
    l3_id = label_meta[i, 0]
    l2_id = label_meta[i, 1]
    l1_id = label_meta[i, 2]
    hierarchy[l1_id][l2_id].append(l3_id)

root_node = Tree('root', [])
for l1_id in sorted(hierarchy.keys()):
    l1_label = f"L1_{l1_id}"
    l1_node = Tree(l1_label, [])
    l2_groups = hierarchy[l1_id]
    for l2_id in sorted(l2_groups.keys()):
        l2_label = f"L2_{l2_id}"
        l3_ids = sorted(l2_groups[l2_id])
        l3_leaves = [f"L3_{l3_id}" for l3_id in l3_ids]
        l2_node = Tree(l2_label, l3_leaves)
        l1_node.append(l2_node)
    root_node.append(l1_node)

# --- 将 root_node 存储到 PKL 文件 ---

# 2. 定义要保存的文件名
pkl_filename = 'fgvc_label_hierarchy_tree.pkl'

# 3. 使用 'wb' (写入二进制) 模式打开文件
try:
    with open(pkl_filename, 'wb') as pkl_file:
        # 4. 使用 pickle.dump() 将对象写入文件
        pickle.dump(root_node, pkl_file)
    print(f"NLTK Tree 已成功保存到 '{pkl_filename}'")

except Exception as e:
    print(f"保存到 PKL 文件时出错: {e}")

# --- （可选）验证：从文件加载并打印 ---
try:
    with open(pkl_filename, 'rb') as pkl_file:
        loaded_tree = pickle.load(pkl_file)

    print(f"\n已从 '{pkl_filename}' 成功加载 Tree:")
    # 验证加载的对象是否与原始对象相同 (或至少结构相同)
    print(loaded_tree)
    # print("\nPretty Print of loaded tree:")
    # loaded_tree.pretty_print()

    # 检查类型是否正确
    print(f"\n加载对象的类型: {type(loaded_tree)}")
    assert isinstance(loaded_tree, Tree)

except FileNotFoundError:
    print(f"错误：找不到文件 '{pkl_filename}' 进行加载验证。")
except Exception as e:
    print(f"从 PKL 文件加载时出错: {e}")
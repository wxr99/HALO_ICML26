import math
from typing import Dict, List, Optional, Tuple, Union, Any
import torch
import torch.nn.functional as F
import re

_L_RE = re.compile(r"^L(\d+)_(\d+)$")

def seen_classes_to_level_tensors(
    seen_classes: List[str],
    max_level: Optional[int] = None,
    one_based_label: bool = False,  # 若标签 n 是从1开始且你希望转为从0开始，可设 True
) -> List[torch.Tensor]:
    """
    将 ['root', 'L1_2', 'L3_5', ...] 解析为按层收集到的标签索引的 list[Tensor].
    约定 root -> 层0，索引固定为0。

    返回 levels: 长度为 max_level+1 的列表，
      - levels[0] 为 [0]（若包含 root），否则为空 tensor
      - levels[m] 为本层见到的标签索引（LongTensor，升序、去重）
    """
    # 暂存为 dict[level] -> set(indices)
    levels_dict: Dict[int, set] = {}

    for s in seen_classes:
        if s == "root":
            levels_dict.setdefault(0, set()).add(0)
            continue
        m = _L_RE.match(s)
        if not m:
            # 跳过不符合格式的项（也可 raise）
            continue
        lvl = int(m.group(1))
        idx = int(m.group(2))
        if one_based_label:
            idx = idx - 1
            if idx < 0:
                # 防御：若原始是1-based，转换后不得为负
                continue
        levels_dict.setdefault(lvl, set()).add(idx)

    # 确定最大层
    if max_level is None:
        max_level = max(levels_dict.keys()) if levels_dict else 0

    # 组装为 list[Tensor]，未出现的层给空 tensor
    levels: List[torch.Tensor] = []
    for lvl in range(0, max_level + 1):
        if lvl in levels_dict and len(levels_dict[lvl]) > 0:
            sorted_idx = sorted(levels_dict[lvl])
            t = torch.tensor(sorted_idx, dtype=torch.long)
        else:
            t = torch.empty(0, dtype=torch.long)
        levels.append(t)
    return levels

def softmax_with_temperature(logits: torch.Tensor, T: float = 1.0) -> torch.Tensor:
    if T <= 0:
        raise ValueError("Temperature T must be > 0.")
    return F.softmax(logits / T, dim=-1)

def normalized_entropy(p: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    # p: [B,C], 返回归一化熵 H/ log(C)，范围 [0,1]
    B, C = p.shape
    logC = math.log(C)
    p_safe = p.clamp_min(eps)
    H = -(p_safe * p_safe.log()).sum(dim=-1)          # [B]
    return H / max(logC, eps)                         # [B]

def entropy_weights(p_list: list, alpha: float = 1.0, eps: float = 1e-8) -> torch.Tensor:
    # p_list: [p1, p2, ...]，每个形状 [B,C]
    # 返回 w: [B,E]，按样本归一化的专家权重
    entrs = [normalized_entropy(p) for p in p_list]   # 各为 [B]
    ent = torch.stack(entrs, dim=-1)                  # [B,E]
    conf = (1.0 - ent).clamp_min(0.0)                 # [B,E]
    conf = conf.pow(alpha)
    w = conf / (conf.sum(dim=-1, keepdim=True) + eps) # [B,E]
    return w

def geometric_mean_ensemble(p_list: list, w: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    # p_list: [p1, p2, ...]，各 [B,C]
    # w: [B,E]，和为 1
    B, C = p_list[0].shape
    E = len(p_list)
    P = torch.stack(p_list, dim=1)            # [B,E,C]
    P = P.clamp_min(eps)
    logP = P.log()                            # [B,E,C]
    w_expanded = w.unsqueeze(-1)              # [B,E,1]
    logP_mix = (w_expanded * logP).sum(dim=1) # [B,C]
    p_ens = F.softmax(logP_mix, dim=-1)       # 保持概率规范
    return p_ens

def arithmetic_mean_ensemble(p_list: list, w: torch.Tensor) -> torch.Tensor:
    P = torch.stack(p_list, dim=1)            # [B,E,C]
    w_expanded = w.unsqueeze(-1)              # [B,E,1]
    p_ens = (w_expanded * P).sum(dim=1)       # [B,C]
    # 已经是凸组合，无需再 softmax
    return p_ens

def ensemble_per_head(
    linear_logits: Dict[str, torch.Tensor],
    proto_logits: Dict[str, torch.Tensor],
    acil_logits: Dict[str, torch.Tensor],
    temps: Dict[str, Tuple[float, float, float]] = None,  # 每个 head 的 (T_linear, T_proto, T_acil)
    alpha: float = 1.0,           # 熵到权重的非线性指数
    method: str = "geo",          # "geo" 或 "arith"
    low_conf_fallback: Optional[float] = None,  # 若设定阈值，如 0.5：所有专家 max prob < 阈值时改用算术平均
    seen_classess: Optional[List[str]] = None,
) -> Dict[str, Dict[str, torch.Tensor]]:
    """
    返回:
      {
        head: {
          'p_linear': [B,C],
          'p_proto':  [B,C],
          'p_acil':   [B,C],
          'w':        [B,3],
          'p_ens':    [B,C],
          'logits_ens':[B,C],
          'pred':     [B]
        }, ...
      }
    """
    seen_levels = seen_classes_to_level_tensors(seen_classess) if seen_classess is not None else None
    heads = linear_logits.keys()
    out = {}
    logits_ens_all = {}
    for i, h in enumerate(heads):
        L = linear_logits[h]
        P = proto_logits[h]
        A = acil_logits[h]
        seen_class = seen_levels[i]
        assert L.shape == P.shape == A.shape, f"Logit shape mismatch at head {h}"
        B, C = L.shape

        T_lin, T_pro, T_aci = (20.0, 20.0, 1.0)
        if temps is not None and h in temps:
            T_lin, T_pro, T_aci = temps[h]

        p_lin = softmax_with_temperature(L, T_lin)   # [B,C]
        p_pro = softmax_with_temperature(P, T_pro)   # [B,C]
        p_aci = softmax_with_temperature(A, T_aci)   # [B,C]

        p_lin_seen = softmax_with_temperature(L[:, seen_class], T_lin)   
        p_pro_seen = softmax_with_temperature(P[:, seen_class], T_pro)   
        p_aci_seen = softmax_with_temperature(A[:, seen_class], T_aci)   


        plist = [p_lin, p_pro, p_aci]
        plist_seen = [p_lin_seen, p_pro_seen, p_aci_seen]
        w = entropy_weights(plist_seen, alpha=alpha)      # [B,3]
        # print(w)

        if low_conf_fallback is not None:
            # 若所有专家的max prob都很低，则改用算术平均（更平滑）
            max_conf = torch.stack([p.max(dim=-1).values for p in plist], dim=-1).max(dim=-1).values  # [B]
            use_arith = (max_conf < low_conf_fallback).float().unsqueeze(-1)  # [B,1]
        else:
            use_arith = None

        if method == "geo":
            p_geo = geometric_mean_ensemble(plist, w)    # [B,C]
            p_arith = arithmetic_mean_ensemble(plist, w) # [B,C]
            if use_arith is not None:
                p_ens = p_geo * (1 - use_arith) + p_arith * use_arith
            else:
                p_ens = p_geo
        elif method == "arith":
            p_ens = arithmetic_mean_ensemble(plist, w)
        else:
            raise ValueError(f"Unknown method: {method}")

        logits_ens = (p_ens + 1e-12).log()             # 对应对数概率，可用于后续联合损失
        pred = p_ens.argmax(dim=-1)

        out[h] = {
            "p_linear": p_lin, "p_proto": p_pro, "p_acil": p_aci,
            "w": w,
            "p_ens": p_ens,
            "logits_ens": logits_ens,
            "pred": pred,
        }
        logits_ens_all[h] = logits_ens
    return out, logits_ens_all
    
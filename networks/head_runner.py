# head_runner.py
from dataclasses import dataclass
from typing import Dict, List, Optional, Union, Any
import torch

HeadTypes = Union[str, List[str]]
LogitsDict = Union[Dict[str, torch.Tensor], Dict[str, Dict[str, torch.Tensor]]]
ListOfLogits = Union[List[torch.Tensor], Dict[str, List[torch.Tensor]]]

@dataclass
class HeadRunResult:
    """
    标准化后的模型输出：
    - logits_dict:
        * 若 head_types 是 str: {head_name: logits}
        * 若 head_types 是 list[str]: {head_type: {head_name: logits}}
    - list_of_logits:
        * 若 head_types 是 str: [logits_of_head_names_in_order]
        * 若 head_types 是 list[str]: {head_type: [logits_in_order]}
    - features: 可选，形如 {'pool': tensor, 'c3': fmap, 'gap:c3': vec, ...}
    """
    logits_dict: LogitsDict
    list_of_logits: ListOfLogits
    features: Optional[Dict[str, torch.Tensor]] = None
    proto_features: Optional[Dict[str, torch.Tensor]] = None


def _normalize_head_types(head_types: HeadTypes) -> List[str]:
    if isinstance(head_types, str):
        return [head_types]
    if isinstance(head_types, list):
        if not head_types:
            raise ValueError("head_types list cannot be empty.")
        if not all(isinstance(h, str) for h in head_types):
            raise ValueError("head_types must be a str or list of str.")
        return head_types
    raise ValueError("head_types must be a str or list of str.")


def _normalize_features_request(
    features_to_return: Optional[Union[bool, str, List[str]]],
    default_return_pool: bool,
) -> Optional[Union[bool, str, List[str]]]:
    """
    归一化特征请求：
    - None 且 default_return_pool=False -> None（不返回特征）
    - None 且 default_return_pool=True  -> True（等价请求 'pool'）
    - 传入 bool/str/list 时直接校验并透传
    """
    if features_to_return is None:
        return True if default_return_pool else None
    if isinstance(features_to_return, bool):
        return features_to_return
    if isinstance(features_to_return, str):
        return features_to_return
    if isinstance(features_to_return, list):
        if not all(isinstance(k, str) for k in features_to_return):
            raise ValueError("features_to_return list must contain strings.")
        return features_to_return
    raise ValueError("features_to_return must be None | bool | str | List[str].")


def run_heads(
    model: Any,
    inputs: torch.Tensor,
    
    head_names: List[str],
    head_types: HeadTypes = "linear",
    features_to_return: Optional[Union[bool, str, List[str]]] = None,
    default_return_pool: bool = False,
    return_protofeatures: bool = False,
) -> HeadRunResult:
    """
    统一调用模型并标准化输出。

    Parameters:
    - model: 你的模型实例，需实现 forward(
        x, head_names, head_types=..., features_to_return=..., return_features=...
      )
      注：forward 内已支持 features_to_return 语义（True 等价 'pool'）
    - inputs: 输入张量
    - head_names: 要计算的 head 名称列表
    - head_types: 'linear' | 'acil' | 'protonet' 或其列表
    - features_to_return:
        * None: 是否返回特征由 default_return_pool 决定
        * True: 返回默认 'pool' 向量
        * 'pool' | 'flat' | 'c2'/'c3'/'c4'/'c5' | 'gap:c2'... 指定需要的特征
        * 列表可组合请求
    - default_return_pool: 当 features_to_return 为 None 时，是否默认返回 'pool'

    Returns:
    - HeadRunResult：包含 logits 的字典、按顺序的 list_of_logits，以及可选的 features
    """
    if not head_names:
        raise ValueError("head_names list cannot be empty.")

    original_is_str = isinstance(head_types, str)
    head_types_list = _normalize_head_types(head_types)
    features_req = _normalize_features_request(features_to_return, default_return_pool)

    # 约定：统一通过 features_to_return 控制是否返回特征，
    # 将 return_features 固定为 False，避免双重语义。
    call_head_types: HeadTypes = head_types if original_is_str else head_types_list
    res = model(
        inputs,
        head_names,
        head_types=call_head_types,
        features_to_return=features_req,
        return_features=False,
        return_protofeatures=return_protofeatures,
    )

    # 模型可能返回两种形态：
    # - 仅 logits 字典
    # - {'logits': logits_dict, 'features': {...}}
    if isinstance(res, dict) and "logits" in res:
        logits_dict: LogitsDict = res["logits"]
        features: Optional[Dict[str, torch.Tensor]] = res.get("features")
        if return_protofeatures:
            proto_featrues = res.get("proto_features")
        else:
            proto_featrues = None
    else:
        logits_dict = res  # type: ignore
        features = None

    # 标准化 list_of_logits（严格按 head_names 顺序）
    if original_is_str:
        if not isinstance(logits_dict, dict):
            raise RuntimeError("Model returned unexpected logits structure for single head_type.")
        list_of_logits: List[torch.Tensor] = [logits_dict[name] for name in head_names]
        list_of_proto_featrues =  [proto_featrues[name] for name in head_names] if return_protofeatures else None
        return HeadRunResult(logits_dict=logits_dict, list_of_logits=list_of_logits, features=features, 
                             proto_features=list_of_proto_featrues)
    else:
        if not isinstance(logits_dict, dict):
            raise RuntimeError("Model returned unexpected logits structure for multiple head_types.")
        per_type_lists: Dict[str, List[torch.Tensor]] = {}
        for ht in head_types_list:
            d = logits_dict.get(ht)
            if not isinstance(d, dict):
                raise RuntimeError(f"Missing logits for head_type '{ht}'.")
            per_type_lists[ht] = [d[name] for name in head_names]
        list_of_proto_featrues = [proto_featrues[name] for name in head_names] if return_protofeatures else None
        return HeadRunResult(logits_dict=logits_dict, list_of_logits=per_type_lists, features=features, 
                             proto_features=list_of_proto_featrues)
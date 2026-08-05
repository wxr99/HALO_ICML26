# heads_setup.py
from typing import Dict, List, Optional
import torch.nn as nn

# 按你的工程路径调整导入
try:
    from .analytic_classifier import ACILDynamicClasses
    from .protonet import ProtoNetHead
    from .iciclenet import IcicleNetHead
    from .hiepronet import HieProNetHead
except ImportError:
    from analytic_classifier import ACILDynamicClasses
    from protonet import ProtoNetHead
    from iciclenet import IcicleNetHead
    from hiepronet import HieProNetHead


def _infer_resnet_stage_channels(backbone) -> Dict[str, Optional[int]]:
    """
    推断 ResNet 各层输出通道数：
    c2 <- layer1, c3 <- layer2, c4 <- layer3, c5 <- layer4
    兼容 BasicBlock 与 Bottleneck。
    """
    out = {"c2": None, "c3": None, "c4": None, "c5": None}
    for key, lname in zip(("c2", "c3", "c4", "c5"), ("layer1", "layer2", "layer3", "layer4")):
        if not hasattr(backbone, lname):
            continue
        layer = getattr(backbone, lname)
        if not hasattr(layer, "__getitem__") or len(layer) == 0:
            continue
        last_block = layer[-1]
        C = None
        if hasattr(last_block, "bn3") and hasattr(last_block.bn3, "num_features"):
            C = last_block.bn3.num_features  # Bottleneck
        elif hasattr(last_block, "bn2") and hasattr(last_block.bn2, "num_features"):
            C = last_block.bn2.num_features  # BasicBlock
        out[key] = C
    return out


def configure_heads_from_config(
    model,
    config: Dict,
    logger=None,
    head_list: Optional[List[str]] = None,
) -> List[str]:
    """
    - 批量添加 heads（支持 head_list 多类型或单一 head_type）
    - 从 config 读取特征来源（全局 feature_source 或逐 head 的 head_sources）
    - 若来源是空间层(c2/c3/c4/c5)，自动重建 head 以匹配通道数
    - 绑定来源：model.set_head_source(head_type, head_name, source)

    返回：已添加的 head 名称列表
    """
    log = getattr(logger, "info", print) if logger else print
    warn = getattr(logger, "warning", print) if logger else print

    add_heads = config.get("add_heads", {})
    if not isinstance(add_heads, dict) or not add_heads:
        warn("No 'add_heads' section found in config or it's empty.")
        return []

    head_types = head_list if isinstance(head_list, list) and head_list else [config.get("head_type", "linear")]
    valid_types = {"linear", "acil", "protonet"}

    # 通道映射：c2..c5 动态推断；pool/flat 使用 model.feature_dim
    stage_channels = _infer_resnet_stage_channels(model.backbone)
    in_dim_map = {
        "c2": stage_channels.get("c2"),
        "c3": stage_channels.get("c3"),
        "c4": stage_channels.get("c4"),
        "c5": stage_channels.get("c5"),
        "pool": model.feature_dim,
        "flat": model.feature_dim,
    }

    default_source = config.get("feature_source", "pool")
    head_sources = config.get("head_sources", {}) if isinstance(config.get("head_sources", {}), dict) else {}

    added_heads_names: List[str] = []
    # 用当前模型参数的设备作为 ACIL 初始化设备，避免硬编码 'cuda'
    try:
        current_device = str(next(model.parameters()).device)
    except StopIteration:
        current_device = "cpu"

    for head_type in head_types:
        if head_type not in valid_types:
            warn(f"Unsupported head_type '{head_type}', skip.")
            continue

        for head_name, num_classes in add_heads.items():
            log(f"Adding head: '{head_name}' with {num_classes} classes (type='{head_type}').")
            model.add_head(head_name, num_classes=num_classes, head_type=head_type)
            added_heads_names.append(head_name)

            # 决定来源：按 head 覆盖 > 全局默认；非法值回退到 pool
            source = head_sources.get(head_name, default_source)
            if source not in ("c2", "c3", "c4", "c5", "pool", "flat"):
                warn(f"Invalid feature source '{source}' for head '{head_name}', fallback to 'pool'.")
                source = "pool"

            # 若为空间层，则按通道数重建 head，避免 in_features 不匹配
            if source not in ("pool", "flat"):
                in_dim = in_dim_map.get(source, model.feature_dim)

                if head_type == "linear":
                    in_dim = in_dim_map.get("pool", model.feature_dim)
                    old = model.heads["linear"][head_name]
                    if getattr(old, "in_features", None) != in_dim:
                        log(f"Rebuilding linear head '{head_name}' for source '{source}' (in_features {getattr(old, 'in_features', None)} -> {in_dim})")
                        model.heads["linear"][head_name] = nn.Linear(in_dim, old.out_features)

                elif head_type == "acil":
                    in_dim = in_dim_map.get("pool", model.feature_dim)
                    log(f"Rebuilding ACIL head '{head_name}' for source '{source}' (feature_dim -> {in_dim})")
                    model.heads["acil"][head_name] = ACILDynamicClasses(
                        feature_dim=in_dim,
                        expansion_dim=getattr(model, "expansion_dim", in_dim) or in_dim,
                        num_classes=num_classes,
                        gamma=0.1,
                        device=current_device,
                    )

                elif head_type == "protonet":
                    proto_format = config.get('proto_format', 'default')
                    old = model.heads["protonet"][head_name]
                    old_in = getattr(old, "feature_dim", model.feature_dim)
                    if old_in != in_dim:
                        log(f"Rebuilding ProtoNet head '{head_name}' for source '{source}' (feature_dim {old_in} -> {in_dim})")
                        if proto_format == 'default':
                            model.heads["protonet"][head_name] = ProtoNetHead(
                                head_name=head_name, in_channels=in_dim, 
                                feature_dim=config.get('proto_dim', 256), num_classes=num_classes)
                        elif proto_format == 'iciclenet':
                            model.heads["protonet"][head_name] = IcicleNetHead(
                                head_name=head_name, in_channels=in_dim, 
                                feature_dim=config.get('proto_dim', 256), num_classes=num_classes)
                        elif proto_format == 'hiepronet':
                            model.heads["protonet"][head_name] = HieProNetHead(
                                head_name=head_name, in_channels=in_dim, 
                                feature_dim=config.get('proto_dim', 256), num_classes=num_classes)
                        

            # 绑定来源（pool/flat 也显式绑定，便于排查）
            model.set_head_source(head_type, head_name, source)
            log(f"Head '{head_name}' uses feature source '{source}'")

    log("Note: 'pool' and 'flat' are equivalent (both are global avg pooled features).")
    return added_heads_names
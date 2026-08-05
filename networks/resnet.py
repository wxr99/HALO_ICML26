import torch
import torch.nn as nn
import torchvision.models as models
from typing import Dict, List, Optional, Tuple, Union, Any, Set
import os
import tarfile # Necessary for handling .tar archives
import io      # Necessary for handling in-memory file streams
from .analytic_classifier import ACILDynamicClasses
from .protonet import ProtoNetHead
import torch.nn.functional as F  # [ADDED] 用于自适应全局平均池化等
from .iciclenet import IcicleNetHead  # [ADDED] 如果需要 IcicleNetHead 支持
from .hiepronet import HieProNetHead  # [ADDED] 如果需要 HieProNetHead 支持

class ResNetMultiHeadHierarchical(nn.Module):
    """
    A ResNet model with a potentially pre-trained backbone and multiple
    classification heads, supporting loading backbone weights from a custom
    .pth or .pth.tar file.
    Compatible with older torchvision versions (using pretrained=True).

    Args:
        backbone_name (str): Name of the ResNet variant (e.g., 'resnet18', 'resnet50').
        custom_weight_path (Optional[str]): Path to a local .pth or .pth.tar file
                                            containing backbone weights. Takes
                                            precedence over 'pretrained'.
        pretrained (bool): If True and custom_weight_path is None, load torchvision's
                           default pre-trained weights for the backbone.
        freeze_backbone (bool): If True, freezes the parameters of the backbone initially.
        tar_member_name (Optional[str]): If using a .tar file, specifies the exact
                                         name of the weight file inside the archive.
                                         If None, attempts to find a suitable file.
    """
    def __init__(self,
                 backbone_name: str = 'resnet50',
                 custom_weight_path: Optional[str] = None,
                 pretrained: bool = True,
                 freeze_backbone: bool = True,
                 tar_member_name: Optional[str] = None,
                 return_features: Optional[bool] = False,
                 expansion_dim: Optional[int] = 4096,
                 proto_dim: Optional[int] = 128,
                 add_heads: Optional[dict] = None,
                 head_list: Optional[List[str]] = None):
        super().__init__()
        self.backbone_name = backbone_name
        self.feature_dim = None
        self.temp_linear_list = None
        self.temp_acil_list = None
        self.return_features = return_features
        self.expansion_dim = expansion_dim
        self.proto_dim = proto_dim
        self.add_heads = add_heads
        self.head_list = head_list
        self.weight_linear = nn.Parameter(torch.ones(len(add_heads)))
        self.weight_acil = nn.Parameter(torch.ones(len(add_heads)))

        # --- 1. Initialize Backbone Architecture ---
        try:
            model_loader = getattr(models, backbone_name)
            load_tv_pretrained = pretrained and (custom_weight_path is None)
            # Initialize with pretrained=False if loading custom weights
            self.backbone = model_loader(pretrained=load_tv_pretrained)

            if load_tv_pretrained:
                print(f"Initialized backbone {backbone_name} with torchvision pretrained=True")
            elif custom_weight_path:
                 print(f"Initialized {backbone_name} architecture (pretrained=False). Weights will be loaded from: {custom_weight_path}")
            else:
                 print(f"Initialized {backbone_name} architecture (pretrained=False) without pretrained weights.")

        except AttributeError:
            raise ValueError(f"Unsupported backbone: {backbone_name}")
        except Exception as e:
             print(f"Error initializing backbone {backbone_name}: {e}")
             raise e

        # --- 2. Get Feature Dimension & Remove Original Head ---
        self._get_feature_dim_and_remove_head()

        # --- 3. Load Weights from Custom File (if provided) ---
        if custom_weight_path:
            # This step handles both .pth and .pth.tar using tarfile logic internally
            self._load_weights_from_path(custom_weight_path, tar_member_name)

        # --- 4. Freeze Backbone (Optional) ---
        if freeze_backbone:
            self.freeze_backbone()
        else:
            self.unfreeze_backbone()

        # --- 5. Heads Storage ---
        # self.heads = nn.ModuleDict()
        # self.heads = {}  # Dict[str, nn.ModuleDict()]
        self.heads = nn.ModuleDict({
            "linear": nn.ModuleDict(),  # 存储所有线性头部
            "acil": nn.ModuleDict(),     # 存储所有 ACIL 头部
            "protonet":  nn.ModuleDict() # 存储所有 ProtoNet 头部
        })

        # 为每个 head 记录其特征来源层，默认 'pool'
        self.heads_source: Dict[str, Dict[str, str]] = {
            "linear": {},
            "acil": {},
            "protonet": {}
        }

        # 多层特征缓存与 hook 注册标记
        self._feature_cache: Dict[str, torch.Tensor] = {}
        self._hooks_registered: bool = False

    def _get_feature_dim_and_remove_head(self):
        """Helper to find feature dim and replace final layer with Identity."""
        # (No changes needed here)
        if hasattr(self.backbone, 'fc') and isinstance(self.backbone.fc, nn.Linear):
             self.feature_dim = self.backbone.fc.in_features
             self.backbone.fc = nn.Identity()
        elif hasattr(self.backbone, 'classifier'):
             # Handle different classifier structures (Linear, Sequential)
             if isinstance(self.backbone.classifier, nn.Linear):
                 self.feature_dim = self.backbone.classifier.in_features
                 self.backbone.classifier = nn.Identity()
             elif isinstance(self.backbone.classifier, nn.Sequential):
                 for layer in reversed(self.backbone.classifier):
                     if isinstance(layer, nn.Linear):
                         self.feature_dim = layer.in_features
                         break
                 else: raise TypeError("Could not find Linear layer in Sequential classifier.")
                 self.backbone.classifier = nn.Identity()
             else: raise TypeError(f"Unsupported classifier type: {type(self.backbone.classifier)}")
        else:
             # Fallback attempt (less reliable)
             try:
                 last_module = list(self.backbone.children())[-1]
                 if isinstance(last_module, nn.Linear):
                     self.feature_dim = last_module.in_features
                     *all_but_last, _ = self.backbone.children()
                     self.backbone = nn.Sequential(*all_but_last, nn.Identity())
                     print("Warning: Inferred feature dimension by replacing the last module.")
                 else: raise TypeError("Could not automatically determine feature dimension.")
             except (IndexError, TypeError): raise TypeError(f"Could not determine feature dimension for {self.backbone_name}.")

        if self.feature_dim is None: raise RuntimeError("Failed to determine backbone feature dimension.")
        print(f"Backbone feature dimension: {self.feature_dim}. Original classifier removed.")


    def _load_weights_from_path(self, weight_path: str, tar_member_name: Optional[str] = None):
        """Loads backbone weights from a .pth or .pth.tar file."""
        if not os.path.exists(weight_path):
            raise FileNotFoundError(f"Weight file not found: {weight_path}")

        print(f"Loading backbone weights from: {weight_path}")
        checkpoint = None # Initialize checkpoint variable
        try:
            # --- Handle .tar archive ---
            # Check if it's likely a tar file based on common extensions
            if weight_path.endswith((".tar", ".tar.gz", ".tgz")):
                print("Detected .tar archive. Attempting to extract weights...")
                read_mode = 'r:gz' if weight_path.endswith(".gz") else 'r'
                with tarfile.open(weight_path, read_mode) as tar:
                    member_to_load = None
                    if tar_member_name: # User specified the exact member name
                        try:
                            member_info = tar.getmember(tar_member_name)
                            member_to_load = member_info
                            print(f"Using specified tar member: {tar_member_name}")
                        except KeyError:
                            raise FileNotFoundError(f"Specified member '{tar_member_name}' not found in {weight_path}. "
                                                    f"Available members: {[m.name for m in tar.getmembers()]}")
                    else: # Auto-detect member name
                        potential_members = [m for m in tar.getmembers() if m.isfile() and m.name.endswith(('.pth', '.pt', '.ckpt'))]
                        if not potential_members: # Fallback: find largest file if no obvious extension
                            potential_members = sorted([m for m in tar.getmembers() if m.isfile()], key=lambda m: m.size, reverse=True)
                        if not potential_members:
                             raise FileNotFoundError(f"Could not find a suitable weight file inside {weight_path}. "
                                                     f"Available members: {[m.name for m in tar.getmembers()]}")
                        member_to_load = potential_members[0]
                        print(f"Auto-detected tar member to load: {member_to_load.name} (size: {member_to_load.size} bytes)")
                        if len(potential_members) > 1:
                             print(f"  (Found other potential members: {[m.name for m in potential_members[1:]]}. Using the first one.)")

                    # Extract the selected member file into memory
                    extracted_file = tar.extractfile(member_to_load)
                    if extracted_file is None:
                         raise IOError(f"Failed to extract member '{member_to_load.name}' from tar file.")

                    # Load checkpoint from the extracted file object (in memory)
                    print(f"Loading weights from extracted member '{member_to_load.name}'...")
                    # Use io.BytesIO to wrap the file-like object for torch.load
                    checkpoint = torch.load(io.BytesIO(extracted_file.read()), map_location='cpu')
                    extracted_file.close()

            # --- Handle direct .pth or other non-tar file ---
            else:
                print("Loading weights directly from file (assuming not a tar archive)...")
                checkpoint = torch.load(weight_path, map_location='cpu')

            # --- Process the loaded checkpoint to get state_dict ---
            if checkpoint is None:
                 raise ValueError("Checkpoint was not loaded.") # Should not happen if file exists and is valid
            state_dict = self._extract_state_dict_from_checkpoint(checkpoint, weight_path)

            # --- Clean Keys (Remove prefixes, filter classifier) ---
            cleaned_state_dict = self._clean_state_dict_keys(state_dict)

            # --- Load the cleaned state_dict into the backbone ---
            print(f"Attempting to load {len(cleaned_state_dict)} keys into the backbone...")
            load_result = self.backbone.load_state_dict(cleaned_state_dict, strict=False)

            # --- Report loading results ---
            if not load_result.missing_keys and not load_result.unexpected_keys:
                print("Successfully loaded all provided backbone weights.")
            else:
                print("Weight loading completed with potential mismatches:")
                if load_result.missing_keys:
                    missing_non_bn_stats = [k for k in load_result.missing_keys if not k.endswith(('.running_mean', '.running_var', '.num_batches_tracked'))]
                    if missing_non_bn_stats:
                         print(f"  Missing keys in model (expected but not found in file): {missing_non_bn_stats}")
                if load_result.unexpected_keys:
                    print(f"  Unexpected keys in file (found but not in model): {load_result.unexpected_keys}")
                    print("    (This is often expected if the weight file included the original classifier head).")

        except tarfile.ReadError:
             print(f"Error: Failed to read '{weight_path}' as a tar file. Please ensure it's a valid archive.")
             raise
        except Exception as e:
            print(f"Error loading weights from {weight_path}: {e}")
            raise

    def _extract_state_dict_from_checkpoint(self, checkpoint: Any, file_path: str) -> Dict[str, torch.Tensor]:
        """Extracts the state_dict from various possible checkpoint formats."""
        # (No changes needed here)
        state_dict = None
        if isinstance(checkpoint, dict):
            if 'state_dict' in checkpoint: state_dict = checkpoint['state_dict']
            elif 'model_state_dict' in checkpoint: state_dict = checkpoint['model_state_dict']
            elif 'model' in checkpoint:
                 if hasattr(checkpoint['model'], 'state_dict'): state_dict = checkpoint['model'].state_dict()
                 elif isinstance(checkpoint['model'], dict): state_dict = checkpoint['model']
                 else: raise KeyError(f"Found 'model' key, but couldn't extract state_dict.")
            elif 'backbone' in checkpoint: state_dict = checkpoint['backbone']
            else:
                if any(k.startswith(('conv', 'bn', 'layer', 'fc', 'downsample')) for k in checkpoint.keys()): state_dict = checkpoint
                else: raise KeyError(f"Could not find state_dict in checkpoint dict. Keys: {list(checkpoint.keys())}")
        elif isinstance(checkpoint, nn.Module):
             print("Warning: Loading weights from a saved nn.Module object.")
             state_dict = checkpoint.state_dict()
        elif hasattr(checkpoint, 'keys'): state_dict = checkpoint
        else: raise TypeError(f"Unsupported data type loaded from {file_path}: {type(checkpoint)}")

        if state_dict is None or not isinstance(state_dict, dict):
             raise TypeError(f"Failed to extract a valid state_dict (dictionary) from {file_path}")
        return state_dict


    def _clean_state_dict_keys(self, state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Removes common prefixes and filters out classifier keys."""
        # (No changes needed here)
        cleaned_state_dict = {}
        if not state_dict: return cleaned_state_dict

        sample_keys = list(state_dict.keys())[:5]
        prefix_to_remove = ""
        if sample_keys:
             if all(k.startswith('module.') for k in sample_keys): prefix_to_remove = 'module.'
             elif all(k.startswith('backbone.') for k in sample_keys): prefix_to_remove = 'backbone.'
             elif all(k.startswith('model.') for k in sample_keys): prefix_to_remove = 'model.'
        if prefix_to_remove: print(f"Detected and removing '{prefix_to_remove}' prefix from keys.")

        for k, v in state_dict.items():
            new_key = k[len(prefix_to_remove):] if prefix_to_remove and k.startswith(prefix_to_remove) else k
            if not new_key.startswith(('fc.', 'classifier.')):
                 cleaned_state_dict[new_key] = v

        if not cleaned_state_dict and state_dict: print("Warning: Key cleaning resulted in an empty state_dict.")
        elif not cleaned_state_dict: print("Warning: Loaded state_dict was empty.")
        return cleaned_state_dict

    # --- Other methods remain the same ---
    # (freeze_backbone, unfreeze_backbone, add_head, freeze_all_heads,
    #  unfreeze_head, freeze_head, forward, get_trainable_parameters)
    def freeze_backbone(self):
        """Freezes all parameters in the backbone."""
        print("Freezing backbone parameters.")
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.eval()

    def unfreeze_backbone(self):
        """Unfreezes all parameters in the backbone."""
        print("Unfreezing backbone parameters.")
        for param in self.backbone.parameters():
            param.requires_grad = True
        self.backbone.train()

    def add_head(self, head_name: str, num_classes: int, head_type: str = "linear", freeze_existing: bool = False):
        """Adds a new classification head."""
        
        if head_name in self.heads: print(f"Warning: Head '{head_name}' already exists. Overwriting.")
        if freeze_existing: self.freeze_all_heads()
        # print(f"Adding head '{head_name}' with {num_classes} classes.")
        if head_type == "linear":
            new_head = nn.Linear(self.feature_dim, num_classes)
        elif head_type == "acil":
            new_head = ACILDynamicClasses(
                feature_dim=self.feature_dim,
                expansion_dim=self.expansion_dim,
                num_classes=num_classes,
                gamma=0.1,
                device="cuda"
            )
        elif head_type == "protonet":
            # 构建 ProtoNetHead，并根据 num_classes 自动生成占位类别（可后续再精调/覆盖
            new_head = ProtoNetHead(
                head_name=head_name, in_channels=self.feature_dim, 
                feature_dim=self.proto_dim, num_classes=num_classes
            )
            # new_head.autofill_classes(num_classes=num_classes)
        else:
            raise ValueError(f"Unsupported head type: {head_type}. Must be 'linear' or 'acil'.")
        self.heads[head_type][head_name] = new_head
        self.heads_source[head_type][head_name] = "pool"  # 默认从 pool 特征读取

    def add_linear_head(self, head_name: str, num_classes: int, freeze_existing: bool = False):
        """Adds a new classification head."""
        if head_name in self.heads["linear"]: print(f"Warning: Head '{head_name}' already exists. Overwriting.")
        if freeze_existing: self.freeze_all_heads()
        # print(f"Adding head '{head_name}' with {num_classes} classes.")
        new_head = nn.Linear(self.feature_dim, num_classes)
        self.heads["linear"][head_name] = new_head
    
    def add_analytic_head(self, head_name: str, num_classes: int, freeze_existing: bool = False):
        """Adds a new classification head."""
        if head_name in self.heads["acil"]: print(f"Warning: Head '{head_name}' already exists. Overwriting.")
        if freeze_existing: self.freeze_all_heads()
        # print(f"Adding head '{head_name}' with {num_classes} classes.")
        new_head = ACILDynamicClasses(feature_dim=self.feature_dim, expansion_dim=self.expansion_dim,
                num_classes=num_classes,gamma=0.1,device="cuda")
        self.heads["acil"][head_name] = new_head
        self.heads_source["acil"][head_name] = "pool"

    def freeze_all_heads(self):
        """Freezes parameters of all existing heads."""
        print("Freezing all existing heads.")
        # for head in self.heads.values():
        #     for param in head.parameters(): param.requires_grad = False
        for family in self.heads.values():               # [CHANGED]
            for module in family.values():
                for p in module.parameters():
                    p.requires_grad = False

    def unfreeze_head(self, head_name: str):
        """Unfreezes parameters of a specific head."""
        # [CHANGED] 在所有 head 类型中查找该名称
        found = False
        for family in self.heads.values():
            if head_name in family:
                print(f"Unfreezing head '{head_name}'.")
                for p in family[head_name].parameters():
                    p.requires_grad = True
                found = True
        if not found:
            print(f"Warning: Head '{head_name}' not found for unfreezing.")


    def freeze_head(self, head_name: str):
        """Freezes parameters of a specific head."""
        # [CHANGED] 在所有 head 类型中查找该名称
        found = False
        for family in self.heads.values():
            if head_name in family:
                print(f"Freezing head '{head_name}'.")
                for p in family[head_name].parameters():
                    p.requires_grad = False
                found = True
        if not found:
            print(f"Warning: Head '{head_name}' not found for freezing.")

    def set_head_source(self, head_type: str, head_name: str, from_feature: str):
        """
        为已存在的 head 指定特征来源：'c2'|'c3'|'c4'|'c5'|'pool'|'flat'
        注意：若 head_type 为 'protonet' 且绑定到 c2/c3/c4/c5，需确保该 ProtoNetHead 的特征维
            与对应层通道数一致；否则请重新以匹配的 feature_dim 创建该 head。
        """
        if head_type not in self.heads or head_name not in self.heads[head_type]:
            raise ValueError(f"Head '{head_name}' not found in type '{head_type}'.")
        if from_feature not in ("c2", "c3", "c4", "c5", "pool", "flat"):
            raise ValueError(f"from_feature must be one of 'c2','c3','c4','c5','pool','flat'")
        self.heads_source[head_type][head_name] = from_feature  # 仅记录来源，不改原有逻辑    

    def _register_feature_hooks(self):  # [ADDED]
        """注册 forward hook，抓取 layer1/2/3/4 与 avgpool 的输出。"""
        if self._hooks_registered:
            return

        self._feature_cache = {}

        def save_output(name):
            def fn(_, __, out):
                self._feature_cache[name] = out
            return fn

        for name in ("layer1", "layer2", "layer3", "layer4", "avgpool"):
            if hasattr(self.backbone, name):
                module = getattr(self.backbone, name)
                module.register_forward_hook(save_output(name))

        self._hooks_registered = True


    def _collect_features_once(self, x: torch.Tensor, needed: set) -> Dict[str, torch.Tensor]:  # [ADDED]
        """
        运行一次 self.backbone(x)，通过 hooks 收集需要的多层特征。
        返回可能包含：'c2','c3','c4','c5','pool','flat'
        """
        assert needed, "needed set must be non-empty"
        self._register_feature_hooks()
        self._feature_cache = {}

        # 与你原 forward 一致的“冻结时临时 eval”处理
        is_backbone_training = self.backbone.training
        is_backbone_frozen = not any(p.requires_grad for p in self.backbone.parameters())
        if is_backbone_frozen:
            self.backbone.eval()
        _ = self.backbone(x)  # 单次前向，hooks 会填充 _feature_cache
        if is_backbone_frozen:
            self.backbone.train(is_backbone_training)

        feats = {}
        if "layer1" in self._feature_cache and "c2" in needed:
            feats["c2"] = self._feature_cache["layer1"]
        if "layer2" in self._feature_cache and "c3" in needed:
            feats["c3"] = self._feature_cache["layer2"]
        if "layer3" in self._feature_cache and "c4" in needed:
            feats["c4"] = self._feature_cache["layer3"]
        if "layer4" in self._feature_cache and "c5" in needed:
            feats["c5"] = self._feature_cache["layer4"]
        if "avgpool" in self._feature_cache and ("pool" in needed or "flat" in needed):
            pooled4d = self._feature_cache["avgpool"]  # [B,2048,1,1]
            feats["pool"] = torch.flatten(pooled4d, 1) # [B,2048]
            feats["flat"] = feats["pool"]

        return feats   

    def forward(
        self,
        x: torch.Tensor,
        head_names: List[str],
        return_features: bool = False,
        head_types: Optional[Union[str, List[str]]] = "linear",
        features_to_return: Optional[Union[bool, str, List[str]]] = None,
        return_protofeatures: bool = False,

        prior_dict: Optional[Dict[str, torch.Tensor]] = None,  # Dict[head_name] -> [B,C] 概率
        prior_power: float = 1.0,
        prior_topk: Optional[int] = None,
        prior_logit_bias: float = 0.0,
        prior_temp: float = 1.0,
    ):
        if not head_names:
            raise ValueError("head_names list cannot be empty.")

        # 1) 统一 head_types
        head_types_list = [head_types] if isinstance(head_types, str) else list(head_types)

        # 2) 需要哪些基础来源（来自 head 绑定）
        needed_sources: Set[str] = set()
        for ht in head_types_list:
            if ht not in self.heads:
                raise ValueError(f"Unknown head_type '{ht}'.")
            for name in head_names:
                if name not in self.heads[ht]:
                    raise ValueError(f"Head '{name}' not found in type '{ht}'. Available: {list(self.heads[ht].keys())}")
                needed_sources.add(self.heads_source.get(ht, {}).get(name, "pool"))

        # 3) 解析要返回的特征
        def norm_req(req) -> Set[str]:
            if req is None:
                return {"pool"} if return_features else set()
            if isinstance(req, bool):
                return {"pool"} if req else set()
                # return {"pool"}
            if isinstance(req, str):
                return {req}
            if isinstance(req, list):
                return set(req)
            raise ValueError("features_to_return must be None|bool|str|List[str].")
        requested = norm_req(features_to_return)

        # 这些请求需要哪些基础来源
        for k in list(requested):
            if k in ("pool", "flat"):
                needed_sources.add("pool")
            elif k in ("c2", "c3", "c4", "c5"):
                needed_sources.add(k)
            elif k.startswith("gap:"):
                _, s = k.split(":", 1)
                if s not in ("c2", "c3", "c4", "c5"):
                    raise ValueError(f"Unsupported gap source '{s}'.")
                needed_sources.add(s)
            else:
                raise ValueError(f"Unsupported feature key '{k}'.")

        # 4) 一次前向收集特征
        feats: Dict[str, torch.Tensor] = self._collect_features_once(x, needed_sources)

        # 5) 简单工具函数
        def pool_vec(fd: Dict[str, torch.Tensor]) -> torch.Tensor:
            v = fd.get("pool", None)
            if v is not None:
                return v
            for src in ("c5", "c4", "c3", "c2"):
                fmap = fd.get(src, None)
                if fmap is not None and fmap.dim() == 4:
                    return F.adaptive_avg_pool2d(fmap, 1).flatten(1)
            raise RuntimeError("Cannot derive pooled feature (no pool/c2-5 available).")

        def vec_for(src: str, fd: Dict[str, torch.Tensor]) -> torch.Tensor:
            if src in ("pool", "flat"):
                return pool_vec(fd)
            fmap = fd.get(src, None)
            if fmap is None:
                raise RuntimeError(f"Missing feature source '{src}'.")
            return F.adaptive_avg_pool2d(fmap, 1).flatten(1)

        # 简单温度函数（仅对 prior 使用，保持它是概率）
        def _prep_prior_for(head_name: str, C_expect: int) -> Optional[torch.Tensor]:
            if prior_dict is None:
                return None
            p = prior_dict.get(head_name, None)
            if p is None:
                return None             
            if p.dim() != 2 or p.size(1) != C_expect:
                raise ValueError(f"prior_dict['{head_name}'] shape mismatch: got {tuple(p.shape)}, expect [B,{C_expect}]")
            if prior_temp != 1.0:
                # p 是概率；用 log-domain temp 调整再归一化，避免放大数值误差
                p = (p.clamp_min(1e-12).log() / float(prior_temp)).softmax(dim=-1)
            else:
                # 轻度归一化，防止数值偏差
                p = p / p.sum(dim=-1, keepdim=True).clamp_min(1e-12)
            return p

        # 6) 计算 logits（按类型最小变换）
        if isinstance(head_types, str):
            outputs: Dict[str, torch.Tensor] = {}
            p_features: Dict[str, torch.Tensor] = {}
            for name in head_names:
                src = self.heads_source.get(head_types, {}).get(name, "pool")
                if head_types == "protonet":
                    fmap = feats.get(src, None)
                    if fmap is None:
                        # 若用户把 protonet 绑定成了 pool/flat，就把向量变成 1x1 的“空间”特征兜底
                        v = pool_vec(feats) if src in ("pool", "flat") else None
                        if v is None:
                            raise RuntimeError(f"ProtoNet head '{name}' needs spatial feature '{src}'.")
                        fmap = v.unsqueeze(-1).unsqueeze(-1)
                    # 仅 protonet 支持 prior 门控
                    head_mod = self.heads["protonet"][name]
                    if isinstance(head_mod, HieProNetHead):  # 只有 HieProNetHead 才支持 class_prior
                        p_prior = _prep_prior_for(name, C_expect=head_mod.C)
                        if return_protofeatures:
                            outputs[name], p_features[name] = head_mod(
                                fmap,
                                class_prior=p_prior,
                                mask_topk=prior_topk if (prior_topk is not None and prior_topk > 0) else None,
                                prior_power=float(prior_power),
                                prior_logit_bias=float(prior_logit_bias),
                                return_protofeatures=True
                            )
                        else:
                            outputs[name] = head_mod(
                                fmap,
                                class_prior=p_prior,
                                mask_topk=prior_topk if (prior_topk is not None and prior_topk > 0) else None,
                                prior_power=float(prior_power),
                                prior_logit_bias=float(prior_logit_bias),
                                return_protofeatures=False
                            )
                    else:
                        if return_protofeatures:
                            outputs[name], p_features[name] = self.heads["protonet"][name](fmap, return_protofeatures=return_protofeatures)
                        else:
                            outputs[name] = self.heads["protonet"][name](fmap, return_protofeatures=return_protofeatures)
                else:
                    vec = vec_for(src, feats)
                    outputs[name] = self.heads[head_types][name](vec)
        else:
            outputs: Dict[str, Dict[str, torch.Tensor]] = {ht: {} for ht in head_types_list}
            p_features: Dict[str, torch.Tensor] = {}
            for ht in head_types_list:
                for name in head_names:
                    src = self.heads_source.get(ht, {}).get(name, "pool")
                    if ht == "protonet":
                        fmap = feats.get(src, None)
                        if fmap is None:
                            v = pool_vec(feats) if src in ("pool", "flat") else None
                            if v is None:
                                raise RuntimeError(f"ProtoNet head '{name}' needs spatial feature '{src}'.")
                            fmap = v.unsqueeze(-1).unsqueeze(-1)
                        if return_protofeatures:
                            outputs[ht][name], p_features[name] = self.heads["protonet"][name](fmap, return_protofeatures=return_protofeatures)
                        else:
                            outputs[ht][name] = self.heads["protonet"][name](fmap, return_protofeatures=return_protofeatures)
                    elif ht == "acil":
                        vec = vec_for('pool', feats)
                        outputs[ht][name] = self.heads[ht][name](vec)
                    elif ht == "linear":
                        vec = vec_for('pool', feats)
                        outputs[ht][name] = self.heads[ht][name](vec)
                    else:
                        vec = vec_for(src, feats)
                        outputs[ht][name] = self.heads[ht][name](vec)

        # 7) 组织返回的特征（可选）
        if requested:
            out_feats: Dict[str, torch.Tensor] = {}
            for k in requested:
                if k in ("pool", "flat"):
                    out_feats["pool"] = pool_vec(feats)
                elif k in ("c2", "c3", "c4", "c5"):
                    fmap = feats.get(k, None)
                    if fmap is None:
                        raise RuntimeError(f"Requested '{k}' but it was not collected.")
                    out_feats[k] = fmap
                else:  # gap:cX
                    _, s = k.split(":", 1)
                    fmap = feats.get(s, None)
                    if fmap is None:
                        raise RuntimeError(f"Requested '{k}' but base '{s}' was not collected.")
                    out_feats[k] = F.adaptive_avg_pool2d(fmap, 1).flatten(1)
            if return_protofeatures:
                return {"logits": outputs, "features": out_feats, "proto_features": p_features}
            else:
                return {"logits": outputs, "features": out_feats}
        if return_protofeatures:
            return {"logits": outputs, "proto_features": p_features}
        else:
            return outputs


    def get_trainable_parameters(self) -> List[Dict[str, Any]]:
        """Gets parameters that require gradients, grouped for optimizer."""
        params_to_optimize = []
        backbone_params = list(filter(lambda p: p.requires_grad, self.backbone.parameters()))
        if backbone_params:
            params_to_optimize.append({'params': backbone_params, 'lr_scale_factor': 0.1})
            print("Including trainable backbone parameters in optimizer groups.")

        for head_name, head in self.heads.items():
            head_params = list(filter(lambda p: p.requires_grad, head.parameters()))
            if head_params:
                params_to_optimize.append({'params': head_params})
                print(f"Including trainable parameters from head '{head_name}' in optimizer groups.")

        if not params_to_optimize: print("Warning: No trainable parameters found!")
        return params_to_optimize

    def train_acil_head(self, X_train: torch.Tensor, 
                        Y_train: torch.Tensor, mode: str = "base",
                        feature_key: str = "pool"):
        """
        Trains specified ACIL heads (base or incremental learning).

        Args:
            X_train (torch.Tensor): Input feature tensor.
            Y_train (Dict[str, torch.Tensor]): Dictionary mapping head names to target labels.
            mode (str): Training mode ('base' or 'incremental').
        """
        if mode not in {"base", "incremental"}:
            raise ValueError("mode must be 'base' or 'incremental'.")

       # 规范化 X_train -> Tensor [B, D]
        if isinstance(X_train, dict):
            # 支持 forward 的两种可能结构：{"features": {...}} 或直接 {...}
            feats_dict = X_train.get("features", X_train)
            if feature_key not in feats_dict:
                raise ValueError(f"X_train is a dict but missing features['{feature_key}']. "
                                f"Keys available: {list(feats_dict.keys())}")
            X_train = feats_dict[feature_key]

        if not isinstance(X_train, torch.Tensor):
            raise TypeError("X_train must be a Tensor or a dict containing the chosen feature tensor.")

       
        # Ensure Y_train matches the number of heads
        num_heads = len(self.heads["acil"])
        if Y_train.shape[0] != num_heads:
            raise ValueError(f"Y_train must have shape (num_heads, batch_size, num_classes), "
                            f"but got {Y_train.shape} with {num_heads} heads.")

        # Iterate over heads and corresponding labels
        for idx, (head_name, head_module) in enumerate(self.heads["acil"].items()):
            head_labels = Y_train[idx]  # Get labels for the current head

            # Only train ACIL heads
            if isinstance(head_module, ACILDynamicClasses):
                # print(f"Training ACIL head '{head_name}' in {mode} mode...")
                if mode == "base":
                    head_module.base_training(X_train, head_labels)
                elif mode == "incremental":
                    head_module.incremental_learning(X_train, head_labels)
            else:
                print(f"Skipping training for non-ACIL head '{head_name}' (type: {type(head_module)}).")

def clone_resnet_snapshot(model: ResNetMultiHeadHierarchical) -> ResNetMultiHeadHierarchical:
    """
    生成一个“新的”同结构模型，并拷贝当前权重。
    返回的模型与原模型参数脱钩（不同对象），默认冻结并设为 eval。
    注意：需要能复现原模型的各个 head（名字、类型、尺寸）。
    """
    # 1) 先根据原模型的构造参数，新建骨干
    model_old = ResNetMultiHeadHierarchical(
        backbone_name=model.backbone_name,
        custom_weight_path=None,
        pretrained=False,
        freeze_backbone=True,             # 先冻结，后面还会统一冻结
        tar_member_name=None,
        return_features=model.return_features,
        expansion_dim=model.expansion_dim,
        proto_dim=model.proto_dim,
        head_list=model.head_list,
        add_heads=model.add_heads
    )

    # 2) 复现 heads 结构（按名字与类型逐个创建）
    # 你的 heads 是一个嵌套的 ModuleDict：{"linear":{}, "acil":{}, "protonet":{}}
    for ht, family in model.heads.items():
        for name, mod in family.items():
            if isinstance(mod, nn.Linear):
                model_old.add_head(name, num_classes=mod.out_features, head_type="linear")
                model_old.set_head_source("linear", name, model.heads_source["linear"].get(name, "pool"))
            elif isinstance(mod, ACILDynamicClasses):
                model_old.add_head(name, num_classes=mod.num_classes, head_type="acil")
                model_old.set_head_source("acil", name, model.heads_source["acil"].get(name, "pool"))
            elif isinstance(mod, ProtoNetHead):
                # 依据你项目里 ProtoNetHead 的构造参数补齐
                model_old.heads["protonet"][name] = ProtoNetHead(
                    head_name=name,
                    in_channels=mod.in_channels,
                    feature_dim=mod.feature_dim,
                    num_classes=mod.num_classes,
                )
                model_old.set_head_source("protonet", name, model.heads_source["protonet"].get(name, "pool"))
            elif isinstance(mod, IcicleNetHead):
                # 如果你把 IcicleNetHead 也放在某个 family 下（例如 "protonet" 或自定义 "icicle"）
                # 这里直接用其可得的形状参数重建
                model_old.heads["protonet"][name] = IcicleNetHead(
                    head_name=mod.head_name,
                    in_channels=mod.add_on.conv1.in_channels,
                    feature_dim=mod.D,
                    num_classes=mod.C,
                    prototypes_per_class=mod.M,
                    mid_channels=mod.add_on.conv1.out_channels,
                    init_scale=0.02,
                    ppnet_eps=mod.eps,
                )
                model_old.set_head_source("protonet", name, model.heads_source["protonet"].get(name, "pool"))
            elif isinstance(mod, HieProNetHead):
                # 如果你把 HieProNetHead 也放在某个 family 下（例如 "protonet" 或自定义 "icicle"）
                # 这里直接用其可得的形状参数重建
                model_old.heads["protonet"][name] = HieProNetHead(
                    head_name=mod.head_name,
                    in_channels=mod.add_on.conv1.in_channels,
                    feature_dim=mod.D,
                    num_classes=mod.C,
                    prototypes_per_class=mod.M,
                    mid_channels=mod.add_on.conv1.out_channels,
                    init_scale=0.02,
                    ppnet_eps=mod.eps,
                )
                model_old.set_head_source("protonet", name, model.heads_source["protonet"].get(name, "pool"))
            else:
                raise TypeError(f"Unknown head type for '{name}': {type(mod)}")

    # 3) 拷贝参数（包含 buffers）
    model_old.load_state_dict(model.state_dict(), strict=False)

    # 4) 冻结并 eval（teacher 不需要梯度）
    for p in model_old.parameters():
        p.requires_grad_(False)
    model_old.eval()

    return model_old

# --- Example Usage ---
if __name__ == '__main__':

    # Define the path to your pretrained weights file
    # IMPORTANT: Ensure this path exists and the file is valid!
    #            Also ensure the backbone_name ('resnet50') matches the weights.
    pretrained_path = './pretrained/dino_resnet50_pretrain.pth' # Your specified path

    print(f"\nAttempting to initialize model with custom weights from: {pretrained_path}")

    # Check if the path exists before attempting to load
    if not os.path.exists(pretrained_path):
         print(f"Error: Pretrained weight file not found at '{pretrained_path}'")
         print("Please ensure the path is correct and the file exists.")
         # Exit or handle the error appropriately in a real script
         exit()

    try:
        # Initialize model using the custom .pth.tar file
        model = ResNetMultiHeadHierarchical(
            backbone_name='resnet50',         # Match the architecture in the file
            custom_weight_path=pretrained_path, # Use your path
            pretrained=False,                 # Set to False when using custom path
            freeze_backbone=True,             # Example: Start with frozen backbone
            # tar_member_name=None            # Let it auto-detect member inside tar, or specify if needed
        )

        # Add heads as needed for your hierarchical task
        model.add_head('head_level_1', num_classes=50)
        model.add_head('head_level_2', num_classes=10)
        # Add more heads...

        # Test forward pass (requires dummy input)
        print("\nAvailable heads:", list(model.heads.keys()))
        try:
            # Create dummy input matching expected ResNet input size
            dummy_input = torch.randn(2, 3, 32, 32) # Batch size 2
            requested_heads = ['head_level_1', 'head_level_2'] # Heads to get output from
            outputs = model(dummy_input, head_names=requested_heads)

            print(f"\nOutputs from model with custom weights ({pretrained_path}):")
            for name, out in outputs.items():
                print(f"  - Head '{name}' output shape: {out.shape}")

            # Example: Prepare for training head_level_1
            model.unfreeze_head('head_level_1')
            # model.freeze_head('head_level_2') # Ensure other heads are frozen if needed
            # model.freeze_backbone() # Ensure backbone is frozen

            trainable_params = model.get_trainable_parameters()
            if trainable_params:
                 # Setup optimizer (example)
                 # optimizer = torch.optim.Adam([p for group in trainable_params for p in group['params']], lr=1e-3)
                 print("\nReady to configure optimizer for trainable parameters.")
            else:
                 print("\nNo trainable parameters found to configure optimizer.")


        except Exception as e:
            print(f"\nAn error occurred during forward pass or setup: {e}")
            print("This might happen if the model initialization failed or input dimensions are wrong.")

    except (FileNotFoundError, tarfile.ReadError, KeyError, TypeError, RuntimeError, ValueError) as e:
         print(f"\nFailed to initialize the model or load weights: {e}")
         print("Please check the weight file path, its format, content, and the backbone_name.")
    except Exception as e:
         print(f"\nAn unexpected error occurred: {e}")
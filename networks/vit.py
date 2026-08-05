# -*- coding: utf-8 -*-
import os, io, tarfile
from typing import Dict, List, Optional, Tuple, Union, Any, Set

import torch
import torch.nn as nn
import torch.nn.functional as F

from .analytic_classifier import ACILDynamicClasses
from .protonet import ProtoNetHead
from .iciclenet import IcicleNetHead
from .hiepronet import HieProNetHead

# 假设你已在同目录提供了 CustomDeiT 或其它 ViT 实现
# from .custom_deit import CustomDeiT  # 若已拆分成单文件，请按需导入
# 这里直接使用你贴出的 CustomDeiT 定义（略）：
# 为方便集成，建议将 CustomDeiT 保持不变，只作为纯特征提取 backbone 使用。

# -*- coding: utf-8 -*-
from functools import partial
from typing import Dict, List, Optional, Tuple, Union, Any
import os

# Helper function for DropPath (Stochastic Depth) - often used in ViT/DeiT
def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""
    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        return f'drop_prob={round(self.drop_prob,3):0.3f}'

# Patch Embedding Layer
class PatchEmbedding(nn.Module):
    """ Image to Patch Embedding
    Args:
        img_size (int): Input image size. Default: 224.
        patch_size (int): Patch token size. Default: 16.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 768.
        norm_layer (nn.Module, optional): Normalization layer. Default: None.
        flatten (bool): Whether to flatten the spatial dimensions of the output. Default: True.
    """
    def __init__(self, img_size=224, patch_size=8, in_chans=3, embed_dim=384, norm_layer=None, flatten=True):
        super().__init__()
        img_size = (img_size, img_size) if isinstance(img_size, int) else img_size
        patch_size = (patch_size, patch_size) if isinstance(patch_size, int) else patch_size
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        # Convolutional layer to extract patches and project them
        # Name 'proj' matches common implementations (timm, DINO likely)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        if not (H == self.img_size[0] and W == self.img_size[1]):
             # Allow dynamic image size if needed, but warn if not matching init size
             # This might cause issues with positional embeddings if not handled carefully
             print(f"Warning: Input image size ({H}x{W}) doesn't match init size ({self.img_size[0]}x{self.img_size[1]})")
             # Potentially resize x here if strict size matching is required:
             # x = F.interpolate(x, size=self.img_size, mode='bilinear', align_corners=False)

        x = self.proj(x) # Shape: (B, embed_dim, grid_size[0], grid_size[1])
        if self.flatten:
            # Flatten the spatial dimensions and transpose to (B, num_patches, embed_dim)
            x = x.flatten(2).transpose(1, 2)  # BCHW -> B C (H*W) -> B (H*W) C
        x = self.norm(x)
        return x

# Multi-Head Self-Attention
class Attention(nn.Module):
    """ Multi-Head Self-Attention module """
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # Linear layer for Q, K, V projections combined (matches common practice)
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        # Output projection layer (matches 'proj' naming)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape # Batch Size, Number of Tokens, Embedding Dimension
        # Calculate Q, K, V for all heads
        # qkv shape: (B, N, 3 * C) -> (B, N, 3, num_heads, head_dim) -> (3, B, num_heads, N, head_dim)
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0) # Make tensors for q, k, v each of shape (B, num_heads, N, head_dim)

        # Calculate attention scores (scaled dot-product)
        # (B, num_heads, N, head_dim) @ (B, num_heads, head_dim, N) -> (B, num_heads, N, N)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # Calculate weighted average of values
        # (B, num_heads, N, N) @ (B, num_heads, N, head_dim) -> (B, num_heads, N, head_dim)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C) # Combine heads: -> (B, N, num_heads, head_dim) -> (B, N, C)

        # Apply output projection
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

# MLP Block
class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        # Use 'fc1' and 'fc2' names, common in timm/DINO
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

# Transformer Encoder Block
class Block(nn.Module):
    """ Transformer Block """
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        # Use 'norm1' and 'norm2' names
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        # Pre-normalization structure (common in ViT/DeiT)
        x = x + self.drop_path(self.attn(self.norm1(x))) # Residual connection after attention
        x = x + self.drop_path(self.mlp(self.norm2(x)))  # Residual connection after MLP
        return x

# The main Custom Vision Transformer (DeiT-Small/8 compatible)
class CustomDeiT(nn.Module):
    """ Vision Transformer inspired by DeiT """
    def __init__(self, img_size=224, patch_size=8, in_chans=3, num_classes=0, # num_classes=0 for feature extraction
                 embed_dim=384, depth=12, num_heads=6, mlp_ratio=4., qkv_bias=True, # DeiT-Small params
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., norm_layer=None,
                 has_cls_token=True): # DINO uses CLS token
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with timm
        self.has_cls_token = has_cls_token
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        self.norm_layer = norm_layer

        # --- Patch Embedding ---
        self.patch_embed = PatchEmbedding(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        # --- CLS Token ---
        if self.has_cls_token:
            # Name 'cls_token' matches DINO/timm
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            # Initialize CLS token (often done, though pre-trained weights will overwrite)
            nn.init.trunc_normal_(self.cls_token, std=.02)
            num_tokens = 1
        else:
            self.cls_token = None
            num_tokens = 0

        # --- Positional Embedding ---
        # Name 'pos_embed' matches DINO/timm
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        # Initialize positional embedding
        nn.init.trunc_normal_(self.pos_embed, std=.02)

        # --- Transformer Blocks ---
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        # Name 'blocks' matches DINO/timm
        self.blocks = nn.Sequential(*[
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])

        # --- Final Normalization ---
        # Name 'norm' matches DINO/timm final layer norm
        self.norm = norm_layer(embed_dim)

        # --- Head (Removed for feature extraction) ---
        # If num_classes > 0, add a head, but for backbone use, we omit it.
        # self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        # We will rely on the ViTMultiHeadHierarchical class to add heads later.

        # Weight init (useful if not loading pre-trained weights, but will be overwritten)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        # elif isinstance(m, nn.Conv2d): # Conv weights init if needed
        #     nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore # type: ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'} # Parameters not typically subject to weight decay

    def forward_features(self, x):
        B = x.shape[0]
        # 1. Patch Embedding
        x = self.patch_embed(x) # Shape: (B, num_patches, embed_dim)

        # 2. Add CLS token
        if self.cls_token is not None:
            # Expand CLS token for the batch
            cls_tokens = self.cls_token.expand(B, -1, -1) # Shape: (B, 1, embed_dim)
            x = torch.cat((cls_tokens, x), dim=1) # Shape: (B, num_patches + 1, embed_dim)

        # 3. Add Positional Embedding
        x = x + self.pos_embed
        x = self.pos_drop(x)

        # 4. Pass through Transformer Blocks
        x = self.blocks(x)

        # 5. Final Normalization
        x = self.norm(x)

        # 6. Return features
        # DINO typically uses the CLS token for downstream tasks
        if self.has_cls_token:
            return x[:, 0] # Return only the CLS token embedding (B, embed_dim)
        else:
            # Alternative: return average of patch tokens
            return x.mean(dim=1)

    def forward(self, x):
        # This model is designed as a backbone, so forward returns features
        x = self.forward_features(x)
        # The head is handled by the ViTMultiHeadHierarchical class
        return x

# Helper function to create the specific DeiT-Small/8 model
def deit_small_patch8_224(**kwargs):
    model = CustomDeiT(
        patch_size=8, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model



class ViTBackboneWrapper(nn.Module):
    """
    对自定义 ViT/DeiT backbone 的薄封装：
    - 保持原始 forward_features/forward 不变；
    - 在需要时注册 forward hooks，抓取每个 block 输出的 token 序列（含或不含 cls）；
    - 提供把 patch tokens 重排为伪 2D 特征图的工具。
    """
    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone
        # 运行期信息（从 backbone 推断）
        self.embed_dim = getattr(backbone, "embed_dim", getattr(backbone, "num_features", None))
        if self.embed_dim is None:
            raise ValueError("Cannot infer ViT embed_dim from backbone. Please ensure backbone.embed_dim or num_features exists.")
        # 获取 patch grid
        patch_embed = getattr(backbone, "patch_embed", None)
        if patch_embed is None or not hasattr(patch_embed, "grid_size"):
            raise ValueError("Backbone must have .patch_embed.grid_size to recover HxW.")
        self.grid_size = patch_embed.grid_size  # (H, W)
        self.num_patches = patch_embed.num_patches
        self.has_cls = getattr(backbone, "has_cls_token", True)

        # block 容器（必须是可迭代的）
        self.blocks = getattr(backbone, "blocks", None)
        if self.blocks is None:
            raise ValueError("Backbone must expose .blocks (nn.Sequential of transformer blocks).")
        self.depth = len(self.blocks)

        # hooks 管理
        self._hooks_registered = False
        self._feature_cache: Dict[str, torch.Tensor] = {}

    def _register_hooks(self):
        if self._hooks_registered:
            return

        def make_hook(i: int):
            # 保存第 i 个 block 的输出 token 序列（B, N, D）
            def hook(_, __, output):
                self._feature_cache[f"tokens.{i}"] = output
                # CLS token 向量（B, D）
                if self.has_cls:
                    self._feature_cache[f"blocks.{i}"] = output[:, 0]
            return hook

        for i, blk in enumerate(self.blocks, start=1):
            blk.register_forward_hook(make_hook(i))

        self._hooks_registered = True

    def tokens_to_map(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        将 patch tokens (B, N[, D]) 或 (B, N, D) 转成伪 2D 特征图 [B, D, H, W]。
        若包含 CLS，则先移除 CLS。
        """
        if tokens.dim() == 2:
            # (B, D) -> (B, D, 1, 1)
            return tokens.unsqueeze(-1).unsqueeze(-1)
        if tokens.dim() != 3:
            raise ValueError(f"tokens_to_map expects (B,N,D) or (B,D), got {tuple(tokens.shape)}")

        B, N, D = tokens.shape
        if self.has_cls and N == self.num_patches + 1:
            tokens = tokens[:, 1:, :]  # remove CLS
            N -= 1
        H, W = self.grid_size
        if N != H * W:
            raise ValueError(f"Token count {N} does not match grid {H}x{W}")
        return tokens.transpose(1, 2).reshape(B, D, H, W)

    def forward_collect(self, x: torch.Tensor, needed: Set[str]) -> Dict[str, torch.Tensor]:
        """
        运行一次 backbone(x)，并根据 needed 收集：
        - cls            -> CLS 向量 (B, D)
        - gap:patch      -> 对 patch tokens GAP 后的 (B, D)
        - blocks.i       -> 第 i 层 block 的 CLS 向量 (B, D)
        - tokens.i       -> 第 i 层 block 的 token 序列 (B, N, D)
        """
        self._feature_cache = {}
        self._register_hooks()

        was_training = self.backbone.training
        is_frozen = not any(p.requires_grad for p in self.backbone.parameters())
        if is_frozen:
            self.backbone.eval()

        # 执行前向：为了拿到 hooks 的中间 tokens，必须跑完整个 ViT
        # 注意：CustomDeiT.forward 返回 CLS 特征（B, D）
        cls_feat = self.backbone(x)  # (B, D)

        if is_frozen:
            self.backbone.train(was_training)

        feats: Dict[str, torch.Tensor] = {}
        # 顶层语义：cls
        feats["cls"] = cls_feat

        # 如果请求 gap:patch，需要用最后一层 tokens
        if "gap:patch" in needed or any(k.startswith("gap:") for k in needed):
            last_tokens_key = f"tokens.{self.depth}"
            if last_tokens_key not in self._feature_cache:
                raise RuntimeError("Missing final block tokens to compute gap:patch.")
            toks = self._feature_cache[last_tokens_key]
            if self.has_cls:
                toks = toks[:, 1:, :]
            feats["gap:patch"] = toks.mean(dim=1)  # (B, D)

        # blocks.i / tokens.i
        for k in list(needed):
            if k.startswith("blocks."):
                if k in self._feature_cache:
                    feats[k] = self._feature_cache[k]  # (B, D)
                else:
                    # 向后兼容：如果没显式请求，但可以从 tokens.i 得到 CLS
                    idx = int(k.split(".")[1])
                    tkey = f"tokens.{idx}"
                    if tkey in self._feature_cache:
                        feats[k] = self._feature_cache[tkey][:, 0] if self.has_cls else None
            elif k.startswith("tokens."):
                if k in self._feature_cache:
                    feats[k] = self._feature_cache[k]  # (B, N, D)

        return feats


class ViTMultiHeadHierarchical(nn.Module):
    """
    ViT 多头层级模型（接口与 ResNetMultiHeadHierarchical 对齐）：
    - heads: {"linear":{}, "acil":{}, "protonet":{}}
    - heads_source: 记录每个 head 的特征来源，默认 "cls"
    - forward 支持：
        head_types: "linear"|"acil"|"protonet"|List
        features_to_return: True/"cls"/"gap:patch"/"blocks.i"/"tokens.i"
        protonet 的空间特征来自 tokens.i -> [B, D, H, W]
    """
    def __init__(self,
                 backbone_ctor: Optional[Any] = None,
                 backbone_kwargs: Optional[Dict[str, Any]] = None,
                 custom_weight_path: Optional[str] = None,
                 pretrained: bool = False,  # 仅为了接口一致，这里无实际作用
                 freeze_backbone: bool = True,
                 tar_member_name: Optional[str] = None,

                 # 与 ResNet 版保持相同的扩展参数
                 return_features: Optional[bool] = False,
                 expansion_dim: Optional[int] = 4096,
                 proto_dim: Optional[int] = 128,
                 add_heads: Optional[dict] = None,
                 head_list: Optional[List[str]] = None):
        super().__init__()
        self.return_features = return_features
        self.expansion_dim = expansion_dim
        self.proto_dim = proto_dim
        self.add_heads = add_heads or {}
        self.head_list = head_list or []
        self.weight_linear = nn.Parameter(torch.ones(len(self.add_heads))) if self.add_heads else nn.Parameter(torch.ones(1))
        self.weight_acil = nn.Parameter(torch.ones(len(self.add_heads))) if self.add_heads else nn.Parameter(torch.ones(1))

        # 1) 构建 backbone
        if backbone_ctor is None:
            # 默认使用你给出的 CustomDeiT 参数
            from functools import partial
            backbone_ctor = CustomDeiT  # 使用你贴出的类
            if backbone_kwargs is None:
                backbone_kwargs = dict(img_size=224, patch_size=8, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4.0)
        self.backbone_core = backbone_ctor(**(backbone_kwargs or {}))
        self.backbone = ViTBackboneWrapper(self.backbone_core)

        self.feature_dim = self.backbone.embed_dim
        print(f"ViT backbone embed_dim (feature_dim): {self.feature_dim}")

        # 2) 加载权重（支持 .pth/.tar）
        if custom_weight_path:
            self._load_weights_from_path(custom_weight_path, tar_member_name)
        else:
            print("Warning: No custom_weight_path provided for ViT. Using random init.")

        # 3) 冻结
        if freeze_backbone:
            self.freeze_backbone()
        else:
            self.unfreeze_backbone()

        # 4) 头字典
        self.heads = nn.ModuleDict({
            "linear": nn.ModuleDict(),
            "acil": nn.ModuleDict(),
            "protonet": nn.ModuleDict(),
        })
        self.heads_source: Dict[str, Dict[str, str]] = {
            "linear": {},
            "acil": {},
            "protonet": {}
        }

    # -------------------- 权重加载，与 ResNet 版对齐 --------------------
    def _load_weights_from_path(self, weight_path: str, tar_member_name: Optional[str] = None):
        if not os.path.exists(weight_path):
            raise FileNotFoundError(f"Weight file not found: {weight_path}")

        print(f"Loading ViT backbone weights from: {weight_path}")
        checkpoint = None
        try:
            if weight_path.endswith((".tar", ".tar.gz", ".tgz")):
                print("Detected .tar archive. Attempting to extract weights...")
                read_mode = 'r:gz' if weight_path.endswith(".gz") else 'r'
                with tarfile.open(weight_path, read_mode) as tar:
                    if tar_member_name:
                        try:
                            member = tar.getmember(tar_member_name)
                        except KeyError:
                            raise FileNotFoundError(f"Member '{tar_member_name}' not in tar. Members: {[m.name for m in tar.getmembers()]}")
                    else:
                        cands = [m for m in tar.getmembers() if m.isfile() and m.name.endswith(('.pth', '.pt', '.ckpt'))]
                        if not cands:
                            cands = sorted([m for m in tar.getmembers() if m.isfile()], key=lambda m: m.size, reverse=True)
                        if not cands:
                            raise FileNotFoundError(f"No suitable member in {weight_path}")
                        member = cands[0]
                        print(f"Auto selected tar member: {member.name}")
                    fobj = tar.extractfile(member)
                    if fobj is None:
                        raise IOError("Failed to extract member from tar.")
                    checkpoint = torch.load(io.BytesIO(fobj.read()), map_location="cpu")
            else:
                checkpoint = torch.load(weight_path, map_location="cpu")

            state_dict = self._extract_state_dict_from_checkpoint(checkpoint, weight_path)
            cleaned = self._clean_state_dict_keys(state_dict)

            print(f"Attempting to load {len(cleaned)} keys into ViT backbone (strict=False)...")
            res = self.backbone_core.load_state_dict(cleaned, strict=False)
            if not res.missing_keys and not res.unexpected_keys:
                print("Successfully loaded ViT backbone weights.")
            else:
                print("ViT weight loading with mismatches:")
                if res.missing_keys:
                    print("  Missing:", res.missing_keys[:20], "..." if len(res.missing_keys) > 20 else "")
                if res.unexpected_keys:
                    print("  Unexpected:", res.unexpected_keys[:20], "..." if len(res.unexpected_keys) > 20 else "")

        except Exception as e:
            print(f"Error loading ViT weights: {e}")
            raise

    def _extract_state_dict_from_checkpoint(self, checkpoint: Any, file_path: str) -> Dict[str, torch.Tensor]:
        state_dict = None
        if isinstance(checkpoint, dict):
            if 'teacher' in checkpoint and isinstance(checkpoint['teacher'], dict):
                print("Found 'teacher' in checkpoint; using teacher state dict.")
                state_dict = checkpoint['teacher']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'model' in checkpoint:
                model = checkpoint['model']
                if hasattr(model, 'state_dict'): state_dict = model.state_dict()
                elif isinstance(model, dict):     state_dict = model
            elif 'backbone' in checkpoint:
                state_dict = checkpoint['backbone']
            else:
                print("Treating checkpoint dict as state_dict directly.")
                state_dict = checkpoint
        elif isinstance(checkpoint, nn.Module):
            state_dict = checkpoint.state_dict()
        elif hasattr(checkpoint, 'keys'):
            state_dict = checkpoint
        else:
            raise TypeError(f"Unsupported checkpoint type from {file_path}: {type(checkpoint)}")

        if not isinstance(state_dict, dict):
            raise TypeError("state_dict extracted is not a dict.")
        return state_dict

    def _clean_state_dict_keys(self, state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        cleaned = {}
        if not state_dict:
            return cleaned
        sample = list(state_dict.keys())[:5]
        prefix = ""
        for pf in ("module.", "backbone.", "model."):
            if sample and all(k.startswith(pf) for k in sample):
                prefix = pf
                break
        if prefix:
            print(f"Removing '{prefix}' prefix from ViT keys.")
        filtered = 0
        for k, v in state_dict.items():
            nk = k[len(prefix):] if prefix and k.startswith(prefix) else k
            if nk.startswith("head."):  # 丢弃预训练分类头
                filtered += 1
                continue
            cleaned[nk] = v
        if filtered:
            print(f"Filtered out {filtered} head.* keys from checkpoint.")
        return cleaned

    # -------------------- 冻结/解冻 --------------------
    def freeze_backbone(self):
        print("Freezing ViT backbone params.")
        for p in self.backbone_core.parameters():
            p.requires_grad = False
        self.backbone_core.eval()

    def unfreeze_backbone(self):
        print("Unfreezing ViT backbone params.")
        for p in self.backbone_core.parameters():
            p.requires_grad = True
        self.backbone_core.train()

    # -------------------- 头管理（接口与 ResNet 版对齐） --------------------
    def add_head(self, head_name: str, num_classes: int, head_type: str = "linear", freeze_existing: bool = False):
        if head_type not in self.heads:
            raise ValueError(f"Unsupported head type: {head_type}. Choose from {list(self.heads.keys())}.")
        if head_name in self.heads[head_type]:
            print(f"Warning: Head '{head_name}' already exists in '{head_type}', overwriting.")
        if freeze_existing:
            self.freeze_all_heads()

        if head_type == "linear":
            mod = nn.Linear(self.feature_dim, num_classes)
        elif head_type == "acil":
            mod = ACILDynamicClasses(feature_dim=self.feature_dim, expansion_dim=self.expansion_dim,
                                     num_classes=num_classes, gamma=0.1, device="cuda")
        elif head_type == "protonet":
            # 注意：默认 in_channels 取 embed_dim；如果绑定 tokens.i，会被转成 [B, D, H, W]，D=embed_dim
            mod = ProtoNetHead(head_name=head_name, in_channels=self.feature_dim, feature_dim=self.proto_dim, num_classes=num_classes)
        else:
            raise ValueError(f"Unsupported head type: {head_type}")

        self.heads[head_type][head_name] = mod
        # ViT 默认来源为 'cls'（等价于 ResNet 的 pool/flat）
        self.heads_source[head_type][head_name] = "cls"

    def add_linear_head(self, head_name: str, num_classes: int, freeze_existing: bool = False):
        if freeze_existing: self.freeze_all_heads()
        self.heads["linear"][head_name] = nn.Linear(self.feature_dim, num_classes)
        self.heads_source["linear"][head_name] = "cls"

    def add_analytic_head(self, head_name: str, num_classes: int, freeze_existing: bool = False):
        if freeze_existing: self.freeze_all_heads()
        self.heads["acil"][head_name] = ACILDynamicClasses(
            feature_dim=self.feature_dim, expansion_dim=self.expansion_dim,
            num_classes=num_classes, gamma=0.1, device="cuda")
        self.heads_source["acil"][head_name] = "cls"

    def freeze_all_heads(self):
        print("Freezing all heads.")
        for fam in self.heads.values():
            for mod in fam.values():
                for p in mod.parameters():
                    p.requires_grad = False

    def unfreeze_head(self, head_name: str):
        found = False
        for fam in self.heads.values():
            if head_name in fam:
                for p in fam[head_name].parameters():
                    p.requires_grad = True
                found = True
        if not found:
            print(f"Warning: Head '{head_name}' not found for unfreezing.")

    def freeze_head(self, head_name: str):
        found = False
        for fam in self.heads.values():
            if head_name in fam:
                for p in fam[head_name].parameters():
                    p.requires_grad = False
                found = True
        if not found:
            print(f"Warning: Head '{head_name}' not found for freezing.")

    # def set_head_source(self, head_type: str, head_name: str, from_feature: str):
    #     """
    #     允许来源：
    #     - 'cls'                 -> CLS token 向量
    #     - 'gap:patch'           -> patch tokens 的全局平均后向量
    #     - 'blocks.i'            -> 第 i 个 block 的 CLS 向量
    #     - 'tokens.i'            -> 第 i 个 block 的 token 序列（B, N, D），仅 protonet 有用
    #     """
    #     if head_type not in self.heads or head_name not in self.heads[head_type]:
    #         raise ValueError(f"Head '{head_name}' not found in type '{head_type}'.")
    #     if not (from_feature == "cls" or
    #             from_feature == "gap:patch" or
    #             (from_feature.startswith("blocks.") and from_feature.split(".")[1].isdigit()) or
    #             (from_feature.startswith("tokens.") and from_feature.split(".")[1].isdigit())):
    #         raise ValueError("from_feature must be one of 'cls','gap:patch','blocks.i','tokens.i'")
    #     self.heads_source[head_type][head_name] = from_feature
    def set_head_source(self, head_type: str, head_name: str, from_feature: str):
        """
        仅修改 vit.py：支持外部别名并映射到 ViT 内部键。
        外部可用名（格式仍是字符串）：
        - ViT 别名：'pool','vit_cls','vit_gap','vit_last','vit_mid','vit_low'
        - ViT 原生：'cls','gap:patch','blocks.i','tokens.i'
        - 若仍有 ResNet 风格：'c2','c3','c4','c5'，也兼容
        """
        if head_type not in self.heads or head_name not in self.heads[head_type]:
            raise ValueError(f"Head '{head_name}' not found in type '{head_type}'.")

        src = str(from_feature).strip().lower()
        depth = int(getattr(self.backbone, "depth", 12))
        last_idx = max(0, depth - 1)

        def map_alias(s: str) -> str:
            # ViT 别名
            if s == "pool":      # 若你更偏好 GAP，可改为 'gap:patch'
                return "cls"
            if s == "vit_cls":
                return "cls"
            if s == "vit_gap":
                return "gap:patch"
            if s == "vit_last":
                return f"tokens.{last_idx}"
            if s == "vit_mid":
                return f"tokens.{max(0, depth // 2 - 1)}"
            if s == "vit_low":
                return f"tokens.{max(0, depth // 4 - 1)}"
            # ResNet 风格（若仍使用）
            if s in ("c2", "c3", "c4", "c5"):
                if s == "c2":
                    i = max(0, depth // 4 - 1)
                elif s == "c3":
                    i = max(0, depth // 2 - 1)
                elif s == "c4":
                    i = max(0, (3 * depth) // 4 - 1)
                else:  # c5
                    i = last_idx
                return f"tokens.{i}"
            # 已是 ViT 原生键
            return s

        mapped = map_alias(src)

        ok = (
            mapped == "cls" or
            mapped == "gap:patch" or
            (mapped.startswith("blocks.") and mapped.split(".")[1].isdigit()) or
            (mapped.startswith("tokens.") and mapped.split(".")[1].isdigit())
        )
        if not ok:
            raise ValueError(
                f"from_feature '{from_feature}' (mapped -> '{mapped}') must be one of "
                f"'cls','gap:patch','blocks.i','tokens.i','pool','vit_cls','vit_gap','vit_last','vit_mid','vit_low','c2','c3','c4','c5'"
            )

        self.heads_source[head_type][head_name] = mapped


    # -------------------- 前向与特征路由 --------------------
    def forward(self,
            x: torch.Tensor,
            head_names: List[str],
            return_features: bool = False,
            head_types: Optional[Union[str, List[str]]] = "linear",
            features_to_return: Optional[Union[bool, str, List[str]]] = None,
            return_protofeatures: bool = False,
            prior_dict: Optional[Dict[str, torch.Tensor]] = None,
            prior_power: float = 1.0,
            prior_topk: Optional[int] = None,
            prior_logit_bias: float = 0.0,
            prior_temp: float = 1.0):
        if not head_names:
            raise ValueError("head_names list cannot be empty.")

        def _norm_head_types(hts) -> List[str]:
            return [hts] if isinstance(hts, str) else list(hts)

        head_types_list = _norm_head_types(head_types)

        # 读取 ViT 深度，设定最后一层索引（0-based）
        depth = int(getattr(self.backbone, "depth", 12))
        last_idx = max(0, depth - 1)

        # 需要哪些来源（来自各 head 的 source 声明）
        needed: Set[str] = set()
        for ht in head_types_list:
            if ht not in self.heads:
                raise ValueError(f"Unknown head_type '{ht}'.")
            for name in head_names:
                if name not in self.heads[ht]:
                    raise ValueError(f"Head '{name}' not found in type '{ht}'. Available: {list(self.heads[ht].keys())}")
                needed.add(self.heads_source.get(ht, {}).get(name, "cls"))

        # 标准化 features_to_return -> requested（外部名，先不映射）
        def norm_req(req) -> Set[str]:
            if req is None:
                return {"cls"} if return_features else set()
            if isinstance(req, bool):
                return {"cls"} if req else set()
            if isinstance(req, str):
                return {req}
            if isinstance(req, list):
                return set(req)
            raise ValueError("features_to_return must be None|bool|str|List[str].")

        requested = norm_req(features_to_return)

        # 别名映射（外部名 -> ViT 内部键），统一 0-based
        def _map_alias(k: str) -> str:
            kk = str(k).strip().lower()
            # ViT 自定义别名
            if kk == "pool":
                return "cls"            # 如偏好 GAP，可改为 'gap:patch'
            if kk == "vit_cls":
                return "cls"
            if kk == "vit_gap":
                return "gap:patch"
            if kk == "vit_last":
                return f"tokens.{last_idx}"
            if kk == "vit_mid":
                return f"tokens.{max(0, depth // 2 - 1)}"
            if kk == "vit_low":
                return f"tokens.{max(0, depth // 4 - 1)}"
            # 兼容 ResNet 风格（可留可去）
            if kk in ("c2", "c3", "c4", "c5"):
                if kk == "c2":
                    i = max(0, depth // 4 - 1)
                elif kk == "c3":
                    i = max(0, depth // 2 - 1)
                elif kk == "c4":
                    i = max(0, (3 * depth) // 4 - 1)
                else:  # c5
                    i = last_idx
                return f"tokens.{i}"
            # 已是 ViT 原生键
            return kk

        # 对 requested 做别名映射，并并入 needed
        if requested:
            requested = {_map_alias(k) for k in requested}
        needed |= requested

        # 运行一次 backbone 收集所需特征
        feats = self.backbone.forward_collect(x, needed)

        # 工具：取向量/取空间图
        def vec_for(key: str) -> torch.Tensor:
            if key == "cls":
                if "cls" not in feats:
                    available = ", ".join(sorted(map(str, feats.keys())))
                    raise KeyError(f"Feature 'cls' not found. Available keys: [{available}]")
                return feats["cls"]
            if key == "gap:patch":
                if "gap:patch" not in feats:
                    available = ", ".join(sorted(map(str, feats.keys())))
                    raise KeyError(f"Feature 'gap:patch' not found. Available keys: [{available}]")
                return feats["gap:patch"]
            if key.startswith("blocks."):
                if key in feats:
                    return feats[key]
                # 若未缓存 blocks.i，则尝试从 tokens.i 退化得到向量（CLS 或 GAP）
                idx = int(key.split(".")[1])
                tkey = f"tokens.{idx}"
                if tkey not in feats:
                    available = ", ".join(sorted(map(str, feats.keys())))
                    raise KeyError(f"Missing tokens for '{key}' (looked for '{tkey}'). Available keys: [{available}]")
                toks = feats[tkey]
                return toks[:, 0] if getattr(self.backbone, "has_cls", True) else toks.mean(dim=1)
            # 兜底：如果 key 是 tokens.i，退化成向量（用于 linear/acil 这类需要向量的头）
            if key.startswith("tokens."):
                if key not in feats:
                    available = ", ".join(sorted(map(str, feats.keys())))
                    raise KeyError(f"Feature '{key}' not found. Available keys: [{available}]")
                toks = feats[key]
                return toks[:, 0] if getattr(self.backbone, "has_cls", True) else toks.mean(dim=1)
            raise ValueError(f"vec_for: Unsupported vector source '{key}'")

        def fmap_for(key: str) -> torch.Tensor:
            # 将 tokens.i 或向量转成 [B, D, H, W]
            if key.startswith("tokens."):
                toks = feats.get(key, None)
                if toks is None:
                    available = ", ".join(sorted(map(str, feats.keys())))
                    raise KeyError(f"Missing tokens for '{key}'. Available keys: [{available}]")
                return self.backbone.tokens_to_map(toks)
            # 允许 cls/gap 退化为 1x1 feature map
            v = vec_for(key)
            return v.unsqueeze(-1).unsqueeze(-1)

        # prior 预处理
        def _prep_prior_for(head_name: str, C_expect: int) -> Optional[torch.Tensor]:
            if prior_dict is None:
                return None
            p = prior_dict.get(head_name, None)
            if p is None:
                return None
            if p.dim() != 2 or p.size(1) != C_expect:
                raise ValueError(f"prior_dict['{head_name}'] shape mismatch: got {tuple(p.shape)}, expect [B,{C_expect}]")
            if prior_temp != 1.0:
                p = (p.clamp_min(1e-12).log() / float(prior_temp)).softmax(dim=-1)
            else:
                p = p / p.sum(dim=-1, keepdim=True).clamp_min(1e-12)
            return p

        # 计算输出
        if isinstance(head_types, str):
            outputs: Dict[str, torch.Tensor] = {}
            p_features: Dict[str, torch.Tensor] = {}
            for name in head_names:
                src = self.heads_source.get(head_types, {}).get(name, "cls")
                if head_types == "protonet":
                    fmap = fmap_for(src)
                    head_mod = self.heads["protonet"][name]
                    if isinstance(head_mod, HieProNetHead):
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
                            outputs[name], p_features[name] = head_mod(fmap, return_protofeatures=True)
                        else:
                            outputs[name] = head_mod(fmap, return_protofeatures=False)
                else:
                    vec = vec_for(src)
                    outputs[name] = self.heads[head_types][name](vec)
        else:
            outputs = {ht: {} for ht in head_types_list}
            p_features: Dict[str, torch.Tensor] = {}
            for ht in head_types_list:
                for name in head_names:
                    src = self.heads_source.get(ht, {}).get(name, "cls")
                    if ht == "protonet":
                        fmap = fmap_for(src)
                        if return_protofeatures:
                            outputs[ht][name], p_features[name] = self.heads["protonet"][name](fmap, return_protofeatures=True)
                        else:
                            outputs[ht][name] = self.heads["protonet"][name](fmap, return_protofeatures=False)
                    elif ht in ("acil", "linear"):
                        vec = vec_for(src)
                        outputs[ht][name] = self.heads[ht][name](vec)
                    else:
                        vec = vec_for(src)
                        outputs[ht][name] = self.heads[ht][name](vec)

        # 返回特征
        if requested:
            # requested 已经做过别名映射，这里只做存在性校验并取出
            out_feats: Dict[str, torch.Tensor] = {}
            for k in requested:
                if k == "cls" or k == "gap:patch" or k.startswith("blocks."):
                    out_feats[k] = vec_for(k)
                elif k.startswith("tokens."):
                    if k not in feats:
                        available = ", ".join(sorted(map(str, feats.keys())))
                        raise KeyError(f"Requested feature '{k}' not found. Available keys: [{available}]")
                    out_feats[k] = feats[k]
                else:
                    raise ValueError(f"Unsupported requested feature key '{k}' for ViT.")
            if return_protofeatures:
                return {"logits": outputs, "features": out_feats, "proto_features": p_features}
            else:
                return {"logits": outputs, "features": out_feats}

        if return_protofeatures:
            return {"logits": outputs, "proto_features": p_features}
        else:
            return outputs

    # -------------------- 训练参数收集，与 ResNet 版一致 --------------------
    def get_trainable_parameters(self) -> List[Dict[str, Any]]:
        params_to_optimize = []
        bb_params = [p for p in self.backbone_core.parameters() if p.requires_grad]
        if bb_params:
            params_to_optimize.append({'params': bb_params, 'lr_scale_factor': 0.1})
            print("Including trainable ViT backbone params.")
        for fam_name, fam in self.heads.items():
            head_params = [p for m in fam.values() for p in m.parameters() if p.requires_grad]
            if head_params:
                params_to_optimize.append({'params': head_params})
                print(f"Including trainable head family '{fam_name}' params.")
        if not params_to_optimize:
            print("Warning: No trainable parameters found!")
        return params_to_optimize

    # -------------------- ACIL 头的训练接口（与 ResNet 版一致） --------------------
    def train_acil_head(self, X_train: torch.Tensor,
                        Y_train: torch.Tensor, mode: str = "base",
                        feature_key: str = "cls"):
        if mode not in {"base", "incremental"}:
            raise ValueError("mode must be 'base' or 'incremental'.")

        # 规范化 X_train
        if isinstance(X_train, dict):
            feats_dict = X_train.get("features", X_train)
            if feature_key not in feats_dict:
                raise ValueError(f"X_train dict missing features['{feature_key}']. Keys: {list(feats_dict.keys())}")
            X_train = feats_dict[feature_key]
        if not isinstance(X_train, torch.Tensor):
            raise TypeError("X_train must be Tensor or dict containing the chosen feature tensor.")

        num_heads = len(self.heads["acil"])
        if Y_train.shape[0] != num_heads:
            raise ValueError(f"Y_train must have shape (num_heads, B, C), got {Y_train.shape} vs heads={num_heads}")

        for idx, (head_name, head_module) in enumerate(self.heads["acil"].items()):
            head_labels = Y_train[idx]
            if isinstance(head_module, ACILDynamicClasses):
                if mode == "base":
                    head_module.base_training(X_train, head_labels)
                else:
                    head_module.incremental_learning(X_train, head_labels)
            else:
                print(f"Skipping non-ACIL head '{head_name}'.")

def clone_vit_snapshot(model: ViTMultiHeadHierarchical) -> ViTMultiHeadHierarchical:
    """
    生成一个“新的”同结构 ViTMultiHeadHierarchical 模型，并拷贝当前权重。
    返回的模型与原模型参数脱钩（不同对象），默认冻结并设为 eval。
    注意：
    - 会尽量复原各个 head 的构造参数（名字、类型、尺寸），并同步 heads_source。
    - 需要 ViTMultiHeadHierarchical 提供 backbone_ctor/backbone_kwargs 等信息；
      若未保存，我们从现有 backbone_core 上推断关键参数。
    """
    # 1) 重建骨干模型的构造参数
    # 优先从实例属性读取（若你在 __init__ 内保存了它们）；否则从 backbone_core 推断
    backbone_ctor = getattr(model, "backbone_ctor", None)
    backbone_kwargs = getattr(model, "backbone_kwargs", None)

    # 当原实现未保存 ctor/kwargs 时，尽量从 backbone_core 推断
    if backbone_ctor is None:
        backbone_ctor = model.backbone_core.__class__
    if backbone_kwargs is None:
        # 从 CustomDeiT 推断关键超参
        core = model.backbone_core
        backbone_kwargs = dict(
            img_size=getattr(core.patch_embed, "img_size", (224, 224))[0] if hasattr(core, "patch_embed") else 224,
            patch_size=getattr(core.patch_embed, "patch_size", (8, 8))[0] if hasattr(core, "patch_embed") else 8,
            in_chans=3,
            num_classes=0,
            embed_dim=getattr(core, "embed_dim", getattr(core, "num_features", 384)),
            depth=getattr(core, "get_num_layers", lambda: len(getattr(core, "blocks", [])))(),
            num_heads=getattr(core.blocks[0].attn, "num_heads", 6) if hasattr(core, "blocks") and len(core.blocks) > 0 else 6,
            mlp_ratio=4.0,
            qkv_bias=True,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.0,
            norm_layer=getattr(core, "norm_layer", None),
            has_cls_token=getattr(core, "has_cls_token", True),
        )
        # norm_layer 可能是 functools.partial 或类，直接复用

    # 保存一些用于新模型构建的辅助参数
    return_features = getattr(model, "return_features", False)
    expansion_dim = getattr(model, "expansion_dim", 4096)
    proto_dim = getattr(model, "proto_dim", 128)
    add_heads = getattr(model, "add_heads", None)
    head_list = getattr(model, "head_list", None)

    # 2) 新建骨干（不加载自定义权重，稍后会整体拷贝 state_dict）
    model_new = ViTMultiHeadHierarchical(
        backbone_ctor=backbone_ctor,
        backbone_kwargs=backbone_kwargs,
        custom_weight_path=None,   # 不从文件加载，直接从旧模型 state_dict 复制
        pretrained=False,
        freeze_backbone=True,
        tar_member_name=None,
        return_features=return_features,
        expansion_dim=expansion_dim,
        proto_dim=proto_dim,
        add_heads=add_heads,
        head_list=head_list
    )

    # 3) 复现 heads 结构（按名字与类型逐个创建）
    # heads 的结构：{"linear": ModuleDict, "acil": ModuleDict, "protonet": ModuleDict}
    for ht, family in model.heads.items():
        for name, mod in family.items():
            if isinstance(mod, nn.Linear):
                # 线性头：out_features 即类别数，in_features 应等于 feature_dim
                model_new.add_head(name, num_classes=mod.out_features, head_type="linear")
                # 同步来源
                src = model.heads_source.get("linear", {}).get(name, "cls")
                model_new.set_head_source("linear", name, src)
            elif isinstance(mod, ACILDynamicClasses):
                # ACIL 头：复用 feature_dim / expansion_dim / num_classes
                model_new.add_head(name, num_classes=mod.num_classes, head_type="acil")
                src = model.heads_source.get("acil", {}).get(name, "cls")
                model_new.set_head_source("acil", name, src)
            elif isinstance(mod, ProtoNetHead):
                # ProtoNetHead 头：根据其内部形状参数重建
                model_new.heads["protonet"][name] = ProtoNetHead(
                    head_name=name,
                    in_channels=getattr(mod, "in_channels", model_new.feature_dim),
                    feature_dim=getattr(mod, "feature_dim", proto_dim),
                    num_classes=getattr(mod, "num_classes", None),
                )
                src = model.heads_source.get("protonet", {}).get(name, "cls")
                model_new.set_head_source("protonet", name, src)
            elif isinstance(mod, IcicleNetHead):
                # IcicleNetHead：按可读参数重建
                model_new.heads["protonet"][name] = IcicleNetHead(
                    head_name=getattr(mod, "head_name", name),
                    in_channels=mod.add_on.conv1.in_channels,
                    feature_dim=mod.D,
                    num_classes=mod.C,
                    prototypes_per_class=mod.M,
                    mid_channels=mod.add_on.conv1.out_channels,
                    init_scale=0.02,
                    ppnet_eps=mod.eps,
                )
                src = model.heads_source.get("protonet", {}).get(name, "cls")
                model_new.set_head_source("protonet", name, src)
            elif isinstance(mod, HieProNetHead):
                # HieProNetHead：按可读参数重建
                model_new.heads["protonet"][name] = HieProNetHead(
                    head_name=getattr(mod, "head_name", name),
                    in_channels=mod.add_on.conv1.in_channels,
                    feature_dim=mod.D,
                    num_classes=mod.C,
                    prototypes_per_class=mod.M,
                    mid_channels=mod.add_on.conv1.out_channels,
                    init_scale=0.02,
                    ppnet_eps=mod.eps,
                )
                src = model.heads_source.get("protonet", {}).get(name, "cls")
                model_new.set_head_source("protonet", name, src)
            else:
                raise TypeError(f"Unknown head type for '{name}': {type(mod)}")

    # 4) 拷贝参数（包含 buffers）
    # 使用 strict=False 以适配可能的轻微命名差异或未用到的参数
    model_new.load_state_dict(model.state_dict(), strict=False)

    # 5) 冻结并设为 eval（teacher/快照用途）
    for p in model_new.parameters():
        p.requires_grad_(False)
    model_new.eval()

    return model_new



# ---------- 使用示例 ----------
if __name__ == "__main__":
    # 构建 ViT（使用你的 CustomDeiT 参数）
    backbone_kwargs = dict(img_size=32, patch_size=2, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4.0)
    model = ViTMultiHeadHierarchical(
        backbone_ctor=CustomDeiT,
        backbone_kwargs=backbone_kwargs,
        custom_weight_path=None,  # 替换为你的权重路径
        freeze_backbone=True
    )

    # 添加头部
    model.add_head("task1", 100, head_type="linear")
    model.add_head("task2", 50, head_type="acil")
    model.add_head("task3", 20, head_type="protonet")

    # 绑定来源（可选）
    model.set_head_source("linear", "task1", "cls")
    model.set_head_source("acil", "task2", "gap:patch")
    # 将 protonet 绑定到中间层 tokens（有空间结构）
    model.set_head_source("protonet", "task3", "tokens.12")

    # 前向
    x = torch.randn(2, 3, 32, 32)
    out = model(x, head_names=["task1", "task2", "task3"], head_types=["linear", "acil", "protonet"], features_to_return=["cls", "tokens.12"])
    print(type(out), list(out.keys()))
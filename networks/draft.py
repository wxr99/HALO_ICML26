import math
import re
from typing import Dict, List, Optional, Tuple, Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------- 工具函数 ----------

_LEVEL_RE = re.compile(r"^L(\d+)_(\d+)$")
_LEVEL_NAME_RE = re.compile(r"^L(\d+)$")  # 用于从 head_name 推断层级，如 'L1'

def parse_hier_id(h: str) -> Tuple[int, int]:
    """
    解析 'Lk_c' 字符串，返回 (level, raw_class_id)
    例如 'L3_9' -> (3, 9)

    注意：raw_class_id 只用于外部可读性；模型内部会对每个 level 维护连续索引。
    """
    m = _LEVEL_RE.match(h)
    if m is None:
        raise ValueError(f"Invalid hierarchical id: {h}. Expected 'L<level>_<class>' e.g. 'L2_12'.")
    return int(m.group(1)), int(m.group(2))


def normalize_l2(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    对最后一维做 L2 归一化，避免除零问题。
    输入形状可为 [*, D]
    """
    return x / (x.norm(dim=-1, keepdim=True) + eps)


def prepare_features(x: torch.Tensor) -> torch.Tensor:
    """
    将任意输入特征统一为 [B, S, D] 的形状，便于后续在“空间位置 S”维度进行聚合。

    支持的输入：
    - [B, D]         -> [B, 1, D]      （单一全局向量）
    - [B, S, D]      -> [B, S, D]      （已是序列形式）
    - [B, H, W, D]   -> [B, H*W, D]    （NHWD: H/W 为空间维，D为通道/特征维）
    - [B, D, H, W]   -> [B, H*W, D]    （NCHW: 常见卷积输出）

    备注：
    - 对于 4D 输入，我们用一个简单启发式判断 NCHW/NHWD：
      若第二维（可能是通道数）较小且不等于 H/W，则更可能是 NCHW。
      若你的数据不符合该启发式，请在外部手动转换成 [B, S, D] 再传入，避免误判。
    """
    if x.dim() == 2:  # [B, D]
        return x.unsqueeze(1)
    if x.dim() == 3:  # [B, S, D]
        return x
    if x.dim() == 4:
        B = x.size(0)
        # 判定是 NHWD 还是 NCHW
        # 经验：若第1维远小于后两维且第3,4维相近，通常是 NCHW；否则按 NHWD 处理
        # 但更稳妥的方式是：若 x.shape[1] <= 512 且 x.shape[1] 是“通道数”，多半是 NCHW
        C_or_H = x.size(1)
        if C_or_H >= 256 and C_or_H not in (x.size(2), x.size(3)):  # NCHW 可能性更大
            # [B, D, H, W] -> [B, H, W, D]
            x = x.permute(0, 2, 3, 1).contiguous()
        # 现在是 [B, H, W, D]
        B, H, W, D = x.shape
        return x.view(B, H * W, D)
    raise ValueError(f"Unsupported feature shape: {tuple(x.shape)}")


def pairwise_sim_spatial(
    Z: torch.Tensor,       # [B, S, D]
    prototypes: torch.Tensor,  # [C, M, D]
    distance: str = "euclidean",
    temperature: float = 1.0,
    normalize: bool = False,
) -> torch.Tensor:
    """
    计算每个空间位置与每个原型的相似度，返回 [B, S, C, M]。相似度越大表示越相似（最大化）。

    - euclidean：sim = -||z - p||^2 / T
      注意：我们用负平方欧氏距离作为“相似度”以便统一为“越大越好”
    - cosine   ：sim = cos(z, p) / T
      可选 normalize=True 时，会对 Z 与原型做 L2 归一化，避免尺度影响
    """
    assert Z.dim() == 3 and prototypes.dim() == 3
    B, S, D = Z.shape
    C, M, Dp = prototypes.shape
    assert D == Dp, f"Feature dim mismatch: {D} vs {Dp}"

    if distance == "cosine":
        if normalize:
            Zn = normalize_l2(Z)
            Pn = normalize_l2(prototypes.view(C * M, D)).view(C, M, D)
        else:
            Zn, Pn = Z, prototypes
        sim = torch.einsum("bsd,cmd->bscm", Zn, Pn)  # [B,S,C,M]
        sim = sim / max(temperature, 1e-8)
        return sim

    elif distance == "euclidean":
        z2 = (Z**2).sum(dim=-1, keepdim=True)          # [B,S,1]
        p2 = (prototypes**2).sum(dim=-1)               # [C,M]
        cross = torch.einsum("bsd,cmd->bscm", Z, prototypes)  # [B,S,C,M]
        d2 = z2[..., None] + p2[None, None, :, :] - 2.0 * cross
        sim = -d2 / max(temperature, 1e-8)
        return sim
    else:
        raise ValueError(f"Unknown distance: {distance}")


def kmeans_pp_init(
    X: torch.Tensor,  # [N, D]
    k: int,
    iters: int = 10,
    seed: Optional[int] = None,
) -> torch.Tensor:
    """
    使用 k-means++ 选初始中心，并做少量 k-means 迭代。
    若支持样本较少（N < k），上层调用会重复/微扰补齐。
    """

    assert X.dim() == 2 and X.size(0) >= 1 #"X must be [N,D] with N>=1"
    device = X.device
    N, D = X.shape
    k = min(k, N)

    if seed is not None:
        g = torch.Generator(device=device)
        g.manual_seed(seed)
        first_idx = torch.randint(0, N, (1,), generator=g, device=device).item()
    else:
        first_idx = torch.randint(0, N, (1,), device=device).item()

    centers = [X[first_idx]]

    # k-means++ 选剩余中心
    for _ in range(1, k):
        dist2 = torch.cdist(X, torch.stack(centers, dim=0), p=2).pow(2)  # [N, len(centers)]
        min_dist2, _ = dist2.min(dim=1)
        probs = min_dist2 / (min_dist2.sum() + 1e-12)
        next_idx = torch.multinomial(probs, 1).item()
        centers.append(X[next_idx])

    centers = torch.stack(centers, dim=0)  # [k, D]

    for _ in range(iters):
        dist = torch.cdist(X, centers, p=2)
        assign = dist.argmin(dim=1)
        new_centers = []
        for c in range(k):
            mask = (assign == c)
            if mask.any():
                new_centers.append(X[mask].mean(dim=0))
            else:
                new_centers.append(X[torch.randint(0, N, (1,), device=device).item()])
        new_centers = torch.stack(new_centers, dim=0)
        if torch.allclose(new_centers, centers):
            break
        centers = new_centers

    return centers # [k, D]


# ---------- 层级原型头（带空间聚合） ----------

class LevelProtoHead(nn.Module):
    """
    管理某一层级的原型参数，并将 [B,S,D] 的特征映射成该层级的 logits [B, C_level]。

    关键点：
    - 每个类别有 M 个原型（M = prototypes_per_class）
    - 对输入的每个空间位置与每个原型计算相似度 -> [B,S,C,M]
    - 先在原型维 M 聚合（取 max / LSE / mean），再在空间维 S 聚合
    - 聚合后得到图像级的证据分数 [B,C]，再做可学习缩放/偏置，输出 logits
    """
    def __init__(
        self,
        level: int,
        feature_dim: int,
        prototypes_per_class: int = 3,
        distance: str = "euclidean",
        temperature: float = 1.0,
        normalize: bool = False,
        proto_aggregator: str = "max",    # M 维聚合：'max' | 'lse' | 'mean'
        spatial_aggregator: str = "max",  # S 维聚合：'max' | 'lse' | 'mean'
        lse_gamma: float = 1.0,           # LSE 平滑强度（越大越接近 max）
    ):
        super().__init__()
        self.level = level
        self.D = feature_dim
        self.M = prototypes_per_class
        self.distance = distance
        self.temperature = temperature
        self.normalize = normalize
        self.proto_aggregator = proto_aggregator
        self.spatial_aggregator = spatial_aggregator
        self.lse_gamma = lse_gamma

        # 每个类别对应一个 [M, D] 的 Parameter；动态增长（类增量）
        self.prototype_params = nn.ParameterList()  # len = num_classes_at_this_level, each [M, D]
        # 记录类别名（外部ID，如 'L2_12'），与 prototype_params 一一对应
        self.class_names: List[str] = []            # e.g., ['L1_0', 'L1_3', ...]

        # 可学习的全局缩放/偏置（对该层级的所有 logits）
        self.logit_scale = nn.Parameter(torch.tensor(1.0))
        self.logit_bias = nn.Parameter(torch.tensor(0.0))

    def num_classes(self) -> int:
        return len(self.class_names)

    def class_index(self, name: str) -> int:
        """
        将外部类ID（如 'L1_3'）映射成该层级内部的连续索引 [0..C-1]
        """
        try:
            return self.class_names.index(name)
        except ValueError:
            raise KeyError(f"Class {name} not found in level L{self.level}")

    # --- 增量添加类别 ---
    def add_class(
        self,
        class_name: str,
        init_embeddings: Optional[torch.Tensor] = None,  # [N,D] 或 [N,H,W,D] / [N,D,H,W] / [N,S,D]
        init_strategy: str = "kmeans",  # 'kmeans' | 'mean' | 'random'
        init_scale: float = 0.02, # random 时的尺度
        seed: Optional[int] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> int:
        """
        为该层级新增一个类别，并初始化该类的原型参数。
        返回该类在该层级内的内部索引（0-based）。

        init_embeddings:
        - 若提供，优先使用数据驱动初始化（kmeans/mean）
        - 未提供则随机初始化（高斯，std=init_scale）
        """
        if device is None or dtype is None:
            ref = next(self.parameters(), None)
            device = ref.device if ref is not None else torch.device("cpu")
            dtype = ref.dtype if ref is not None else torch.float32

        if init_embeddings is not None:
            # 将任意形状支持样本特征展开成 [N_total, D]
            if init_embeddings.dim() == 4:  # 可能是 NHWD 或 NCHW
                emb = prepare_features(init_embeddings)  # [N, S, D]
                X = emb.reshape(-1, emb.size(-1))
            elif init_embeddings.dim() == 3:  # [N, S, D]
                X = init_embeddings.reshape(-1, init_embeddings.size(-1))
            elif init_embeddings.dim() == 2:  # [N, D]
                X = init_embeddings
            else:
                raise ValueError(f"Unsupported init_embeddings shape: {tuple(init_embeddings.shape)}")
            X = X.to(device=device, dtype=dtype)
            # 余弦模式下通常先做 L2 归一化，减少尺度影响
            if self.normalize and self.distance == "cosine":
                X = normalize_l2(X)

            # 根据策略初始化原型
            if init_strategy == "kmeans":
                centers = kmeans_pp_init(X, k=min(self.M, X.size(0)), seed=seed)  # [k,D]
                if centers.size(0) < self.M:
                    # 样本不足：重复+微扰补齐
                    reps = []
                    for i in range(self.M):
                        reps.append(centers[i % centers.size(0)] + 0.01 * torch.randn_like(centers[0]))
                    centers = torch.stack(reps, dim=0)
                proto = centers[: self.M]
            elif init_strategy == "mean":
                mean = X.mean(dim=0, keepdim=True)
                proto = mean.repeat(self.M, 1) + 0.01 * torch.randn(self.M, X.size(1), device=device, dtype=dtype)
            elif init_strategy == "random":
                proto = init_scale * torch.randn(self.M, self.D, device=device, dtype=dtype)
            else:
                raise ValueError(f"Unknown init_strategy: {init_strategy}")
        else:
            # 无支持数据：随机初始化
            proto = init_scale * torch.randn(self.M, self.D, device=device, dtype=dtype)

        # 注册为可学习参数
        p = nn.Parameter(proto)
        self.prototype_params.append(p)
        self.class_names.append(class_name)
        return len(self.class_names) - 1

    def update_class_prototypes(
        self,
        class_id: str,
        new_embeddings: torch.Tensor,  # 同 add_class 的 init_embeddings
        strategy: str = "kmeans",
        seed: Optional[int] = None,
    ) -> None:
        level, _ = parse_hier_id(class_id)
        if level != self.level:
            raise ValueError(f"class_id {class_id} not in this level L{self.level}")
        cidx = self.class_index(class_id)

        # [N_total, D]
        if new_embeddings.dim() >= 3:
            emb = prepare_features(new_embeddings)
            X = emb.reshape(-1, emb.size(-1))
        else:
            X = new_embeddings.reshape(-1, new_embeddings.size(-1))

        X = X.to(next(self.parameters()).device)
        if self.normalize and self.distance == "cosine":
            X = normalize_l2(X)

        if strategy == "kmeans":
            centers = kmeans_pp_init(X, k=self.M, seed=seed)  # [M,D]
        elif strategy == "mean":
            centers = X.mean(dim=0, keepdim=True).repeat(self.M, 1) + 0.01 * torch.randn_like(X[: self.M])
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        with torch.no_grad():
            self.prototype_params[cidx].copy_(centers)

    def _stack_prototypes(self) -> Optional[torch.Tensor]:
        """
        将 ParameterList 堆叠成 [C, M, D] 的原型张量。
        若该层级当前无类，返回 None。
        """
        if self.num_classes() == 0:
            return None
        return torch.stack([p for p in self.prototype_params], dim=0)  # [C,M,D]

    # --- 聚合策略 ---
    def _aggregate(self, sim_bscm: torch.Tensor) -> torch.Tensor:
        """
        将 输入的每个空间位置与每个原型的相似度（[B, S, C, M]） 相似度按两步聚合成 [B,C]：
        1) 原型维 M 聚合（proto_aggregator）
        2) 空间维 S 聚合（spatial_aggregator）
        """
        # Step 1: 原型聚合（在每个空间位置上，把该类的 M 个原型合并为单一得分）
        # 原型聚合 -> [B,S,C]
        if self.proto_aggregator == "max":
            s_bsc, _ = sim_bscm.max(dim=3) # [B,S,C]
        elif self.proto_aggregator == "mean":
            s_bsc = sim_bscm.mean(dim=3)
        elif self.proto_aggregator == "lse":
            gamma = max(self.lse_gamma, 1e-6)
            s_bsc = (1.0 / gamma) * torch.logsumexp(gamma * sim_bscm, dim=3)
        else:
            raise ValueError(f"Unknown proto_aggregator: {self.proto_aggregator}")

        # Step 2: 空间聚合（把 S 个空间位置的证据合并为图像级得分
        if self.spatial_aggregator == "max":
            s_bc, _ = s_bsc.max(dim=1)
        elif self.spatial_aggregator == "mean":
            s_bc = s_bsc.mean(dim=1)
        elif self.spatial_aggregator == "lse":
            gamma = max(self.lse_gamma, 1e-6)
            s_bc = (1.0 / gamma) * torch.logsumexp(gamma * s_bsc, dim=1)
        else:
            raise ValueError(f"Unknown spatial_aggregator: {self.spatial_aggregator}")

        return s_bc # [B,C]

    def forward(self, Z_bsd: torch.Tensor) -> torch.Tensor:
        """
        输入：
        - Z_bsd: [B,S,D]（假定外部已用 prepare_features 统一好形状）

        输出：
        - logits: [B, C_level]（该层级的图像级分类logits）
        """
        C = self.num_classes()
        if C == 0:
            # 若该层级尚无类，返回空形状的张量，避免打断上游（外部需自行跳过）
            return Z_bsd.new_zeros((Z_bsd.size(0), 0))
        prototypes = self._stack_prototypes()  # [C,M,D]
        # 计算每个位置与每个原型的相似度
        sim = pairwise_sim_spatial(
            Z_bsd, prototypes,
            distance=self.distance, temperature=self.temperature, normalize=self.normalize
        )  # [B,S,C,M]

        # 两步聚合 -> 图像级得分 [B,C]
        s_bc = self._aggregate(sim)  # [B,C]
        # 线性缩放+偏置，得到 logits
        logits = self.logit_scale * s_bc + self.logit_bias
        return logits


# =========================
# ProtoNetHead：可挂载到 backbone.heads['protonet'][head_name]
# =========================

class ProtoNetHead(nn.Module):
    """
    可作为“一个 head”挂到 backbone 的 heads['protonet'][head_name] 中。
    它内含一个 LevelProtoHead，并在前向中自动将输入转换为 [B,S,D]。
    """
    def __init__(
        self,
        head_name: str,         # e.g., 'L1' 或任意自定义名
        feature_dim: int,
        prototypes_per_class: int = 3,
        distance: str = "euclidean",
        temperature: float = 1.0,
        normalize: bool = False,
        proto_aggregator: str = "max",
        spatial_aggregator: str = "max",
        lse_gamma: float = 1.0,
    ):
        super().__init__()
        # 尝试从 head_name 推断层级（如 'L1' -> level=1），否则默认 level=1（不影响功能）
        m = _LEVEL_NAME_RE.match(head_name)
        level = int(m.group(1)) if m else 1

        self.head_name = head_name
        self.feature_dim = feature_dim

        self.level_head = LevelProtoHead(
            level=level,
            feature_dim=feature_dim,
            prototypes_per_class=prototypes_per_class,
            distance=distance,
            temperature=temperature,
            normalize=normalize,
            proto_aggregator=proto_aggregator,
            spatial_aggregator=spatial_aggregator,
            lse_gamma=lse_gamma,
        )

    # ---- Head API ----
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        输入：features 支持 [B,D] / [B,S,D] / [B,H,W,D] / [B,D,H,W]
        输出：该 head 的 logits [B, C]
        """
        Z_bsd = prepare_features(features)
        return self.level_head(Z_bsd)

    def num_classes(self) -> int:
        return self.level_head.num_classes()

    def class_names(self) -> List[str]:
        return self.level_head.class_names_list()

    def add_classes(
        self,
        class_ids: Iterable[str],
        init_embeddings: Optional[Dict[str, torch.Tensor]] = None,
        init_strategy: str = "kmeans",
        init_scale: float = 0.02,
        seed: Optional[int] = None,
    ) -> Dict[str, int]:
        """
        增量添加一批类别到该 head 对应的层级。
        返回 {class_id -> internal_index}
        """
        init_embeddings = init_embeddings or {}
        out: Dict[str, int] = {}
        for cid in class_ids:
            idx = self.level_head.add_class(
                class_name=cid,
                init_embeddings=init_embeddings.get(cid, None),
                init_strategy=init_strategy,
                init_scale=init_scale,
                seed=seed,
            )
            out[cid] = idx
        return out

    def update_class_prototypes(
        self,
        class_id: str,
        new_embeddings: torch.Tensor,
        strategy: str = "kmeans",
        seed: Optional[int] = None,
    ) -> None:
        self.level_head.update_class_prototypes(class_id, new_embeddings, strategy=strategy, seed=seed)

    # ---- 便捷：按 num_classes 自动填充类（用于与现有 add_head(num_classes=...) 对齐）----
    def autofill_classes(self, num_classes: int) -> None:
        """
        为了与 add_head(head_name, num_classes=K) 对齐：
        - 若 head_name 形如 'Lk'，则自动生成类ID：['Lk_0', 'Lk_1', ..., 'Lk_{K-1}']
        - 否则生成 ['{head_name}_0', '{head_name}_1', ...]
        注意：原型会随机初始化；若你有支持样本，请用 add_classes(...) 传入 init_embeddings 覆盖。
        """
        if num_classes <= 0:
            return
        m = _LEVEL_NAME_RE.match(self.head_name)
        prefix = f"L{m.group(1)}" if m else self.head_name
        to_add = [f"{prefix}_{i}" for i in range(num_classes)]
        self.add_classes(to_add, init_embeddings=None, init_strategy="random")

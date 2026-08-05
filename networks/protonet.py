import re
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------- 基础工具 ----------

_LEVEL_NAME_RE = re.compile(r"^L(\d+)$")

def normalize_l2(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    对最后一维做 L2 归一化，避免除零问题。
    输入形状可为 [*, D]
    """
    return x / (x.norm(dim=-1, keepdim=True) + eps)

def prepare_features(x: torch.Tensor) -> torch.Tensor:
    """
    统一特征到 [B, S, D] 的形状，S 为空间位置数/patch 数。
    支持：
    - [B, D]         -> [B, 1, D]
    - [B, S, D]      -> [B, S, D]
    - [B, H, W, D]   -> [B, H*W, D]
    - [B, D, H, W]   -> [B, H*W, D]（自动 NCHW->NHWD）
    """
    if x.dim() == 2:
        return x.unsqueeze(1)                  # [B,1,D]
    if x.dim() == 3:
        return x                               # [B,S,D]
    if x.dim() == 4:
        # 简单启发：若第二维像“通道”，转为 NHWD
        if x.size(1) >= 256 and x.size(1) not in (x.size(2), x.size(3)):
            x = x.permute(0, 2, 3, 1).contiguous()  # [B,H,W,D]
        B, H, W, D = x.shape
        return x.view(B, H * W, D)             # [B,S=H*W,D]
    raise ValueError(f"Unsupported feature shape: {tuple(x.shape)}")


# ---------- ProtoPNet 激活（仅欧氏距离） ----------

def _pairwise_sqdist_spatial(
    Z_bsd: torch.Tensor,   # [B, S, D]
    P_cmd: torch.Tensor,   # [C, M, D]
) -> torch.Tensor:
    """
    返回 d2[b,s,c,m] = ||Z[b,s,:] - P[c,m,:]||_2^2
    形状：[B, S, C, M]
    """
    z2 = (Z_bsd ** 2).sum(dim=-1, keepdim=True)                # [B,S,1]
    p2 = (P_cmd ** 2).sum(dim=-1)                              # [C,M]
    cross = torch.einsum("bsd,cmd->bscm", Z_bsd, P_cmd)        # [B,S,C,M]
    d2 = z2[..., None] + p2[None, None, :, :] - 2.0 * cross
    return d2.clamp_min_(0.0)

def _protopnet_activation_from_d2(
    d2_bscm: torch.Tensor,  # [B, S, C, M]
    eps: float = 1e-4
) -> torch.Tensor:
    """
    g_p(z) = log((d2 + 1) / (d2 + eps))，对 d2 单调递减
    输出形状同输入：[B, S, C, M]
    """
    eps = float(max(min(eps, 0.99), 1e-8))
    return torch.log((d2_bscm + 1.0) / (d2_bscm + eps))


# ---------- Add-on layers f_A：两层 1×1 conv + ReLU + Sigmoid ----------

class AddOnLayers(nn.Module):
    """
    输入（来自 backbone 的卷积特征）: x [B, C_in, H, W]
    输出: a [B, D, H, W]，D=out_dim，末端经 Sigmoid∈[0,1]
    """
    def __init__(self, in_channels: int, mid_channels: int, out_dim: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(mid_channels, out_dim, kernel_size=1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,C_in,H,W]
        x = self.conv1(x)      # [B,mid,H,W]
        x = self.relu(x)       # [B,mid,H,W]
        x = self.conv2(x)      # [B,D,H,W]
        x = self.sigmoid(x)    # [B,D,H,W]
        return x


# ---------- 单一 ProtoNetHead（含 f_A、固定最后一层、原型一次性初始化） ----------

class ProtoNetHead(nn.Module):
    """
    该 Head 集成：
    - f_A（两个 1×1 卷积 + ReLU + Sigmoid）可训练；
    - 原型参数 P[c,m,:]（一次性随机初始化，训练中参与梯度更新）；
    - 固定的最后一层 h（assigned_1_0），不参与训练。

    输入：
    - 若 x 为 [B, C_in, H, W]：先过 f_A -> [B, D, H, W] -> reshape 为 Z [B, S, D]
    - 若 x 为 [B, S, D] / [B, D] / [B, H, W, D]：视为已经是 f_A 之后的特征，直接 prepare_features -> Z [B, S, D]

    输出：
    - logits [B, C]（与原代码保持一致）
    """
    def __init__(
        self,
        head_name: str,           # e.g. 'L1'
        in_channels: int,         # f_A 的输入通道（来自 backbone）
        feature_dim: int,         # D：原型/特征维度，也是 f_A 的输出通道
        num_classes: int,         # C：类别数
        prototypes_per_class: int = 3,  # M：每类原型数
        mid_channels: int = 256,  # f_A 中间通道数
        init_scale: float = 0.02, # 原型随机初始化尺度
        ppnet_eps: float = 1e-4,  # g_p 的 ε
    ):
        super().__init__()
        # 解析层级（可选，仅用于命名/日志）
        m = _LEVEL_NAME_RE.match(head_name)
        self.level = int(m.group(1)) if m else 1
        self.head_name = head_name

        # 结构超参
        self.C = int(num_classes)           # 类别数
        self.M = int(prototypes_per_class)  # 每类原型数
        self.D = int(feature_dim)           # 特征维度
        self.P = self.C * self.M            # 原型总数
        self.eps = ppnet_eps

        # f_A：两个 1×1 conv + ReLU + Sigmoid（可训练）
        self.add_on = AddOnLayers(in_channels=in_channels, mid_channels=mid_channels, out_dim=self.D)

        # 原型参数：P[c,m,:]，形状 [C, M, D]，一次性随机初始化，训练可更新
        proto = init_scale * torch.randn(self.C, self.M, self.D)
        self.prototypes = nn.Parameter(proto)  # [C,M,D]

        # 固定的最后一层 h：Linear(P -> C)，权重为 assigned_1_0（属于该类=1，否则=0）
        # 注意：不训练该层参数（requires_grad=False）
        self.last_layer = nn.Linear(self.P, self.C, bias=False)
        with torch.no_grad():
            W = torch.zeros(self.C, self.P)   # [C,P]
            for c in range(self.C):
                W[c, c * self.M:(c + 1) * self.M] = 1.0
            self.last_layer.weight.copy_(W)
        for p in self.last_layer.parameters():
            p.requires_grad = False

        # 可学习的全局缩放与偏置（不改变 h 的结构连接）
        self.logit_scale = nn.Parameter(torch.tensor(1.0))
        self.logit_bias = nn.Parameter(torch.tensor(0.0))

        # 便于可视化/正则化的原型归属矩阵（P×C），属于类为 1，否则 0
        pid = torch.zeros(self.P, self.C)     # [P,C]
        for c in range(self.C):
            pid[c * self.M:(c + 1) * self.M, c] = 1.0
        self.register_buffer("prototype_class_identity", pid)  # [P,C]

    # ---- 内部：将任意输入转为 Z [B,S,D] ----
    def _to_Z(self, x: torch.Tensor) -> torch.Tensor:
        """
        若 x 是 [B,C_in,H,W]：先过 f_A -> [B,D,H,W] -> permute -> [B,H,W,D] -> reshape -> [B,S,D]
        否则走 prepare_features 统一为 [B,S,D]。
        """
        if x.dim() == 4 and x.size(1) != self.D:
            # print(x.dim(), x.size(1), self.D)
            # 认为是 backbone 的卷积特征
            a = self.add_on(x)                       # [B,D,H,W]
            a = a.permute(0, 2, 3, 1).contiguous()   # [B,H,W,D]
            Z = prepare_features(a)                  # [B,S=H*W,D]
            return Z
        else:
            # 已是 f_A 后的特征（或用户自备特征）
            return prepare_features(x)               # [B,S,D]

    # ---- 前向 ----
    def forward(self, x: torch.Tensor, return_protofeatures=False) -> torch.Tensor:
        """
        输入：
          - x: [B,C_in,H,W] 或 [B,S,D] 或 [B,D] 或 [B,H,W,D]
        输出：
          - logits: [B, C]
        """
        # 1) 得到 Z（若是卷积特征，则内含 f_A；否则直接统一为 [B,S,D]）
        Z_bsd = self._to_Z(x)                        # [B,S,D]
        B, S, D = Z_bsd.shape                        # B=batch, S=位置数, D=feature_dim
        assert D == self.D, f"Feature dim mismatch: got {D}, expect {self.D}"

        # 2) 计算到所有原型的平方欧氏距离
        P_cmd = self.prototypes                      # [C,M,D]
        d2_bscm = _pairwise_sqdist_spatial(Z_bsd, P_cmd)  # [B,S,C,M]

        # 3) ProtoPNet 激活 g_p(d2)
        act_bscm = _protopnet_activation_from_d2(d2_bscm, eps=self.eps)  # [B,S,C,M]

        # 4) 每个原型在空间维取 max，得到该原型对整张图的响应
        proto_act_bcm, _ = act_bscm.max(dim=1)       # [B,C,M]

        # 5) 展平成 [B,P]，通过固定的最后一层 h（assigned_1_0）汇总到类别 logits
        proto_act_bp = proto_act_bcm.view(B, self.P) # [B,P=C*M]
        logits = self.last_layer(proto_act_bp)       # [B,C]
        logits = self.logit_scale * logits + self.logit_bias  # [B,C]
        if return_protofeatures:
            return logits, Z_bsd
        else:
            return logits
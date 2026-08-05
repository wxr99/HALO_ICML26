import torch
import torch.nn.functional as F

def temperature_scaled_softmax_list(logits_list, temp_list):
    # logits_list: List[[B,C]], temp_list: List[scalar]
    probs = []
    for z, t in zip(logits_list, temp_list):
        t = float(t)
        probs.append(F.softmax(z / t, dim=-1) if t != 0 else F.softmax(z, dim=-1))
    return probs

def weighted_combine_probs(probs_a_list, probs_b_list, w_a_list, w_b_list):
    # elementwise: p = w_a * p_a + w_b * p_b, 然后归一化
    out = []
    for p_a, p_b, w_a, w_b in zip(probs_a_list, probs_b_list, w_a_list, w_b_list):
        p = float(w_a) * p_a + float(w_b) * p_b
        p = p / p.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        out.append(p)
    return out

def log_prob_fuse(p_la: torch.Tensor, z_proto: torch.Tensor, alpha=1.0, beta=0.5) -> torch.Tensor:
    # p_la: [B,C] 概率（已是 linear+analytic 融合）
    # z_proto: [B,C] logits（proto 引导或未引导）
    log_p_la = (p_la + 1e-12).log()
    log_p_pr = F.log_softmax(z_proto, dim=-1)
    log_p = alpha * log_p_la + beta * log_p_pr
    # 返回概率形式，便于与你现有 outputs_dict 对齐
    return log_p.exp()

def js_divergence(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    m = 0.5 * (p + q)
    kl_pm = torch.sum(p * (p.add(eps).log() - m.add(eps).log()), dim=-1)
    kl_qm = torch.sum(q * (q.add(eps).log() - m.add(eps).log()), dim=-1)
    return 0.5 * (kl_pm + kl_qm)

def cosine_sim(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    num = (p * q).sum(dim=-1)
    den = (p.norm(dim=-1) * q.norm(dim=-1)).clamp_min(eps)
    return num / den

def top1_margin(p: torch.Tensor) -> torch.Tensor:
    vals, _ = torch.topk(p, 2, dim=-1)
    return vals[:, 0] - vals[:, 1]


def entropy(p: torch.Tensor, eps=1e-12) -> torch.Tensor:
    # p: [B,C]
    return -(p.clamp_min(eps) * p.clamp_min(eps).log()).sum(dim=-1)


def topk_masked_geom_mean(p_base: torch.Tensor, p_aux: torch.Tensor, k: int, alpha: float, beta: float) -> torch.Tensor:
    # 仅对 base 的 top-k 类做几何均值融合，其它类保持 base 原值
    B, C = p_base.shape
    topk_val, topk_idx = torch.topk(p_base, k=min(k, C), dim=-1)
    log_p_base = (p_base + 1e-12).log()
    log_p_aux  = (p_aux + 1e-12).log()
    log_out = log_p_base.clone()
    # 聚合到一个 mask 上
    mask = torch.zeros_like(p_base, dtype=torch.bool)
    mask.scatter_(1, topk_idx, True)
    # 只在 mask 上融合
    log_out = torch.where(mask, alpha * log_p_base + beta * log_p_aux, log_p_base)
    out = log_out.exp()
    out = out / out.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    return out
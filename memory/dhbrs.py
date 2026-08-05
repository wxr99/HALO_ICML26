import random
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Set

import torch


@dataclass
class _Record:
    x: torch.Tensor
    labels_raw: List[torch.Tensor]      # 原始每粒度单值标签（含 -1）
    score: float
    labels_gc: List[Tuple[int, int]]    # 仅用于内部统计的 (g,c)，过滤掉 -1


class DHBMemory:
    """
    动态粒度/类别均衡的水塘记忆（与给定 ReservoirMemory 的接口兼容）：
    - add_batch(inputs, labels): labels 是长度为 G 的 [B]-shape 张量列表，值域包含 -1。
    - sample(num_samples): 返回 [(input, list_of_labels)]，其中 list_of_labels 为长度 G 的张量标量列表（含 -1），
      以便你的后续 stack 代码可以直接工作。
    - 内部在统计时忽略 -1，仅用实际 (g,c) 做均衡。
    """

    def __init__(
        self,
        max_size: int,
        balance_strength: float = 0.3,
        difficulty_bias: float = 0.0,
        min_keep_prob: float = 0.01,
    ):
        self.max_size = max_size
        self.data: List[_Record] = []
        self.count_total_seen = 0

        # 动态结构：仅跟踪真实出现过的 (g,c)
        self.granularities: Set[int] = set()
        self.classes_by_g: Dict[int, Set[int]] = defaultdict(set)
        self.counts: Dict[int, Dict[int, int]] = defaultdict(lambda: defaultdict(int))
        self.buckets: Dict[int, Dict[int, Set[int]]] = defaultdict(lambda: defaultdict(set))

        # 控制项
        self.balance_strength = float(balance_strength)
        self.difficulty_bias = float(difficulty_bias)
        self.min_keep_prob = float(min_keep_prob)

    @staticmethod
    def _extract_labels_gc_for_sample(labels_per_g: List[torch.Tensor], sample_idx: int) -> Tuple[List[torch.Tensor], List[Tuple[int,int]]]:
        """
        从按粒度列表中提取：
        - labels_raw: 长度 G 的标量张量列表（保留 -1）
        - labels_gc: 过滤 -1 后的 (g,c) 列表，用于内部统计
        """
        labels_raw: List[torch.Tensor] = []
        labels_gc: List[Tuple[int, int]] = []
        for g, t in enumerate(labels_per_g):
            c_tensor = t[sample_idx]
            labels_raw.append(c_tensor)
            c = int(c_tensor.item())
            if c != -1:
                labels_gc.append((g, c))
        # 防御式去重
        labels_gc = list({(g, c) for (g, c) in labels_gc})
        return labels_raw, labels_gc

    def _register(self, idx: int, labels_gc: List[Tuple[int, int]]):
        for g, c in labels_gc:
            self.granularities.add(g)
            self.classes_by_g[g].add(c)
            self.counts[g][c] += 1
            self.buckets[g][c].add(idx)

    def _unregister(self, idx: int, labels_gc: List[Tuple[int, int]]):
        for g, c in labels_gc:
            if c in self.counts[g]:
                self.counts[g][c] -= 1
                if self.counts[g][c] <= 0:
                    self.counts[g].pop(c, None)
                    if c in self.buckets[g]:
                        self.buckets[g][c].discard(idx)
                        if not self.buckets[g][c]:
                            self.buckets[g].pop(c, None)
                    if g in self.classes_by_g and c in self.classes_by_g[g]:
                        self.classes_by_g[g].discard(c)
            if c in self.buckets[g]:
                self.buckets[g][c].discard(idx)

        # 清理空粒度
        empty_g = [g for g in list(self.granularities) if len(self.classes_by_g[g]) == 0]
        for g in empty_g:
            self.granularities.discard(g)
            self.counts.pop(g, None)
            self.buckets.pop(g, None)
            self.classes_by_g.pop(g, None)

    def _compute_bucket_pressure(self) -> Dict[int, Dict[int, float]]:
        """
        pressure[g][c] = n_{g,c} / ideal_g,  ideal_g = N / |C_g|
        """
        N = max(len(self.data), 1)
        pressure: Dict[int, Dict[int, float]] = {}
        for g in self.granularities:
            Cg = max(len(self.classes_by_g[g]), 1)
            ideal = N / Cg
            p_g: Dict[int, float] = {}
            for c in self.classes_by_g[g]:
                n_gc = self.counts[g].get(c, 0)
                p_g[c] = n_gc / max(ideal, 1e-6)
            pressure[g] = p_g
        return pressure

    @staticmethod
    def _compress_score(score: float) -> float:
        s = max(score, 0.0)
        return float(torch.log(torch.tensor(1.0 + s)).item())

    def _keep_probability(self, labels_gc: List[Tuple[int, int]], score: float) -> float:
        pressure = self._compute_bucket_pressure()
        if not labels_gc:
            p_avg = 1.0
        else:
            plist = [pressure.get(g, {}).get(c, 0.0) for g, c in labels_gc]
            p_avg = sum(plist) / len(plist)

        w_bal = (1.0 / max(p_avg, 1e-6)) ** self.balance_strength
        z = self._compress_score(score)
        w_diff = float(torch.exp(torch.tensor(self.difficulty_bias * z)).item())

        raw = w_bal + w_diff - 1.0
        raw = max(raw, 1e-6)
        keep_prob = 1.0 / (1.0 + (1.0 / raw))
        keep_prob = max(self.min_keep_prob, min(keep_prob, 1.0))
        return keep_prob

    def _choose_replacement_index(self) -> int:
        pressure = self._compute_bucket_pressure()
        candidate_indices: Set[int] = set()

        # 聚合所有过饱和桶
        for g in self.granularities:
            over = [(c, pr) for c, pr in pressure[g].items() if pr >= 1.0 and len(self.buckets[g][c]) > 0]
            if not over:
                continue
            classes, weights = zip(*over)
            chosen_c = random.choices(classes, weights=weights, k=1)[0]
            candidate_indices.update(self.buckets[g][chosen_c])

        if not candidate_indices:
            return random.randrange(0, len(self.data))

        candidates = list(candidate_indices)
        eps = 1e-6
        # 偏向替换低分样本
        weights = [1.0 / (eps + (1.0 + max(self.data[i].score, 0.0))) for i in candidates]
        return random.choices(candidates, weights=weights, k=1)[0]

    def add_batch(self, inputs: torch.Tensor, labels: List[torch.Tensor], scores: Optional[torch.Tensor] = None):
        """
        Args:
            inputs: [B, ...]
            labels: 长度为 G 的列表，每个元素是 [B] 张量，值域包含 -1 与稀疏类 id
            scores: [B] 或 None（可传入损失/不确定度），未提供则置 1。
        """
        B = inputs.shape[0]
        assert all(len(t) == B for t in labels), "每个粒度的标签长度必须等于 batch 大小"

        if scores is None:
            scores = torch.ones(B, device=inputs.device)
        else:
            assert scores.shape[0] == B

        for i in range(B):
            self.count_total_seen += 1
            x_i = inputs[i]
            labels_raw_i, labels_gc_i = self._extract_labels_gc_for_sample(labels, i)
            score_i = float(scores[i].detach().cpu().item())

            # 经典水塘基础概率
            base_prob = min(1.0, self.max_size / max(1, self.count_total_seen))
            # 动态均衡/困难度概率
            keep_prob = self._keep_probability(labels_gc_i, score_i)
            # 独立门融合
            accept_prob = 1.0 - (1.0 - base_prob) * (1.0 - keep_prob)

            if random.random() > accept_prob:
                continue

            rec = _Record(x=x_i, labels_raw=labels_raw_i, score=score_i, labels_gc=labels_gc_i)

            if len(self.data) < self.max_size:
                self.data.append(rec)
                new_idx = len(self.data) - 1
                self._register(new_idx, labels_gc_i)
            else:
                rep_idx = self._choose_replacement_index()
                self._unregister(rep_idx, self.data[rep_idx].labels_gc)
                self.data[rep_idx] = rec
                self._register(rep_idx, labels_gc_i)

    def sample(self, num_samples: int):
        """
        返回与 ReservoirMemory 相同的格式：
        List[(input, list_of_labels)]，其中 list_of_labels 为长度 G 的“标量张量”列表（含 -1）。
        这保证你随后：
            memory_inputs = torch.stack([item[0] for item in sampled_data])
            memory_labels_list = [torch.stack([item[1][i] for item in sampled_data]) for i in range(len(sampled_data[0][1]))]
        可以直接工作。
        """
        if not self.data:
            raise ValueError("No data available for sampling.")
        sampled = random.sample(self.data, min(num_samples, len(self.data)))
        return [(rec.x, rec.labels_raw) for rec in sampled]

    # 可选：查看当前计数
    def stats(self) -> Dict[int, Dict[int, int]]:
        return {g: dict(self.counts[g]) for g in self.granularities}
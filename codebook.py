"""Multi-scale patch feature codebook (Sec. 3.3 + Algorithm 1).

Each scale has its own codebook. Features are L2-normalised; merging uses
cosine similarity; spatial hash keys preserve locality for retrieval.

Implemented as `nn.Module` so the codebook persists in `state_dict`
(features and counts are buffers, hash keys are a Python list).
"""
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from fps_grouping import spatial_hash


class PatchCodebook(nn.Module):
    """One scale's codebook. Algorithm 1 of the paper."""

    def __init__(
        self,
        feature_dim: int = 32,
        threshold: float = 0.85,
        hash_grid: int = 32,
        max_entries: int = 4096,
    ):
        super().__init__()
        self.dim = feature_dim
        self.threshold = threshold
        self.hash_grid = hash_grid
        self.max_entries = max_entries

        # Buffers grow during training; we keep a fixed capacity and a size cursor.
        self.register_buffer("features", torch.zeros(max_entries, feature_dim))
        self.register_buffer("counts",   torch.zeros(max_entries))
        self.register_buffer("size",     torch.zeros(1, dtype=torch.long))
        # Hash keys for each entry: kept as a Python list of sets (non-tensor state).
        self.hash_keys: List[set] = [set() for _ in range(max_entries)]

    def reset(self):
        self.features.zero_()
        self.counts.zero_()
        self.size.zero_()
        self.hash_keys = [set() for _ in range(self.max_entries)]

    @torch.no_grad()
    def update(self, feature: torch.Tensor, position: torch.Tensor):
        """Add ONE patch entry. Algorithm 1.

        feature : [D]      raw (unnormalised) patch feature
        position: [3]      patch centre in canonical [-1, 1]^3 space
        """
        f = F.normalize(feature, dim=-1)
        h = int(spatial_hash(position, self.hash_grid).item())
        size = int(self.size.item())
        if size > 0:
            sims = self.features[:size] @ f                  # [size]
            best = int(sims.argmax().item())
            if sims[best] >= self.threshold:
                # Count-weighted merge (Alg.1 line 5).
                n_old = self.counts[best]
                s = sims[best]
                self.features[best] = (n_old * self.features[best] + s * f) / (n_old + s)
                self.features[best] = F.normalize(self.features[best], dim=-1)
                self.counts[best] = n_old + 1.0
                self.hash_keys[best].add(h)
                return
        # New entry
        if size >= self.max_entries:
            return                                           # cap reached
        self.features[size] = f
        self.counts[size] = 1.0
        self.hash_keys[size] = {h}
        self.size += 1

    @torch.no_grad()
    def batch_update(self, features: torch.Tensor, positions: torch.Tensor):
        """Convenience: add many patches in sequence."""
        for f, p in zip(features.detach(), positions.detach()):
            self.update(f, p)

    def query(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """For each query patch feature, return nearest codebook feature + score.

        features: [..., D]
        returns:
          retrieved: [..., D]  L2-normalised matched template feature
          score:     [...]     cosine similarity in [-1, 1]
        """
        size = int(self.size.item())
        if size == 0:
            # Codebook empty (e.g. first batch). Fall back to query itself.
            return F.normalize(features, dim=-1), torch.zeros(features.shape[:-1], device=features.device)

        q = F.normalize(features, dim=-1)
        bank = self.features[:size]                          # [S, D]
        # [..., D] @ [D, S] = [..., S]
        sims = q @ bank.t()
        score, idx = sims.max(dim=-1)                        # [...]
        retrieved = bank[idx]                                # [..., D]
        return retrieved, score


class MultiScaleCodebook(nn.Module):
    """Wrapper holding one PatchCodebook per scale."""

    def __init__(
        self,
        num_scales: int,
        feature_dim: int = 32,
        threshold: float = 0.85,
        hash_grid: int = 32,
        max_entries: int = 4096,
    ):
        super().__init__()
        self.books = nn.ModuleList([
            PatchCodebook(feature_dim, threshold, hash_grid, max_entries)
            for _ in range(num_scales)
        ])

    def reset(self):
        for b in self.books:
            b.reset()

    @torch.no_grad()
    def update_scale(self, scale: int, features: torch.Tensor, positions: torch.Tensor):
        self.books[scale].batch_update(features, positions)

    def query_all(self, features_per_scale: List[torch.Tensor]):
        """Returns lists of (retrieved, score) for each scale."""
        results = []
        for book, feats in zip(self.books, features_per_scale):
            results.append(book.query(feats))
        return results

    @torch.no_grad()
    def select_scale(self, features_per_scale: List[torch.Tensor]) -> int:
        """Eq. (2): the scale whose summed similarities are largest."""
        totals = []
        for book, feats in zip(self.books, features_per_scale):
            _, score = book.query(feats)
            totals.append(score.sum().item())
        return int(max(range(len(totals)), key=lambda i: totals[i]))

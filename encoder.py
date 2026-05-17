"""Self-contained PointNet++-style encoder used as a stand-in for the
pre-trained MinkUNet34C in the paper.

If you have MinkowskiEngine installed, instantiate your own MinkUNet34C and
pass it as `external_encoder` to `HierarchicalAnomalyNet`. This module only
expects an `nn.Module` taking [B, N, 3] and returning [B, N, feature_dim].
"""
import torch
import torch.nn as nn

from fps_grouping import farthest_point_sample, knn_indices, index_points


class SetAbstraction(nn.Module):
    """Sample, group, point-wise MLP, max-pool.  Reduces N → M points."""
    def __init__(self, in_dim: int, out_dim: int, k: int = 16):
        super().__init__()
        self.k = k
        self.mlp = nn.Sequential(
            nn.Linear(in_dim + 3, out_dim), nn.GELU(),
            nn.Linear(out_dim, out_dim),    nn.GELU(),
        )

    def forward(self, xyz, feat, new_xyz):
        # xyz: [B, N, 3], feat: [B, N, C], new_xyz: [B, M, 3]
        idx = knn_indices(new_xyz, xyz, self.k)              # [B, M, k]
        gx = index_points(xyz, idx)                          # [B, M, k, 3]
        gf = index_points(feat, idx)                         # [B, M, k, C]
        rel = gx - new_xyz.unsqueeze(2)
        x = torch.cat([rel, gf], dim=-1)
        x = self.mlp(x)
        return x.max(dim=-2).values                          # [B, M, out_dim]


class FeaturePropagation(nn.Module):
    """Interpolate sparse features back to dense via inverse-distance weighting."""
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, out_dim), nn.GELU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, xyz_dense, xyz_sparse, feat_dense, feat_sparse):
        dist = torch.cdist(xyz_dense, xyz_sparse)            # [B, N, M]
        knn = dist.topk(3, dim=-1, largest=False)
        w = 1.0 / (knn.values + 1e-8)
        w = w / w.sum(-1, keepdim=True)
        interp = (index_points(feat_sparse, knn.indices) * w.unsqueeze(-1)).sum(-2)
        return self.mlp(torch.cat([feat_dense, interp], dim=-1))


class PointEncoder(nn.Module):
    """PointNet++-style encoder, returns 32-D per-point features by default.

    Two SA stages with a single FP back to full resolution. Lightweight on
    purpose — the paper's strength comes from the codebook + patch fusion,
    not the encoder.
    """
    def __init__(self, feature_dim: int = 32, hidden: int = 64):
        super().__init__()
        self.input_mlp = nn.Sequential(
            nn.Linear(3, hidden), nn.GELU(),
            nn.Linear(hidden, hidden),
        )
        self.sa1 = SetAbstraction(hidden,       hidden,     k=16)
        self.sa2 = SetAbstraction(hidden,       hidden * 2, k=16)
        self.fp2 = FeaturePropagation(hidden * 2 + hidden, hidden)
        self.fp1 = FeaturePropagation(hidden     + hidden, hidden)
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, feature_dim),
        )

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        # points: [B, N, 3]
        B, N, _ = points.shape
        f0 = self.input_mlp(points)                          # [B, N, H]

        n1 = max(N // 4, 64)
        n2 = max(N // 16, 32)

        idx1 = farthest_point_sample(points, n1)
        p1 = index_points(points, idx1)
        f1 = self.sa1(points, f0, p1)                        # [B, n1, H]

        idx2 = farthest_point_sample(p1, n2)
        p2 = index_points(p1, idx2)
        f2 = self.sa2(p1, f1, p2)                            # [B, n2, 2H]

        f1_up = self.fp2(p1, p2, f1, f2)                     # [B, n1, H]
        f0_up = self.fp1(points, p1, f0, f1_up)              # [B, N, H]
        return self.head(f0_up)                              # [B, N, feature_dim]

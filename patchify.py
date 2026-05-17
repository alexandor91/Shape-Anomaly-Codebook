"""Adaptive multi-scale FPS patchification (Sec. 3.1)."""
from typing import List, Sequence, Tuple

import torch
import torch.nn as nn

from fps_grouping import farthest_point_sample, knn_indices, index_points


class AdaptivePatchifier(nn.Module):
    """Produces patches at multiple spatial resolutions.

    Each scale is a (num_patches, patch_size) pair. The default in
    config.yaml is the Anomaly-ShapeNet / Real3D-AD setting from Sec. 4.1.
    """

    def __init__(self, scales: Sequence[Tuple[int, int]] = ((32, 64), (64, 32), (192, 8))):
        super().__init__()
        self.scales = list(scales)

    @torch.no_grad()
    def forward(self, points: torch.Tensor) -> List[dict]:
        """points: [B, N, 3]

        Returns a list of length len(scales). Each entry is a dict:
          centers:    [B, M, 3]
          point_idx:  [B, M, K]    (indices into the input cloud)
          patches:    [B, M, K, 3] (grouped 3-D coords)
          local:      [B, M, K, 3] (patches - centers)
          centroid:   [B, M, 3]    (mean of `local`, used as the encoder query)
          num_patches, patch_size : ints
        """
        out = []
        for M, K in self.scales:
            center_idx = farthest_point_sample(points, M)
            centers = index_points(points, center_idx)
            patch_idx = knn_indices(centers, points, K)
            patches = index_points(points, patch_idx)
            local = patches - centers.unsqueeze(2)
            centroid = local.mean(dim=2)                     # [B, M, 3]
            out.append({
                "centers": centers,
                "point_idx": patch_idx,
                "patches": patches,
                "local": local,
                "centroid": centroid,
                "num_patches": M,
                "patch_size": K,
            })
        return out

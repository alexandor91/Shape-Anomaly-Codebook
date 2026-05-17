"""Negative augmentation: six pseudo-anomaly types from Sec. 3.1 / Fig. 6.

  1. sink                — inward normal shift
  2. concavity           — outward normal shift
  3. bulges              — sine-wave normal modulation
  4. holes               — random spherical crop
  5. angle_displacement  — random cube cut-off
  6. plane_missing       — random cylinder cut-off

Each call returns (anomalous_points, gt_offset, gt_mask), where:
  - anomalous_points : [N, 3] same shape as input
  - gt_offset        : [N, 3] vector from anomalous → normal (Sec. 3.6: o_gt = S - S̃)
  - gt_mask          : [N]    1 where the point is anomalous, 0 otherwise
"""
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Sequence

import torch


@dataclass
class AugConfig:
    severities: Sequence[float] = (1e-3, 1e-2, 1e-1)
    types: Sequence[str] = (
        "sink", "concavity", "bulges", "holes",
        "angle_displacement", "plane_missing",
    )


def estimate_normals(points: torch.Tensor, k: int = 20) -> torch.Tensor:
    """Quick PCA-based normals. Returns [N, 3] unit-norm normals."""
    N = points.shape[0]
    dist = torch.cdist(points, points)
    knn = dist.topk(min(k, N), largest=False).indices                # [N, k]
    neighbours = points[knn]                                         # [N, k, 3]
    centred = neighbours - neighbours.mean(1, keepdim=True)
    cov = torch.einsum("nki,nkj->nij", centred, centred) / k         # [N, 3, 3]
    _, eigvecs = torch.linalg.eigh(cov)
    return eigvecs[:, :, 0]                                          # smallest eig


def _random_rotation(device) -> torch.Tensor:
    """Uniform random 3-D rotation matrix (Shoemake's quaternion method)."""
    u1, u2, u3 = torch.rand(3, device=device).tolist()
    q = torch.tensor([
        math.sqrt(1 - u1) * math.sin(2 * math.pi * u2),
        math.sqrt(1 - u1) * math.cos(2 * math.pi * u2),
        math.sqrt(u1)      * math.sin(2 * math.pi * u3),
        math.sqrt(u1)      * math.cos(2 * math.pi * u3),
    ], device=device)
    w, x, y, z = q
    return torch.tensor([
        [1 - 2*y*y - 2*z*z,  2*x*y - 2*z*w,      2*x*z + 2*y*w     ],
        [2*x*y + 2*z*w,      1 - 2*x*x - 2*z*z,  2*y*z - 2*x*w     ],
        [2*x*z - 2*y*w,      2*y*z + 2*x*w,      1 - 2*x*x - 2*y*y ],
    ], device=device)


class NegativeAugmentation:
    """Stateless callable. One call → one pseudo-anomaly sample."""

    def __init__(self, cfg: AugConfig | None = None):
        self.cfg = cfg or AugConfig()

    def __call__(self, points, normals=None, atype=None, severity=None):
        if normals is None:
            normals = estimate_normals(points)
        atype = atype or self.cfg.types[torch.randint(len(self.cfg.types), (1,)).item()]
        severity = severity if severity is not None else \
            self.cfg.severities[torch.randint(len(self.cfg.severities), (1,)).item()]

        fn = getattr(self, f"_{atype}")
        return fn(points, normals, float(severity))

    # ---------- Normal-direction shifts ----------
    def _sink(self, points, normals, sev):
        return self._normal_shift(points, normals, sev, sign=-1)

    def _concavity(self, points, normals, sev):
        return self._normal_shift(points, normals, sev, sign=+1)

    @staticmethod
    def _normal_shift(points, normals, sev, sign):
        """Gaussian-weighted displacement along ±normal."""
        N = points.shape[0]
        c = points[torch.randint(N, (1,))]
        sigma = max(sev * 5.0, 1e-3)
        w = torch.exp(-((points - c) ** 2).sum(-1) / (2 * sigma ** 2))  # [N]
        displacement = sign * sev * w.unsqueeze(-1) * normals            # [N, 3]
        anomaly = points + displacement
        gt_offset = -displacement                                        # back-shift
        gt_mask = (w > 0.1).float()
        return anomaly, gt_offset, gt_mask

    # ---------- Sine-wave bulges ----------
    @staticmethod
    def _bulges(points, normals, sev):
        N = points.shape[0]
        c = points[torch.randint(N, (1,))]
        sigma = max(sev * 5.0, 1e-3)
        d = (points - c).norm(dim=-1)
        w = torch.exp(-d ** 2 / (2 * sigma ** 2))
        amp = sev * torch.sin(d * 20.0)
        displacement = (w * amp).unsqueeze(-1) * normals
        anomaly = points + displacement
        gt_offset = -displacement
        gt_mask = (w > 0.1).float()
        return anomaly, gt_offset, gt_mask

    # ---------- Region removals (holes / cube / cylinder) ----------
    @staticmethod
    def _holes(points, normals, sev):
        """A random spherical region is "broken": points inside are flagged
        anomalous; we push them radially outward to the boundary as the
        synthetic restoration offset."""
        N = points.shape[0]
        c = points[torch.randint(N, (1,))].squeeze(0)
        r = max(sev * 3.0, 1e-3)
        rel = points - c
        d = rel.norm(dim=-1)
        in_region = d < r
        anomaly = points.clone()
        gt_offset = torch.zeros_like(points)
        if in_region.any():
            direction = rel[in_region] / (d[in_region].unsqueeze(-1) + 1e-8)
            gt_offset[in_region] = direction * (r - d[in_region].unsqueeze(-1)) * 0.5
        return anomaly, gt_offset, in_region.float()

    def _angle_displacement(self, points, normals, sev):
        """Random oriented cube cut-off."""
        N = points.shape[0]
        c = points[torch.randint(N, (1,))].squeeze(0)
        R = _random_rotation(points.device)
        size = max(sev * 5.0, 1e-3)
        local = (points - c) @ R
        in_region = (local.abs() < size).all(-1)
        anomaly = points.clone()
        gt_offset = torch.zeros_like(points)
        if in_region.any():
            rel = points[in_region] - c
            direction = rel / (rel.norm(dim=-1, keepdim=True) + 1e-8)
            gt_offset[in_region] = direction * size * 0.5
        return anomaly, gt_offset, in_region.float()

    def _plane_missing(self, points, normals, sev):
        """Random cylinder cut-off (mimics planar/laminar displacement)."""
        N = points.shape[0]
        c = points[torch.randint(N, (1,))].squeeze(0)
        axis = torch.randn(3, device=points.device)
        axis = axis / (axis.norm() + 1e-8)
        r = max(sev * 4.0, 1e-3)
        rel = points - c
        proj = (rel * axis).sum(-1, keepdim=True) * axis
        perp = rel - proj
        d_perp = perp.norm(dim=-1)
        in_region = d_perp < r
        anomaly = points.clone()
        gt_offset = torch.zeros_like(points)
        if in_region.any():
            direction = perp[in_region] / (d_perp[in_region].unsqueeze(-1) + 1e-8)
            gt_offset[in_region] = direction * (r - d_perp[in_region].unsqueeze(-1)) * 0.5
        return anomaly, gt_offset, in_region.float()

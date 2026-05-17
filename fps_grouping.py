"""Farthest-point sampling and k-NN / ball grouping in pure PyTorch.

No CUDA extensions required. For very large point clouds, swap in a fast FPS
library (e.g. torch_cluster.fps) by overriding `farthest_point_sample`.
"""
import torch


@torch.no_grad()
def farthest_point_sample(xyz: torch.Tensor, n_sample: int) -> torch.LongTensor:
    """Returns indices of `n_sample` farthest points.

    xyz: [B, N, 3] or [N, 3]
    return: [B, n_sample] or [n_sample]
    """
    squeeze = xyz.dim() == 2
    if squeeze:
        xyz = xyz.unsqueeze(0)
    B, N, _ = xyz.shape
    device = xyz.device
    centroids = torch.zeros(B, n_sample, dtype=torch.long, device=device)
    distance = torch.full((B, N), float("inf"), device=device)
    farthest = torch.randint(0, N, (B,), dtype=torch.long, device=device)
    batch = torch.arange(B, device=device)
    for i in range(n_sample):
        centroids[:, i] = farthest
        center = xyz[batch, farthest].unsqueeze(1)        # [B, 1, 3]
        dist = ((xyz - center) ** 2).sum(-1)               # [B, N]
        distance = torch.minimum(distance, dist)
        farthest = distance.argmax(-1)
    return centroids.squeeze(0) if squeeze else centroids


def knn_indices(query: torch.Tensor, key: torch.Tensor, k: int) -> torch.LongTensor:
    """Returns k-NN indices from `key` for each `query` point.

    query: [B, M, 3], key: [B, N, 3]
    return: [B, M, k]
    """
    dist = torch.cdist(query, key)                          # [B, M, N]
    return dist.topk(k, dim=-1, largest=False).indices


def index_points(points: torch.Tensor, idx: torch.LongTensor) -> torch.Tensor:
    """Gather points along the N dimension.

    points: [B, N, C], idx: [B, ...]
    return: [B, ..., C]
    """
    B = points.shape[0]
    # Build a batch index broadcastable to `idx`
    batch_idx = torch.arange(B, device=points.device).view(
        B, *([1] * (idx.dim() - 1))
    ).expand_as(idx)
    return points[batch_idx, idx]


def spatial_hash(xyz: torch.Tensor, grid: int = 32, scene_range: float = 1.0) -> torch.LongTensor:
    """Quantise points in [-scene_range, scene_range]^3 to a single integer key.

    xyz: [..., 3]
    return: [...]  int64 hash keys in [0, grid^3).
    """
    normalised = (xyz + scene_range) / (2.0 * scene_range)   # to [0, 1]
    cells = (normalised * grid).long().clamp_(0, grid - 1)
    return cells[..., 0] * (grid * grid) + cells[..., 1] * grid + cells[..., 2]

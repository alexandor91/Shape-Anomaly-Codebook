"""Point-cloud dataset for shape anomaly detection.

Expected layout (per the README):
    data_root/<class_name>/train/normal/*.npy
    data_root/<class_name>/test/good/*.npy
    data_root/<class_name>/test/anomaly/*.npy
    data_root/<class_name>/test/anomaly/*_mask.npy   (per-point binary labels)

Supports `.npy` and `.ply`. Points are loaded as float32 [N, 3] (or [N, 6] with
the trailing 3 dims interpreted as normals).
"""
from __future__ import annotations
import os
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch.utils.data import Dataset


def _load_npy_or_ply(path: Path) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        arr = np.load(path).astype(np.float32)
    elif suffix == ".ply":
        from plyfile import PlyData
        ply = PlyData.read(str(path))
        v = ply["vertex"].data
        xyz = np.stack([v["x"], v["y"], v["z"]], axis=-1).astype(np.float32)
        if all(c in v.dtype.names for c in ("nx", "ny", "nz")):
            nrm = np.stack([v["nx"], v["ny"], v["nz"]], axis=-1).astype(np.float32)
            arr = np.concatenate([xyz, nrm], axis=-1)
        else:
            arr = xyz
    else:
        raise ValueError(f"Unsupported file: {path}")
    return arr


def normalise_unit_sphere(points: np.ndarray) -> np.ndarray:
    """Center and rescale into unit sphere. Operates only on the xyz columns."""
    xyz = points[:, :3]
    centre = xyz.mean(0, keepdims=True)
    xyz = xyz - centre
    scale = np.linalg.norm(xyz, axis=-1).max() + 1e-8
    xyz = xyz / scale
    if points.shape[-1] > 3:
        return np.concatenate([xyz, points[:, 3:]], axis=-1)
    return xyz


def uniform_sample(points: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    N = points.shape[0]
    if N == n:
        return points
    replace = N < n
    idx = rng.choice(N, size=n, replace=replace)
    return points[idx]


class PointCloudAnomalyDataset(Dataset):
    """Loads one (class_name, split) pair.

    split = 'train_normal' | 'test_good' | 'test_anomaly'
    """

    def __init__(
        self,
        root: str,
        class_name: str,
        split: Literal["train_normal", "test_good", "test_anomaly"],
        num_points: int = 10_000,
        normalize: bool = True,
        seed: int = 0,
    ):
        self.root = Path(root) / class_name
        self.num_points = num_points
        self.normalize = normalize
        self.rng = np.random.default_rng(seed)

        if split == "train_normal":
            d = self.root / "train" / "normal"
        elif split == "test_good":
            d = self.root / "test" / "good"
        elif split == "test_anomaly":
            d = self.root / "test" / "anomaly"
        else:
            raise ValueError(split)

        self.split = split
        self.files = sorted(
            f for f in d.glob("*")
            if f.suffix.lower() in (".npy", ".ply") and not f.stem.endswith("_mask")
        )
        if not self.files:
            raise FileNotFoundError(f"No point clouds under {d}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> dict:
        path = self.files[idx]
        pts = _load_npy_or_ply(path)
        if self.normalize:
            pts = normalise_unit_sphere(pts)
        pts = uniform_sample(pts, self.num_points, self.rng)

        xyz = torch.from_numpy(pts[:, :3]).float()
        normals = torch.from_numpy(pts[:, 3:6]).float() if pts.shape[-1] >= 6 else None

        sample = {"xyz": xyz, "name": path.stem, "class": self.root.name}
        if normals is not None:
            sample["normals"] = normals

        if self.split == "test_anomaly":
            mask_path = path.with_name(path.stem + "_mask.npy")
            if mask_path.exists():
                sample["gt_mask"] = torch.from_numpy(np.load(mask_path).astype(np.float32))
        return sample

"""Training entry point.

Usage:
  python train.py --config config.yaml --class_name airplane --data_root /path/to/data

Per-class training (one model + one codebook per object class), matching
Real3D-AD / Anomaly-ShapeNet protocol in Sec. 4.1.
"""
import argparse
import os
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import PointCloudAnomalyDataset
from augmentation import NegativeAugmentation, AugConfig, estimate_normals
from network import HierarchicalAnomalyNet
from losses import AnomalyLoss


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--class_name", required=True)
    p.add_argument("--data_root", required=True)
    p.add_argument("--out_dir", default="runs")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def build_model(cfg) -> HierarchicalAnomalyNet:
    return HierarchicalAnomalyNet(
        scales=[tuple(s) for s in cfg["patches"]["scales"]],
        feature_dim=cfg["model"]["feature_dim"],
        attention_heads=cfg["model"]["attention_heads"],
        attention_head_dim=cfg["model"]["attention_head_dim"],
        codebook_threshold=cfg["codebook"]["threshold"],
        codebook_hash_grid=cfg["codebook"]["hash_grid"],
        codebook_max_entries=cfg["codebook"]["max_entries"],
    )


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    cfg = yaml.safe_load(open(args.config))
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    out_dir = Path(args.out_dir) / args.class_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ data
    train_set = PointCloudAnomalyDataset(
        root=args.data_root,
        class_name=args.class_name,
        split="train_normal",
        num_points=cfg["data"]["num_points"],
        normalize=cfg["data"]["normalize"],
        seed=args.seed,
    )
    loader = DataLoader(
        train_set,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        num_workers=2,
        drop_last=False,
    )

    # ------------------------------------------------------------------ model
    model = build_model(cfg).to(device)
    augment = NegativeAugmentation(AugConfig(
        severities=tuple(cfg["augmentation"]["severities"]),
        types=tuple(cfg["augmentation"]["types"]),
    ))
    loss_fn = AnomalyLoss(
        lambda_sim=cfg["loss"]["lambda_sim"],
        lambda_bce=cfg["loss"]["lambda_bce"],
        eps=cfg["loss"]["eps"],
    )
    optim = torch.optim.Adam(model.parameters(), lr=cfg["train"]["lr"])

    # ====================================================== Phase 1: codebook
    print("Phase 1 — populating codebook from normal samples")
    model.codebook.reset()
    warmup_epochs = cfg["train"]["codebook_warmup_epochs"]
    for ep in range(warmup_epochs):
        for batch in tqdm(loader, desc=f"warmup {ep + 1}/{warmup_epochs}"):
            xyz = batch["xyz"].to(device)
            model.update_codebook(xyz)
    for s, book in enumerate(model.codebook.books):
        print(f"  scale {s}: {int(book.size.item())} codebook entries")

    # ====================================================== Phase 2: training
    print("Phase 2 — training with negative augmentation")
    model.train()
    n_epochs = cfg["train"]["epochs"]
    save_every = cfg["train"]["save_every"]
    best_loss = float("inf")
    for ep in range(n_epochs):
        running = {"L_dist": 0.0, "L_sim": 0.0, "L_bce": 0.0, "total": 0.0}
        n_batches = 0
        for batch in tqdm(loader, desc=f"epoch {ep + 1}/{n_epochs}"):
            xyz = batch["xyz"].to(device)
            B, N, _ = xyz.shape

            # Build batch of pseudo-anomalies (one per sample)
            anom_pts, gt_off, gt_mask = [], [], []
            normals = batch.get("normals")
            if normals is not None:
                normals = normals.to(device)
            for b in range(B):
                n_b = normals[b] if normals is not None else estimate_normals(xyz[b])
                a, o, m = augment(xyz[b], n_b)
                anom_pts.append(a); gt_off.append(o); gt_mask.append(m)
            anom_pts = torch.stack(anom_pts, 0)
            gt_off   = torch.stack(gt_off, 0)
            gt_mask  = torch.stack(gt_mask, 0)

            # Refresh codebook with the current normal sample (continuous update).
            model.update_codebook(xyz)

            offset, logits, _ = model(anom_pts)
            loss, parts = loss_fn(offset, logits, gt_off, gt_mask)

            optim.zero_grad()
            loss.backward()
            optim.step()

            for k, v in parts.items():
                running[k] += v
            running["total"] += loss.item()
            n_batches += 1

        for k in running:
            running[k] /= max(n_batches, 1)
        print(f"  ep {ep + 1}: " + ", ".join(f"{k}={v:.4f}" for k, v in running.items()))

        if running["total"] < best_loss:
            best_loss = running["total"]
            torch.save({
                "model": model.state_dict(),
                "config": cfg,
                "class_name": args.class_name,
                "epoch": ep + 1,
            }, out_dir / "best.pt")

        if (ep + 1) % save_every == 0:
            torch.save({
                "model": model.state_dict(),
                "config": cfg,
                "class_name": args.class_name,
                "epoch": ep + 1,
            }, out_dir / f"epoch_{ep + 1:04d}.pt")

    print(f"Done. Best loss {best_loss:.4f}. Checkpoints in {out_dir}")


if __name__ == "__main__":
    main()

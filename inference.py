"""Inference and evaluation.

  Single-sample mode:
    python inference.py --checkpoint runs/airplane/best.pt --input shape.npy --output preds.npy

  Whole-test-set mode (computes AUC-ROC / AUC-PR):
    python inference.py --checkpoint runs/airplane/best.pt --data_root /path/to/data
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from tqdm import tqdm

from dataset import PointCloudAnomalyDataset, normalise_unit_sphere, uniform_sample
from losses import offset_to_score
from network import HierarchicalAnomalyNet
from train import build_model

try:
    from sklearn.metrics import roc_auc_score, average_precision_score
except ImportError:
    roc_auc_score = average_precision_score = None


def load_checkpoint(path: str, device: torch.device):
    ckpt = torch.load(path, map_location=device)
    model = build_model(ckpt["config"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt["config"], ckpt["class_name"]


def predict_single(model, xyz: torch.Tensor, top_k: int, score_thr: float):
    """xyz: [N, 3] on the model's device.  Returns dict with scores + detections."""
    with torch.no_grad():
        offset, logits, aux = model(xyz.unsqueeze(0))         # [1, N, 3], [1, N]
        score = offset_to_score(offset, logits).squeeze(0)    # [N]

    # Top-K stable detections (Sec. 7.2)
    sorted_idx = score.argsort(descending=True)
    keep_topk = sorted_idx[:top_k]

    # Threshold-filtered points
    keep_thr = (score >= score_thr).nonzero(as_tuple=True)[0]

    return {
        "score": score.cpu().numpy(),
        "offset": offset.squeeze(0).cpu().numpy(),
        "top_k_indices": keep_topk.cpu().numpy(),
        "above_thr_indices": keep_thr.cpu().numpy(),
        "best_scale": int(aux["best_scale"][0].item()),
    }


def cmd_single(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model, cfg, _ = load_checkpoint(args.checkpoint, device)

    pts = np.load(args.input).astype(np.float32)
    if cfg["data"]["normalize"]:
        pts = normalise_unit_sphere(pts)
    rng = np.random.default_rng(0)
    pts = uniform_sample(pts, cfg["data"]["num_points"], rng)
    xyz = torch.from_numpy(pts[:, :3]).to(device)

    result = predict_single(
        model, xyz,
        top_k=cfg["infer"]["top_k_points"],
        score_thr=cfg["infer"]["score_threshold"],
    )
    np.savez(args.output, **result)
    print(f"Saved predictions to {args.output} "
          f"(top-K={len(result['top_k_indices'])}, "
          f"above-threshold={len(result['above_thr_indices'])})")


def cmd_evaluate(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model, cfg, class_name = load_checkpoint(args.checkpoint, device)

    test_set = PointCloudAnomalyDataset(
        root=args.data_root,
        class_name=class_name,
        split="test_anomaly",
        num_points=cfg["data"]["num_points"],
        normalize=cfg["data"]["normalize"],
    )

    if roc_auc_score is None:
        raise RuntimeError("scikit-learn is required for metric computation")

    point_scores, point_labels = [], []
    object_scores, object_labels = [], []

    for sample in tqdm(test_set, desc=f"eval {class_name}"):
        xyz = sample["xyz"].to(device)
        result = predict_single(
            model, xyz,
            top_k=cfg["infer"]["top_k_points"],
            score_thr=cfg["infer"]["score_threshold"],
        )
        s = result["score"]
        if "gt_mask" in sample:
            g = sample["gt_mask"].cpu().numpy()
            point_scores.append(s); point_labels.append(g)
        # Object-level: max-score aggregation (common in 3D AD benchmarks)
        object_scores.append(float(s.max()))
        object_labels.append(1)  # all anomaly samples

    # Add good samples for object-level metric (label = 0)
    good_set = PointCloudAnomalyDataset(
        root=args.data_root, class_name=class_name, split="test_good",
        num_points=cfg["data"]["num_points"], normalize=cfg["data"]["normalize"],
    )
    for sample in tqdm(good_set, desc=f"eval good {class_name}"):
        xyz = sample["xyz"].to(device)
        result = predict_single(model, xyz,
                                top_k=cfg["infer"]["top_k_points"],
                                score_thr=cfg["infer"]["score_threshold"])
        object_scores.append(float(result["score"].max()))
        object_labels.append(0)

    metrics = {
        "object_AUROC": float(roc_auc_score(object_labels, object_scores)),
        "object_AUPR":  float(average_precision_score(object_labels, object_scores)),
    }
    if point_scores:
        ps = np.concatenate(point_scores)
        pl = np.concatenate(point_labels)
        metrics["point_AUROC"] = float(roc_auc_score(pl, ps))
        metrics["point_AUPR"]  = float(average_precision_score(pl, ps))

    out = Path(args.checkpoint).parent / f"metrics_{class_name}.json"
    json.dump(metrics, open(out, "w"), indent=2)
    print(f"Metrics for {class_name}:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
    print(f"(Saved to {out})")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--input")
    p.add_argument("--output", default="preds.npz")
    p.add_argument("--data_root", help="If given, runs full test-set evaluation")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    if args.data_root:
        cmd_evaluate(args)
    elif args.input:
        cmd_single(args)
    else:
        p.error("Pass --input <file> for single inference or --data_root <dir> for full eval.")


if __name__ == "__main__":
    main()

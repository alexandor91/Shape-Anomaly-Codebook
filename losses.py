"""Combined anomaly loss (Sec. 3.6, Eq. 8–11) + inference scoring.

  L_anomaly = L_dist + λ_sim · L_sim + λ_bce · L_BCE
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class AnomalyLoss(nn.Module):
    def __init__(self, lambda_sim: float = 0.5, lambda_bce: float = 0.5, eps: float = 1e-6):
        super().__init__()
        self.lambda_sim = lambda_sim
        self.lambda_bce = lambda_bce
        self.eps = eps

    def forward(
        self,
        pred_offset:  torch.Tensor,    # [B, N, 3]
        pred_logits:  torch.Tensor,    # [B, N]
        gt_offset:    torch.Tensor,    # [B, N, 3]
        gt_mask:      torch.Tensor,    # [B, N]
    ):
        # L_dist: mean L1 over xyz (Eq. 9)
        l_dist = (pred_offset - gt_offset).abs().mean()

        # L_sim: cosine direction (Eq. 10). Skip points with near-zero GT offset
        # to avoid noisy direction supervision on clean points.
        valid = gt_offset.norm(dim=-1) > 1e-4
        if valid.any():
            cos = F.cosine_similarity(
                pred_offset[valid], gt_offset[valid], dim=-1, eps=self.eps,
            )
            l_sim = -0.5 * (1.0 + cos).mean()
        else:
            l_sim = pred_offset.new_zeros(())

        # L_BCE on the sign / validity mask (Eq. 11)
        l_bce = F.binary_cross_entropy_with_logits(pred_logits, gt_mask)

        total = l_dist + self.lambda_sim * l_sim + self.lambda_bce * l_bce
        return total, {"L_dist": l_dist.item(), "L_sim": l_sim.item(), "L_bce": l_bce.item()}


def offset_to_score(
    pred_offset: torch.Tensor,
    pred_logits: torch.Tensor,
) -> torch.Tensor:
    """Convert predicted offsets + sign mask into per-point anomaly scores in [0,1].

    Follows PO3AD's L1-normalisation strategy (Sec. 3.6 last paragraph):
      - take the L1 magnitude of the offset
      - mask by the predicted sign / validity head
      - per-sample normalise so the strongest offset reaches 1.0
    """
    mag = pred_offset.abs().sum(dim=-1)                       # [B, N]
    gate = torch.sigmoid(pred_logits)                          # [B, N]
    raw = mag * gate
    # Per-batch min-max normalisation to [0, 1]
    flat_min = raw.amin(dim=-1, keepdim=True)
    flat_max = raw.amax(dim=-1, keepdim=True)
    score = (raw - flat_min) / (flat_max - flat_min + 1e-8)
    return score

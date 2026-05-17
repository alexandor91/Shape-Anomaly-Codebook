"""End-to-end HierarchicalAnomalyNet.

Wires the patchifier, encoder, multi-scale codebook, RoPE cross-attention,
and patch score modulation together per Sec. 3.
"""
from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from patchify import AdaptivePatchifier
from encoder import PointEncoder
from codebook import MultiScaleCodebook
from rope_attention import RoPECrossAttention
from modulation import PatchScoreModulation, OffsetHead
from fps_grouping import index_points


class HierarchicalAnomalyNet(nn.Module):
    def __init__(
        self,
        scales=((32, 64), (64, 32), (192, 8)),
        feature_dim: int = 32,
        attention_heads: int = 8,
        attention_head_dim: int = 64,
        codebook_threshold: float = 0.85,
        codebook_hash_grid: int = 32,
        codebook_max_entries: int = 4096,
        external_encoder: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.scales = list(scales)
        self.feature_dim = feature_dim

        self.encoder = external_encoder or PointEncoder(feature_dim=feature_dim)
        self.patchifier = AdaptivePatchifier(scales=scales)
        self.codebook = MultiScaleCodebook(
            num_scales=len(scales),
            feature_dim=feature_dim,
            threshold=codebook_threshold,
            hash_grid=codebook_hash_grid,
            max_entries=codebook_max_entries,
        )
        # Tiny MLP that turns a relative centroid offset (3-D) into a
        # patch-feature query (paper Eq. (1): p_j = f(centroid_offset)).
        # In our implementation we additionally pool encoder features inside
        # the patch and add them — gives the network access to learned context.
        self.patch_encoder = nn.Sequential(
            nn.Linear(3, feature_dim * 2), nn.GELU(),
            nn.Linear(feature_dim * 2, feature_dim),
        )

        self.cross_attn = RoPECrossAttention(
            dim=feature_dim, num_heads=attention_heads, head_dim=attention_head_dim,
        )
        self.modulation = PatchScoreModulation(feature_dim=feature_dim)
        self.head = OffsetHead(feature_dim=feature_dim)

    # --------------------------------------------------------------- patch feats
    def _patch_features(
        self,
        patch_info: List[dict],
        point_feats: torch.Tensor,
    ) -> List[torch.Tensor]:
        """Eq. (1): patch feature from mean point coordinate relative to centre,
        augmented with the average of encoder point features inside the patch.

        Returns: list (one per scale) of [B, M, D] features.
        """
        feats_per_scale = []
        for info in patch_info:
            centroid = info["centroid"]                       # [B, M, 3]
            geom_feat = self.patch_encoder(centroid)          # [B, M, D]
            # Pool encoder features within each patch
            grouped = index_points(point_feats, info["point_idx"])  # [B, M, K, D]
            pooled = grouped.mean(dim=2)                      # [B, M, D]
            feats_per_scale.append(geom_feat + pooled)
        return feats_per_scale

    # ---------------------------------------------------------------- training
    @torch.no_grad()
    def update_codebook(self, normal_points: torch.Tensor):
        """Run the encoder on a normal sample and push its patch features into
        the codebook (Algorithm 1). Call before / during training on normals."""
        self.eval()
        feats = self.encoder(normal_points)                   # [B, N, D]
        patch_info = self.patchifier(normal_points)
        patch_feats = self._patch_features(patch_info, feats)
        for scale_idx, (info, pf) in enumerate(zip(patch_info, patch_feats)):
            # Flatten (B, M, D) over batch
            B, M, D = pf.shape
            self.codebook.update_scale(
                scale_idx,
                pf.reshape(B * M, D),
                info["centers"].reshape(B * M, 3),
            )
        self.train()

    # --------------------------------------------------------------- inference
    def forward(
        self,
        points: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        """
        points: [B, N, 3]

        Returns:
          offset      : [B, N, 3]  predicted ô
          mask_logits : [B, N]
          aux         : dict (selected scale, per-scale similarity sums, ...)
        """
        feats = self.encoder(points)                          # [B, N, D]
        patch_info = self.patchifier(points)
        patch_feats = self._patch_features(patch_info, feats) # list of [B, M_l, D]

        # Codebook retrieval at every scale; pick best via Eq. (2)
        retrieved, scores = [], []
        for book, pf in zip(self.codebook.books, patch_feats):
            r, s = book.query(pf)
            retrieved.append(r)
            scores.append(s.sum(dim=-1))                      # [B]
        # Per-sample best scale
        score_stack = torch.stack(scores, dim=-1)             # [B, L]
        best_scale = score_stack.argmax(dim=-1)               # [B]

        # For simplicity (and matching most-relevant-scale ablation winner in
        # Table 6: 84.2 vs 81.2), we use one scale per batch element.
        B = points.shape[0]
        # Build per-point retrieved-template and per-point anomalous-patch features.
        D = self.feature_dim
        N = points.shape[1]
        t_per_point = torch.zeros(B, N, D, device=points.device, dtype=feats.dtype)
        p_per_point = torch.zeros_like(t_per_point)
        kv_pos      = torch.zeros(B, 0, 3, device=points.device)  # will rebuild per-sample

        # We do a small Python-level loop over batch since each sample may
        # select a different scale (typically B is tiny — Sec. 4.1 uses 1).
        kv_feat_list, kv_pos_list = [], []
        for b in range(B):
            l = int(best_scale[b].item())
            info = patch_info[l]
            pf = patch_feats[l][b]                            # [M, D]
            rf = retrieved[l][b]                              # [M, D]
            point_idx = info["point_idx"][b]                  # [M, K]
            # For each point, find which patch it belongs to (use the nearest centre)
            # Build a (N,) assignment by scattering — most points lie inside one patch.
            assignment = torch.full((N,), -1, dtype=torch.long, device=points.device)
            for j in range(point_idx.shape[0]):
                idx = point_idx[j]
                # Earlier writes can be overwritten by later ones; this just means
                # a point shared between patches takes its last-seen patch.
                assignment[idx] = j
            # Points that ended up unassigned: fall back to the nearest centre.
            unassigned = (assignment == -1)
            if unassigned.any():
                centres = info["centers"][b]                  # [M, 3]
                dists = torch.cdist(points[b][unassigned], centres)
                assignment[unassigned] = dists.argmin(dim=-1)
            t_per_point[b] = rf[assignment]
            p_per_point[b] = pf[assignment]
            kv_feat_list.append(rf)
            kv_pos_list.append(info["centers"][b])

        # Cross-attention (RoPE).  Points attend to retrieved templates at all
        # patch centres of the chosen scale.  We pad to the max # of patches.
        max_M = max(f.shape[0] for f in kv_feat_list)
        kv_feat = torch.zeros(B, max_M, D, device=points.device, dtype=feats.dtype)
        kv_pos  = torch.zeros(B, max_M, 3, device=points.device)
        for b, (f, p) in enumerate(zip(kv_feat_list, kv_pos_list)):
            kv_feat[b, :f.shape[0]] = f
            kv_pos[b, :p.shape[0]] = p

        z_hat = self.cross_attn(feats, points, kv_feat, kv_pos)   # [B, N, D]

        # Modulation + offset head
        z_prime = self.modulation(z_hat, t_per_point, p_per_point)
        offset, mask_logits = self.head(z_prime, z_hat)

        aux = {
            "best_scale": best_scale,
            "scale_scores": score_stack,
        }
        return offset, mask_logits, aux

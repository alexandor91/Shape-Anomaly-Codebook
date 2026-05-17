"""Patch score modulation + offset head (Sec. 3.5, Fig. 11).

  Δf_kj   = 1 - <t_k, p_j>                       patch discrepancy
  ρ_i     = σ(MLP_gate(Δf_kj))                   gate
  γ_i, β_i = MLP_mod(Δf_kj)                      modulation
  z'_i    = ρ_i ⊙ (γ_i ⊙ ẑ_i + β_i)              gated FiLM modulation
  ô_i     = MLP(concat[z'_i, ẑ_i]) + ẑ_i         residual offset head
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchScoreModulation(nn.Module):
    def __init__(self, feature_dim: int = 32, hidden: int = 64):
        super().__init__()
        # Δf is a scalar per patch; we lift it to a small embedding before
        # producing gate / scale / shift to give the network more capacity.
        self.gate_mlp = nn.Sequential(
            nn.Linear(1, hidden), nn.GELU(),
            nn.Linear(hidden, feature_dim),
        )
        self.mod_mlp = nn.Sequential(
            nn.Linear(1, hidden), nn.GELU(),
            nn.Linear(hidden, feature_dim * 2),               # γ ∥ β
        )

    def forward(
        self,
        z_hat: torch.Tensor,        # [B, N, D] cross-attended point features
        t_per_point: torch.Tensor,  # [B, N, D] retrieved normal template per point
        p_per_point: torch.Tensor,  # [B, N, D] anomalous patch feature per point
    ) -> torch.Tensor:
        # Patch discrepancy (Sec. 3.5)
        t = F.normalize(t_per_point, dim=-1)
        p = F.normalize(p_per_point, dim=-1)
        delta = 1.0 - (t * p).sum(dim=-1, keepdim=True)       # [B, N, 1]

        gate = torch.sigmoid(self.gate_mlp(delta))            # [B, N, D]
        gamma, beta = self.mod_mlp(delta).chunk(2, dim=-1)    # each [B, N, D]
        z_prime = gate * (gamma * z_hat + beta)               # Eq. (6)
        return z_prime


class OffsetHead(nn.Module):
    """Eq. (7) + sign-mask head.

    Predicts:
      ô_i :  [B, N, 3]  point-wise anomaly offset (regressed)
      m̂_i: [B, N]      sign / validity mask (binary)
    """

    def __init__(self, feature_dim: int = 32, hidden: int = 64):
        super().__init__()
        self.offset_mlp = nn.Sequential(
            nn.Linear(feature_dim * 2, hidden), nn.GELU(),
            nn.Linear(hidden, hidden),         nn.GELU(),
            nn.Linear(hidden, 3),
        )
        # We project ẑ to 3-D for the residual term o_hat = MLP(...) + proj(ẑ)
        self.residual_proj = nn.Linear(feature_dim, 3, bias=False)

        self.mask_mlp = nn.Sequential(
            nn.Linear(feature_dim * 2, hidden), nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, z_prime: torch.Tensor, z_hat: torch.Tensor):
        fused = torch.cat([z_prime, z_hat], dim=-1)           # [B, N, 2D]
        offset = self.offset_mlp(fused) + self.residual_proj(z_hat)
        mask_logits = self.mask_mlp(fused).squeeze(-1)        # [B, N]
        return offset, mask_logits

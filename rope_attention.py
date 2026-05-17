"""RoPE-enhanced cross-attention (Sec. 3.4 + Sec. 6.5).

  - Rotary position embedding extended to 3-D by splitting the head dim
    into three equal axis blocks (x, y, z).
  - Linear attention with elu(x) + 1 feature maps (Eq. 5; Katharopoulos
    et al., 2020 — ref [13]).
"""
import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def build_rope_freqs(half_dim: int, base: float = 10000.0, device=None) -> torch.Tensor:
    """RoPE frequency vector of length `half_dim`."""
    return 1.0 / (base ** (torch.arange(0, half_dim, device=device).float() / half_dim))


def apply_rope_axis(x: torch.Tensor, pos: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """Rotary embedding along one axis.

    x:     [..., 2 * half_dim]  (pairs of (even, odd) dims)
    pos:   [...]                position along this axis
    freqs: [half_dim]
    """
    half = freqs.shape[-1]
    angles = pos.unsqueeze(-1) * freqs                       # [..., half]
    c, s = angles.cos(), angles.sin()
    x_even = x[..., 0::2]                                    # [..., half]
    x_odd  = x[..., 1::2]
    rot_even = x_even * c - x_odd * s
    rot_odd  = x_even * s + x_odd * c
    out = torch.stack([rot_even, rot_odd], dim=-1)           # [..., half, 2]
    return out.flatten(-2)                                   # [..., 2*half]


def apply_rope_3d(x: torch.Tensor, positions: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """Apply RoPE separately to three equal axis-blocks of the head dim.

    x:         [..., H, D]   per-head feature, D divisible by 6
    positions: [..., 3]
    freqs:     [D // 6]      shared across heads and axes
    """
    *prefix, H, D = x.shape
    assert D % 6 == 0, f"head dim {D} must be divisible by 6 (3 axes × pairs)"
    block = D // 3                                           # per-axis block size
    x_blocks = x.view(*prefix, H, 3, block)                  # split into [x, y, z]
    # Broadcast positions over heads
    pos = positions.unsqueeze(-2)                            # [..., 1, 3]
    rotated = []
    for axis in range(3):
        rotated.append(apply_rope_axis(
            x_blocks[..., axis, :],                          # [..., H, block]
            pos[..., axis],                                  # [..., 1]
            freqs,
        ))
    return torch.stack(rotated, dim=-2).view(*prefix, H, D)


class RoPECrossAttention(nn.Module):
    """Linear cross-attention with 3-D rotary positions.

    Queries come from point features {z_i} at positions {x_i}.
    Keys / values come from retrieved patch features {t_k} at positions {x̄_k}.
    Linear-attention complexity: O(N · D²).
    """

    def __init__(self, dim: int = 32, num_heads: int = 8, head_dim: int = 64):
        super().__init__()
        assert head_dim % 6 == 0, "head_dim must be divisible by 6 for 3-D RoPE"
        self.num_heads = num_heads
        self.head_dim = head_dim
        inner = num_heads * head_dim
        self.q = nn.Linear(dim, inner, bias=False)
        self.k = nn.Linear(dim, inner, bias=False)
        self.v = nn.Linear(dim, inner, bias=False)
        self.o = nn.Linear(inner, dim, bias=False)
        # Per-axis half_dim = head_dim // 6
        self.register_buffer("freqs", build_rope_freqs(head_dim // 6), persistent=False)

    @staticmethod
    def _phi(x: torch.Tensor) -> torch.Tensor:
        return F.elu(x) + 1.0

    def forward(
        self,
        q_feat: torch.Tensor, q_pos: torch.Tensor,
        kv_feat: torch.Tensor, kv_pos: torch.Tensor,
    ) -> torch.Tensor:
        """
        q_feat:  [B, N, D]   point features
        q_pos:   [B, N, 3]
        kv_feat: [B, M, D]   patch features
        kv_pos:  [B, M, 3]
        returns: [B, N, D]
        """
        B, N, _ = q_feat.shape
        M = kv_feat.shape[1]
        H, D = self.num_heads, self.head_dim

        q = self.q(q_feat).view(B, N, H, D)
        k = self.k(kv_feat).view(B, M, H, D)
        v = self.v(kv_feat).view(B, M, H, D)

        # 3-D RoPE
        q = apply_rope_3d(q, q_pos, self.freqs)
        k = apply_rope_3d(k, kv_pos, self.freqs)

        # Linear attention with φ(x) = elu(x) + 1
        q = self._phi(q)                                     # [B, N, H, D]
        k = self._phi(k)                                     # [B, M, H, D]

        # KV (sum over keys/values): [B, H, D, D]
        kv = torch.einsum("bmhd,bmhe->bhde", k, v)
        # K sum: [B, H, D]
        k_sum = k.sum(dim=1).transpose(1, 2).contiguous()    # [B, H, D]  via permute
        # Actually clearer: k_sum[b, h, d] = sum_m k[b, m, h, d]
        k_sum = k.sum(dim=1)                                 # [B, H, D]

        # Numerator: [B, N, H, D]
        num = torch.einsum("bnhd,bhde->bnhe", q, kv)
        # Denominator: [B, N, H, 1]
        den = torch.einsum("bnhd,bhd->bnh", q, k_sum).unsqueeze(-1) + 1e-6

        out = (num / den).reshape(B, N, H * D)
        return self.o(out)

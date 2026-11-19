import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """Root-mean-square normalization with a learned per-feature scale (no bias)."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_f32 = x.float()
        inv_rms = torch.rsqrt(x_f32.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x_f32 * inv_rms * self.weight).type_as(x)


def precompute_rope(head_dim: int, max_seq_len: int, theta: float = 10000.0):
    """Precompute cos/sin tables for rotary position embeddings.

    Returns:
            cos, sin: float32 tensors of shape (max_seq_len, head_dim // 2).
    """
    # One frequency per pair of dimensions; decreases from 1.0 towards 1/theta.
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    positions = torch.arange(max_seq_len).float()

    # angles[m, i] = m * inv_freq[i]: rotation angle of pair i at position m.
    angles = torch.outer(positions, inv_freq)
    return angles.cos(), angles.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate pairs of features of x by position-dependent angles.

    Args:
        x: (batch, n_heads, seq_len, head_dim) queries or keys.
        cos, sin: (max_seq_len, head_dim // 2) tables from precompute_rope.

    Returns:
        Tensor of the same shape and dtype as x.
    """
    seq_len = x.shape[-2]
    cos, sin = cos[:seq_len], sin[:seq_len]

    # First half and second half of the features form the rotated pairs.
    x1, x2 = x.float().chunk(2, dim=-1)
    rotated = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    return rotated.type_as(x)



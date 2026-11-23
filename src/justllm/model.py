import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    """Hyperparameters of the model (GPT-2 small sized)."""

    vocab_size: int = 50304  # GPT-2 has 50257 tokens; padded to a multiple of 64 for speed
    block_size: int = 1024   # maximum context length
    n_layer: int = 12
    n_head: int = 12
    d_model: int = 768
    d_ff: int = 2048         # SwiGLU hidden size
    rope_theta: float = 10000.0
    norm_eps: float = 1e-6

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_head

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


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention with rotary position embeddings."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.n_head = cfg.n_head
        self.head_dim = cfg.head_dim
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)  # Q, K, V in one matmul
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)     # mixes the heads' outputs

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape

        q, k, v = self.qkv(x).split(C, dim=-1)

        # (B, T, C) -> (B, n_head, T, head_dim): every head attends independently.
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        # Merge heads back: (B, n_head, T, head_dim) -> (B, T, C).
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)


class SwiGLU(nn.Module):
    """Feed-forward block with a gated SiLU activation, applied to each token independently."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.gate = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.up = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.down = nn.Linear(cfg.d_ff, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))

class Block(nn.Module):
    """One transformer layer: pre-norm attention and pre-norm MLP, both with residual connections."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.attn = CausalSelfAttention(cfg)
        self.mlp_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.mlp = SwiGLU(cfg)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x), cos, sin)
        x = x + self.mlp(self.mlp_norm(x))
        return x


class GPT(nn.Module):
    """Decoder-only transformer language model."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg

        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.final_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight  # weight tying

        # RoPE tables are not trained and are recomputed from the config, so they stay out of checkpoints.
        cos, sin = precompute_rope(cfg.head_dim, cfg.block_size, cfg.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self.apply(self._init_weights)
        # Layers writing into the residual stream get a smaller std, so variance does not grow with depth.
        for name, p in self.named_parameters():
            if name.endswith("attn.proj.weight") or name.endswith("mlp.down.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        """Run the model.

        Args:
            idx: (B, T) token ids, T <= block_size.
            targets: (B, T) next-token ids; if given, the loss is computed.

        Returns:
            logits (B, T, vocab_size) and the cross-entropy loss (or None).
        """
        B, T = idx.shape
        assert T <= self.cfg.block_size, f"sequence length {T} exceeds block_size {self.cfg.block_size}"

        x = self.tok_emb(idx)
        for block in self.blocks:
            x = block(x, self.rope_cos, self.rope_sin)
        x = self.final_norm(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss











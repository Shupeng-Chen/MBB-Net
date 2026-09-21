import torch
import torch.nn as nn


class CrossAttention(nn.Module):
    def __init__(self, dim=512, num_heads=8):
        super().__init__()

        if dim % num_heads != 0:
            raise ValueError(
                f"dim={dim} must be divisible by num_heads={num_heads}"
            )

        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5

        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, dim * 2)
        self.proj = nn.Linear(dim, dim)

    def forward(self, q_x, kv_x):
        B, N, C = q_x.shape
        B2, M, C2 = kv_x.shape

        if B != B2:
            raise ValueError(
                f"Batch mismatch: {B} vs {B2}"
            )

        if C != C2:
            raise ValueError(
                f"Channel mismatch: {C} vs {C2}"
            )

        q = (
            self.q(q_x)
            .reshape(
                B,
                N,
                self.num_heads,
                -1,
            )
            .transpose(1, 2)
        )

        kv = (
            self.kv(kv_x)
            .reshape(
                B,
                M,
                2,
                self.num_heads,
                -1,
            )
            .permute(
                2,
                0,
                3,
                1,
                4,
            )
        )

        k, v = kv[0], kv[1]

        attn = (
            q @ k.transpose(-2, -1)
        ) * self.scale

        attn = attn.softmax(
            dim=-1
        )

        x = (
            (attn @ v)
            .transpose(1, 2)
            .reshape(B, N, C)
        )

        return self.proj(x)


class ModernBidirectionalBridge(nn.Module):
    """
    Final canonical MBB used by the released PCN model.

    G2S:
        Q   = sem_token
        K/V = geo_tokens.detach()

    S2G:
        Q   = geo_tokens
        K/V = original sem_token

    The two directions are computed in parallel from the
    original input tokens.

    Both branches retain:
        cross-attention
        learnable zero-initialized gate
        residual connection
        LayerNorm
    """

    def __init__(self, dim=512, num_heads=8):
        super().__init__()

        self.sem_attn = CrossAttention(
            dim=dim,
            num_heads=num_heads,
        )

        self.geo_attn = CrossAttention(
            dim=dim,
            num_heads=num_heads,
        )

        self.sem_norm = nn.LayerNorm(dim)
        self.geo_norm = nn.LayerNorm(dim)

        self.sem_gate = nn.Parameter(
            torch.zeros(1)
        )

        self.geo_gate = nn.Parameter(
            torch.zeros(1)
        )

    def forward(
        self,
        sem_token,
        geo_tokens,
    ):
        # G2S
        geo_for_sem = geo_tokens.detach()

        sem_feat = self.sem_attn(
            sem_token,
            geo_for_sem,
        )

        # S2G
        # IMPORTANT: original sem_token is used here.
        geo_feat = self.geo_attn(
            geo_tokens,
            sem_token,
        )

        # Parallel residual outputs
        sem_out = self.sem_norm(
            sem_token
            + self.sem_gate * sem_feat
        )

        geo_out = self.geo_norm(
            geo_tokens
            + self.geo_gate * geo_feat
        )

        return sem_out, geo_out

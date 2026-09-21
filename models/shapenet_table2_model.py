#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Static ShapeNet-55 models for the Table-2 ablations.

Supported modes
---------------
baseline
    Completion-only SnowflakeNet.

cls_only
    Full parameter structure is retained for exact checkpoint loading.
    Forward:
        semantic token -> sem_norm -> classifier
    Completion is intentionally not reported.

no_bridge
    Full parameter structure is retained.
    No cross-attention is used.
    Forward:
        semantic token -> sem_norm -> classifier
        geometry token -> geo_norm -> decoder

full
    Bidirectional cross-attention without stop-gradient.

g2s
    Geometry -> Semantic only.

s2g
    Semantic -> Geometry only.

ours
    Canonical asymmetric-decoupled MBB:
        G2S uses detached geometry K/V.
        S2G uses the original semantic token.

The module/parameter names intentionally match the original released
checkpoints exactly.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from backbones.snowflakenet.models.model_completion import (
    Decoder,
    FeatureExtractor,
)

from .mbb_bridge import CrossAttention


VALID_MODES = {
    "baseline",
    "cls_only",
    "no_bridge",
    "full",
    "g2s",
    "s2g",
    "ours",
}


class ShapeNetTable2Bridge(nn.Module):
    """
    Static implementation of the original ShapeNet-55 ablation bridge.

    Parameter names are intentionally identical to the original bridge:
        sem_attn
        geo_attn
        sem_norm
        geo_norm
        sem_gate
        geo_gate
    """

    def __init__(
        self,
        dim: int = 512,
        num_heads: int = 8,
        mode: str = "ours",
    ):
        super().__init__()

        mode = mode.lower()

        if mode not in {
            "full",
            "g2s",
            "s2g",
            "ours",
        }:
            raise ValueError(
                f"Unsupported bridge mode: {mode}"
            )

        self.mode = mode

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
        sem_token: torch.Tensor,
        geo_tokens: torch.Tensor,
    ):
        # --------------------------------------------------------
        # FULL:
        # bidirectional interaction, no stop-gradient
        # --------------------------------------------------------
        if self.mode == "full":
            sem_feat = self.sem_attn(
                sem_token,
                geo_tokens,
            )

            geo_feat = self.geo_attn(
                geo_tokens,
                sem_token,
            )

        # --------------------------------------------------------
        # OURS:
        # canonical asymmetric decoupling
        # --------------------------------------------------------
        elif self.mode == "ours":
            geo_for_sem = geo_tokens.detach()

            sem_feat = self.sem_attn(
                sem_token,
                geo_for_sem,
            )

            geo_feat = self.geo_attn(
                geo_tokens,
                sem_token,
            )

        # --------------------------------------------------------
        # S2G:
        # Semantic -> Geometry only
        # --------------------------------------------------------
        elif self.mode == "s2g":
            sem_feat = torch.zeros_like(
                sem_token
            )

            geo_feat = self.geo_attn(
                geo_tokens,
                sem_token,
            )

        # --------------------------------------------------------
        # G2S:
        # Geometry -> Semantic only
        # --------------------------------------------------------
        elif self.mode == "g2s":
            sem_feat = self.sem_attn(
                sem_token,
                geo_tokens,
            )

            geo_feat = torch.zeros_like(
                geo_tokens
            )

        else:
            raise RuntimeError(
                f"Unexpected bridge mode: {self.mode}"
            )

        sem_out = self.sem_norm(
            sem_token
            + self.sem_gate * sem_feat
        )

        geo_out = self.geo_norm(
            geo_tokens
            + self.geo_gate * geo_feat
        )

        return sem_out, geo_out


class ShapeNet55Table2Model(nn.Module):
    """
    Static model implementing all seven canonical Table-2 modes.
    """

    def __init__(
        self,
        cfg,
        mode: str = "ours",
    ):
        super().__init__()

        mode = mode.lower()

        if mode not in VALID_MODES:
            raise ValueError(
                f"Unsupported Table-2 mode: {mode}. "
                f"Valid modes: {sorted(VALID_MODES)}"
            )

        self.cfg = cfg
        self.mode = mode

        dim_feat = 512

        # --------------------------------------------------------
        # Geometry encoder
        # --------------------------------------------------------
        self.feat_extractor = FeatureExtractor(
            out_dim=dim_feat
        )

        # --------------------------------------------------------
        # Decoder
        #
        # baseline and all joint models contain the decoder.
        # cls_only checkpoint also contains the decoder because
        # the original unified trainer instantiated the full model.
        # --------------------------------------------------------
        self.decoder = Decoder(
            dim_feat=dim_feat,
            num_pc=cfg.get(
                "NUM_PC",
                256,
            ),
            num_p0=cfg.get(
                "NUM_P0",
                512,
            ),
            up_factors=cfg.get(
                "UP_FACTORS",
                [1, 4, 4],
            ),
        )

        # --------------------------------------------------------
        # BASELINE is the only 287-tensor architecture.
        # It contains no MBB / semantic encoder / classifier.
        # --------------------------------------------------------
        if self.mode == "baseline":
            return

        # --------------------------------------------------------
        # All remaining modes retain the complete 328-tensor
        # parameter structure.
        #
        # no_bridge / cls_only were historically instantiated
        # with the full MBB module even though attention is
        # bypassed in forward().
        # --------------------------------------------------------
        bridge_mode = (
            self.mode
            if self.mode in {
                "full",
                "g2s",
                "s2g",
                "ours",
            }
            else "ours"
        )

        self.mbb = ShapeNetTable2Bridge(
            dim=dim_feat,
            mode=bridge_mode,
        )

        self.sem_enc = nn.Sequential(
            nn.Conv1d(
                3,
                128,
                1,
            ),
            nn.BatchNorm1d(
                128
            ),
            nn.GELU(),

            nn.Conv1d(
                128,
                dim_feat,
                1,
            ),
            nn.BatchNorm1d(
                dim_feat
            ),
            nn.GELU(),
        )

        self.classifier = nn.Sequential(
            nn.Linear(
                dim_feat,
                256,
            ),
            nn.BatchNorm1d(
                256
            ),
            nn.GELU(),

            nn.Linear(
                256,
                cfg.get(
                    "NUM_CLASSES",
                    55,
                ),
            ),
        )

    def forward(
        self,
        point_cloud: torch.Tensor,
    ):
        # Input convention:
        #   B x N x 3
        pcd_bnc = point_cloud

        pcd_bcn = (
            point_cloud
            .permute(
                0,
                2,
                1,
            )
            .contiguous()
        )

        # --------------------------------------------------------
        # Geometry encoder
        # feat: B x 512 x 1
        # --------------------------------------------------------
        feat = self.feat_extractor(
            pcd_bcn
        )

        # --------------------------------------------------------
        # BASELINE: completion only
        # --------------------------------------------------------
        if self.mode == "baseline":
            out_list = self.decoder(
                feat,
                pcd_bnc,
            )

            return out_list, None

        # --------------------------------------------------------
        # Semantic token:
        # B x 1 x 512
        # --------------------------------------------------------
        sem_token = torch.max(
            self.sem_enc(
                pcd_bcn
            ),
            dim=2,
            keepdim=True,
        )[0].transpose(
            1,
            2,
        )

        geo_tokens = feat.transpose(
            1,
            2,
        )

        # --------------------------------------------------------
        # NO-BRIDGE / CLS-ONLY
        #
        # IMPORTANT:
        # Both paths still pass through their LayerNorms.
        # This reproduces the historical patched forward exactly.
        # --------------------------------------------------------
        if self.mode in {
            "no_bridge",
            "cls_only",
        }:
            enhanced_sem = self.mbb.sem_norm(
                sem_token
            )

            enhanced_geo = self.mbb.geo_norm(
                geo_tokens
            )

        # --------------------------------------------------------
        # FULL / G2S / S2G / OURS
        # --------------------------------------------------------
        else:
            enhanced_sem, enhanced_geo = (
                self.mbb(
                    sem_token,
                    geo_tokens,
                )
            )

        logits = self.classifier(
            enhanced_sem.squeeze(1)
        )

        # --------------------------------------------------------
        # CLS-ONLY:
        # decoder was not optimized by the supplied trainer,
        # so completion must not be reported.
        # --------------------------------------------------------
        if self.mode == "cls_only":
            return [], logits

        # --------------------------------------------------------
        # Completion branch
        # --------------------------------------------------------
        out_list = self.decoder(
            enhanced_geo
            .transpose(
                1,
                2,
            )
            .contiguous(),
            pcd_bnc,
        )

        return out_list, logits

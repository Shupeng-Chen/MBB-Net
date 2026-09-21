#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Portable canonical MBB-Net model for ShapeNet-55.

Canonical paper configuration:
    NUM_CLASSES = 55
    NUM_PC      = 256
    NUM_P0      = 512
    UP_FACTORS  = [1, 4, 4]

The MBB implementation is shared with the released PCN model:

G2S:
    Q = semantic token
    K/V = detached geometry token

S2G:
    Q = geometry token
    K/V = semantic token

Both paths use independent gated residual updates followed by LayerNorm.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from backbones.snowflakenet.models.model_completion import (
    Decoder,
    FeatureExtractor,
)

from .mbb_bridge import ModernBidirectionalBridge


class MBB_Model_ShapeNet(nn.Module):
    """
    Canonical ShapeNet MBB-Net.

    Input:
        point_cloud: [B, N, 3]

    Returns:
        out_list:
            SnowflakeNet progressive point-cloud outputs.

        logits:
            [B, NUM_CLASSES]
    """

    def __init__(
        self,
        cfg,
        use_mbb: bool = True,
    ):
        super().__init__()

        self.cfg = cfg
        self.use_mbb = use_mbb

        dim_feat = 512

        # ------------------------------------------------------
        # Geometry encoder
        # ------------------------------------------------------
        self.feat_extractor = FeatureExtractor(
            out_dim=dim_feat
        )

        # ------------------------------------------------------
        # SnowflakeNet decoder
        #
        # ShapeNet-55:
        # 256 -> 512 -> 2048 -> 8192
        # ------------------------------------------------------
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

        if self.use_mbb:

            # --------------------------------------------------
            # Canonical static MBB
            # --------------------------------------------------
            self.mbb = ModernBidirectionalBridge(
                dim=dim_feat
            )

            # --------------------------------------------------
            # Independent semantic encoder
            # --------------------------------------------------
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

            # --------------------------------------------------
            # Classification head
            # --------------------------------------------------
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
        """
        point_cloud:
            [B, N, 3]
        """

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

        # ------------------------------------------------------
        # Geometry encoding
        # [B, 512, 1]
        # ------------------------------------------------------
        feat = self.feat_extractor(
            pcd_bcn
        )

        logits = None

        if self.use_mbb:

            # --------------------------------------------------
            # Semantic token
            # [B, 1, 512]
            # --------------------------------------------------
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

            # --------------------------------------------------
            # Canonical asymmetric-decoupled MBB
            # --------------------------------------------------
            enhanced_sem, enhanced_geo = (
                self.mbb(
                    sem_token,
                    geo_tokens,
                )
            )

            # --------------------------------------------------
            # Classification
            # --------------------------------------------------
            logits = self.classifier(
                enhanced_sem.squeeze(1)
            )

            # --------------------------------------------------
            # Geometry back to SnowflakeNet convention
            # --------------------------------------------------
            feat = enhanced_geo.transpose(
                1,
                2,
            )

        # ------------------------------------------------------
        # Progressive completion
        # ------------------------------------------------------
        out_list = self.decoder(
            feat,
            pcd_bnc,
        )

        return out_list, logits

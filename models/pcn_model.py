import torch
import torch.nn as nn

from backbones.snowflakenet.models.model_completion import (
    FeatureExtractor,
    Decoder,
)

from models.mbb_bridge import (
    ModernBidirectionalBridge,
)


class MBB_Model_PCN(nn.Module):
    """
    Canonical PCN MBB-Net.

    This class preserves the module names and tensor shapes of the
    final PCN model used to produce the released checkpoint.
    """

    def __init__(
        self,
        cfg,
        use_mbb=True,
    ):
        super().__init__()

        self.cfg = cfg
        self.use_mbb = use_mbb

        dim_feat = 512

        # --------------------------------------------------
        # Geometry encoder + SnowflakeNet decoder
        # --------------------------------------------------

        self.feat_extractor = FeatureExtractor(
            out_dim=dim_feat
        )

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
                [1, 4, 8],
            ),
        )

        # --------------------------------------------------
        # Semantic branch + canonical MBB
        # --------------------------------------------------

        if self.use_mbb:

            self.mbb = ModernBidirectionalBridge(
                dim=dim_feat
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
                        8,
                    ),
                ),
            )

    def forward(
        self,
        point_cloud,
    ):
        """
        Args:
            point_cloud: [B, N, 3]

        Returns:
            coarse: [B, 256, 3]
            fine:   final completed point cloud
            logits: classification logits
        """

        pcd_bnc = point_cloud

        pcd_bcn = (
            point_cloud
            .permute(0, 2, 1)
            .contiguous()
        )

        # Geometry feature: [B, 512, 1]
        feat = self.feat_extractor(
            pcd_bcn
        )

        logits = None

        if self.use_mbb:

            # Semantic token: [B, 1, 512]
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

            # Canonical static MBB.
            enhanced_sem, enhanced_geo = (
                self.mbb(
                    sem_token,
                    feat.transpose(
                        1,
                        2,
                    ),
                )
            )

            logits = self.classifier(
                enhanced_sem.squeeze(1)
            )

            feat = enhanced_geo.transpose(
                1,
                2,
            )

        out_list = self.decoder(
            feat,
            pcd_bnc,
        )

        return (
            out_list[0],
            out_list[-1],
            logits,
        )

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""MBB transfer model for the official SymmCompletion backbone.

This file does not modify any official SymmCompletion source. It imports the
official ``models.SymmCompletion.SymmCompletion`` class and wraps it with:

- an independent semantic encoder;
- the current MBB cross-attention and stop-gradient definition;
- a classification head;
- a geometry adapter that feeds both official SymmCompletion feature streams
  back into the unchanged SGFormer decoder.

The initial completion output is exactly the official pretrained output because
the MBB gates are initialized to zero and only the bridge-induced delta is
injected into the official geometric streams.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Dict, Iterable, Optional, Tuple

import torch
import torch.nn as nn

from third_party.symmcompletion.models.SymmCompletion import SymmCompletion


class CrossAttention(nn.Module):
    """Cross-attention used by the current MBB implementation."""

    def __init__(self, dim: int = 512, num_heads: int = 8) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(
                f"dim={dim} must be divisible by num_heads={num_heads}."
            )
        self.num_heads = int(num_heads)
        self.scale = (dim // num_heads) ** -0.5
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, dim * 2)
        self.proj = nn.Linear(dim, dim)

    def forward(
        self,
        q_x: torch.Tensor,
        kv_x: torch.Tensor,
    ) -> torch.Tensor:
        if q_x.ndim != 3 or kv_x.ndim != 3:
            raise ValueError(
                "CrossAttention expects BNC tensors, "
                f"got {tuple(q_x.shape)} and {tuple(kv_x.shape)}."
            )

        batch, nq, channels = q_x.shape
        batch_kv, nkv, channels_kv = kv_x.shape
        if batch != batch_kv or channels != channels_kv:
            raise ValueError(
                f"Incompatible attention shapes: "
                f"{tuple(q_x.shape)} vs {tuple(kv_x.shape)}."
            )

        q = (
            self.q(q_x)
            .reshape(
                batch,
                nq,
                self.num_heads,
                channels // self.num_heads,
            )
            .transpose(1, 2)
        )
        kv = (
            self.kv(kv_x)
            .reshape(
                batch,
                nkv,
                2,
                self.num_heads,
                channels // self.num_heads,
            )
            .permute(2, 0, 3, 1, 4)
        )
        key, value = kv[0], kv[1]

        attention = (
            q @ key.transpose(-2, -1)
        ) * self.scale
        attention = attention.softmax(dim=-1)

        output = (
            attention @ value
        ).transpose(1, 2).reshape(
            batch,
            nq,
            channels,
        )
        return self.proj(output)


class ModernBidirectionalBridge(nn.Module):
    """Current MBB directional definitions.

    ``ours`` performs bidirectional forward interaction. Geometry is detached
    on the G2S path, preventing the classification objective from updating the
    geometry input through that path. The S2G path remains differentiable.
    """

    VALID_MODES = {"full", "ours", "s2g", "g2s"}

    def __init__(
        self,
        dim: int = 512,
        mode: str = "ours",
    ) -> None:
        super().__init__()
        normalized = mode.lower()
        if normalized not in self.VALID_MODES:
            raise ValueError(
                f"Unknown MBB mode: {mode}. "
                f"Expected one of {sorted(self.VALID_MODES)}."
            )

        self.mode = normalized
        self.sem_attn = CrossAttention(dim=dim)
        self.geo_attn = CrossAttention(dim=dim)
        self.sem_norm = nn.LayerNorm(dim)
        self.geo_norm = nn.LayerNorm(dim)

        # Same zero-initialized learnable gates as the current MBB code.
        self.sem_gate = nn.Parameter(torch.zeros(1))
        self.geo_gate = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        sem_token: torch.Tensor,
        geo_tokens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.mode == "full":
            sem_feature = self.sem_attn(
                sem_token,
                geo_tokens,
            )
            geo_feature = self.geo_attn(
                geo_tokens,
                sem_token,
            )

        elif self.mode == "ours":
            sem_feature = self.sem_attn(
                sem_token,
                geo_tokens.detach(),
            )
            geo_feature = self.geo_attn(
                geo_tokens,
                sem_token,
            )

        elif self.mode == "s2g":
            sem_feature = torch.zeros_like(
                sem_token
            )
            geo_feature = self.geo_attn(
                geo_tokens,
                sem_token,
            )

        else:  # g2s
            sem_feature = self.sem_attn(
                sem_token,
                geo_tokens,
            )
            geo_feature = torch.zeros_like(
                geo_tokens
            )

        sem_output = self.sem_norm(
            sem_token
            + self.sem_gate * sem_feature
        )
        geo_output = self.geo_norm(
            geo_tokens
            + self.geo_gate * geo_feature
        )
        return sem_output, geo_output


class SemanticEncoder(nn.Module):
    """Independent semantic encoder used by MBB-Net."""

    def __init__(self, dim: int = 512) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv1d(3, 128, 1),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Conv1d(128, dim, 1),
            nn.BatchNorm1d(dim),
            nn.GELU(),
        )

    def forward(
        self,
        point_cloud: torch.Tensor,
    ) -> torch.Tensor:
        # B,N,3 -> B,1,C
        features = self.network(
            point_cloud.transpose(
                1,
                2,
            ).contiguous()
        )
        return (
            features.max(
                dim=2,
                keepdim=True,
            )[0]
            .transpose(1, 2)
            .contiguous()
        )


def extract_state_dict(
    checkpoint: Dict,
) -> Dict[str, torch.Tensor]:
    """Extract a model state dict from common checkpoint wrappers."""
    if not isinstance(checkpoint, dict):
        raise TypeError(
            "Checkpoint must be a dictionary, "
            f"got {type(checkpoint).__name__}."
        )

    for key in (
        "base_model",
        "model",
        "model_state_dict",
        "state_dict",
    ):
        value = checkpoint.get(key)
        if isinstance(value, dict) and value:
            return value

    if checkpoint and all(
        torch.is_tensor(value)
        for value in checkpoint.values()
    ):
        return checkpoint

    raise RuntimeError(
        "Could not locate a model state dict. "
        "Expected base_model/model/model_state_dict/state_dict."
    )


def strip_parallel_prefix(
    state_dict: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    cleaned: Dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        new_key = key
        while new_key.startswith("module."):
            new_key = new_key[
                len("module.") :
            ]
        if new_key in cleaned:
            raise RuntimeError(
                "Duplicate checkpoint key after prefix removal: "
                f"{new_key}"
            )
        cleaned[new_key] = value
    return cleaned


def infer_up_factors(
    state_dict: Dict[str, torch.Tensor],
) -> Tuple[int, int]:
    cleaned = strip_parallel_prefix(
        state_dict
    )
    first_key = (
        "sgformer_1.fc.4.weight"
    )
    second_key = (
        "sgformer_2.fc.4.weight"
    )

    missing = [
        key
        for key in (
            first_key,
            second_key,
        )
        if key not in cleaned
    ]
    if missing:
        raise RuntimeError(
            "Cannot infer SymmCompletion up-factors; "
            f"missing keys: {missing}"
        )

    return (
        int(cleaned[first_key].shape[0] // 3),
        int(cleaned[second_key].shape[0] // 3),
    )


class SymmMBBTransfer(nn.Module):
    """Transfer MBB onto an official pretrained SymmCompletion backbone.

    SymmCompletion provides two decoder feature streams:

    - partial features:  B x 128 x 512
    - symmetry features: B x 128 x 512

    They are concatenated into 1024 geometric tokens, projected from 128 to 512
    channels, processed by MBB, projected back to 128 channels, split, and fed
    into the unchanged official SGFormer decoder.

    Variants
    --------
    no_bridge:
        Independent classifier plus the original completion path.
    full:
        Unconstrained bidirectional MBB interaction.
    ours:
        Asymmetric bidirectional MBB with stop-gradient on G2S geometry input.
    s2g:
        Semantic-to-geometry only.
    g2s:
        Geometry-to-semantic only.

    Tune policies
    -------------
    full:
        Fine-tune the official backbone and new modules, optionally with a
        smaller backbone learning rate.
    adapter:
        Freeze the official backbone. Gradients still pass through the frozen
        decoder to train the geometric MBB adapter.
    """

    VALID_DATASETS = {
        "pcn",
        "shapenet55",
    }
    VALID_VARIANTS = {
        "no_bridge",
        "full",
        "ours",
        "s2g",
        "g2s",
    }
    VALID_TUNE_POLICIES = {
        "full",
        "adapter",
    }

    def __init__(
        self,
        dataset: str,
        num_classes: int,
        official_checkpoint: str,
        variant: str = "ours",
        tune_policy: str = "full",
        dim: int = 512,
        geometric_channel_dim: int = 128,
        classifier_dropout: float = 0.3,
    ) -> None:
        super().__init__()

        dataset = dataset.lower()
        variant = variant.lower()
        tune_policy = tune_policy.lower()

        if dataset not in self.VALID_DATASETS:
            raise ValueError(
                f"Unsupported dataset: {dataset}."
            )
        if variant not in self.VALID_VARIANTS:
            raise ValueError(
                f"Unsupported variant: {variant}."
            )
        if tune_policy not in (
            self.VALID_TUNE_POLICIES
        ):
            raise ValueError(
                f"Unsupported tune policy: {tune_policy}."
            )

        self.dataset = dataset
        self.num_classes = int(
            num_classes
        )
        self.variant = variant
        self.tune_policy = tune_policy
        self.dim = int(dim)
        self.geometric_channel_dim = int(
            geometric_channel_dim
        )
        self.official_checkpoint = str(
            official_checkpoint
        )

        raw_checkpoint = torch.load(
            official_checkpoint,
            map_location="cpu",
        )
        official_state = extract_state_dict(
            raw_checkpoint
        )
        up_first, up_second = (
            infer_up_factors(
                official_state
            )
        )

        expected_up_factors = (
            (2, 8)
            if dataset == "pcn"
            else (2, 3)
        )
        if (
            up_first,
            up_second,
        ) != expected_up_factors:
            raise RuntimeError(
                "The supplied official checkpoint does not match "
                f"{dataset}. Detected up-factors "
                f"{(up_first, up_second)}, expected "
                f"{expected_up_factors}."
            )

        include_input = (
            dataset == "shapenet55"
        )
        official_config = SimpleNamespace(
            up_factors=(
                f"{up_first}, {up_second}"
            ),
            include_input=include_input,
        )

        self.symm = SymmCompletion(
            official_config
        )
        cleaned_state = strip_parallel_prefix(
            official_state
        )
        incompatible = (
            self.symm.load_state_dict(
                cleaned_state,
                strict=True,
            )
        )
        if (
            incompatible.missing_keys
            or incompatible.unexpected_keys
        ):
            raise RuntimeError(
                "Strict official checkpoint loading failed. "
                f"Missing={incompatible.missing_keys}; "
                f"unexpected={incompatible.unexpected_keys}."
            )

        self.official_epoch = int(
            raw_checkpoint.get(
                "epoch",
                -1,
            )
        )
        self.official_metrics = (
            raw_checkpoint.get(
                "metrics",
                {},
            )
        )
        self.up_factors = (
            up_first,
            up_second,
        )
        self.include_input = (
            include_input
        )

        self.semantic_encoder = (
            SemanticEncoder(
                dim=dim,
            )
        )
        self.semantic_norm = nn.LayerNorm(
            dim
        )
        self.classifier = nn.Sequential(
            nn.Linear(dim, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(
                classifier_dropout
            ),
            nn.Linear(
                256,
                self.num_classes,
            ),
        )

        if variant == "no_bridge":
            self.geometry_projection = None
            self.geometry_back_projection = None
            self.mbb = None
        else:
            self.geometry_projection = nn.Linear(
                geometric_channel_dim,
                dim,
            )
            # No bias: a zero MBB delta maps to an exact zero update.
            self.geometry_back_projection = nn.Linear(
                dim,
                geometric_channel_dim,
                bias=False,
            )
            self.mbb = (
                ModernBidirectionalBridge(
                    dim=dim,
                    mode=variant,
                )
            )

        self._backbone_trainable = True
        self.configure_tuning(
            tune_policy
        )

    def configure_tuning(
        self,
        policy: str,
    ) -> None:
        normalized = policy.lower()
        if normalized not in (
            self.VALID_TUNE_POLICIES
        ):
            raise ValueError(
                f"Unsupported tune policy: {policy}."
            )

        self.tune_policy = normalized
        self.set_backbone_trainable(
            normalized == "full"
        )

    def set_backbone_trainable(
        self,
        trainable: bool,
    ) -> None:
        self._backbone_trainable = bool(
            trainable
        )
        for parameter in (
            self.symm.parameters()
        ):
            parameter.requires_grad = bool(
                trainable
            )

        if not trainable:
            self.symm.eval()

    @property
    def backbone_trainable(
        self,
    ) -> bool:
        return self._backbone_trainable

    def train(
        self,
        mode: bool = True,
    ):
        super().train(mode)
        if not self._backbone_trainable:
            # Keep official BN/statistics fixed while adapters train.
            self.symm.eval()
        return self

    def backbone_parameters(
        self,
    ) -> Iterable[nn.Parameter]:
        return self.symm.parameters()

    def transfer_parameters(
        self,
    ) -> Iterable[nn.Parameter]:
        for name, parameter in (
            self.named_parameters()
        ):
            if not name.startswith(
                "symm."
            ):
                yield parameter

    def parameter_report(
        self,
    ) -> Dict[str, int]:
        total = sum(
            parameter.numel()
            for parameter in self.parameters()
        )
        trainable = sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )
        return {
            "total": total,
            "trainable": trainable,
            "frozen": total - trainable,
        }

    def _extract_official_features(
        self,
        point_cloud: torch.Tensor,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        point_cloud_bcn = (
            point_cloud.transpose(
                2,
                1,
            ).contiguous()
        )

        if self._backbone_trainable:
            coarse, symmetry_points, partial_features = (
                self.symm.lstnet(
                    point_cloud_bcn
                )
            )
            symmetry_features = (
                self.symm.local_encoder(
                    symmetry_points
                )
            )
        else:
            with torch.no_grad():
                coarse, symmetry_points, partial_features = (
                    self.symm.lstnet(
                        point_cloud_bcn
                    )
                )
                symmetry_features = (
                    self.symm.local_encoder(
                        symmetry_points
                    )
                )

        return (
            coarse,
            partial_features,
            symmetry_features,
        )

    def _adapt_geometric_streams(
        self,
        semantic_token: torch.Tensor,
        partial_features: torch.Tensor,
        symmetry_features: torch.Tensor,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        if self.variant == "no_bridge":
            semantic_output = (
                self.semantic_norm(
                    semantic_token
                )
            )
            return (
                semantic_output,
                partial_features,
                symmetry_features,
            )

        partial_tokens = (
            partial_features.transpose(
                1,
                2,
            ).contiguous()
        )
        symmetry_tokens = (
            symmetry_features.transpose(
                1,
                2,
            ).contiguous()
        )
        raw_geometry = torch.cat(
            [
                partial_tokens,
                symmetry_tokens,
            ],
            dim=1,
        )
        projected_geometry = (
            self.geometry_projection(
                raw_geometry
            )
        )

        semantic_output, enhanced_geometry = (
            self.mbb(
                semantic_token,
                projected_geometry,
            )
        )

        # Removing the LayerNorm-only baseline makes gate=0 exactly preserve
        # both official geometric streams.
        normalized_baseline = (
            self.mbb.geo_norm(
                projected_geometry
            )
        )
        bridge_delta = (
            enhanced_geometry
            - normalized_baseline
        )
        raw_delta = (
            self.geometry_back_projection(
                bridge_delta
            )
        )
        adapted_geometry = (
            raw_geometry + raw_delta
        )

        partial_count = (
            partial_tokens.size(1)
        )
        partial_adapted = (
            adapted_geometry[
                :,
                :partial_count,
            ]
            .transpose(1, 2)
            .contiguous()
        )
        symmetry_adapted = (
            adapted_geometry[
                :,
                partial_count:,
            ]
            .transpose(1, 2)
            .contiguous()
        )

        return (
            semantic_output,
            partial_adapted,
            symmetry_adapted,
        )

    def forward(
        self,
        point_cloud: torch.Tensor,
    ) -> Tuple[list, torch.Tensor]:
        if (
            point_cloud.ndim != 3
            or point_cloud.shape[-1] != 3
        ):
            raise ValueError(
                "Expected point cloud B,N,3; "
                f"got {tuple(point_cloud.shape)}."
            )

        coarse, partial_features, symmetry_features = (
            self._extract_official_features(
                point_cloud
            )
        )

        semantic_token = (
            self.semantic_encoder(
                point_cloud
            )
        )
        (
            semantic_output,
            partial_adapted,
            symmetry_adapted,
        ) = self._adapt_geometric_streams(
            semantic_token,
            partial_features,
            symmetry_features,
        )

        logits = self.classifier(
            semantic_output.squeeze(1)
        )

        fine_first = self.symm.sgformer_1(
            coarse,
            symmetry_adapted,
            partial_adapted,
        )
        fine_second = self.symm.sgformer_2(
            fine_first.transpose(
                2,
                1,
            ).contiguous(),
            symmetry_adapted,
            partial_adapted,
        )

        if self.symm.include_input:
            fine_second = torch.cat(
                [
                    fine_second,
                    point_cloud,
                ],
                dim=1,
            ).contiguous()

        outputs = [
            coarse.transpose(
                2,
                1,
            ).contiguous(),
            fine_first,
            fine_second,
        ]
        self.symm.pred_dense_point = (
            outputs[-1]
        )
        return outputs, logits

    def completion_loss(
        self,
        outputs: list,
        ground_truth: torch.Tensor,
    ):
        return self.symm.get_loss(
            outputs,
            ground_truth,
        )

    def bridge_state(
        self,
    ) -> Dict[str, Optional[float]]:
        if self.mbb is None:
            return {
                "sem_gate": None,
                "geo_gate": None,
            }
        return {
            "sem_gate": float(
                self.mbb.sem_gate.detach().cpu().item()
            ),
            "geo_gate": float(
                self.mbb.geo_gate.detach().cpu().item()
            ),
        }

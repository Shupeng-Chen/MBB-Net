#!/usr/bin/env python
from __future__ import annotations

import types
from typing import Tuple

import torch

from models.pcn_model import MBB_Model_PCN


PAPER_PCN_MODES = (
    "completionOnly",
    "full",
    "ours",
)

# Public/paper-facing names are intentionally distinct from implementation flags.
#
# CompletionOnly:
#   completion network only; no semantic encoder, classifier, or MBB.
#
# No-Bridge (used elsewhere, e.g. ShapeNet-55):
#   both tasks remain present, but cross-task bridge interaction is disabled.
#
# Historical aliases are accepted so old scripts/checkpoint names remain usable.
_MODE_ALIASES = {
    "completiononly": "completionOnly",
    "completion_only": "completionOnly",
    "comp_only": "completionOnly",
    "full": "full",
    "ours": "ours",
    "mbb": "ours",
    "mbb-net": "ours",
    "mbb_net": "ours",
}


def normalize_pcn_mode(mode: str) -> str:
    raw = str(mode).strip()

    # Preserve the canonical camel-case public spelling.
    if raw == "completionOnly":
        return "completionOnly"

    key = raw.lower()

    if key not in _MODE_ALIASES:
        raise ValueError(
            f"Unsupported PCN paper mode: {mode!r}. "
            f"Expected one of {PAPER_PCN_MODES}."
        )

    return _MODE_ALIASES[key]


def _apply_full_patch(model: MBB_Model_PCN) -> None:
    """
    Historical PCN Full bridge.

    IMPORTANT:
    This is the original PCN sequential/cascaded bidirectional bridge:

        1. Geometry -> Semantic
        2. Update semantic token
        3. Updated Semantic -> Geometry

    There is NO stop-gradient in this configuration.

    This is intentionally different from the final canonical MBB-Net bridge.
    """
    if not hasattr(model, "mbb"):
        raise AttributeError(
            "PCN Full requires a model with an MBB module."
        )

    def full_forward(
        bridge_self,
        sem_token: torch.Tensor,
        geo_tokens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        # G2S: fully differentiable.
        sem_feat = bridge_self.sem_attn(
            sem_token,
            geo_tokens,
        )

        # Historical PCN Full updates semantic representation first.
        sem_token_out = bridge_self.sem_norm(
            sem_token
            + bridge_self.sem_gate * sem_feat
        )

        # S2G uses the UPDATED semantic representation.
        geo_feat = bridge_self.geo_attn(
            geo_tokens,
            sem_token_out,
        )

        geo_tokens_out = bridge_self.geo_norm(
            geo_tokens
            + bridge_self.geo_gate * geo_feat
        )

        return (
            sem_token_out,
            geo_tokens_out,
        )

    model.mbb.forward = types.MethodType(
        full_forward,
        model.mbb,
    )

    print(
        "[PCN mode] full: historical sequential/cascaded "
        "bidirectional bridge; no stop-gradient."
    )


def _apply_ours_patch(model: MBB_Model_PCN) -> None:
    """
    Final canonical MBB-Net used by the released PCN Ours checkpoint.

    G2S:
        sem_attn(sem_token, geo_tokens.detach())

    S2G:
        geo_attn(geo_tokens, sem_token)

    Both branches use the ORIGINAL inputs in parallel.
    """
    if not hasattr(model, "mbb"):
        raise AttributeError(
            "PCN Ours requires a model with an MBB module."
        )

    def ours_forward(
        bridge_self,
        sem_token: torch.Tensor,
        geo_tokens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        geo_for_sem = geo_tokens.detach()

        sem_feat = bridge_self.sem_attn(
            sem_token,
            geo_for_sem,
        )

        geo_feat = bridge_self.geo_attn(
            geo_tokens,
            sem_token,
        )

        sem_token_out = bridge_self.sem_norm(
            sem_token
            + bridge_self.sem_gate * sem_feat
        )

        geo_tokens_out = bridge_self.geo_norm(
            geo_tokens
            + bridge_self.geo_gate * geo_feat
        )

        return (
            sem_token_out,
            geo_tokens_out,
        )

    model.mbb.forward = types.MethodType(
        ours_forward,
        model.mbb,
    )

    print(
        "[PCN mode] ours: canonical parallel MBB; "
        "stop-gradient only on the G2S geometry source."
    )


def build_pcn_model(
    cfg: dict,
    mode: str,
) -> MBB_Model_PCN:

    mode = normalize_pcn_mode(mode)

    if mode == "completionOnly":
        model = MBB_Model_PCN(
            cfg,
            use_mbb=False,
        )

        print(
            "[PCN mode] CompletionOnly: SnowflakeNet completion "
            "branch only; no semantic encoder, classifier, or MBB."
        )

        return model

    model = MBB_Model_PCN(
        cfg,
        use_mbb=True,
    )

    if mode == "full":
        _apply_full_patch(model)

    elif mode == "ours":
        _apply_ours_patch(model)

    return model

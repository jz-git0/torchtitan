# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Shared helpers for model-specific ladder adapters.

Shape suffix legend:
  B=batch, L=seq, D=model dim, R=routed local tokens.
"""

from __future__ import annotations

import torch

from torchtitan.models.common.moe import MoE
from ..ladder_moe import LadderMoE, Pending


def build_ladder_moe(moe_config: MoE.Config) -> LadderMoE:
    """Build a LadderMoE wrapper from a TorchTitan MoE config.

    Input: moe_config describes router, experts, and sharding.
    Output: a LadderMoE with the same MoE sharding_config attached to the wrapper.
    """
    return LadderMoE.Config(
        moe=moe_config,
        sharding_config=moe_config.sharding_config,
    ).build()


def drain_pending(
    moe: LadderMoE, res_BLD: torch.Tensor, pending: Pending
) -> torch.Tensor:
    """Add a pending expert result to a residual tensor.

    Input: res_BLD is [B, L, D], pending is (routed_out_RD [R, D], DispatchState).
    Output: updated residual [B, L, D]. CUDA output is recorded on the
    current stream before combine.
    """
    e_RD, state = pending
    if e_RD.is_cuda:
        e_RD.record_stream(torch.cuda.current_stream(e_RD.device))
    return res_BLD + moe.combine_bld(e_RD, state)

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Sketch Comet-style fused comm/compute MoE backend.

A fused backend would pipeline token exchange, expert GEMM, and combine at chunk
granularity. That is orthogonal to ladder scheduling, but it collapses the
dispatch/experts/combine seams exposed by LadderMoE.
"""

from __future__ import annotations

import torch

from ..ladder_moe import RouteInfo


class CometFusedMoE(torch.nn.Module):
    """Fused MoE placeholder: route is separate; MoE body is one call.

    Input: num_token_chunks configures future chunking for [B, L, D] tokens.
    Output: placeholder module exposing fused_moe once a backend exists.
    """

    supports_phase_split = False

    def __init__(self, *, num_token_chunks: int = 4) -> None:
        super().__init__()
        self.num_token_chunks = num_token_chunks

    def fused_moe(self, x_BLD: torch.Tensor, route_info: RouteInfo) -> torch.Tensor:
        """Placeholder for fused dispatch, experts, and combine.

        Input: x_BLD is [B, L, D], route_info carries flattened routing metadata [T, K].
        Output: future MoE output [B, L, D].
        """
        raise NotImplementedError(
            "sketch: tile-pipelined a2a<->grouped_mm; fused kernel or stream pipeline"
        )

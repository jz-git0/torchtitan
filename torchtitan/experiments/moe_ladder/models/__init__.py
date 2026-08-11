# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Model adapters for the MoE ladder experiment."""

from .qwen3 import (
    ladderize_qwen3_config,
    model_registry,
    Qwen3HoistedGateABlock,
    Qwen3HoistedGateBBlock,
    Qwen3LadderModel,
    Qwen3LadderMoEBlock,
    Qwen3ParallelMoEBlock,
)

__all__ = [
    "Qwen3HoistedGateABlock",
    "Qwen3HoistedGateBBlock",
    "Qwen3LadderMoEBlock",
    "Qwen3LadderModel",
    "Qwen3ParallelMoEBlock",
    "ladderize_qwen3_config",
    "model_registry",
]

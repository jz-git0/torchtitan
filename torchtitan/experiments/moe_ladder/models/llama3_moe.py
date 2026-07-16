# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Llama3/MoLE adapter slot for the MoE ladder experiment.

The next step is to mirror the MoE block layout from origin/ahmed-mole2 here,
then reuse the shared phase-split MoE and schedule code from this package.
"""

from __future__ import annotations

from torchtitan.experiments.moe_ladder.schedules import normalize_schedule


def ladderize_llama3_moe_config(config, schedule: str):
    """Validate schedule and reject the unimplemented Llama3/MoLE adapter.

    Input: config is the future Llama3/MoLE model config, schedule is a schedule string.
    Output: NotImplementedError until the block layout exists.
    """
    normalize_schedule(schedule)
    raise NotImplementedError(
        "Llama3/MoLE laddering needs the origin/ahmed-mole2 block layout to be "
        "ported into torchtitan.experiments.moe_ladder.models.llama3_moe"
    )

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""DeepSeek-V3 adapter scaffold for the MoE ladder experiment."""

from __future__ import annotations

from torchtitan.experiments.moe_ladder.schedules import normalize_schedule


def ladderize_deepseek_v3_config(config, schedule: str):
    """Validate schedule and reject the unimplemented DeepSeek-V3 adapter.

    Input: config is the future DeepSeek-V3 config, schedule is a schedule string.
    Output: NotImplementedError until MLA and expert phases are mapped.
    """
    normalize_schedule(schedule)
    raise NotImplementedError(
        "DeepSeek-V3 laddering needs MLA attention and shared/routed expert "
        "phase boundaries to be mapped before enabling training"
    )

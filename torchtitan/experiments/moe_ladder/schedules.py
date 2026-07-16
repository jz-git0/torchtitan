# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Schedule names shared by model adapters and profiling harnesses."""

from __future__ import annotations

from typing import get_args, Literal

ScheduleName = Literal["parallel", "ladder", "hoisted_gateA", "hoisted_gateB"]
SCHEDULE_NAMES: tuple[str, ...] = get_args(ScheduleName)


def normalize_schedule(schedule: str) -> ScheduleName:
    """Return a validated ladder schedule name.

    Input: schedule is a string from user config.
    Output: a ScheduleName literal, or ValueError if the string is unknown.
    """
    if schedule not in SCHEDULE_NAMES:
        raise ValueError(
            f"unknown ladder schedule {schedule!r}; expected one of {SCHEDULE_NAMES}"
        )
    return schedule  # pyrefly: ignore[bad-return]

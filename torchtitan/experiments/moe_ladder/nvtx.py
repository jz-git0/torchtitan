# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""NVTX helpers for MoE ladder profiling."""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager

import torch

_NVTX_ENABLED = os.environ.get("MOE_LADDER_NVTX") == "1"


@contextmanager
def nvtx_range(name: str) -> Iterator[None]:
    """Emit an NVTX range when explicitly enabled on CUDA."""
    if not _NVTX_ENABLED or not torch.cuda.is_available():
        yield
        return

    torch.cuda.nvtx.range_push(name)
    try:
        yield
    finally:
        torch.cuda.nvtx.range_pop()

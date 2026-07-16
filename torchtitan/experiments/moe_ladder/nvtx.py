# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""NVTX helpers for MoE ladder profiling."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

import torch


@contextmanager
def nvtx_range(name: str) -> Iterator[None]:
    """Emit an NVTX range when CUDA NVTX is available, otherwise no-op."""
    nvtx = getattr(torch.cuda, "nvtx", None)
    if nvtx is None or not torch.cuda.is_available():
        yield
        return

    try:
        nvtx.range_push(name)
    except RuntimeError:
        yield
        return

    try:
        yield
    finally:
        nvtx.range_pop()

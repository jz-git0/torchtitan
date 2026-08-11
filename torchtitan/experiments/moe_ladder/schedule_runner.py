# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Shared execution helpers for MoE ladder schedules.

Shape suffix legend:
  B=batch, L=seq, D=model dim, R=routed local tokens.
"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import nullcontext

import torch
from torch.distributed.tensor import DTensor

from .ladder_moe import LadderMoE, Pending, RouteInfo
from .nvtx import nvtx_range

AttentionFn = Callable[[torch.Tensor], torch.Tensor]
NormFn = Callable[[torch.Tensor], torch.Tensor]

_MOE_SIDE_STREAMS: dict[int, torch.cuda.Stream] = {}


def _moe_side_stream(x_BLD: torch.Tensor) -> torch.cuda.Stream | None:
    if not x_BLD.is_cuda:
        return None
    device_index = x_BLD.device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    stream = _MOE_SIDE_STREAMS.get(device_index)
    if stream is None:
        with torch.cuda.device(device_index):
            stream = torch.cuda.Stream()
        _MOE_SIDE_STREAMS[device_index] = stream
    return stream


def _record_stream(tensor: torch.Tensor, stream: torch.cuda.Stream) -> None:
    local_tensor = (
        tensor.to_local(grad_placements=tensor.placements)
        if isinstance(tensor, DTensor)
        else tensor
    )
    if local_tensor.is_cuda:
        local_tensor.record_stream(stream)


def _wait_moe_stream_after_current(
    stream: torch.cuda.Stream | None,
    *side_inputs: torch.Tensor,
) -> None:
    if stream is not None:
        assert side_inputs
        stream.wait_stream(torch.cuda.current_stream(side_inputs[0].device))
        for tensor in side_inputs:
            _record_stream(tensor, stream)


def _wait_current_after_moe_stream(
    stream: torch.cuda.Stream | None,
    x_BLD: torch.Tensor,
    *side_outputs: torch.Tensor,
) -> None:
    if stream is not None:
        # TODO: Delayed callers conservatively join the full side stream. A
        # residual-ready event would allow cross-block pipelining.
        current_stream = torch.cuda.current_stream(x_BLD.device)
        current_stream.wait_stream(stream)
        for output in side_outputs:
            _record_stream(output, current_stream)


def _moe_stream_context(stream: torch.cuda.Stream | None):
    if stream is None:
        return nullcontext()
    return torch.cuda.stream(stream)


def dispatch_experts(
    moe: LadderMoE,
    moe_in_BLD: torch.Tensor,
    route_info: RouteInfo,
) -> Pending:
    """Run one dispatch phase followed by local expert compute.

    Input: moe_in_BLD is [B, L, D], route_info has top-k metadata [T, K].
    Output: Pending tuple (expert output [R, D], DispatchState).
    """
    with nvtx_range("moe_ladder/dispatch_experts/dispatch"):
        routed_RD, counts_e, state = moe.dispatch(moe_in_BLD, route_info)
    with nvtx_range("moe_ladder/dispatch_experts/experts"):
        e_RD = moe.experts_forward(routed_RD, counts_e)
    return e_RD, state


def parallel_step(
    moe: LadderMoE,
    res_BLD: torch.Tensor,
    *,
    attention: AttentionFn,
    ffn_norm: NormFn,
) -> torch.Tensor:
    """Run option 1, where attention and MoE read the same residual.

    Input: res_BLD is [B, L, D]; attention and ffn_norm preserve [B, L, D].
    Output: updated residual [B, L, D].

    Dispatch is one indivisible phase; combine exposes separate launch and wait
    calls. Attention is submitted on the current stream before dispatch,
    experts, and combine are submitted on the MoE stream. Any of those MoE
    phases may overlap attention. Both branches read the same residual; this
    concurrency changes execution order, not the value defined by the parallel
    block.
    """
    with nvtx_range("moe_ladder/parallel/ffn_norm_prepare"):
        moe_in_BLD = moe.prepare_input(ffn_norm(res_BLD))
    with nvtx_range("moe_ladder/parallel/route"):
        route_info = moe.route(moe_in_BLD)

    moe_stream = _moe_side_stream(res_BLD)
    _wait_moe_stream_after_current(
        moe_stream,
        moe_in_BLD,
        route_info.topk_scores_TK,
        route_info.topk_expert_ids_TK,
        route_info.num_local_tokens_per_expert_E,
    )

    with nvtx_range("moe_ladder/parallel/attention"):
        attn_BLD = attention(res_BLD)
    with _moe_stream_context(moe_stream):
        e_RD, state = dispatch_experts(moe, moe_in_BLD, route_info)
        with nvtx_range("moe_ladder/parallel/begin_combine"):
            combine_h = moe.begin_combine(e_RD, state)
        with nvtx_range("moe_ladder/parallel/finish_combine"):
            moe_out_BLD = moe.finish_combine_bld(combine_h, state)
    _wait_current_after_moe_stream(moe_stream, res_BLD, moe_out_BLD)
    return res_BLD + attn_BLD + moe_out_BLD


def ladder_step(
    moe: LadderMoE,
    res_BLD: torch.Tensor,
    pending: Pending | None,
    *,
    attention: AttentionFn,
    ffn_norm: NormFn,
) -> tuple[torch.Tensor, Pending]:
    """Run option 2, delaying MoE output by one block.

    Input: res_BLD is [B, L, D], pending is None or (expert output [R, D], state).
    Output: (updated residual [B, L, D], next Pending).
    """
    base_res_BLD = res_BLD
    moe_stream = _moe_side_stream(base_res_BLD)
    if pending is not None:
        with _moe_stream_context(moe_stream):
            with nvtx_range("moe_ladder/ladder/begin_combine"):
                combine_h = moe.begin_combine(*pending)
    _wait_moe_stream_after_current(moe_stream, base_res_BLD)

    with nvtx_range("moe_ladder/ladder/attention"):
        attn_BLD = attention(base_res_BLD)

    with _moe_stream_context(moe_stream):
        if pending is not None:
            with nvtx_range("moe_ladder/ladder/finish_combine"):
                res_BLD = base_res_BLD + moe.finish_combine_bld(combine_h, pending[1])
        with nvtx_range("moe_ladder/ladder/ffn_norm_prepare"):
            moe_in_BLD = moe.prepare_input(ffn_norm(res_BLD))
        with nvtx_range("moe_ladder/ladder/route"):
            route_info = moe.route(moe_in_BLD)
        with nvtx_range("moe_ladder/ladder/dispatch_experts"):
            pending_next = dispatch_experts(moe, moe_in_BLD, route_info)
    _wait_current_after_moe_stream(moe_stream, base_res_BLD, res_BLD)
    return res_BLD + attn_BLD, pending_next


def hoisted_gate_a_step(
    moe: LadderMoE,
    res_BLD: torch.Tensor,
    pending: Pending | None,
    *,
    attention: AttentionFn,
    ffn_norm: NormFn,
) -> tuple[torch.Tensor, Pending]:
    """Run option 3A with stale gate decisions and combine launched first.

    Input: res_BLD is [B, L, D], pending is None or (expert output [R, D], state).
    Output: (updated residual [B, L, D], next Pending).

    The previous block's combine is launched before gate computation, so its
    communication can overlap the stale gate and attention. Token dispatch
    occurs later, after combine completes, using the stale gate decisions.
    """
    base_res_BLD = res_BLD
    moe_stream = _moe_side_stream(base_res_BLD)
    if pending is not None:
        with _moe_stream_context(moe_stream):
            with nvtx_range("moe_ladder/hoisted_gateA/begin_combine"):
                combine_h = moe.begin_combine(*pending)

    with nvtx_range("moe_ladder/hoisted_gateA/stale_ffn_norm_prepare"):
        stale_moe_in_BLD = moe.prepare_input(ffn_norm(base_res_BLD))
    with nvtx_range("moe_ladder/hoisted_gateA/stale_route"):
        route_info = moe.route(stale_moe_in_BLD)

    _wait_moe_stream_after_current(
        moe_stream,
        base_res_BLD,
        stale_moe_in_BLD,
        route_info.topk_scores_TK,
        route_info.topk_expert_ids_TK,
        route_info.num_local_tokens_per_expert_E,
    )
    if pending is None:
        with nvtx_range("moe_ladder/hoisted_gateA/attention"):
            attn_BLD = attention(base_res_BLD)
        with _moe_stream_context(moe_stream):
            with nvtx_range("moe_ladder/hoisted_gateA/dispatch_experts"):
                pending_next = dispatch_experts(moe, stale_moe_in_BLD, route_info)
        _wait_current_after_moe_stream(moe_stream, base_res_BLD)
        return base_res_BLD + attn_BLD, pending_next

    with nvtx_range("moe_ladder/hoisted_gateA/attention"):
        attn_BLD = attention(base_res_BLD)
    with _moe_stream_context(moe_stream):
        with nvtx_range("moe_ladder/hoisted_gateA/finish_combine"):
            res_BLD = base_res_BLD + moe.finish_combine_bld(combine_h, pending[1])
        with nvtx_range("moe_ladder/hoisted_gateA/fresh_ffn_norm_prepare"):
            moe_in_BLD = moe.prepare_input(ffn_norm(res_BLD))
        with nvtx_range("moe_ladder/hoisted_gateA/dispatch_experts"):
            pending_next = dispatch_experts(moe, moe_in_BLD, route_info)

    _wait_current_after_moe_stream(moe_stream, base_res_BLD, res_BLD)
    return res_BLD + attn_BLD, pending_next


def hoisted_gate_b_step(
    moe: LadderMoE,
    res_BLD: torch.Tensor,
    pending: Pending | None,
    *,
    attention: AttentionFn,
    ffn_norm: NormFn,
) -> tuple[torch.Tensor, Pending]:
    """Run option 3B with stale gate decisions computed before combine.

    Input: res_BLD is [B, L, D], pending is None or (expert output [R, D], state).
    Output: (updated residual [B, L, D], next Pending).

    Gate computation runs on the stale residual before the previous combine
    launch, so it remains exposed while the combine can overlap attention.
    Token dispatch occurs after combine using the stale gate decisions.
    """
    base_res_BLD = res_BLD
    moe_stream = _moe_side_stream(base_res_BLD)
    with nvtx_range("moe_ladder/hoisted_gateB/stale_ffn_norm_prepare"):
        stale_moe_in_BLD = moe.prepare_input(ffn_norm(base_res_BLD))
    with nvtx_range("moe_ladder/hoisted_gateB/stale_route"):
        route_info = moe.route(stale_moe_in_BLD)

    _wait_moe_stream_after_current(
        moe_stream,
        base_res_BLD,
        stale_moe_in_BLD,
        route_info.topk_scores_TK,
        route_info.topk_expert_ids_TK,
        route_info.num_local_tokens_per_expert_E,
    )
    if pending is None:
        with nvtx_range("moe_ladder/hoisted_gateB/attention"):
            attn_BLD = attention(base_res_BLD)
        with _moe_stream_context(moe_stream):
            with nvtx_range("moe_ladder/hoisted_gateB/dispatch_experts"):
                pending_next = dispatch_experts(moe, stale_moe_in_BLD, route_info)
        _wait_current_after_moe_stream(moe_stream, base_res_BLD)
        return base_res_BLD + attn_BLD, pending_next

    with _moe_stream_context(moe_stream):
        with nvtx_range("moe_ladder/hoisted_gateB/begin_combine"):
            combine_h = moe.begin_combine(*pending)
    with nvtx_range("moe_ladder/hoisted_gateB/attention"):
        attn_BLD = attention(base_res_BLD)
    with _moe_stream_context(moe_stream):
        with nvtx_range("moe_ladder/hoisted_gateB/finish_combine"):
            res_BLD = base_res_BLD + moe.finish_combine_bld(combine_h, pending[1])
        with nvtx_range("moe_ladder/hoisted_gateB/fresh_ffn_norm_prepare"):
            moe_in_BLD = moe.prepare_input(ffn_norm(res_BLD))
        with nvtx_range("moe_ladder/hoisted_gateB/dispatch_experts"):
            pending_next = dispatch_experts(moe, moe_in_BLD, route_info)

    _wait_current_after_moe_stream(moe_stream, base_res_BLD, res_BLD)
    return res_BLD + attn_BLD, pending_next

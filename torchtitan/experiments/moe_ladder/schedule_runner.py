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

import torch

from torchtitan.models.common.token_dispatcher import TokenDispatchSplits

from .ladder_moe import LadderMoE, Pending, RouteInfo
from .nvtx import nvtx_range

AttentionFn = Callable[[torch.Tensor], torch.Tensor]
NormFn = Callable[[torch.Tensor], torch.Tensor]


def dispatch_experts(
    moe: LadderMoE,
    moe_in_BLD: torch.Tensor,
    route_info: RouteInfo,
) -> Pending:
    """Run atomic dispatch followed by local expert compute.

    Input: moe_in_BLD is [B, L, D], route_info has top-k metadata [T, K].
    Output: Pending tuple (expert output [R, D], DispatchState).
    """
    with nvtx_range("moe_ladder/dispatch_experts/dispatch"):
        routed_RD, counts_e, state = moe.dispatch(moe_in_BLD, route_info)
    with nvtx_range("moe_ladder/dispatch_experts/experts"):
        e_RD = moe.experts_forward(routed_RD, counts_e)
    return e_RD, state


def dispatch_experts_from_splits(
    moe: LadderMoE,
    moe_in_BLD: torch.Tensor,
    route_info: RouteInfo,
    splits: TokenDispatchSplits,
) -> Pending:
    """Dispatch tokens using precomputed EP splits, then run experts.

    Input: moe_in_BLD is [B, L, D], route_info has [T, K] routing, and splits are precomputed token counts.
    Output: Pending (expert output [R, D], state).
    """
    with nvtx_range("moe_ladder/dispatch_from_splits/begin_token_dispatch"):
        dispatch_h = moe.begin_token_dispatch(moe_in_BLD, route_info, splits)
    with nvtx_range("moe_ladder/dispatch_from_splits/finish_dispatch"):
        routed_RD, counts_e, state = moe.finish_dispatch(dispatch_h, moe_in_BLD)
    with nvtx_range("moe_ladder/dispatch_from_splits/experts"):
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

    Under EP the token dispatch is split around attention: the large token
    all-to-all is launched by begin_token_dispatch, attention runs, then
    finish_dispatch waits it -- so attention overlaps the big collective. The
    host split sync (sync_counts) runs before attention; it waits only on the
    tiny counts exchange, so the small sync (not the large dispatch) is the
    exposed cost. The combine all-to-all stays exposed because a single
    attention can hide only one large collective.

    Dispatchers without a host split sync (DeepEP) take the atomic-dispatch
    branch: dispatch cannot be split around attention there, so combine is
    split instead -- begin_combine launches before attention and
    finish_combine_bld waits after it. Both attention and the MoE read the
    same residual, so this reorder changes launch order only, not values.
    With DeepEP the combine runs on its own stream and genuinely overlaps
    attention; with the EP=1 local dispatcher the phases are pass-throughs.
    """
    with nvtx_range("moe_ladder/parallel/ffn_norm_prepare"):
        moe_in_BLD = moe.prepare_input(ffn_norm(res_BLD))
    with nvtx_range("moe_ladder/parallel/route"):
        route_info = moe.route(moe_in_BLD)

    if moe.ep_size > 1 and moe.needs_split_sync:
        with nvtx_range("moe_ladder/parallel/begin_counts"):
            counts_h = moe.begin_counts(route_info)
        with nvtx_range("moe_ladder/parallel/sync_counts"):
            splits = moe.sync_counts(counts_h, route_info)
        with nvtx_range("moe_ladder/parallel/begin_token_dispatch"):
            dispatch_h = moe.begin_token_dispatch(moe_in_BLD, route_info, splits)
        with nvtx_range("moe_ladder/parallel/attention"):
            attn_BLD = attention(res_BLD)
        with nvtx_range("moe_ladder/parallel/finish_dispatch"):
            routed_RD, counts_e, state = moe.finish_dispatch(dispatch_h, moe_in_BLD)
        with nvtx_range("moe_ladder/parallel/experts"):
            e_RD = moe.experts_forward(routed_RD, counts_e)
        with nvtx_range("moe_ladder/parallel/combine"):
            moe_out_BLD = moe.combine_bld(e_RD, state)
        return res_BLD + attn_BLD + moe_out_BLD

    e_RD, state = dispatch_experts(moe, moe_in_BLD, route_info)
    with nvtx_range("moe_ladder/parallel/begin_combine"):
        combine_h = moe.begin_combine(e_RD, state)
    with nvtx_range("moe_ladder/parallel/attention"):
        attn_BLD = attention(res_BLD)
    with nvtx_range("moe_ladder/parallel/finish_combine"):
        moe_out_BLD = moe.finish_combine_bld(combine_h, state)
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
    if pending is not None:
        with nvtx_range("moe_ladder/ladder/begin_combine"):
            combine_h = moe.begin_combine(*pending)
    else:
        combine_h = None

    with nvtx_range("moe_ladder/ladder/attention"):
        attn_BLD = attention(res_BLD)
    if pending is not None:
        with nvtx_range("moe_ladder/ladder/finish_combine"):
            res_BLD = res_BLD + moe.finish_combine_bld(combine_h, pending[1])

    with nvtx_range("moe_ladder/ladder/ffn_norm_prepare"):
        moe_in_BLD = moe.prepare_input(ffn_norm(res_BLD))
    with nvtx_range("moe_ladder/ladder/route"):
        route_info = moe.route(moe_in_BLD)
    with nvtx_range("moe_ladder/ladder/dispatch_experts"):
        pending_next = dispatch_experts(moe, moe_in_BLD, route_info)
    return res_BLD + attn_BLD, pending_next


def hoisted_gate_a_step(
    moe: LadderMoE,
    res_BLD: torch.Tensor,
    pending: Pending | None,
    *,
    attention: AttentionFn,
    ffn_norm: NormFn,
) -> tuple[torch.Tensor, Pending]:
    """Run option 3A with stale routing and combine before counts.

    Input: res_BLD is [B, L, D], pending is None or (expert output [R, D], state).
    Output: (updated residual [B, L, D], next Pending).

    The previous block's combine all-to-all is launched before routing, so it
    overlaps both the (stale) routing gate and attention. Combine is issued
    before counts, so counts queue behind the large combine on the shared EP
    communicator and the host split sync resolves later. Compare hoisted_gateB,
    which issues counts first for a faster sync but cannot overlap routing.
    """
    if pending is not None:
        with nvtx_range("moe_ladder/hoisted_gateA/begin_combine"):
            combine_h = moe.begin_combine(*pending)
    else:
        combine_h = None

    with nvtx_range("moe_ladder/hoisted_gateA/stale_ffn_norm_prepare"):
        stale_moe_in_BLD = moe.prepare_input(ffn_norm(res_BLD))
    with nvtx_range("moe_ladder/hoisted_gateA/stale_route"):
        route_info = moe.route(stale_moe_in_BLD)

    if pending is None:
        with nvtx_range("moe_ladder/hoisted_gateA/attention"):
            attn_BLD = attention(res_BLD)
        with nvtx_range("moe_ladder/hoisted_gateA/dispatch_experts"):
            pending_next = dispatch_experts(moe, stale_moe_in_BLD, route_info)
        return res_BLD + attn_BLD, pending_next

    if moe.ep_size > 1 and moe.needs_split_sync:
        with nvtx_range("moe_ladder/hoisted_gateA/begin_counts"):
            counts_h = moe.begin_counts(route_info)
        with nvtx_range("moe_ladder/hoisted_gateA/attention"):
            attn_BLD = attention(res_BLD)
        with nvtx_range("moe_ladder/hoisted_gateA/sync_counts"):
            splits = moe.sync_counts(counts_h, route_info)
        with nvtx_range("moe_ladder/hoisted_gateA/finish_combine"):
            res_BLD = res_BLD + moe.finish_combine_bld(combine_h, pending[1])
        with nvtx_range("moe_ladder/hoisted_gateA/fresh_ffn_norm_prepare"):
            moe_in_BLD = moe.prepare_input(ffn_norm(res_BLD))
        with nvtx_range("moe_ladder/hoisted_gateA/dispatch_from_splits"):
            pending_next = dispatch_experts_from_splits(
                moe, moe_in_BLD, route_info, splits
            )
    else:
        with nvtx_range("moe_ladder/hoisted_gateA/attention"):
            attn_BLD = attention(res_BLD)
        with nvtx_range("moe_ladder/hoisted_gateA/finish_combine"):
            res_BLD = res_BLD + moe.finish_combine_bld(combine_h, pending[1])
        with nvtx_range("moe_ladder/hoisted_gateA/fresh_ffn_norm_prepare"):
            moe_in_BLD = moe.prepare_input(ffn_norm(res_BLD))
        with nvtx_range("moe_ladder/hoisted_gateA/dispatch_experts"):
            pending_next = dispatch_experts(moe, moe_in_BLD, route_info)

    return res_BLD + attn_BLD, pending_next


def hoisted_gate_b_step(
    moe: LadderMoE,
    res_BLD: torch.Tensor,
    pending: Pending | None,
    *,
    attention: AttentionFn,
    ffn_norm: NormFn,
) -> tuple[torch.Tensor, Pending]:
    """Run option 3B with stale routing and counts before combine.

    Input: res_BLD is [B, L, D], pending is None or (expert output [R, D], state).
    Output: (updated residual [B, L, D], next Pending).

    Counts are issued before combine on the shared EP communicator, and the host
    split sync runs before attention. The tiny counts exchange completes first,
    so the sync resolves eagerly -- it neither queues behind the large combine
    nor waits on attention (its split sums do not sit behind attention on the
    compute stream) -- and the host then runs ahead while attention overlaps the
    large combine. Routing precedes every collective (counts depend on it), so
    its gate compute is not overlapped. The eager sync exposes only the small
    counts latency, which does not gate the combine-bound critical path (dispatch
    waits on the combine-updated residual, not on the splits). Compare
    hoisted_gateA, which launches combine first to overlap routing too but delays
    the sync behind the large combine.
    """
    with nvtx_range("moe_ladder/hoisted_gateB/stale_ffn_norm_prepare"):
        stale_moe_in_BLD = moe.prepare_input(ffn_norm(res_BLD))
    with nvtx_range("moe_ladder/hoisted_gateB/stale_route"):
        route_info = moe.route(stale_moe_in_BLD)

    if pending is None:
        with nvtx_range("moe_ladder/hoisted_gateB/attention"):
            attn_BLD = attention(res_BLD)
        with nvtx_range("moe_ladder/hoisted_gateB/dispatch_experts"):
            pending_next = dispatch_experts(moe, stale_moe_in_BLD, route_info)
        return res_BLD + attn_BLD, pending_next

    if moe.ep_size > 1 and moe.needs_split_sync:
        with nvtx_range("moe_ladder/hoisted_gateB/begin_counts"):
            counts_h = moe.begin_counts(route_info)
        with nvtx_range("moe_ladder/hoisted_gateB/begin_combine"):
            combine_h = moe.begin_combine(*pending)
        with nvtx_range("moe_ladder/hoisted_gateB/sync_counts"):
            splits = moe.sync_counts(counts_h, route_info)
        with nvtx_range("moe_ladder/hoisted_gateB/attention"):
            attn_BLD = attention(res_BLD)
        with nvtx_range("moe_ladder/hoisted_gateB/finish_combine"):
            res_BLD = res_BLD + moe.finish_combine_bld(combine_h, pending[1])
        with nvtx_range("moe_ladder/hoisted_gateB/fresh_ffn_norm_prepare"):
            moe_in_BLD = moe.prepare_input(ffn_norm(res_BLD))
        with nvtx_range("moe_ladder/hoisted_gateB/dispatch_from_splits"):
            pending_next = dispatch_experts_from_splits(
                moe, moe_in_BLD, route_info, splits
            )
    else:
        with nvtx_range("moe_ladder/hoisted_gateB/begin_combine"):
            combine_h = moe.begin_combine(*pending)
        with nvtx_range("moe_ladder/hoisted_gateB/attention"):
            attn_BLD = attention(res_BLD)
        with nvtx_range("moe_ladder/hoisted_gateB/finish_combine"):
            res_BLD = res_BLD + moe.finish_combine_bld(combine_h, pending[1])
        with nvtx_range("moe_ladder/hoisted_gateB/fresh_ffn_norm_prepare"):
            moe_in_BLD = moe.prepare_input(ffn_norm(res_BLD))
        with nvtx_range("moe_ladder/hoisted_gateB/dispatch_experts"):
            pending_next = dispatch_experts(moe, moe_in_BLD, route_info)

    return res_BLD + attn_BLD, pending_next

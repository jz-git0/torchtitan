# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Expert-parallel MoE with phase-split dispatch/combine over DeepEP.

This module exposes the common MoE forward as smaller phases so schedule code
can overlap expert-parallel communication with attention. The methods still use
the common router, token dispatcher, and grouped experts; this file owns only
the phase boundaries and the state that must survive between those phases.

DeepEP is the only supported dispatcher. It runs collectives on its own stream.
It also owns the dispatch layout, so this scheduler has no explicit count
exchange or host-materialized split list to reorder. Dispatch is one
indivisible phase; only combine is phase-split. DeepEP also requires an EP mesh,
so EP=1 is rejected.

Shape suffix legend:
  B=batch, L=seq, D=model dim, F=FFN hidden, E=experts, e=local experts (E/ep),
  K=top-k, T=B*L tokens, N=T*K routed slots, R=routed tokens on local experts.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, Partial, Placement, Replicate

from torchtitan.distributed import ParallelDims
from torchtitan.distributed.spmd_types import maybe_set_sparse_mesh
from torchtitan.distributed.utils import check_dtensor_placements_match
from torchtitan.models.common.moe import MoE
from torchtitan.models.common.token_dispatcher import (
    DeepEPDispatchMetadata,
    DeepEPTokenDispatcher,
    TokenCombineHandle,
)
from torchtitan.protocols.module import Module
from torchtitan.protocols.sharding import resolve_placements


@dataclass
class RouteInfo:
    """Gate metadata carried from route() into token dispatch.

    Top-k tensors use the token-major [T, K] shape expected by the dispatcher.
    DeepEP derives expert counts on device; the count tensor is retained only
    to satisfy the common dispatcher interface.
    """

    topk_scores_TK: torch.Tensor  # noqa: N815
    topk_expert_ids_TK: torch.Tensor  # noqa: N815
    num_local_tokens_per_expert_E: torch.Tensor  # noqa: N815


@dataclass
class DispatchState:
    """Metadata needed to combine dispatched expert output later.

    It records the DeepEP handle, local shape, and output placements. Delayed
    schedules carry it with expert output as Pending.
    """

    metadata: DeepEPDispatchMetadata
    batch_size: int
    local_seq_len: int
    dim: int
    output_device_mesh: DeviceMesh | None = None
    output_src_placements: tuple[Placement, ...] | None = None
    output_dst_placements: tuple[Placement, ...] | None = None


# Delayed expert output plus the DispatchState needed to combine it later.
Pending = tuple[torch.Tensor, DispatchState]


@dataclass(frozen=True)
class _PhaseInput:
    """Placement-aware MoE input and its local phase tensor."""

    x_BLD: torch.Tensor  # noqa: N815
    local_x_BLD: torch.Tensor  # noqa: N815


class LadderMoE(MoE):
    """Common MoE with ladder-specific phase boundaries.

    The wrapper keeps the common router, experts, and token dispatcher, but
    rejects features whose phase boundaries are not implemented here yet:
    shared experts and any dispatch backend other than DeepEP.

    Input: Config wraps a common MoE config with router, experts, and optional sharding_config.
    Output: [B, L, D] forward result plus route, dispatch, experts, and combine phase methods.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        """Config for a LadderMoE built from a common MoE config."""

        moe: MoE.Config

    def __init__(self, *, config: Config) -> None:
        moe_config = config.moe
        if moe_config.shared_experts is not None:
            raise ValueError("ladder MoE does not support shared experts yet")
        super().__init__(moe_config)
        self.token_dispatcher = self.experts.token_dispatcher
        if not isinstance(self.token_dispatcher, DeepEPTokenDispatcher):
            raise ValueError("ladder MoE supports the DeepEP dispatcher only")

        self._phase_input_device_mesh: DeviceMesh | None = None
        self._phase_input_src_placements: tuple[Placement, ...] | None = None
        self._phase_input_dst_placements: tuple[Placement, ...] | None = None
        self._phase_plain_input_allowed = True
        self._phase_output_device_mesh: DeviceMesh | None = None
        self._phase_output_src_placements: tuple[Placement, ...] | None = None
        self._phase_output_dst_placements: tuple[Placement, ...] | None = None

    @staticmethod
    def _to_local(tensor: torch.Tensor) -> torch.Tensor:
        """Return a local tensor view for DTensor inputs.

        Input: tensor is a Tensor or DTensor with any shape.
        Output: local Tensor with the same local shape, or the input Tensor unchanged.
        """
        return (
            tensor.to_local(grad_placements=tensor.placements)
            if isinstance(tensor, DTensor)
            else tensor
        )

    def _validate_sp_dispatch_shape(
        self, x_BLD: torch.Tensor, local_x_BLD: torch.Tensor
    ) -> None:
        """Validate EP+TP sequence-shard assumptions for phase methods.

        Input: x_BLD is global/local DTensor [B, L, D]; local_x_BLD is [B, L_local, D].
        Output: None or ValueError for unsupported sequence padding.
        """
        sp_size = getattr(self.token_dispatcher, "sp_size", 1)
        if (
            sp_size == 1
            or not self.seq_dim_tp_sharded
            or not isinstance(x_BLD, DTensor)
        ):
            return

        global_seq_len = x_BLD.shape[1]
        if global_seq_len < sp_size or global_seq_len % sp_size != 0:
            raise ValueError(
                "ladder MoE EP+TP currently requires sequence length to be "
                "divisible by TP and at least TP; common MoE padding is not "
                "implemented for phase-split schedules yet"
            )

        expected_local_seq_len = global_seq_len // sp_size
        if local_x_BLD.shape[1] != expected_local_seq_len:
            raise ValueError(
                "ladder MoE EP+TP received an uneven sequence shard; common "
                "MoE sequence padding is not implemented for phase-split "
                "schedules yet"
            )

    def _configure_phase_dtensor_specs(self, parallel_dims: ParallelDims) -> None:
        """Cache DTensor placement specs used by public phase methods.

        Input: parallel_dims resolves meshes from the common MoE sharding config.
        Output: updates input and output placement metadata on this module.
        """
        sharding_config = self._sharding_config
        if sharding_config is None:
            return

        self._phase_plain_input_allowed = parallel_dims.spmd_backend != "full_dtensor"
        x_src = (sharding_config.in_src_shardings or {}).get("x_BLD")
        x_dst = (sharding_config.in_dst_shardings or {}).get("x_BLD")
        x_mesh = parallel_dims.resolve_shared_mesh([x_src, x_dst])
        if x_mesh is not None:
            self._phase_input_device_mesh = x_mesh
            if x_src is not None:
                self._phase_input_src_placements = resolve_placements(x_src, x_mesh)
            if x_dst is not None:
                self._phase_input_dst_placements = resolve_placements(x_dst, x_mesh)

        out_src = sharding_config.out_src_shardings
        if isinstance(out_src, tuple):
            raise ValueError("ladder MoE expects a single tensor output")
        out_dst = sharding_config.out_dst_shardings
        out_mesh = parallel_dims.resolve_shared_mesh([out_src, out_dst])
        if out_mesh is not None and out_src is not None:
            self._phase_output_device_mesh = out_mesh
            self._phase_output_src_placements = resolve_placements(out_src, out_mesh)
            if out_dst is not None:
                self._phase_output_dst_placements = resolve_placements(
                    out_dst,
                    out_mesh,
                )

    def prepare_input(self, x_BLD: torch.Tensor) -> torch.Tensor:
        """Redistribute a MoE input to the common MoE compute layout.

        Input: x_BLD is [B, L, D] Tensor or DTensor.
        Output: [B, L, D] with the requested placements.
        """
        mesh = self._phase_input_device_mesh
        src = self._phase_input_src_placements
        dst = self._phase_input_dst_placements
        if mesh is None:
            return x_BLD
        if not isinstance(x_BLD, DTensor):
            if not self._phase_plain_input_allowed:
                raise ValueError(
                    "phase-split MoE requires a DTensor input in full_dtensor mode"
                )
            if src is None:
                return x_BLD
            x_BLD = DTensor.from_local(
                x_BLD,
                mesh,
                list(src),
                run_check=False,
                grad_placements=list(src),
            )
        if src is not None and not check_dtensor_placements_match(
            x_BLD.placements,
            src,
            x_BLD.ndim,
        ):
            raise ValueError(
                "phase-split MoE input placements "
                f"{x_BLD.placements} do not match the expected source {src}"
            )
        if dst is None or check_dtensor_placements_match(
            x_BLD.placements,
            dst,
            x_BLD.ndim,
        ):
            return x_BLD
        return x_BLD.redistribute(placements=dst, async_op=True)

    def _prepare_phase_input(self, x_BLD: torch.Tensor) -> _PhaseInput:
        """Apply the DTensor input boundary shared by phase methods."""
        local_x_BLD = self._to_local(x_BLD)
        self._validate_sp_dispatch_shape(x_BLD, local_x_BLD)
        return _PhaseInput(x_BLD=x_BLD, local_x_BLD=local_x_BLD)

    def _output_specs(
        self,
        x_BLD: torch.Tensor,
    ) -> tuple[
        DeviceMesh | None,
        tuple[Placement, ...] | None,
        tuple[Placement, ...] | None,
    ]:
        """Return output mesh and placement metadata for phase combine.

        Input: x_BLD is [B, L, D] Tensor or DTensor.
        Output: (mesh, source placements, destination placements), or all None for non-DTensor.
        """
        if not isinstance(x_BLD, DTensor):
            return None, None, None
        if (
            self._phase_output_device_mesh is None
            or self._phase_output_src_placements is None
        ):
            raise ValueError(
                "ladder MoE phase output needs sharding metadata; call "
                "parallelize before passing DTensor inputs"
            )
        return (
            self._phase_output_device_mesh,
            self._phase_output_src_placements,
            self._phase_output_dst_placements,
        )

    def _wrap_output_src(
        self,
        out_BLD: torch.Tensor,
        state: DispatchState,
    ) -> torch.Tensor:
        """Wrap local combine output as the common MoE source placement.

        Input: out_BLD is local [B, L, D] and state carries output placements.
        Output: DTensor [B, L, D] when placements exist, else the input Tensor.
        """
        if state.output_device_mesh is None:
            return out_BLD
        assert state.output_src_placements is not None
        # A Partial forward output receives replicated local gradients.
        return DTensor.from_local(
            out_BLD,
            state.output_device_mesh,
            list(state.output_src_placements),
            run_check=False,
            grad_placements=[
                Replicate() if isinstance(p, Partial) else p
                for p in state.output_src_placements
            ],
        )

    def _restore_output(
        self,
        out_BLD: torch.Tensor,
        state: DispatchState,
    ) -> torch.Tensor:
        """Wrap and redistribute combine output to the residual layout.

        Input: out_BLD is local [B, L, D] and state has output specs.
        Output: Tensor or DTensor [B, L, D] in the schedule residual placement.
        """
        out_BLD = self._wrap_output_src(out_BLD, state)
        if not isinstance(out_BLD, DTensor):
            return out_BLD
        dst = state.output_dst_placements
        if dst is None or check_dtensor_placements_match(
            out_BLD.placements,
            dst,
            out_BLD.ndim,
        ):
            return out_BLD
        return out_BLD.redistribute(placements=dst, async_op=True)

    def parallelize(self, parallel_dims: ParallelDims) -> None:
        """Apply common MoE parallelization and wire phase metadata.

        Input: parallel_dims defines EP/TP meshes and sharding placements.
        Output: module parameters and dispatcher are parallelized in place.
        """
        ep_mesh = parallel_dims.get_optional_mesh("ep")
        if ep_mesh is None:
            raise ValueError(
                "ladder MoE requires expert parallelism: the DeepEP dispatcher "
                "needs an EP mesh, so set expert_parallel_degree > 1"
            )
        num_experts = self.token_dispatcher.num_experts
        ep_size = ep_mesh.size()
        if num_experts % ep_size != 0:
            raise ValueError(
                f"num_experts ({num_experts}) must be divisible by EP ({ep_size})"
            )
        super().parallelize(parallel_dims)
        self._configure_phase_dtensor_specs(parallel_dims)

    def route(self, x_BLD: torch.Tensor) -> RouteInfo:
        """Compute the gate and flatten top-k metadata for later dispatch.

        This is the first schedulable MoE phase. It uses the inherited common
        MoE router and updates load-balancing accounting through _route(). It
        then converts DTensor results to local tensors because the dispatcher
        phase APIs operate on local dynamic token counts.

        The returned RouteInfo can be consumed immediately or saved for token
        dispatch after other work.

        Input: x_BLD is [B, L, D] Tensor or DTensor.
        Output: RouteInfo contains scores [T, K], ids [T, K], and counts [E].
        """
        topk_scores_BLK, topk_ids_BLK, counts_E = self._route(x_BLD)
        topk_scores_BLK = self._to_local(topk_scores_BLK)
        topk_ids_BLK = self._to_local(topk_ids_BLK)
        counts_E = self._to_local(counts_E)
        B, L, K = topk_scores_BLK.shape
        return RouteInfo(
            topk_scores_BLK.reshape(B * L, K),
            topk_ids_BLK.reshape(B * L, K),
            counts_E,
        )

    def _state(
        self,
        x_BLD: torch.Tensor,
        metadata: DeepEPDispatchMetadata,
        *,
        local_x_BLD: torch.Tensor,
    ) -> DispatchState:
        """Build DispatchState from input shape and dispatcher metadata.

        This state lets combine run after attention or in the next block.

        Input: x_BLD is [B, L, D], metadata is dispatch state, and
        local_x_BLD is [B, L_local, D].
        Output: DispatchState.
        """
        B, L, D = local_x_BLD.shape
        output_mesh, output_src, output_dst = self._output_specs(x_BLD)
        return DispatchState(
            metadata=metadata,
            batch_size=B,
            local_seq_len=L,
            dim=D,
            output_device_mesh=output_mesh,
            output_src_placements=output_src,
            output_dst_placements=output_dst,
        )

    def experts_forward(
        self, routed_RD: torch.Tensor, counts_e: torch.Tensor
    ) -> torch.Tensor:
        """Run local grouped expert MLPs on dispatched tokens.

        This reuses the common GroupedExperts._experts_forward() so the ladder
        path keeps the production grouped-GEMM implementation.

        Input: routed_RD is [R, D], counts_e gives tokens per local expert [e].
        Output: routed expert output [R, D].
        """
        with maybe_set_sparse_mesh():
            return self.experts._experts_forward(routed_RD, counts_e)

    def begin_combine(
        self, routed_out_RD: torch.Tensor, state: DispatchState
    ) -> TokenCombineHandle:
        """Launch asynchronous inverse token exchange for combine.

        This consumes the expert output and the DispatchState from the matching
        dispatch. Delayed schedules call it on a Pending value from an earlier
        block so the large combine all-to-all can overlap with current attention.

        Input: routed_out_RD is expert output [R, D], state carries dispatch metadata.
        Output: TokenCombineHandle.
        """
        return self.token_dispatcher.begin_token_combine(
            routed_out_RD,
            state.metadata,
        )

    def finish_combine(
        self, combine_h: TokenCombineHandle, state: DispatchState
    ) -> torch.Tensor:
        """Wait for combine and return tokens in local order.

        Score application and local unpermutation happen in begin_combine()
        before the DeepEP exchange. Use finish_combine_bld() or combine_bld()
        to restore the residual shape and placements.

        Input: combine_h is from begin_combine, and state has token shape T and model dim D.
        Output: flattened combined tokens [T, D].
        """
        # DeepEP ignores x_TD; reuse the handle tensor for the shared signature.
        return self.token_dispatcher.finish_token_combine(
            combine_h,
            state.metadata,
            combine_h.routed_output_RD,
            num_local_tokens_after_padding=state.batch_size * state.local_seq_len,
            local_seq_len_after_padding=state.local_seq_len,
        )

    def finish_combine_bld(
        self,
        combine_h: TokenCombineHandle,
        state: DispatchState,
    ) -> torch.Tensor:
        """Finish combine and restore residual tensor shape.

        Input: combine_h is from begin_combine and state stores B, L, D.
        Output: Tensor or DTensor residual [B, L, D].
        """
        out_BLD = self.finish_combine(combine_h, state).view(
            state.batch_size,
            -1,
            state.dim,
        )
        return self._restore_output(out_BLD, state)

    def combine_bld(
        self, routed_out_RD: torch.Tensor, state: DispatchState
    ) -> torch.Tensor:
        """Run begin_combine and finish_combine_bld as one call.

        Input: routed_out_RD is [R, D] with its DispatchState.
        Output: residual tensor [B, L, D].
        """
        combine_h = self.begin_combine(routed_out_RD, state)
        return self.finish_combine_bld(combine_h, state)

    def dispatch(
        self, x_BLD: torch.Tensor, route_info: RouteInfo
    ) -> tuple[torch.Tensor, torch.Tensor, DispatchState]:
        """Run token dispatch as one phase.

        DeepEP owns its dispatch layout, so this API exposes no count or
        split-list synchronization phase. Dispatch remains one phase; only
        combine is phase-split.

        Input: x_BLD is [B, L, D] and route_info has [T, K] routing metadata.
        Output: routed tokens [R, D], expert counts [e], and DispatchState.
        """
        phase_input = self._prepare_phase_input(x_BLD)
        x_TD = phase_input.local_x_BLD.flatten(0, 1)
        routed_RD, counts_e, metadata = self.token_dispatcher.dispatch(
            x_TD,
            route_info.topk_scores_TK,
            route_info.topk_expert_ids_TK,
            route_info.num_local_tokens_per_expert_E,
        )
        return (
            routed_RD,
            counts_e,
            self._state(
                phase_input.x_BLD,
                metadata,
                local_x_BLD=phase_input.local_x_BLD,
            ),
        )

    def combine(
        self, routed_out_RD: torch.Tensor, state: DispatchState
    ) -> torch.Tensor:
        """Run begin_combine and finish_combine as one call.

        Input: routed_out_RD is [R, D] plus DispatchState.
        Output: flattened combined tokens [T, D].
        """
        return self.finish_combine(self.begin_combine(routed_out_RD, state), state)

    def forward(self, x_BLD: torch.Tensor) -> torch.Tensor:
        """Run all phases atomically as the parity-test reference path.

        Schedules call the phase methods directly.

        Input: x_BLD is [B, L, D] Tensor or DTensor.
        Output: MoE result [B, L, D] in the common MoE source output placement.
        """
        x_BLD = self.prepare_input(x_BLD)
        route_info = self.route(x_BLD)
        routed_RD, counts_e, state = self.dispatch(x_BLD, route_info)
        out_TD = self.combine(self.experts_forward(routed_RD, counts_e), state)
        out_BLD = out_TD.view(
            state.batch_size,
            -1,
            state.dim,
        )
        return self._wrap_output_src(out_BLD, state)

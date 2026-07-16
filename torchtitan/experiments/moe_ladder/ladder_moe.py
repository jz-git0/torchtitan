# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Expert-parallel MoE with phase-split dispatch/combine.

This module exposes the common MoE forward as smaller phases so schedule code
can overlap expert-parallel communication with attention. The methods still use
the common router, token dispatcher, and grouped experts; this file owns only
the phase boundaries and the state that must survive between those phases.

Shape suffix legend:
  B=batch, L=seq, D=model dim, F=FFN hidden, E=experts, e=local experts (E/ep),
  K=top-k, T=B*L tokens, N=T*K routed slots, R=routed tokens on local experts.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, Placement
from torch.profiler import record_function

from torchtitan.distributed.spmd_types import maybe_set_sparse_mesh
from torchtitan.distributed.utils import check_dtensor_placements_match
from torchtitan.models.common.moe import GroupedExperts, MoE, TokenChoiceTopKRouter
from torchtitan.models.common.nn_modules import Linear
from torchtitan.models.common.token_dispatcher import (
    AllToAllDispatchMetadata,
    AllToAllTokenDispatcher,
    DeepEPDispatchMetadata,
    DeepEPTokenDispatcher,
    LocalDispatchMetadata,
    TokenCombineHandle,
    TokenDispatchHandle,
    TokenDispatchSplits,
)
from torchtitan.protocols.module import Module
from torchtitan.protocols.sharding import resolve_placements

# Dispatcher-specific state that combine needs to invert the matching dispatch.
DispatchMetadata = (
    LocalDispatchMetadata | AllToAllDispatchMetadata | DeepEPDispatchMetadata
)


@dataclass
class RouteInfo:
    """Routing metadata carried from route() into dispatch phases.

    Created by LadderMoE.route(). The count-exchange phases consume
    num_local_tokens_per_expert_E, while token dispatch consumes the flattened
    top-k scores and expert ids. Keeping this in one object lets schedule code
    run routing early, then decide later when to launch counts and token
    dispatch. The top-k tensors are flattened to [T, K] because the dispatcher
    APIs operate on token-major metadata rather than router-shaped [B, L, K]
    metadata.
    """

    topk_scores_TK: torch.Tensor  # noqa: N815
    topk_expert_ids_TK: torch.Tensor  # noqa: N815
    num_local_tokens_per_expert_E: torch.Tensor  # noqa: N815


@dataclass
class DispatchState:
    """State needed to invert dispatch after expert computation.

    Created by LadderMoE.finish_dispatch() through _state(), then carried with
    the routed expert output as Pending when schedules delay combine. The token
    dispatcher metadata is required to undo token permutation and EP all-to-all;
    the shape, device, and dtype fields let finish_combine() allocate the token
    buffer it passes back to the dispatcher; the output placement fields let
    finish_combine_bld() restore DTensor outputs to the residual layout.
    """

    metadata: DispatchMetadata
    batch_size: int
    local_seq_len: int
    num_tokens: int
    dim: int
    device: torch.device
    dtype: torch.dtype
    output_device_mesh: DeviceMesh | None = None
    output_src_placements: tuple[Placement, ...] | None = None
    output_dst_placements: tuple[Placement, ...] | None = None


# Delayed expert output plus the DispatchState needed to combine it later.
Pending = tuple[torch.Tensor, DispatchState]


@dataclass(frozen=True)
class _PhaseInput:
    """Prepared MoE input for manually invoked phase methods.

    Module.forward() normally lets the sharding wrapper redistribute DTensor
    inputs and wrap outputs. Phase scheduling calls route/dispatch/combine
    manually, so LadderMoE has to keep the same boundary semantics locally.
    """

    x_BLD: torch.Tensor  # noqa: N815
    local_x_BLD: torch.Tensor  # noqa: N815

    @property
    def batch_size(self) -> int:
        return self.local_x_BLD.shape[0]

    @property
    def local_seq_len(self) -> int:
        return self.local_x_BLD.shape[1]

    @property
    def dim(self) -> int:
        return self.local_x_BLD.shape[2]

    @property
    def num_tokens(self) -> int:
        return self.batch_size * self.local_seq_len

    def flatten_tokens(self) -> torch.Tensor:
        return self.local_x_BLD.reshape(self.num_tokens, self.dim)


def _default_expert_init(init_std: float) -> dict[str, Callable]:
    """Return fallback expert parameter initializers.

    This is installed only when the incoming MoE config has
    experts.param_init=None, or when from_dims() builds a standalone LadderMoE
    for tests. Stock model configs such as Qwen3 normally provide their own
    per-layer expert initializer, so this fallback is not used there.

    It is needed because GroupedExperts allocates w1_EFD, w2_EDF, and w3_EFD
    with torch.empty(), and TorchTitan Module.init_states() requires either a
    param_init mapping or a reset_parameters() implementation for direct
    parameters. GroupedExperts has no reset_parameters().
    """
    init = partial(nn.init.trunc_normal_, std=init_std)
    return {"w1_EFD": init, "w2_EDF": init, "w3_EFD": init}


def _cpu_experts_forward(
    x_RD: torch.Tensor,
    counts_E: torch.Tensor,
    w1_EFD: torch.Tensor,
    w2_EDF: torch.Tensor,
    w3_EFD: torch.Tensor,
) -> torch.Tensor:
    """Run grouped expert MLPs on CPU without torch._grouped_mm.

    LadderMoE.experts_forward() calls this only when routed_RD is on CPU. The
    common GroupedExperts._experts_forward() path uses torch._grouped_mm, which
    is the optimized grouped-GEMM path used by the real GPU MoE implementation
    and is not a portable CPU-test contract. This fallback mirrors the same
    SwiGLU expert math with ordinary PyTorch ops so CPU unit tests can exercise
    route/dispatch/combine behavior.

    Input: x_RD is [R, D], counts_E is [E], w1_EFD/w3_EFD are [E, F, D], and
    w2_EDF is [E, D, F]. Output: [R, D].
    """
    if x_RD.numel() == 0:
        return x_RD.new_empty((0, w2_EDF.shape[1]))
    pieces = []
    for x_rD, w1_FD, w2_DF, w3_FD in zip(
        x_RD.split(counts_E.cpu().tolist(), dim=0), w1_EFD, w2_EDF, w3_EFD
    ):
        h_rF = F.silu(F.linear(x_rD, w1_FD)) * F.linear(x_rD, w3_FD)
        pieces.append(F.linear(h_rF, w2_DF))
    return torch.cat(pieces, dim=0).type_as(x_RD)


class LadderMoE(MoE):
    """Common MoE with ladder-specific phase boundaries.

    The wrapper keeps the common router, experts, and token dispatcher, but
    rejects features whose phase boundaries are not implemented here yet:
    shared experts and dispatch backends other than standard all-to-all and
    DeepEP. If expert initialization is missing, __init__ installs
    _default_expert_init() before the common MoE is built.

    Input: Config wraps a common MoE config with router, experts, and optional sharding_config.
    Output: [B, L, D] forward result plus route, dispatch, experts, and combine phase methods.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        """Config for a LadderMoE built from a common MoE config.

        init_std is used only by _default_expert_init() when the wrapped common
        MoE config did not already provide experts.param_init.
        """

        moe: MoE.Config
        init_std: float = 0.02

    def __init__(self, *, config: Config) -> None:
        moe_config = config.moe
        if moe_config.shared_experts is not None:
            raise ValueError("ladder MoE does not support shared experts yet")
        if moe_config.experts.param_init is None:
            moe_config = replace(
                moe_config,
                experts=replace(
                    moe_config.experts,
                    param_init=_default_expert_init(config.init_std),
                ),
            )

        super().__init__(moe_config)
        self.dim = moe_config.experts.dim
        self.num_experts = moe_config.num_experts
        self.token_dispatcher = self.experts.token_dispatcher
        if not isinstance(
            self.token_dispatcher, (AllToAllTokenDispatcher, DeepEPTokenDispatcher)
        ):
            raise ValueError("ladder MoE supports standard all-to-all EP and DeepEP")
        # Standard all-to-all must exchange expert counts and sync split sizes
        # to the host before the token all-to-all, so schedules order the
        # counts phases explicitly. DeepEP computes its dispatch layout
        # internally (no host split lists), so schedules skip the counts
        # phases and use atomic dispatch instead.
        self.needs_split_sync = isinstance(
            self.token_dispatcher, AllToAllTokenDispatcher
        )

        self.ep_mesh: DeviceMesh | None = None
        self.ep_size = 1
        self.num_local_experts = self.num_experts
        self._phase_input_dst_placements: tuple[Placement, ...] | None = None
        self._phase_output_device_mesh: DeviceMesh | None = None
        self._phase_output_src_placements: tuple[Placement, ...] | None = None
        self._phase_output_dst_placements: tuple[Placement, ...] | None = None

    @classmethod
    def from_dims(
        cls,
        *,
        dim: int,
        hidden_dim: int,
        num_experts: int,
        top_k: int,
        ep_mesh: DeviceMesh | None = None,
        score_func: str = "softmax",
        route_norm: bool = True,
        route_scale: float = 1.0,
        init_std: float = 0.02,
    ) -> "LadderMoE":
        """Build and initialize a standalone LadderMoE for tests.

        This bypasses a model-specific config builder, so it always supplies
        _default_expert_init(). Model adapters should usually pass through the
        model's own MoE.Config instead, preserving its expert initialization.

        Input: dim D, hidden_dim F, num_experts E, top_k K, and optional EP
        mesh. Output: LadderMoE whose forward input/output shape is [B, L, D].
        """
        moe_config = MoE.Config(
            num_experts=num_experts,
            load_balance_coeff=None,
            router=TokenChoiceTopKRouter.Config(
                num_experts=num_experts,
                gate=Linear.Config(in_features=dim, out_features=num_experts),
                top_k=top_k,
                score_func=score_func,
                route_norm=route_norm,
                route_scale=route_scale,
            ),
            experts=GroupedExperts.Config(
                dim=dim,
                hidden_dim=hidden_dim,
                num_experts=num_experts,
                param_init=_default_expert_init(init_std),
                token_dispatcher=AllToAllTokenDispatcher.Config(
                    num_experts=num_experts,
                    top_k=top_k,
                ),
            ),
        )
        moe = cls(config=cls.Config(moe=moe_config, init_std=init_std))
        moe.init_states()
        if ep_mesh is not None:
            moe.wire_meshes(ep_mesh=ep_mesh, shard_plain_experts=True)
        return moe

    def _set_ep_mesh(self, ep_mesh: DeviceMesh | None) -> None:
        """Record EP mesh metadata and derive local expert count.

        Input: ep_mesh is None for EP=1 or a DeviceMesh over expert ranks.
        Output: updates ep_size and num_local_experts, or raises ValueError.
        """
        self.ep_mesh = ep_mesh
        self.ep_size = ep_mesh.size() if ep_mesh is not None else 1
        if ep_mesh is not None and self.num_experts % self.ep_size != 0:
            raise ValueError(
                f"num_experts ({self.num_experts}) must be divisible by "
                f"EP ({self.ep_size})"
            )
        self.num_local_experts = self.num_experts // self.ep_size

    @staticmethod
    def _to_local(tensor: torch.Tensor) -> torch.Tensor:
        """Return a local tensor view for DTensor inputs.

        Input: tensor is a Tensor or DTensor with any shape.
        Output: local Tensor with the same local shape, or the input Tensor unchanged.
        """
        return tensor.to_local() if isinstance(tensor, DTensor) else tensor

    def _validate_sp_dispatch_shape(
        self, x_BLD: torch.Tensor, local_x_BLD: torch.Tensor
    ) -> None:
        """Validate EP+TP sequence-shard assumptions for phase methods.

        Input: x_BLD is global/local DTensor [B, L, D]; local_x_BLD is [B, L_local, D].
        Output: None or ValueError on bad shape. The phase methods repeat this cheap check intentionally.
        """
        sp_size = getattr(self.token_dispatcher, "sp_size", 1)
        if (
            self.ep_size == 1
            or sp_size == 1
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

    def _configure_phase_dtensor_specs(self, parallel_dims) -> None:
        """Cache DTensor placement specs used by public phase methods.

        Input: parallel_dims resolves meshes from the common MoE sharding config.
        Output: updates input and output placement metadata on this module.
        """
        sharding_config = self._sharding_config
        if sharding_config is None:
            return

        x_dst = (sharding_config.in_dst_shardings or {}).get("x_BLD")
        x_mesh = parallel_dims.resolve_shared_mesh([x_dst])
        if x_mesh is not None and x_dst is not None:
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
        Output: [B, L, D] with requested placements; schedule phases may call this redundantly.
        """
        if not isinstance(x_BLD, DTensor):
            return x_BLD
        dst = self._phase_input_dst_placements
        if dst is None or check_dtensor_placements_match(
            x_BLD.placements,
            dst,
            x_BLD.ndim,
        ):
            return x_BLD
        return x_BLD.redistribute(placements=dst, async_op=True)

    def _prepare_phase_input(self, x_BLD: torch.Tensor) -> _PhaseInput:
        """Apply the DTensor input boundary shared by phase methods."""
        x_BLD = self.prepare_input(x_BLD)
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
        return DTensor.from_local(
            out_BLD,
            state.output_device_mesh,
            list(state.output_src_placements),
            run_check=False,
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

    def wire_meshes(
        self,
        *,
        ep_mesh: DeviceMesh | None,
        tp_mesh: DeviceMesh | None = None,
        shard_plain_experts: bool = False,
    ) -> None:
        """Attach EP/TP meshes for dispatch and optional expert slicing.

        Input: ep_mesh/tp_mesh identify axes; shard_plain_experts enables standalone slicing.
        Output: mutates dispatcher and possibly expert parameters in place.
        """
        self.token_dispatcher.wire_meshes(ep_mesh=ep_mesh, tp_mesh=tp_mesh)
        self._set_ep_mesh(ep_mesh)
        if ep_mesh is None or not shard_plain_experts:
            return
        if isinstance(self.experts.w1_EFD, DTensor):
            return
        if self.experts.w1_EFD.shape[0] != self.num_experts:
            return

        # Standalone from_dims() checks can wire a real EP mesh without running
        # super().parallelize(); trainer paths shard experts as DTensors first.
        rank = ep_mesh.get_local_rank()
        lo = rank * self.num_local_experts
        hi = lo + self.num_local_experts
        with torch.no_grad():
            self.experts.w1_EFD = nn.Parameter(self.experts.w1_EFD[lo:hi].contiguous())
            self.experts.w2_EDF = nn.Parameter(self.experts.w2_EDF[lo:hi].contiguous())
            self.experts.w3_EFD = nn.Parameter(self.experts.w3_EFD[lo:hi].contiguous())

    def parallelize(self, parallel_dims) -> None:
        """Apply common MoE parallelization and wire phase metadata.

        Input: parallel_dims defines EP/TP meshes and sharding placements.
        Output: module parameters and dispatcher are parallelized in place.
        """
        super().parallelize(parallel_dims)
        self._configure_phase_dtensor_specs(parallel_dims)
        ep_mesh = parallel_dims.get_optional_mesh("ep")
        tp_mesh = parallel_dims.get_optional_mesh("tp")
        shard_plain_experts = not isinstance(self.experts.w1_EFD, DTensor)
        self.wire_meshes(
            ep_mesh=ep_mesh, tp_mesh=tp_mesh, shard_plain_experts=shard_plain_experts
        )

    def route(self, x_BLD: torch.Tensor) -> RouteInfo:
        """Run routing and flatten top-k metadata for later dispatch.

        This is the first schedulable MoE phase. It uses the inherited common
        MoE router, updates load-balancing accounting through _route(), and
        then converts DTensor results to local tensors because the dispatcher
        phase APIs operate on local dynamic token counts.

        The returned RouteInfo can be consumed immediately by dispatch(), or
        saved so a schedule can launch counts/token dispatch after other work.

        Input: x_BLD is [B, L, D] Tensor or DTensor. Output: RouteInfo with
        scores [T, K], expert ids [T, K], and local counts [E].
        """
        with record_function("ladder_moe/route"):
            phase_input = self._prepare_phase_input(x_BLD)
            topk_scores_BLK, topk_ids_BLK, counts_E = self._route(phase_input.x_BLD)
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
        metadata: DispatchMetadata,
        *,
        local_x_BLD: torch.Tensor | None = None,
    ) -> DispatchState:
        """Build DispatchState from input shape and dispatcher metadata.

        Called when token dispatch finishes, before expert output exists. The
        state intentionally captures the original local shape and output
        placement metadata at dispatch time so combine can run later, possibly
        after attention or in the next transformer block.

        Input: x_BLD is [B, L, D], metadata is dispatch state, and local_x_BLD
        is [B, L_local, D] or None.
        Output: DispatchState.
        """
        local_x_BLD = self._to_local(x_BLD) if local_x_BLD is None else local_x_BLD
        B, L, D = local_x_BLD.shape
        output_mesh, output_src, output_dst = self._output_specs(x_BLD)
        return DispatchState(
            metadata=metadata,
            batch_size=B,
            local_seq_len=L,
            num_tokens=B * L,
            dim=D,
            device=local_x_BLD.device,
            dtype=local_x_BLD.dtype,
            output_device_mesh=output_mesh,
            output_src_placements=output_src,
            output_dst_placements=output_dst,
        )

    def begin_counts(self, route_info: RouteInfo) -> torch.Tensor:
        """Launch asynchronous expert-count exchange.

        This is separated from token dispatch because EP token all-to-all needs
        send/receive splits first. Schedules can launch this tiny exchange early
        and later call sync_counts() when Python split lists are required.

        Input: route_info carries local counts [E].
        Output: dispatcher-specific tensor handle for the global count exchange.
        """
        with record_function("ladder_moe/counts_a2a"):
            return self.token_dispatcher.begin_count_exchange(
                route_info.num_local_tokens_per_expert_E,
            )

    def sync_counts(
        self, global_counts_pe: torch.Tensor, route_info: RouteInfo
    ) -> TokenDispatchSplits:
        """Finish count exchange and build token all-to-all splits.

        This is the host-visible sync point for standard EP dispatch: the
        dispatcher must turn exchanged device counts into concrete split sizes
        before token all-to-all can launch.

        Input: global_counts_pe is the count-exchange handle and route_info has
        local counts [E].
        Output: TokenDispatchSplits for token dispatch.
        """
        with record_function("ladder_moe/counts_host_sync"):
            return self.token_dispatcher.finish_count_exchange(
                route_info.num_local_tokens_per_expert_E,
                global_counts_pe,
            )

    def begin_token_dispatch(
        self,
        x_BLD: torch.Tensor,
        route_info: RouteInfo,
        splits: TokenDispatchSplits,
    ) -> TokenDispatchHandle:
        """Gather routed tokens and launch asynchronous token dispatch.

        This consumes RouteInfo.topk_* to flatten/gather each token's top-k
        expert slots and sends those routed slots to the ranks owning the local
        experts. The returned handle lets a schedule overlap the all-to-all with
        attention before calling finish_dispatch().

        Input: x_BLD is [B, L, D], route_info has scores/ids [T, K], and splits
        come from count exchange.
        Output: TokenDispatchHandle.
        """
        phase_input = self._prepare_phase_input(x_BLD)
        x_TD = phase_input.flatten_tokens()
        with record_function("ladder_moe/dispatch_token_a2a"):
            return self.token_dispatcher.begin_token_dispatch(
                x_TD,
                route_info.topk_scores_TK,
                route_info.topk_expert_ids_TK,
                splits,
            )

    def finish_dispatch(
        self,
        handle: TokenDispatchHandle,
        x_BLD: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, DispatchState]:
        """Finish token dispatch and create combine state.

        This is where the asynchronous dispatch handle becomes local expert
        input. The returned DispatchState must stay paired with the routed tokens
        and their expert output; combine cannot correctly invert dispatch
        without it.

        Input: handle is from begin_token_dispatch and x_BLD is [B, L, D].
        Output: routed tokens [R, D], per-local-expert counts [e], DispatchState.
        """
        phase_input = self._prepare_phase_input(x_BLD)
        with record_function("ladder_moe/permute"):
            routed_RD, counts_e, metadata = self.token_dispatcher.finish_token_dispatch(
                handle
            )
        state = self._state(
            phase_input.x_BLD,
            metadata,
            local_x_BLD=phase_input.local_x_BLD,
        )
        return routed_RD, counts_e, state

    def experts_forward(
        self, routed_RD: torch.Tensor, counts_e: torch.Tensor
    ) -> torch.Tensor:
        """Run local grouped expert MLPs on dispatched tokens.

        CPU routed tensors use _cpu_experts_forward() for test portability.
        Non-CPU tensors use the common GroupedExperts._experts_forward() so the
        ladder path keeps the production grouped-GEMM implementation.

        Input: routed_RD is [R, D], counts_e gives tokens per local expert [e].
        Output: routed expert output [R, D].
        """
        with record_function("ladder_moe/experts_grouped_mm"):
            if routed_RD.device.type == "cpu":
                return _cpu_experts_forward(
                    routed_RD,
                    counts_e,
                    self.experts.w1_EFD,
                    self.experts.w2_EDF,
                    self.experts.w3_EFD,
                )
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
        with record_function("ladder_moe/combine_token_a2a"):
            return self.token_dispatcher.begin_token_combine(
                routed_out_RD,
                state.metadata,
            )

    def finish_combine(
        self, combine_h: TokenCombineHandle, state: DispatchState
    ) -> torch.Tensor:
        """Finish combine, apply route scores, and scatter to token order.

        This completes the inverse of dispatch using state.metadata, then
        reconstructs flattened token order. The returned tensor is still [T, D];
        callers that need residual shape should use finish_combine_bld() or
        combine_bld().

        Input: combine_h is from begin_combine, and state has token shape T and model dim D.
        Output: flattened combined tokens [T, D].
        """
        with record_function("ladder_moe/combine_scatter_add"):
            # EP combine only needs shape/device/dtype metadata here; avoid
            # allocating a full token buffer that the dispatcher will discard.
            num_tokens = state.num_tokens if self.ep_mesh is None else 0
            x_TD = torch.empty(
                num_tokens,
                state.dim,
                device=state.device,
                dtype=state.dtype,
            )
            return self.token_dispatcher.finish_token_combine(
                combine_h,
                state.metadata,
                x_TD,
                num_local_tokens_after_padding=state.num_tokens,
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
        """Run full token dispatch as one phase.

        Input: x_BLD is [B, L, D] and route_info has [T, K] routing metadata.
        Output: routed tokens [R, D], expert counts [e], and DispatchState.

        Dispatchers without a host split sync (DeepEP) always take the atomic
        dispatcher.dispatch() path, at any EP size.
        """
        phase_input = self._prepare_phase_input(x_BLD)
        if self.ep_size == 1 or not self.needs_split_sync:
            x_TD = phase_input.flatten_tokens()
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

        counts_h = self.begin_counts(route_info)
        splits = self.sync_counts(counts_h, route_info)
        handle = self.begin_token_dispatch(phase_input.x_BLD, route_info, splits)
        return self.finish_dispatch(handle, phase_input.x_BLD)

    def combine(
        self, routed_out_RD: torch.Tensor, state: DispatchState
    ) -> torch.Tensor:
        """Run begin_combine and finish_combine as one call.

        Input: routed_out_RD is [R, D] plus DispatchState.
        Output: flattened combined tokens [T, D].
        """
        return self.finish_combine(self.begin_combine(routed_out_RD, state), state)

    def forward(self, x_BLD: torch.Tensor) -> torch.Tensor:
        """Run route, dispatch, experts, and combine atomically.

        Input: x_BLD is [B, L, D] Tensor or DTensor.
        Output: MoE result [B, L, D] in the common MoE source output placement.
        """
        phase_input = self._prepare_phase_input(x_BLD)
        route_info = self.route(phase_input.x_BLD)
        routed_RD, counts_e, state = self.dispatch(phase_input.x_BLD, route_info)
        out_TD = self.combine(self.experts_forward(routed_RD, counts_e), state)
        out_BLD = out_TD.view(
            state.batch_size,
            -1,
            state.dim,
        )
        return self._wrap_output_src(out_BLD, state)

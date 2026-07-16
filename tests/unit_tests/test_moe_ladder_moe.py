# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from unittest.mock import patch

import pytest
import spmd_types as spmd
import torch
import torch.nn.functional as F
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import distribute_tensor, DTensor, Shard
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)

from torchtitan.distributed import ParallelDims
from torchtitan.experiments.moe_ladder.ladder_moe import LadderMoE
from torchtitan.experiments.moe_ladder.models.common import drain_pending
from torchtitan.experiments.moe_ladder.schedule_runner import (
    hoisted_gate_a_step,
    hoisted_gate_b_step,
    ladder_step,
    parallel_step,
)
from torchtitan.models.common.moe import GroupedExperts, MoE, TokenChoiceTopKRouter
from torchtitan.models.common.moe_sharding import set_moe_sharding_config
from torchtitan.models.common.nn_modules import Linear
from torchtitan.models.common.token_dispatcher import AllToAllTokenDispatcher


# Shape legend: B=batch, L=seq, D=model dim, F=FFN hidden, E=experts, K=top-k.
def _make_moe(
    *, dim: int = 16, hidden_dim: int = 32, num_experts: int = 4, top_k: int = 2
) -> LadderMoE:
    return LadderMoE.from_dims(
        dim=dim, hidden_dim=hidden_dim, num_experts=num_experts, top_k=top_k
    )


def _dense_reference(
    x_BLD: torch.Tensor,
    moe: LadderMoE,
    topk_scores_TK: torch.Tensor,
    topk_ids_TK: torch.Tensor,
) -> torch.Tensor:
    B, L, D = x_BLD.shape
    x_TD = x_BLD.reshape(B * L, D)
    out_TD = torch.zeros_like(x_TD)
    for expert_idx in range(moe.num_experts):
        scores_T = (topk_scores_TK * (topk_ids_TK == expert_idx)).sum(dim=-1)
        selected_T = scores_T > 0
        if not selected_T.any():
            continue
        x_eD = x_TD[selected_T]
        h_eF = F.silu(F.linear(x_eD, moe.experts.w1_EFD[expert_idx]))
        h_eF = h_eF * F.linear(x_eD, moe.experts.w3_EFD[expert_idx])
        out_eD = F.linear(h_eF, moe.experts.w2_EDF[expert_idx])
        out_TD[selected_T] += out_eD * scores_T[selected_T, None]
    return out_TD.view(B, L, D)


def test_ladder_moe_phase_split_matches_dense_reference():
    torch.manual_seed(1234)
    B, L, D = 2, 5, 16
    moe = _make_moe(dim=D)
    x_BLD = torch.randn(B, L, D)

    route_info = moe.route(x_BLD)
    routed_RD, counts_E, state = moe.dispatch(x_BLD, route_info)
    identity_TD = moe.combine(routed_RD, state)
    torch.testing.assert_close(identity_TD, x_BLD.reshape(B * L, D))
    identity_BLD = moe.combine_bld(routed_RD, state)
    torch.testing.assert_close(identity_BLD, x_BLD)

    e_RD = moe.experts_forward(routed_RD, counts_E)
    out_BLD = moe.combine_bld(e_RD, state)
    ref_BLD = _dense_reference(
        x_BLD,
        moe,
        route_info.topk_scores_TK,
        route_info.topk_expert_ids_TK,
    )
    torch.testing.assert_close(out_BLD, ref_BLD, atol=1e-6, rtol=1e-6)


def test_ladder_moe_forward_matches_dense_reference():
    torch.manual_seed(1234)
    B, L, D = 2, 5, 16
    moe = _make_moe(dim=D)
    x_BLD = torch.randn(B, L, D)

    route_info = moe.route(x_BLD)
    ref_BLD = _dense_reference(
        x_BLD,
        moe,
        route_info.topk_scores_TK,
        route_info.topk_expert_ids_TK,
    )

    torch.testing.assert_close(moe(x_BLD), ref_BLD, atol=1e-6, rtol=1e-6)


def test_ladder_moe_rejects_uneven_expert_parallel_degree():
    class FakeMesh:
        def size(self):
            return 3

    moe = _make_moe(num_experts=4)

    with pytest.raises(ValueError, match="must be divisible by EP"):
        moe.wire_meshes(
            ep_mesh=FakeMesh(),  # pyrefly: ignore [bad-argument-type]
            shard_plain_experts=False,
        )


class TestLadderMoETensorParallelSequenceParallel(DTensorTestBase):
    @property
    def world_size(self):
        return 2

    def _parallel_dims(self):
        parallel_dims = ParallelDims(
            dp_replicate=1,
            dp_shard=1,
            cp=1,
            tp=self.world_size,
            pp=1,
            ep=1,
            world_size=self.world_size,
        )
        with patch(
            "torchtitan.distributed.parallel_dims.device_type", self.device_type
        ):
            parallel_dims.build_mesh()
        return parallel_dims

    def _make_moe(self, *, ladder: bool, initialize: bool = True) -> MoE:
        dim = 16
        hidden_dim = 32
        num_experts = 4
        top_k = 2
        moe_config = MoE.Config(
            num_experts=num_experts,
            load_balance_coeff=None,
            router=TokenChoiceTopKRouter.Config(
                num_experts=num_experts,
                gate=Linear.Config(in_features=dim, out_features=num_experts),
                top_k=top_k,
                score_func="softmax",
                route_norm=True,
            ),
            experts=GroupedExperts.Config(
                dim=dim,
                hidden_dim=hidden_dim,
                num_experts=num_experts,
                token_dispatcher=AllToAllTokenDispatcher.Config(
                    num_experts=num_experts,
                    top_k=top_k,
                ),
            ),
        )
        set_moe_sharding_config(
            moe_config,
            enable_ep=False,
            enable_sp=True,
            expert_param_layout={
                "w1_EFD": spmd.S(1),
                "w2_EDF": spmd.S(2),
                "w3_EFD": spmd.S(1),
            },
        )
        if ladder:
            moe = LadderMoE.Config(
                moe=moe_config,
                sharding_config=moe_config.sharding_config,
            ).build()
        else:
            moe = moe_config.build()
        moe.to(self.device_type)
        if initialize:
            moe.init_states(buffer_device=torch.device(self.device_type))
        return moe

    def _make_parallelized_moe(self) -> LadderMoE:
        moe = self._make_moe(ladder=True)
        assert isinstance(moe, LadderMoE)
        moe.parallelize(self._parallel_dims())
        return moe

    @with_comms
    def test_tp_sp_forward_returns_sequence_sharded_dtensor(self):
        torch.manual_seed(2024)
        moe = self._make_parallelized_moe()
        x_BLD = torch.randn(2, 4, 16, device=self.device_type)
        x_dtensor = distribute_tensor(
            x_BLD,
            moe.experts.w1_EFD.device_mesh["tp"],
            [Shard(1)],
        )

        out = moe(x_dtensor)

        self.assertIsInstance(out, DTensor)
        self.assertEqual(tuple(out.placements), (Shard(1),))
        self.assertEqual(tuple(out.to_local().shape), (2, 2, 16))
        self.assertEqual(tuple(out.full_tensor().shape), tuple(x_BLD.shape))

    @with_comms
    def test_tp_sp_forward_backward_matches_common_moe(self):
        torch.manual_seed(2024)
        ladder_moe = self._make_moe(ladder=True)
        common_moe = self._make_moe(ladder=False, initialize=False)
        assert isinstance(ladder_moe, LadderMoE)
        common_moe.load_state_dict(ladder_moe.state_dict())

        parallel_dims = self._parallel_dims()
        ladder_moe.parallelize(parallel_dims)
        common_moe.parallelize(parallel_dims)
        tp_mesh = ladder_moe.experts.w1_EFD.device_mesh["tp"]

        x_BLD = torch.randn(2, 4, 16, device=self.device_type)
        ladder_x = distribute_tensor(x_BLD.clone(), tp_mesh, [Shard(1)])
        common_x = distribute_tensor(x_BLD.clone(), tp_mesh, [Shard(1)])
        ladder_x.requires_grad_()
        common_x.requires_grad_()

        ladder_out = ladder_moe(ladder_x)
        common_out = common_moe(common_x)
        torch.testing.assert_close(ladder_out.to_local(), common_out.to_local())

        ladder_out.to_local().float().square().sum().backward()
        common_out.to_local().float().square().sum().backward()

        ladder_x_grad = ladder_x.grad
        common_x_grad = common_x.grad
        assert isinstance(ladder_x_grad, DTensor)
        assert isinstance(common_x_grad, DTensor)
        torch.testing.assert_close(
            ladder_x_grad.to_local(),
            common_x_grad.to_local(),
        )
        for (ladder_name, ladder_param), (common_name, common_param) in zip(
            ladder_moe.named_parameters(),
            common_moe.named_parameters(),
            strict=True,
        ):
            self.assertEqual(ladder_name, common_name)
            ladder_grad = ladder_param.grad
            common_grad = common_param.grad
            assert isinstance(ladder_grad, DTensor)
            assert isinstance(common_grad, DTensor)
            torch.testing.assert_close(
                ladder_grad.to_local(),
                common_grad.to_local(),
            )


class TestLadderMoEExpertParallel(DTensorTestBase):
    """EP>1 tests for the phase-split MoE and the four ladder schedules.

    Runs on GPU (NCCL) when >= world_size devices are available, else CPU
    (Gloo). Both the reference and the EP module are ``LadderMoE`` so the CPU
    grouped-expert fallback keeps the test device-agnostic. Input is identical
    on every rank (replicated), so each rank's EP output equals the full,
    single-rank MoE output over the same tokens.
    """

    @property
    def world_size(self):
        return 2

    def _make_ep_pair(
        self,
        ep_mesh,
        *,
        dim: int = 16,
        hidden_dim: int = 32,
        num_experts: int = 4,
        top_k: int = 2,
    ) -> tuple[LadderMoE, LadderMoE]:
        """Build (single-rank reference, EP-sharded) MoEs with identical weights.

        The reference keeps all experts local (ep_mesh unset -> local dispatch);
        the EP module loads the same full weights, then ``wire_meshes`` slices
        this rank's expert shard and installs the all-to-all dispatch path.
        """
        torch.manual_seed(2024)
        ref_moe = LadderMoE.from_dims(
            dim=dim, hidden_dim=hidden_dim, num_experts=num_experts, top_k=top_k
        )
        ref_moe.to(self.device_type)

        ep_moe = LadderMoE.from_dims(
            dim=dim, hidden_dim=hidden_dim, num_experts=num_experts, top_k=top_k
        )
        ep_moe.load_state_dict(ref_moe.state_dict())
        ep_moe.to(self.device_type)
        ep_moe.wire_meshes(ep_mesh=ep_mesh, shard_plain_experts=True)
        assert ep_moe.ep_size == self.world_size
        return ref_moe, ep_moe

    def _expert_shard(self, ep_mesh, param: torch.Tensor, n_local: int) -> slice:
        lo = ep_mesh.get_local_rank() * n_local
        return slice(lo, lo + n_local)

    def _assert_grad_close(self, actual, expected, *, rtol: float = 2e-3) -> None:
        """Compare gradients by relative L2 norm.

        Distributed EP changes gradient accumulation order (all-to-all token
        reordering plus grouped-GEMM regrouping), so a few elements drift at the
        1e-5 level. A relative-L2 check tolerates that noise while still catching
        a genuinely wrong gradient, which would move the whole tensor.
        """
        denom = expected.detach().norm().clamp_min(1e-12)
        rel = (actual.detach() - expected.detach()).norm() / denom
        self.assertLess(
            rel.item(), rtol, f"relative grad L2 error {rel.item():.2e} >= {rtol}"
        )

    @with_comms
    def test_ep_forward_backward_matches_single_rank(self):
        ep_mesh = init_device_mesh(
            self.device_type, (self.world_size,), mesh_dim_names=("ep",)
        )
        ref_moe, ep_moe = self._make_ep_pair(ep_mesh)

        torch.manual_seed(7)
        x_BLD = torch.randn(2, 4, 16, device=self.device_type)
        ref_x = x_BLD.clone().requires_grad_()
        ep_x = x_BLD.clone().requires_grad_()

        ref_out = ref_moe(ref_x)
        ep_out = ep_moe(ep_x)
        torch.testing.assert_close(ep_out, ref_out)

        ref_out.float().square().sum().backward()
        ep_out.float().square().sum().backward()
        self._assert_grad_close(ep_x.grad, ref_x.grad)

        # Router gate is replicated (not EP-sharded): full gradient must match.
        self._assert_grad_close(
            ep_moe.router.gate.weight.grad, ref_moe.router.gate.weight.grad
        )
        # Expert weights are EP-sharded: this rank's shard grad must match the
        # corresponding slice of the single-rank reference grad, scaled by
        # ep_size. Input is replicated across EP ranks, so each rank's local
        # experts accumulate gradient over the replicated tokens from every EP
        # rank (ep_size copies of each token); the forward output stays per-rank
        # exact because combine returns each token's result to its origin. Real
        # training DP-shards the input, so each expert sees each token once.
        shard = self._expert_shard(
            ep_mesh, ep_moe.experts.w1_EFD, ep_moe.num_local_experts
        )
        for name in ("w1_EFD", "w2_EDF", "w3_EFD"):
            ep_grad = getattr(ep_moe.experts, name).grad
            ref_grad = getattr(ref_moe.experts, name).grad
            self._assert_grad_close(ep_grad, ep_moe.ep_size * ref_grad[shard])

    @with_comms
    def test_ep_parallel_schedule_matches_atomic(self):
        ep_mesh = init_device_mesh(
            self.device_type, (self.world_size,), mesh_dim_names=("ep",)
        )
        _, ep_moe = self._make_ep_pair(ep_mesh)

        torch.manual_seed(11)
        res_BLD = torch.randn(2, 4, 16, device=self.device_type)

        def ffn_norm(x):
            return x

        def attention(res):
            return res * 0.5

        out = parallel_step(ep_moe, res_BLD, attention=attention, ffn_norm=ffn_norm)
        # parallel_step only reorders the phases around attention; the MoE
        # contribution must equal the atomic EP forward over the same input.
        ref = res_BLD + attention(res_BLD) + ep_moe(ffn_norm(res_BLD))
        torch.testing.assert_close(out, ref)

    @with_comms
    def test_ep_delayed_schedules_finite_forward_backward(self):
        ep_mesh = init_device_mesh(
            self.device_type, (self.world_size,), mesh_dim_names=("ep",)
        )

        def ffn_norm(x):
            return x

        def attention(res):
            return res * 0.5

        for schedule_fn in (ladder_step, hoisted_gate_a_step, hoisted_gate_b_step):
            _, ep_moe = self._make_ep_pair(ep_mesh)
            torch.manual_seed(3)
            res_BLD = torch.randn(2, 4, 16, device=self.device_type, requires_grad=True)

            # Two steps so a cross-block Pending is produced and consumed under
            # EP, then drain the final Pending into the residual.
            cur = res_BLD
            pending = None
            for _ in range(2):
                cur, pending = schedule_fn(
                    ep_moe, cur, pending, attention=attention, ffn_norm=ffn_norm
                )
            cur = drain_pending(ep_moe, cur, pending)

            self.assertTrue(torch.isfinite(cur).all())
            cur.float().square().sum().backward()
            assert res_BLD.grad is not None
            self.assertTrue(torch.isfinite(res_BLD.grad).all())


if __name__ == "__main__":
    unittest.main()

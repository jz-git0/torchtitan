# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import sys
from dataclasses import replace
from types import MethodType, SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.tensor import distribute_tensor, DTensor, Replicate, Shard
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)
from torchtitan.distributed import ParallelDims
from torchtitan.experiments.moe_ladder.ladder_moe import LadderMoE
from torchtitan.experiments.moe_ladder.models.common import build_ladder_moe
from torchtitan.experiments.moe_ladder.models.qwen3 import (
    _copy_block_config,
    Qwen3HoistedGateABlock,
    Qwen3HoistedGateBBlock,
    Qwen3LadderMoEBlock,
    Qwen3ParallelMoEBlock,
)
from torchtitan.models.common.moe import MoE
from torchtitan.models.common.moe_sharding import _moe_sharding_config
from torchtitan.models.common.token_dispatcher import (
    BaseEPTokenDispatcher,
    DeepEPDispatchMetadata,
    DeepEPTokenDispatcher,
    LocalTokenDispatcher,
    TokenCombineHandle,
)
from torchtitan.models.qwen3 import qwen3_configs


class _ScaledAttention(nn.Module):
    def forward(self, x_BLD, attention_masks, positions):
        del attention_masks, positions
        return 3 * x_BLD


def _install_local_deepep_stub(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace only DeepEP communication with local token permutation."""

    def init(self, config) -> None:
        BaseEPTokenDispatcher.__init__(self, config)

    def dispatch(
        self,
        x_TD,
        topk_scores_TK,
        topk_expert_ids_TK,
        num_local_tokens_per_expert_E,
    ):
        routed_RD, counts_e, metadata = LocalTokenDispatcher.dispatch(
            self,
            x_TD,
            topk_scores_TK,
            topk_expert_ids_TK,
            num_local_tokens_per_expert_E,
        )
        return routed_RD, counts_e, DeepEPDispatchMetadata(state=metadata)

    def begin_token_combine(self, routed_output_RD, metadata):
        return LocalTokenDispatcher.begin_token_combine(
            self, routed_output_RD, metadata.state
        )

    def finish_token_combine(
        self,
        handle,
        metadata,
        x_TD,
        *,
        num_local_tokens_after_padding,
        local_seq_len_after_padding,
    ):
        del x_TD
        local_x_TD = handle.routed_output_RD.new_empty(
            num_local_tokens_after_padding,
            handle.routed_output_RD.shape[-1],
        )
        return LocalTokenDispatcher.finish_token_combine(
            self,
            handle,
            metadata.state,
            local_x_TD,
            num_local_tokens_after_padding=num_local_tokens_after_padding,
            local_seq_len_after_padding=local_seq_len_after_padding,
        )

    def combine(
        self,
        routed_output_RD,
        metadata,
        x_TD,
        *,
        num_local_tokens_after_padding,
        local_seq_len_after_padding,
    ):
        handle = begin_token_combine(self, routed_output_RD, metadata)
        return finish_token_combine(
            self,
            handle,
            metadata,
            x_TD,
            num_local_tokens_after_padding=num_local_tokens_after_padding,
            local_seq_len_after_padding=local_seq_len_after_padding,
        )

    monkeypatch.setattr(DeepEPTokenDispatcher, "__init__", init)
    monkeypatch.setattr(DeepEPTokenDispatcher, "dispatch", dispatch)
    monkeypatch.setattr(
        DeepEPTokenDispatcher, "begin_token_combine", begin_token_combine
    )
    monkeypatch.setattr(
        DeepEPTokenDispatcher, "finish_token_combine", finish_token_combine
    )
    monkeypatch.setattr(DeepEPTokenDispatcher, "combine", combine)
    monkeypatch.setitem(
        sys.modules,
        "torchtitan.distributed.deepep.deepep",
        SimpleNamespace(sync_combine=lambda: None),
    )


def _tiny_qwen_block_config():
    source = qwen3_configs["debugmodel_moe"](
        attn_backend="flex",
        moe_comm_backend="deepep",
    ).layers[0]
    assert source.moe is not None

    moe = source.moe
    dispatcher = replace(
        moe.experts.token_dispatcher,
        num_experts=2,
        top_k=1,
    )
    experts = replace(
        moe.experts,
        hidden_dim=8,
        num_experts=2,
        token_dispatcher=dispatcher,
    )
    router = replace(
        moe.router,
        num_experts=2,
        top_k=1,
        gate=replace(moe.router.gate, out_features=2),
    )
    source = replace(
        source,
        moe=replace(
            moe,
            num_experts=2,
            experts=experts,
            router=router,
        ),
    )
    return source


def _tiny_block_config(block_cls):
    return _copy_block_config(_tiny_qwen_block_config(), block_cls)


def _double_expert_output(self, routed_RD, counts_e):
    del counts_e
    return 2 * routed_RD


def _build_block(block_cls, device: str):
    block = _tiny_block_config(block_cls).build()
    block.attention = _ScaledAttention()
    block.attention_norm = nn.Identity()
    block.ffn_norm = nn.Identity()
    block.init_states()
    block.moe.experts_forward = MethodType(_double_expert_output, block.moe)
    return block.to(device)


def _manual_pending(block, moe_input_BLD, route_input_BLD):
    route_info = block.moe.route(route_input_BLD)
    routed_RD, counts_e, state = block.moe.dispatch(moe_input_BLD, route_info)
    return block.moe.experts_forward(routed_RD, counts_e), state


_DEVICES = [
    "cpu",
    pytest.param(
        "cuda",
        marks=pytest.mark.skipif(
            not torch.cuda.is_available(),
            reason="requires CUDA",
        ),
    ),
]


def test_ladder_moe_matches_stock_moe(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_local_deepep_stub(monkeypatch)
    source = _tiny_qwen_block_config()
    assert source.moe is not None
    ladder_moe = build_ladder_moe(source.moe)
    stock_moe = source.moe.build()
    torch.manual_seed(11)
    ladder_moe.init_states()
    stock_moe.load_state_dict(ladder_moe.state_dict())
    x_BLD = torch.randn(2, 3, 256)

    torch.testing.assert_close(ladder_moe(x_BLD), stock_moe(x_BLD))


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.filterwarnings(
    "ignore:The AccumulateGrad node's stream does not match.*:UserWarning"
)
def test_parallel_block_matches_explicit_phase_composition(
    monkeypatch: pytest.MonkeyPatch,
    device: str,
) -> None:
    _install_local_deepep_stub(monkeypatch)
    block = _build_block(Qwen3ParallelMoEBlock, device)
    x_BLD = torch.randn(2, 3, 256, device=device, requires_grad=True)

    actual_BLD = block(x_BLD, attention_masks=None)
    pending = _manual_pending(block, x_BLD, x_BLD)
    expected_BLD = x_BLD + 3 * x_BLD + block.moe.combine_bld(*pending)

    torch.testing.assert_close(actual_BLD, expected_BLD)
    actual_BLD.sum().backward()
    assert x_BLD.grad is not None
    assert torch.isfinite(x_BLD.grad).all()


@pytest.mark.parametrize(
    "block_cls",
    [Qwen3LadderMoEBlock, Qwen3HoistedGateABlock, Qwen3HoistedGateBBlock],
)
@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.filterwarnings(
    "ignore:The AccumulateGrad node's stream does not match.*:UserWarning"
)
def test_delayed_blocks_match_two_block_recurrence(
    monkeypatch: pytest.MonkeyPatch,
    block_cls,
    device: str,
) -> None:
    _install_local_deepep_stub(monkeypatch)
    first = _build_block(block_cls, device)
    second = _build_block(block_cls, device)
    x_BLD = torch.randn(2, 3, 256, device=device, requires_grad=True)

    res1_BLD, pending1 = first(
        x_BLD,
        attention_masks=None,
    )
    res2_BLD, pending2 = second(
        res1_BLD,
        attention_masks=None,
        pending=pending1,
    )

    expected_res1_BLD = 4 * x_BLD
    previous_moe_BLD = second.moe.combine_bld(*pending1)
    moe_input_BLD = expected_res1_BLD + previous_moe_BLD
    route_input_BLD = (
        moe_input_BLD if block_cls is Qwen3LadderMoEBlock else expected_res1_BLD
    )
    expected_pending2 = _manual_pending(second, moe_input_BLD, route_input_BLD)
    expected_res2_BLD = moe_input_BLD + 3 * expected_res1_BLD

    torch.testing.assert_close(res1_BLD, expected_res1_BLD)
    torch.testing.assert_close(res2_BLD, expected_res2_BLD)
    torch.testing.assert_close(pending2[0], expected_pending2[0])

    actual_final_BLD = second.drain(res2_BLD, pending2)
    expected_final_BLD = expected_res2_BLD + second.moe.combine_bld(*expected_pending2)
    torch.testing.assert_close(actual_final_BLD, expected_final_BLD)

    actual_final_BLD.sum().backward()
    assert x_BLD.grad is not None
    assert torch.isfinite(x_BLD.grad).all()


class TestLadderMoETensorParallel(DTensorTestBase):
    @property
    def world_size(self):
        return 2

    def _parallel_dims(self) -> ParallelDims:
        parallel_dims = ParallelDims(
            dp_replicate=1,
            dp_shard=1,
            cp=1,
            tp=self.world_size,
            pp=1,
            ep=self.world_size,
            world_size=self.world_size,
        )
        with patch(
            "torchtitan.distributed.parallel_dims.device_type", self.device_type
        ):
            parallel_dims.build_mesh()
        return parallel_dims

    def _phase_moe(self, parallel_dims: ParallelDims) -> LadderMoE:
        moe = LadderMoE.__new__(LadderMoE)
        nn.Module.__init__(moe)
        moe._sharding_config = _moe_sharding_config(
            enable_ep=True,
            enable_sp=True,
        )
        moe.seq_dim_tp_sharded = True
        moe.token_dispatcher = SimpleNamespace(sp_size=self.world_size, num_experts=2)
        moe._phase_input_device_mesh = None
        moe._phase_input_src_placements = None
        moe._phase_input_dst_placements = None
        moe._phase_plain_input_allowed = True
        moe._phase_output_device_mesh = None
        moe._phase_output_src_placements = None
        moe._phase_output_dst_placements = None
        with patch.object(MoE, "parallelize", autospec=True) as common_parallelize:
            moe.parallelize(parallel_dims)
        common_parallelize.assert_called_once_with(moe, parallel_dims)
        return moe

    @with_comms
    def test_phase_input_boundary_with_tp(self) -> None:
        parallel_dims = self._parallel_dims()
        moe = self._phase_moe(parallel_dims)
        tp_mesh = parallel_dims.get_mesh("tp")

        local_x_BLD = torch.randn(2, 2, 4, device=self.device_type, requires_grad=True)
        phase_input = moe._prepare_phase_input(moe.prepare_input(local_x_BLD))
        self.assertIsInstance(phase_input.x_BLD, DTensor)
        self.assertEqual(phase_input.x_BLD.placements, (Shard(1),))
        self.assertEqual(phase_input.x_BLD.shape, torch.Size((2, 4, 4)))
        torch.testing.assert_close(phase_input.local_x_BLD, local_x_BLD)
        phase_input.local_x_BLD.sum().backward()
        torch.testing.assert_close(local_x_BLD.grad, torch.ones_like(local_x_BLD))

        replicated = distribute_tensor(
            torch.randn(2, 4, 4, device=self.device_type),
            tp_mesh,
            (Replicate(),),
        )
        with self.assertRaisesRegex(ValueError, "expected source"):
            moe.prepare_input(replicated)

        uneven = distribute_tensor(
            torch.randn(2, 3, 4, device=self.device_type),
            tp_mesh,
            (Shard(1),),
        )
        with self.assertRaisesRegex(ValueError, "divisible by TP"):
            moe._prepare_phase_input(moe.prepare_input(uneven))

    @with_comms
    def test_phase_output_boundary_with_tp(self) -> None:
        parallel_dims = self._parallel_dims()
        moe = self._phase_moe(parallel_dims)
        local_x_BLD = torch.zeros(2, 2, 4, device=self.device_type)
        phase_input = moe._prepare_phase_input(moe.prepare_input(local_x_BLD))
        metadata = DeepEPDispatchMetadata(state=object())
        state = moe._state(
            phase_input.x_BLD,
            metadata,
            local_x_BLD=phase_input.local_x_BLD,
        )

        rank = dist.get_rank()
        dispatcher = DeepEPTokenDispatcher.__new__(DeepEPTokenDispatcher)
        dispatcher.sp_size = self.world_size
        dispatcher.sp_rank = rank
        combined_TD = torch.full(
            (4, 4),
            float(rank + 1),
            device=self.device_type,
            requires_grad=True,
        )
        handle = TokenCombineHandle(routed_output_RD=combined_TD)
        deepep_module = SimpleNamespace(sync_combine=lambda: None)
        with patch.dict(
            sys.modules,
            {"torchtitan.distributed.deepep.deepep": deepep_module},
        ):
            partial_TD = dispatcher.finish_token_combine(
                handle,
                metadata,
                combined_TD,
                num_local_tokens_after_padding=4,
                local_seq_len_after_padding=2,
            )

        restored_BLD = moe._restore_output(partial_TD.view(2, 4, 4), state)
        self.assertIsInstance(restored_BLD, DTensor)
        self.assertEqual(restored_BLD.placements, (Shard(1),))
        local_out_BLD = restored_BLD.to_local(grad_placements=restored_BLD.placements)
        torch.testing.assert_close(
            local_out_BLD,
            torch.full_like(local_x_BLD, float(rank + 1)),
        )
        local_out_BLD.sum().backward()
        torch.testing.assert_close(combined_TD.grad, torch.ones_like(combined_TD))

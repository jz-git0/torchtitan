# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
from torchtitan.experiments.moe_ladder.models.common import drain_pending
from torchtitan.experiments.moe_ladder.schedule_runner import (
    hoisted_gate_a_step,
    hoisted_gate_b_step,
    ladder_step,
    parallel_step,
)


class _Route:
    __slots__ = (
        "value",
        "topk_scores_TK",
        "topk_expert_ids_TK",
        "num_local_tokens_per_expert_E",
    )

    def __init__(self, value: torch.Tensor) -> None:
        self.value = value
        self.topk_scores_TK = value
        self.topk_expert_ids_TK = value
        self.num_local_tokens_per_expert_E = value


class _FakeMoE:
    """Small differentiable phase implementation for schedule tests."""

    def __init__(self) -> None:
        self.calls: list[str] = []
        self.streams: dict[str, int] = {}
        self.combine_in_flight = False

    def record(self, name: str, tensor: torch.Tensor) -> None:
        self.calls.append(name)
        if tensor.is_cuda:
            self.streams[name] = torch.cuda.current_stream(tensor.device).cuda_stream

    def prepare_input(self, x_BLD: torch.Tensor) -> torch.Tensor:
        self.record("prepare", x_BLD)
        return x_BLD

    def route(self, x_BLD: torch.Tensor) -> _Route:
        self.record("route", x_BLD)
        return _Route(10 * x_BLD)

    def dispatch(
        self,
        x_BLD: torch.Tensor,
        route: _Route,
    ) -> tuple[torch.Tensor, torch.Tensor, object]:
        self.record("dispatch", x_BLD)
        counts = torch.empty(0, device=x_BLD.device, dtype=torch.int32)
        return x_BLD + route.value, counts, object()

    def experts_forward(
        self,
        routed_RD: torch.Tensor,
        counts_e: torch.Tensor,
    ) -> torch.Tensor:
        del counts_e
        self.record("experts", routed_RD)
        return 2 * routed_RD

    def begin_combine(
        self,
        routed_out_RD: torch.Tensor,
        state: object,
    ) -> torch.Tensor:
        del state
        assert not self.combine_in_flight
        self.combine_in_flight = True
        self.record("begin", routed_out_RD)
        return routed_out_RD

    def finish_combine_bld(
        self,
        handle: torch.Tensor,
        state: object,
    ) -> torch.Tensor:
        del state
        assert self.combine_in_flight
        self.combine_in_flight = False
        self.record("finish", handle)
        return handle

    def combine_bld(
        self,
        routed_out_RD: torch.Tensor,
        state: object,
    ) -> torch.Tensor:
        del state
        assert not self.combine_in_flight
        self.record("combine", routed_out_RD)
        return routed_out_RD


def _attention(moe: _FakeMoE, res_BLD: torch.Tensor) -> torch.Tensor:
    moe.record("attention", res_BLD)
    return 3 * res_BLD


def _identity(x_BLD: torch.Tensor) -> torch.Tensor:
    return x_BLD


def test_parallel_schedule_value_and_order() -> None:
    moe = _FakeMoE()
    res_BLD = torch.ones(2, 3, 4)

    out_BLD = parallel_step(
        moe,  # pyrefly: ignore [bad-argument-type]
        res_BLD,
        attention=lambda x: _attention(moe, x),
        ffn_norm=_identity,
    )

    torch.testing.assert_close(out_BLD, 26 * res_BLD)
    assert moe.calls == [
        "prepare",
        "route",
        "attention",
        "dispatch",
        "experts",
        "begin",
        "finish",
    ]


@pytest.mark.parametrize(
    ("step_fn", "expected_calls", "expected_pending_scale", "expected_final_scale"),
    [
        (
            ladder_step,
            ["begin", "attention", "finish", "prepare", "route", "dispatch", "experts"],
            572,
            610,
        ),
        (
            hoisted_gate_a_step,
            [
                "begin",
                "prepare",
                "route",
                "attention",
                "finish",
                "prepare",
                "dispatch",
                "experts",
            ],
            132,
            170,
        ),
        (
            hoisted_gate_b_step,
            [
                "prepare",
                "route",
                "begin",
                "attention",
                "finish",
                "prepare",
                "dispatch",
                "experts",
            ],
            132,
            170,
        ),
    ],
)
def test_delayed_schedule_value_and_order(
    step_fn,
    expected_calls: list[str],
    expected_pending_scale: int,
    expected_final_scale: int,
) -> None:
    moe = _FakeMoE()
    initial_BLD = torch.ones(2, 3, 4)
    res_BLD, pending = step_fn(
        moe,
        initial_BLD,
        None,
        attention=lambda x: _attention(moe, x),
        ffn_norm=_identity,
    )
    torch.testing.assert_close(res_BLD, 4 * initial_BLD)
    torch.testing.assert_close(pending[0], 22 * initial_BLD)

    moe.calls.clear()
    res_BLD, pending = step_fn(
        moe,
        res_BLD,
        pending,
        attention=lambda x: _attention(moe, x),
        ffn_norm=_identity,
    )

    assert moe.calls == expected_calls
    torch.testing.assert_close(pending[0], expected_pending_scale * initial_BLD)
    final_BLD = drain_pending(
        moe,  # pyrefly: ignore [bad-argument-type]
        res_BLD,
        pending,
    )
    torch.testing.assert_close(final_BLD, expected_final_scale * initial_BLD)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_parallel_schedule_uses_side_stream() -> None:
    moe = _FakeMoE()
    res_BLD = torch.ones(2, 3, 4, device="cuda")
    caller_stream = torch.cuda.current_stream().cuda_stream

    parallel_step(
        moe,
        res_BLD,
        attention=lambda x: _attention(moe, x),
        ffn_norm=_identity,
    )
    torch.cuda.synchronize()

    assert moe.streams["route"] == caller_stream
    assert moe.streams["attention"] == caller_stream
    for phase in ("dispatch", "experts", "begin", "finish"):
        assert moe.streams[phase] != caller_stream


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize(
    "step_fn", [ladder_step, hoisted_gate_a_step, hoisted_gate_b_step]
)
def test_delayed_schedule_overlaps_combine_with_attention(step_fn) -> None:
    moe = _FakeMoE()
    initial_BLD = torch.ones(2, 3, 4, device="cuda")
    caller_stream = torch.cuda.current_stream().cuda_stream

    res_BLD, pending = step_fn(
        moe,
        initial_BLD,
        None,
        attention=lambda x: _attention(moe, x),
        ffn_norm=_identity,
    )
    moe.streams.clear()
    step_fn(
        moe,
        res_BLD,
        pending,
        attention=lambda x: _attention(moe, x),
        ffn_norm=_identity,
    )
    torch.cuda.synchronize()

    assert moe.streams["attention"] == caller_stream
    for phase in ("begin", "finish", "dispatch", "experts"):
        assert moe.streams[phase] != caller_stream


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize(
    ("step_fn", "expected_gradient"),
    [(ladder_step, 610), (hoisted_gate_a_step, 170), (hoisted_gate_b_step, 170)],
)
@pytest.mark.filterwarnings(
    "ignore:The AccumulateGrad node's stream does not match.*:UserWarning"
)
def test_delayed_schedule_cuda_backward(step_fn, expected_gradient: int) -> None:
    """Exercise side-stream dependencies on small, non-DeepEP CUDA tensors."""
    moe = _FakeMoE()
    initial_BLD = torch.randn(2, 3, 4, device="cuda", requires_grad=True)

    res_BLD, pending = step_fn(
        moe,
        initial_BLD,
        None,
        attention=lambda x: _attention(moe, x),
        ffn_norm=_identity,
    )
    res_BLD, pending = step_fn(
        moe,
        res_BLD,
        pending,
        attention=lambda x: _attention(moe, x),
        ffn_norm=_identity,
    )
    final_BLD = drain_pending(
        moe,  # pyrefly: ignore [bad-argument-type]
        res_BLD,
        pending,
    )
    final_BLD.sum().backward()
    torch.cuda.synchronize()

    assert initial_BLD.grad is not None
    torch.testing.assert_close(
        initial_BLD.grad, torch.full_like(initial_BLD, expected_gradient)
    )

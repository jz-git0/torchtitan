# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch

from torchtitan.experiments.moe_ladder.models.deepseek_v3 import (
    ladderize_deepseek_v3_config,
)
from torchtitan.experiments.moe_ladder.models.llama3_moe import (
    ladderize_llama3_moe_config,
)
from torchtitan.experiments.moe_ladder.models.qwen3 import (
    ladderize_qwen3_config,
    model_registry,
    Qwen3HoistedGateABlock,
    Qwen3HoistedGateBBlock,
    Qwen3LadderModel,
    Qwen3LadderMoEBlock,
    Qwen3ParallelMoEBlock,
)
from torchtitan.experiments.moe_ladder.profile_block import _block_step
from torchtitan.models.common import (
    CosSinRoPE,
    Embedding,
    GQAttention,
    Linear,
    QKVLinear,
    RMSNorm,
    ScaledDotProductAttention,
)
from torchtitan.models.common.moe import GroupedExperts, MoE, TokenChoiceTopKRouter
from torchtitan.models.common.token_dispatcher import AllToAllTokenDispatcher
from torchtitan.models.qwen3.model import Qwen3Model, Qwen3TransformerBlock
from torchtitan.protocols.sharding import ShardingConfig


# Shape legend: B=batch, L=seq, D=model dim, V=vocab.
def _constant_gate_init(param: torch.Tensor) -> None:
    torch.nn.init.constant_(param, 0.125)


def _tiny_qwen3_moe_config(num_layers: int = 3) -> Qwen3Model.Config:
    dim = 16
    head_dim = 8
    num_heads = 2
    num_kv_heads = 1
    num_experts = 4
    top_k = 2
    vocab_size = 32

    attention = GQAttention.Config(
        n_heads=num_heads,
        n_kv_heads=num_kv_heads,
        head_dim=head_dim,
        dim=dim,
        qkv_linear=QKVLinear.Config(
            head_dim=head_dim,
            wq=Linear.Config(in_features=dim, out_features=num_heads * head_dim),
            wkv=Linear.Config(in_features=dim, out_features=num_kv_heads * head_dim),
        ),
        wo=Linear.Config(in_features=num_heads * head_dim, out_features=dim),
        qk_norm=RMSNorm.Config(normalized_shape=head_dim),
        inner_attention=ScaledDotProductAttention.Config(),
        rope=CosSinRoPE.Config(dim=head_dim, max_seq_len=16, theta=10000.0),
    )
    moe = MoE.Config(
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
            hidden_dim=32,
            num_experts=num_experts,
            token_dispatcher=AllToAllTokenDispatcher.Config(
                num_experts=num_experts,
                top_k=top_k,
            ),
        ),
    )
    layers = [
        Qwen3TransformerBlock.Config(
            attention=attention,
            attention_norm=RMSNorm.Config(normalized_shape=dim),
            ffn_norm=RMSNorm.Config(normalized_shape=dim),
            moe=moe,
        )
        for _ in range(num_layers)
    ]
    return Qwen3Model.Config(
        vocab_size=vocab_size,
        dim=dim,
        norm=RMSNorm.Config(normalized_shape=dim),
        tok_embeddings=Embedding.Config(
            num_embeddings=vocab_size,
            embedding_dim=dim,
        ),
        lm_head=Linear.Config(in_features=dim, out_features=vocab_size),
        layers=layers,
    )


@pytest.mark.parametrize(
    ("schedule", "block_cls"),
    [
        ("parallel", Qwen3ParallelMoEBlock),
        ("ladder", Qwen3LadderMoEBlock),
        ("hoisted_gateA", Qwen3HoistedGateABlock),
        ("hoisted_gateB", Qwen3HoistedGateBBlock),
    ],
)
def test_ladderize_qwen3_config_uses_schedule_blocks(schedule, block_cls):
    config = ladderize_qwen3_config(_tiny_qwen3_moe_config(), schedule)
    model = config.build()

    assert isinstance(model, Qwen3LadderModel)
    assert all(isinstance(layer, block_cls) for layer in model.layers.values())


def test_ladder_moe_preserves_router_param_init_after_init_states():
    config = ladderize_qwen3_config(_tiny_qwen3_moe_config(num_layers=1), "parallel")
    assert config.layers[0].moe is not None
    config.layers[0].moe.router.gate.param_init = {"weight": _constant_gate_init}

    model = config.build()
    model.init_states(buffer_device=torch.device("cpu"))

    layer = next(iter(model.layers.values()))
    torch.testing.assert_close(
        layer.moe.router.gate.weight,
        torch.full_like(layer.moe.router.gate.weight, 0.125),
    )


def test_ladder_moe_preserves_common_moe_sharding_config():
    config = ladderize_qwen3_config(_tiny_qwen3_moe_config(num_layers=1), "parallel")
    assert config.layers[0].moe is not None
    sharding_config = ShardingConfig()
    config.layers[0].moe.sharding_config = sharding_config

    model = config.build()
    layer = next(iter(model.layers.values()))

    assert layer.moe._sharding_config is sharding_config


def test_qwen3_ladder_parallelize_allows_tp(monkeypatch):
    config = ladderize_qwen3_config(_tiny_qwen3_moe_config(num_layers=1), "parallel")
    model = config.build()
    calls = []

    def fake_parallelize(self, parallel_dims):
        calls.append((self, parallel_dims))

    class FakeParallelDims:
        tp_enabled = True

    monkeypatch.setattr(Qwen3Model, "parallelize", fake_parallelize)
    parallel_dims = FakeParallelDims()

    model.parallelize(parallel_dims)

    assert calls == [(model, parallel_dims)]


@pytest.mark.parametrize(
    "schedule", ["parallel", "ladder", "hoisted_gateA", "hoisted_gateB"]
)
def test_qwen3_ladder_model_forward_backward(schedule):
    torch.manual_seed(0)
    config = ladderize_qwen3_config(_tiny_qwen3_moe_config(), schedule)
    model = config.build()
    model.init_states(buffer_device=torch.device("cpu"))

    B, L = 2, 5
    tokens_BL = torch.randint(0, config.vocab_size, (B, L))
    positions_BL = torch.arange(L).expand(B, L)

    out_BLV = model(tokens_BL, positions=positions_BL, attention_masks=None)
    assert out_BLV.shape == (B, L, config.vocab_size)
    assert torch.isfinite(out_BLV).all()

    loss = out_BLV.float().square().mean()
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert grads and all(g is not None for g in grads)


@pytest.mark.parametrize(
    "schedule", ["parallel", "ladder", "hoisted_gateA", "hoisted_gateB"]
)
def test_qwen3_profile_block_step_smoke(schedule):
    torch.manual_seed(0)
    config = ladderize_qwen3_config(_tiny_qwen3_moe_config(num_layers=1), schedule)
    model = config.build()
    model.init_states(buffer_device=torch.device("cpu"))
    layer = next(iter(model.layers.values()))

    B, L = 2, 5
    h_BLD = torch.randn(B, L, config.dim)
    positions_BL = torch.arange(L).expand(B, L)

    out_BLD, pending = _block_step(layer, h_BLD, None, positions_BL, None)
    assert out_BLD.shape == h_BLD.shape
    assert torch.isfinite(out_BLD).all()

    if schedule == "parallel":
        assert pending is None
    else:
        assert pending is not None
        out_BLD, pending = _block_step(layer, h_BLD, None, positions_BL, pending)
        assert out_BLD.shape == h_BLD.shape
        assert torch.isfinite(out_BLD).all()
        assert pending is not None


def test_qwen3_ladder_model_calls_delayed_blocks_through_module_call():
    torch.manual_seed(0)
    config = ladderize_qwen3_config(_tiny_qwen3_moe_config(), "ladder")
    model = config.build()
    model.init_states(buffer_device=torch.device("cpu"))

    calls = []
    handles = [
        layer.register_forward_hook(lambda module, args, output: calls.append(module))
        for layer in model.layers.values()
    ]
    try:
        B, L = 2, 5
        tokens_BL = torch.randint(0, config.vocab_size, (B, L))
        positions_BL = torch.arange(L).expand(B, L)
        model(tokens_BL, positions=positions_BL, attention_masks=None)
    finally:
        for handle in handles:
            handle.remove()

    assert len(calls) == len(model.layers)


def test_qwen3_ladder_model_detects_checkpoint_wrapped_delayed_blocks():
    class WrappedBlock(torch.nn.Module):
        def __init__(self, block):
            super().__init__()
            self._checkpoint_wrapped_module = block
            self.return_pending_values = []

        def forward(self, *args, **kwargs):
            self.return_pending_values.append(kwargs.get("return_pending", False))
            return self._checkpoint_wrapped_module(*args, **kwargs)

    torch.manual_seed(0)
    config = ladderize_qwen3_config(_tiny_qwen3_moe_config(), "ladder")
    model = config.build()
    model.init_states(buffer_device=torch.device("cpu"))

    wrappers = []
    for name, layer in list(model.layers.items()):
        wrapper = WrappedBlock(layer)
        model.layers[name] = wrapper
        wrappers.append(wrapper)

    B, L = 2, 5
    tokens_BL = torch.randint(0, config.vocab_size, (B, L))
    positions_BL = torch.arange(L).expand(B, L)
    out_BLV = model(tokens_BL, positions=positions_BL, attention_masks=None)

    assert out_BLV.shape == (B, L, config.vocab_size)
    assert all(wrapper.return_pending_values == [True] for wrapper in wrappers)


def test_qwen3_ladder_registry_rejects_unsupported_comm_backend():
    with pytest.raises(ValueError, match="'standard' all-to-all and 'deepep'"):
        model_registry(
            "debugmodel_moe",
            schedule="parallel",
            attn_backend="flex",
            moe_comm_backend="hybridep",
        )


def test_qwen3_ladder_registry_accepts_deepep_comm_backend():
    spec = model_registry(
        "debugmodel_moe",
        schedule="parallel",
        attn_backend="flex",
        moe_comm_backend="deepep",
    )

    assert isinstance(spec.model, Qwen3LadderModel.Config)


def test_qwen3_model_registry_returns_ladder_spec():
    spec = model_registry("debugmodel_moe", schedule="parallel", attn_backend="flex")

    assert spec.name == "moe_ladder_qwen3"
    assert isinstance(spec.model, Qwen3LadderModel.Config)
    assert spec.post_optimizer_build_fn is not None


def test_non_qwen_adapters_are_explicit_scaffolds():
    with pytest.raises(NotImplementedError):
        ladderize_llama3_moe_config(object(), "parallel")
    with pytest.raises(NotImplementedError):
        ladderize_deepseek_v3_config(object(), "parallel")

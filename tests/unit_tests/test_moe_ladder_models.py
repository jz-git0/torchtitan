# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
from types import SimpleNamespace

import pytest
import torch
import torch.distributed.checkpoint as dcp
from safetensors.torch import save_file
from torchtitan.config import CompileConfig, ParallelismConfig
from torchtitan.experiments.moe_ladder import config_registry as ladder_configs
from torchtitan.experiments.moe_ladder.config_registry import (
    qwen3_30b_a3b_ladder,
    qwen3_30b_a3b_stock,
)
from torchtitan.experiments.moe_ladder.profile_inference import (
    _chunked_cross_entropy,
    _load_model_weights,
    _model_spec,
    _parallel_configs,
    _worker_cli_args,
)
from torchtitan.models.qwen3.state_dict_adapter import Qwen3StateDictAdapter
from torchtitan.protocols.state_dict_adapter import StateDictAdapter


class _IdentityStateDictAdapter(StateDictAdapter):
    """Minimal adapter for the profiler's direct Hugging Face load branch."""

    def to_hf(self, state_dict):
        return state_dict

    def from_hf(self, hf_state_dict):
        return hf_state_dict


def test_worker_cli_args_forward_worker_options(tmp_path) -> None:
    output = tmp_path / "worker.json"
    args = argparse.Namespace(
        worker=False,
        variant=None,
        variants=["stock"],
        output=tmp_path / "all.json",
        nproc_per_node=2,
        timeout=30,
        batch_size=4,
        loss_only=True,
        checkpoint_path=None,
    )

    assert _worker_cli_args(args, "ladder", output) == [
        "--worker",
        "--variant",
        "ladder",
        "--output",
        str(output),
        "--batch-size",
        "4",
        "--loss-only",
    ]


def test_chunked_cross_entropy_matches_full_loss() -> None:
    generator = torch.Generator().manual_seed(17)
    logits_BLV = torch.randn(2, 5, 11, generator=generator, dtype=torch.bfloat16)
    labels_BL = torch.randint(0, 11, (2, 5), generator=generator)

    actual = _chunked_cross_entropy(logits_BLV, labels_BL, chunk_tokens=3)
    expected = torch.nn.functional.cross_entropy(
        logits_BLV.float().reshape(-1, 11), labels_BL.reshape(-1)
    )

    torch.testing.assert_close(actual, expected)


def test_model_spec_rejects_invalid_layer_truncation() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        _model_spec("stock", "flex", n_layers=-1)
    with pytest.raises(ValueError, match="exceeds"):
        _model_spec("stock", "flex", n_layers=100)


def test_dcp_loader_updates_model_parameters(tmp_path) -> None:
    source = torch.nn.Linear(4, 3)
    target = torch.nn.Linear(4, 3)
    with torch.no_grad():
        source.weight.fill_(2.5)
        source.bias.fill_(-1.25)
        target.weight.zero_()
        target.bias.zero_()

    checkpoint_path = tmp_path / "checkpoint"
    dcp.save(source.state_dict(), checkpoint_id=str(checkpoint_path))
    _load_model_weights(
        target,
        SimpleNamespace(),
        None,
        checkpoint_path=checkpoint_path,
        hf_model_path=None,
    )

    torch.testing.assert_close(target.weight, source.weight)
    torch.testing.assert_close(target.bias, source.bias)


def test_hf_loader_updates_model_parameters(tmp_path) -> None:
    source = torch.nn.Linear(4, 3)
    target = torch.nn.Linear(4, 3)
    with torch.no_grad():
        source.weight.fill_(4.5)
        source.bias.fill_(0.75)
        target.weight.zero_()
        target.bias.zero_()

    hf_model_path = tmp_path / "hf_model"
    hf_model_path.mkdir()
    save_file(source.state_dict(), hf_model_path / "model.safetensors")
    _load_model_weights(
        target,
        SimpleNamespace(),  # pyrefly: ignore [bad-argument-type]
        _IdentityStateDictAdapter,
        checkpoint_path=None,
        hf_model_path=hf_model_path,
    )

    torch.testing.assert_close(target.weight, source.weight)
    torch.testing.assert_close(target.bias, source.bias)
    assert not (hf_model_path / ".metadata").exists()


def test_30b_training_config_uses_ladder_model_and_ep4() -> None:
    config = qwen3_30b_a3b_ladder()

    assert config.model_spec.name == "moe_ladder_qwen3"
    assert config.model_spec.flavor == "30B-A3B_ladder"
    assert config.model_spec.state_dict_adapter is Qwen3StateDictAdapter
    assert config.parallelism.expert_parallel_degree == 4
    assert config.activation_checkpoint is None


def test_delayed_config_rejects_pipeline_parallelism() -> None:
    spec = _model_spec("ladder", "flex", n_layers=2)
    parallelism = ParallelismConfig(
        expert_parallel_degree=2,
        pipeline_parallel_degree=2,
    )

    with pytest.raises(
        ValueError,
        match="pending MoE state cannot cross stages",
    ):
        spec.model.update_from_config(
            config=SimpleNamespace(
                parallelism=parallelism,
                compile=CompileConfig(),
            )
        )


def test_delayed_config_rejects_activation_checkpointing() -> None:
    spec = _model_spec("ladder", "flex", n_layers=2)
    config = SimpleNamespace(
        parallelism=ParallelismConfig(expert_parallel_degree=2),
        activation_checkpoint=object(),
        compile=CompileConfig(),
    )

    with pytest.raises(ValueError, match="do not support activation checkpointing"):
        spec.model.update_from_config(config=config)


def test_ladder_config_rejects_model_compile() -> None:
    spec = _model_spec("parallel", "flex", n_layers=2)
    config = SimpleNamespace(
        parallelism=ParallelismConfig(expert_parallel_degree=2),
        activation_checkpoint=None,
        compile=CompileConfig(enable=True),
    )

    with pytest.raises(ValueError, match="do not support torch.compile"):
        spec.model.update_from_config(config=config)


@pytest.mark.parametrize(
    "variant", ["parallel", "ladder", "hoisted_gateA", "hoisted_gateB"]
)
def test_ladder_variant_builds_for_inference(variant: str) -> None:
    spec = _model_spec(variant, "flex", n_layers=2)
    parallelism = ParallelismConfig(expert_parallel_degree=2)

    spec.model.update_from_config(
        config=SimpleNamespace(
            parallelism=parallelism,
            activation_checkpoint=None,
            compile=CompileConfig(),
        )
    )
    with torch.device("meta"):
        model = spec.model.build()

    assert len(model.layers) == 2


def test_profiler_rejects_uneven_tp_sequence() -> None:
    args = argparse.Namespace(
        tp=2,
        ep=2,
        loss_only=True,
        steps=1,
        loss_chunk_tokens=1,
        seq_len=7,
    )
    with pytest.raises(ValueError, match="seq_len to be divisible by tp"):
        _parallel_configs(args, world_size=4)


@pytest.mark.parametrize(
    "factory_name",
    [
        "qwen3_debugmodel_moe_stock",
        "qwen3_debugmodel_moe_parallel",
        "qwen3_debugmodel_moe_ladder",
        "qwen3_debugmodel_moe_hoisted_gateA",
        "qwen3_debugmodel_moe_hoisted_gateB",
    ],
)
def test_debug_config_factory_builds(factory_name: str) -> None:
    config = getattr(ladder_configs, factory_name)()
    assert config.parallelism.expert_parallel_degree == 2
    assert config.activation_checkpoint is None


def test_30b_stock_config_uses_deepep_and_ep4() -> None:
    config = qwen3_30b_a3b_stock()

    assert config.model_spec.name == "qwen3"
    assert config.parallelism.expert_parallel_degree == 4
    assert config.activation_checkpoint is None
    assert (
        "DeepEPTokenDispatcher"
        in type(
            config.model_spec.model.layers[0].moe.experts.token_dispatcher
        ).__qualname__
    )

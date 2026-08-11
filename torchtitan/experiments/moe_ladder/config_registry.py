# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""TorchTitan configs for the MoE ladder experiment."""

from __future__ import annotations

from dataclasses import replace

from torchtitan.components.checkpoint import CheckpointManager
from torchtitan.components.loss import ChunkedLossWrapper, CrossEntropyLoss
from torchtitan.components.lr_scheduler import LRSchedulersContainer
from torchtitan.components.metrics import MetricsProcessor
from torchtitan.components.optimizer import default_adamw
from torchtitan.config import ParallelismConfig, TrainingConfig
from torchtitan.hf_datasets.text_datasets import HuggingFaceTextDataLoader
from torchtitan.models.common.config_utils import decoder_vocab_size
from torchtitan.models.qwen3 import model_registry as qwen3_model_registry
from torchtitan.models.qwen3.config_registry import qwen3_30b_a3b as stock_qwen3_30b_a3b
from torchtitan.trainer import Trainer
from .models.qwen3 import model_registry, MOE_COMM_BACKEND


# These configs only validate schedule execution. The debug geometry is too small
# for meaningful compute/communication overlap measurements.
def _qwen3_debugmodel_moe(schedule: str | None) -> Trainer.Config:
    """Build a Qwen3 debug MoE training config for a baseline or schedule.

    Input: schedule is a valid ladder schedule string, or None for stock Qwen3.
    Output: Trainer.Config for Qwen3 debug MoE, EP=2, seq_len=1024, and 10 steps.
    """
    model_spec = (
        qwen3_model_registry("debugmodel_moe", moe_comm_backend=MOE_COMM_BACKEND)
        if schedule is None
        else model_registry("debugmodel_moe", schedule=schedule)
    )
    config = Trainer.Config(
        loss=ChunkedLossWrapper.Config(
            loss_fn=CrossEntropyLoss.Config(
                global_vocab_size=decoder_vocab_size(model_spec),
            ),
        ),
        hf_assets_path="./tests/assets/tokenizer",
        metrics=MetricsProcessor.Config(log_freq=1),
        model_spec=model_spec,
        dataloader=HuggingFaceTextDataLoader.Config(dataset="c4_test"),
        optimizer=default_adamw(lr=3e-4),
        lr_scheduler=LRSchedulersContainer.Config(warmup_steps=2),
        training=TrainingConfig(local_batch_size=4, seq_len=1024, steps=10),
        parallelism=ParallelismConfig(expert_parallel_degree=2),
        activation_checkpoint=None,
        checkpoint=CheckpointManager.Config(
            interval=10,
            last_save_model_only=False,
            export_dtype="float16",
        ),
    )
    return config


def _qwen3_30b_a3b(schedule: str | None) -> Trainer.Config:
    """Adapt the stock 30B-A3B recipe to a DeepEP baseline or schedule."""
    config = stock_qwen3_30b_a3b()
    model_spec = (
        qwen3_model_registry("30B-A3B", moe_comm_backend=MOE_COMM_BACKEND)
        if schedule is None
        else model_registry("30B-A3B", schedule=schedule)
    )
    parallelism = replace(config.parallelism, expert_parallel_degree=4)
    return replace(
        config,
        model_spec=model_spec,
        parallelism=parallelism,
        activation_checkpoint=None,
    )


def qwen3_debugmodel_moe_stock() -> Trainer.Config:
    """Return the stock Qwen3 debug MoE config with DeepEP."""
    return _qwen3_debugmodel_moe(None)


def qwen3_debugmodel_moe_parallel() -> Trainer.Config:
    """Return the Qwen3 debug MoE config for the parallel schedule."""
    return _qwen3_debugmodel_moe("parallel")


def qwen3_debugmodel_moe_ladder() -> Trainer.Config:
    """Return the Qwen3 debug MoE config for the ladder schedule."""
    return _qwen3_debugmodel_moe("ladder")


def qwen3_debugmodel_moe_hoisted_gateA() -> Trainer.Config:  # noqa: N802
    """Return the Qwen3 debug MoE config for hoisted_gateA."""
    return _qwen3_debugmodel_moe("hoisted_gateA")


def qwen3_debugmodel_moe_hoisted_gateB() -> Trainer.Config:  # noqa: N802
    """Return the Qwen3 debug MoE config for hoisted_gateB."""
    return _qwen3_debugmodel_moe("hoisted_gateB")


def qwen3_30b_a3b_stock() -> Trainer.Config:
    """Return the stock Qwen3 30B-A3B config with DeepEP and EP=4."""
    return _qwen3_30b_a3b(None)


def qwen3_30b_a3b_parallel() -> Trainer.Config:
    """Return the Qwen3 30B-A3B config for the parallel schedule."""
    return _qwen3_30b_a3b("parallel")


def qwen3_30b_a3b_ladder() -> Trainer.Config:
    """Return the Qwen3 30B-A3B config for the ladder schedule."""
    return _qwen3_30b_a3b("ladder")


def qwen3_30b_a3b_hoisted_gateA() -> Trainer.Config:  # noqa: N802
    """Return the Qwen3 30B-A3B config for hoisted_gateA."""
    return _qwen3_30b_a3b("hoisted_gateA")


def qwen3_30b_a3b_hoisted_gateB() -> Trainer.Config:  # noqa: N802
    """Return the Qwen3 30B-A3B config for hoisted_gateB."""
    return _qwen3_30b_a3b("hoisted_gateB")

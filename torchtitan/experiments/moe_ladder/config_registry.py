# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""TorchTitan configs for the MoE ladder experiment."""

from __future__ import annotations

from torchtitan.components.checkpoint import CheckpointManager
from torchtitan.components.loss import ChunkedLossWrapper, CrossEntropyLoss
from torchtitan.components.lr_scheduler import LRSchedulersContainer
from torchtitan.components.metrics import MetricsProcessor
from torchtitan.components.optimizer import default_adamw
from torchtitan.config import ParallelismConfig, TrainingConfig
from torchtitan.hf_datasets.text_datasets import HuggingFaceTextDataLoader
from torchtitan.models.common.config_utils import decoder_vocab_size
from torchtitan.trainer import Trainer

from .models.qwen3 import model_registry


def _qwen3_debugmodel_moe(schedule: str) -> Trainer.Config:
    """Build a Qwen3 debug MoE training config for one ladder schedule.

    Input: schedule is a valid ladder schedule string.
    Output: Trainer.Config for Qwen3 debug MoE, EP=2, seq_len=1024, and 10 steps.
    """
    model_spec = model_registry("debugmodel_moe", schedule=schedule)
    return Trainer.Config(
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
        checkpoint=CheckpointManager.Config(
            interval=10,
            last_save_model_only=False,
            export_dtype="float16",
        ),
    )


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

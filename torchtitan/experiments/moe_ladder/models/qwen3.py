# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Qwen3 adapter for the MoE ladder experiment.

Shape suffix legend:
  B=batch, L=seq, D=model dim, R=routed local tokens.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from dataclasses import dataclass
from typing import ClassVar

import torch

from torchtitan.components.optimizer import register_moe_load_balancing_hook
from torchtitan.distributed.pipeline_parallel import pipeline_llm
from torchtitan.models.common.attention import AttentionMasksType
from torchtitan.models.common.decoder import TransformerBlock
from torchtitan.models.qwen3.model import Qwen3Model, Qwen3TransformerBlock
from torchtitan.models.qwen3.parallelize import parallelize_qwen3
from torchtitan.models.qwen3.state_dict_adapter import Qwen3StateDictAdapter
from torchtitan.models.utils import validate_converter_order
from torchtitan.protocols.model import ModelConfigConverter
from torchtitan.protocols.model_spec import ModelSpec
from ..ladder_moe import Pending
from ..schedule_runner import (
    hoisted_gate_a_step,
    hoisted_gate_b_step,
    ladder_step,
    parallel_step,
)
from ..schedules import normalize_schedule, ScheduleName
from .common import build_ladder_moe, drain_pending

# The only MoE communication backend the ladder supports. Stock Qwen3 baselines
# must be built with it too so comparisons hold the dispatcher fixed.
MOE_COMM_BACKEND = "deepep"


class _Qwen3LadderBlock(TransformerBlock):
    """Base Qwen3 block that owns attention, norms, and LadderMoE.

    Input: Qwen3TransformerBlock.Config with an MoE config.
    Output: residual [B, L, D], or (residual [B, L, D], Pending).
    """

    schedule: ClassVar[ScheduleName]

    def __init__(self, config: Qwen3TransformerBlock.Config):
        super().__init__()
        if config.moe is None:
            raise ValueError("Qwen3 ladder blocks require a MoE config")

        # Match Qwen3TransformerBlock's child registration order. This keeps
        # seeded initialization identical between stock and ladder variants.
        self.attention = config.attention.build()
        self.moe = build_ladder_moe(config.moe)
        self.attention_norm = config.attention_norm.build()
        self.ffn_norm = config.ffn_norm.build()
        self.moe_enabled = True

    def _attention(
        self,
        res_BLD: torch.Tensor,
        attention_masks: AttentionMasksType | None,
        positions: torch.Tensor | None,
    ) -> torch.Tensor:
        """Apply Qwen3 attention to a residual tensor.

        Input: res_BLD is [B, L, D], attention_masks is optional, and positions is [B, L] or None.
        Output: attention result [B, L, D].
        """
        return self.attention(self.attention_norm(res_BLD), attention_masks, positions)


class Qwen3ParallelMoEBlock(_Qwen3LadderBlock):
    """Option 1: attention and MoE read the same residual."""

    schedule: ClassVar[ScheduleName] = "parallel"

    @dataclass(kw_only=True, slots=True)
    class Config(Qwen3TransformerBlock.Config):
        """Config for the Qwen3 parallel ladder block."""

    def forward(
        self,
        res_BLD: torch.Tensor,
        attention_masks: AttentionMasksType | None,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run the parallel ladder schedule for one Qwen3 block.

        Input: res_BLD is [B, L, D], optional masks and positions match Qwen3 attention.
        Output: updated residual [B, L, D].
        """
        return parallel_step(
            self.moe,
            res_BLD,
            attention=lambda res: self._attention(res, attention_masks, positions),
            ffn_norm=self.ffn_norm,
        )


class _Qwen3DelayedMoEBlock(_Qwen3LadderBlock):
    """Base class for schedules that pass Pending across blocks."""

    step_fn: ClassVar[Callable[..., tuple[torch.Tensor, Pending]]]

    def drain(self, res_BLD: torch.Tensor, pending: Pending) -> torch.Tensor:
        """Combine pending expert output into the residual."""
        return drain_pending(self.moe, res_BLD, pending)

    def forward_step(
        self,
        res_BLD: torch.Tensor,
        pending: Pending | None,
        attention_masks: AttentionMasksType | None,
        positions: torch.Tensor | None,
    ) -> tuple[torch.Tensor, Pending]:
        """Run one delayed schedule step without draining next Pending.

        Input: res_BLD is [B, L, D], pending is None or (expert output [R, D], state).
        Output: (residual [B, L, D], next Pending).
        """
        return type(self).step_fn(
            self.moe,
            res_BLD,
            pending,
            attention=lambda res: self._attention(res, attention_masks, positions),
            ffn_norm=self.ffn_norm,
        )

    def forward(
        self,
        res_BLD: torch.Tensor,
        attention_masks: AttentionMasksType | None,
        positions: torch.Tensor | None = None,
        *,
        pending: Pending | None = None,
    ) -> tuple[torch.Tensor, Pending]:
        """Run a delayed block and return its next Pending."""
        return self.forward_step(res_BLD, pending, attention_masks, positions)


class Qwen3LadderMoEBlock(_Qwen3DelayedMoEBlock):
    """Option 2: delayed MoE output, fresh routing after combine."""

    schedule: ClassVar[ScheduleName] = "ladder"
    step_fn: ClassVar[Callable[..., tuple[torch.Tensor, Pending]]] = ladder_step

    @dataclass(kw_only=True, slots=True)
    class Config(Qwen3TransformerBlock.Config):
        """Config for the Qwen3 delayed ladder block."""


class Qwen3HoistedGateABlock(_Qwen3DelayedMoEBlock):
    """Option 3A: launch combine, then compute stale gate decisions."""

    schedule: ClassVar[ScheduleName] = "hoisted_gateA"
    step_fn: ClassVar[Callable[..., tuple[torch.Tensor, Pending]]] = hoisted_gate_a_step

    @dataclass(kw_only=True, slots=True)
    class Config(Qwen3TransformerBlock.Config):
        """Config for the Qwen3 hoisted_gateA block."""


class Qwen3HoistedGateBBlock(_Qwen3DelayedMoEBlock):
    """Option 3B: compute stale gate decisions, then launch combine."""

    schedule: ClassVar[ScheduleName] = "hoisted_gateB"
    step_fn: ClassVar[Callable[..., tuple[torch.Tensor, Pending]]] = hoisted_gate_b_step

    @dataclass(kw_only=True, slots=True)
    class Config(Qwen3TransformerBlock.Config):
        """Config for the Qwen3 hoisted_gateB block."""


_BLOCK_BY_SCHEDULE: dict[ScheduleName, type[_Qwen3LadderBlock]] = {
    "parallel": Qwen3ParallelMoEBlock,
    "ladder": Qwen3LadderMoEBlock,
    "hoisted_gateA": Qwen3HoistedGateABlock,
    "hoisted_gateB": Qwen3HoistedGateBBlock,
}


def _unwrap_ladder_block(layer: torch.nn.Module) -> _Qwen3LadderBlock | None:
    """Return the inner ladder block through checkpoint wrappers.

    Input: layer is a Qwen3 layer module, possibly checkpoint-wrapped.
    Output: the underlying _Qwen3LadderBlock or None for non-ladder layers.
    """
    while True:
        if isinstance(layer, _Qwen3LadderBlock):
            return layer
        wrapped = getattr(layer, "_checkpoint_wrapped_module", None)
        if wrapped is None:
            return None
        layer = wrapped


class Qwen3LadderModel(Qwen3Model):
    """Qwen3 decoder whose MoE blocks use one ladder schedule."""

    @dataclass(kw_only=True, slots=True)
    class Config(Qwen3Model.Config):
        """Config for the Qwen3 model using ladder MoE blocks."""

        def update_from_config(self, *, config, **kwargs) -> None:
            Qwen3Model.Config.update_from_config(self, config=config, **kwargs)
            if config.compile.enable and "model" in config.compile.components:
                raise ValueError("MoE ladder schedules do not support torch.compile")
            delayed_config_types = (
                Qwen3LadderMoEBlock.Config,
                Qwen3HoistedGateABlock.Config,
                Qwen3HoistedGateBBlock.Config,
            )
            if not isinstance(self.layers[0], delayed_config_types):
                return
            if config.parallelism.pipeline_parallel_degree > 1:
                raise ValueError(
                    "delayed MoE ladder schedules do not support pipeline "
                    "parallelism because pending MoE state cannot cross stages"
                )
            if getattr(config, "activation_checkpoint", None) is not None:
                raise ValueError(
                    "delayed MoE ladder schedules do not support activation "
                    "checkpointing because DeepEP state crosses block boundaries"
                )

    def forward(
        self,
        tokens: torch.Tensor,
        positions: torch.Tensor | None = None,
        attention_masks: AttentionMasksType | None = None,
    ):
        """Run the Qwen3 model while carrying delayed MoE state.

        Input: tokens is [B, L] ids or [B, L, D] embeddings; positions is [B, L] or None.
        Output: logits [B, L, V], or hidden states [B, L, D] when lm_head is skipped.
        """
        h_BLD = (
            self.tok_embeddings(tokens) if self.tok_embeddings is not None else tokens
        )
        layers = self.layers.values()
        first_block = _unwrap_ladder_block(next(iter(layers)))
        assert isinstance(first_block, _Qwen3LadderBlock)
        if isinstance(first_block, Qwen3ParallelMoEBlock):
            for layer in layers:
                h_BLD = layer(h_BLD, attention_masks, positions)
        else:
            assert isinstance(first_block, _Qwen3DelayedMoEBlock)
            pending: Pending | None = None
            for layer in layers:
                h_BLD, pending = layer(
                    h_BLD, attention_masks, positions, pending=pending
                )
            assert pending is not None
            last_block = _unwrap_ladder_block(layer)
            assert isinstance(last_block, _Qwen3DelayedMoEBlock)
            h_BLD = last_block.drain(h_BLD, pending)

        h_BLD = self.norm(h_BLD) if self.norm is not None else h_BLD
        if self._skip_lm_head:
            return h_BLD
        return self.lm_head(h_BLD) if self.lm_head is not None else h_BLD


def _init_field_values(config: object) -> dict[str, object]:
    """Return values for the public init fields of a dataclass instance."""
    return {
        field.name: getattr(config, field.name)
        for field in dataclasses.fields(config)
        if field.init and not field.name.startswith("_")
    }


def _copy_block_config(
    layer_config: Qwen3TransformerBlock.Config,
    block_cls: type[_Qwen3LadderBlock],
) -> Qwen3TransformerBlock.Config:
    """Copy a Qwen3 block config into a ladder block config class.

    Input: layer_config is the source block config and block_cls selects the schedule.
    Output: a new block config with the same init fields.
    """
    return block_cls.Config(**_init_field_values(layer_config))


def ladderize_qwen3_config(
    config: Qwen3Model.Config,
    schedule: ScheduleName,
) -> Qwen3LadderModel.Config:
    """Convert a Qwen3 MoE config to a Qwen3LadderModel config.

    Input: config has MoE transformer blocks and schedule is a ladder schedule string.
    Output: a model config whose layers use schedule-specific blocks.
    """
    block_cls = _BLOCK_BY_SCHEDULE[schedule]
    layers = []
    for layer_config in config.layers:
        if layer_config.moe is None:
            raise ValueError("Qwen3 ladder configs require every layer to be MoE")
        layers.append(_copy_block_config(layer_config, block_cls))

    kwargs = _init_field_values(config)
    kwargs["layers"] = layers
    return Qwen3LadderModel.Config(**kwargs)


def model_registry(
    flavor: str,
    *,
    schedule: str,
    attn_backend: str = "flex",
    converters: list[ModelConfigConverter.Config] | None = None,
) -> ModelSpec:
    """Build the TorchTitan ModelSpec for Qwen3 ladder training.

    Input: flavor names a Qwen3 config, schedule selects the block, and converters may update the config.
    Output: ModelSpec for training.

    The MoE communication backend is pinned to DeepEP: LadderMoE phase-splits
    combine over DeepEP's async combine and supports no other dispatcher.
    """
    from torchtitan.models.qwen3 import qwen3_configs

    config = qwen3_configs[flavor](
        attn_backend=attn_backend,
        moe_comm_backend=MOE_COMM_BACKEND,
    )
    if converters is not None:
        validate_converter_order(converters)
        for converter in converters:
            config = converter.build().convert(config)

    schedule_name = normalize_schedule(schedule)
    ladder_config = ladderize_qwen3_config(config, schedule_name)
    return ModelSpec(
        name="moe_ladder_qwen3",
        flavor=f"{flavor}_{schedule_name}",
        model=ladder_config,
        parallelize_fn=parallelize_qwen3,
        pipelining_fn=pipeline_llm,
        post_optimizer_build_fn=register_moe_load_balancing_hook,
        # Ladder blocks preserve stock Qwen3 state-dict keys, so the same
        # adapter supports direct Hugging Face checkpoint loading and export.
        state_dict_adapter=Qwen3StateDictAdapter,
    )

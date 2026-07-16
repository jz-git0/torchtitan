# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Forward-only benchmark for one Qwen3 MoE block.

This isolates the schedule-local question that the full-model benchmark cannot:
how much one Qwen3 attention+MoE block costs after EP/TP wiring, and whether the
schedule exposes useful overlap inside that block. Build and parallelization
reuse profile_inference.py so the profiled block has the same sharding as the
end-to-end model.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor

from torchtitan.experiments.moe_ladder.ladder_moe import Pending
from torchtitan.experiments.moe_ladder.models.qwen3 import (
    Qwen3HoistedGateABlock,
    Qwen3HoistedGateBBlock,
    Qwen3LadderMoEBlock,
    Qwen3ParallelMoEBlock,
)
from torchtitan.experiments.moe_ladder.nvtx import nvtx_range
from torchtitan.experiments.moe_ladder.profile_inference import (
    _build_model,
    _make_batch,
    _parallel_configs,
    _summarize_ms,
    VARIANTS,
)
from torchtitan.models.common.attention import AttentionMasksType
from torchtitan.models.qwen3.model import Qwen3TransformerBlock


DelayedBlock = Qwen3LadderMoEBlock | Qwen3HoistedGateABlock | Qwen3HoistedGateBBlock
ProfileBlock = Qwen3TransformerBlock | Qwen3ParallelMoEBlock | DelayedBlock


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--variant", choices=VARIANTS)
    parser.add_argument(
        "--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS)
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/moe_ladder_block_profile.json"),
    )
    parser.add_argument("--nproc-per-node", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--ep", type=int, default=2)
    parser.add_argument("--layer-index", type=int, default=0)
    parser.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    parser.add_argument(
        "--attn-backend",
        choices=("flex", "flex_flash", "varlen"),
        default="flex",
    )
    parser.add_argument(
        "--moe-comm-backend",
        choices=("standard", "deepep"),
        default="standard",
    )
    return parser.parse_args()


def _select_block(model: torch.nn.Module, layer_index: int) -> ProfileBlock:
    """Return one Qwen3 block from a built and parallelized model."""
    layers = list(model.layers.values())
    if layer_index < 0 or layer_index >= len(layers):
        raise ValueError(
            f"layer-index ({layer_index}) must be in [0, {len(layers) - 1}]"
        )
    block = layers[layer_index]
    if not isinstance(
        block,
        (
            Qwen3TransformerBlock,
            Qwen3ParallelMoEBlock,
            Qwen3LadderMoEBlock,
            Qwen3HoistedGateABlock,
            Qwen3HoistedGateBBlock,
        ),
    ):
        raise TypeError(f"expected a Qwen3 block, got {type(block).__name__}")
    return block


def _block_step(
    block: ProfileBlock,
    h_BLD: torch.Tensor,
    attention_masks: AttentionMasksType | None,
    positions_BL: torch.Tensor,
    pending: Pending | None,
) -> tuple[torch.Tensor, Pending | None]:
    """Run one profiled block step and carry delayed schedule state."""
    if isinstance(
        block,
        (Qwen3LadderMoEBlock, Qwen3HoistedGateABlock, Qwen3HoistedGateBBlock),
    ):
        out_BLD, pending_next = block(
            h_BLD,
            attention_masks,
            positions_BL,
            pending=pending,
            return_pending=True,
        )
        return out_BLD, pending_next
    return block(h_BLD, attention_masks, positions_BL), None


def _local_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """Return a local tensor for scalar reporting."""
    return tensor.to_local() if isinstance(tensor, DTensor) else tensor


def _run_worker(args: argparse.Namespace) -> None:
    """Run one distributed worker for a single block variant."""
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    parallel_dims, parallelism, training = _parallel_configs(args, world_size)
    model = _build_model(args, parallel_dims, parallelism, training)
    block = _select_block(model, args.layer_index)

    tokens_BL, _, positions_BL = _make_batch(args, rank)
    attention_masks = model.get_attention_masks(positions=positions_BL)
    with torch.no_grad():
        h_BLD = (
            model.tok_embeddings(tokens_BL)
            if model.tok_embeddings is not None
            else tokens_BL
        )
        pending: Pending | None = None
        out_BLD: torch.Tensor | None = None

        for _ in range(args.warmup_steps):
            with nvtx_range(f"moe_ladder_block/{args.variant}/warmup_step"):
                out_BLD, pending = _block_step(
                    block,
                    h_BLD,
                    attention_masks,
                    positions_BL,
                    pending,
                )
        torch.cuda.synchronize(device)
        dist.barrier()

        local_samples_ms: list[float] = []
        for _ in range(args.steps):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            with nvtx_range(f"moe_ladder_block/{args.variant}/measure_step"):
                start.record()
                out_BLD, pending = _block_step(
                    block,
                    h_BLD,
                    attention_masks,
                    positions_BL,
                    pending,
                )
                end.record()
                end.synchronize()
            local_samples_ms.append(start.elapsed_time(end))

        assert out_BLD is not None
        checksum = _local_tensor(out_BLD).float().mean()
        dist.all_reduce(checksum, op=dist.ReduceOp.AVG)

    rank_samples: list[list[float] | None] = [None] * world_size
    dist.all_gather_object(rank_samples, local_samples_ms)

    if rank == 0:
        if any(
            samples is None or len(samples) != args.steps for samples in rank_samples
        ):
            raise RuntimeError("ranks produced inconsistent latency sample counts")
        samples_ms = [
            max(samples[step] for samples in rank_samples if samples is not None)
            for step in range(args.steps)
        ]
        latency = _summarize_ms(samples_ms)
        num_data_groups = world_size // args.tp
        tokens_per_step = args.batch_size * args.seq_len * num_data_groups
        median_tps = tokens_per_step / (latency["median_ms"] / 1000.0)
        result: dict[str, Any] = {
            "profile_scope": "single_qwen3_block",
            "variant": args.variant,
            "world_size": world_size,
            "tp": args.tp,
            "ep": args.ep,
            "layer_index": args.layer_index,
            "batch_size": args.batch_size,
            "seq_len": args.seq_len,
            "local_tokens_per_rank": args.batch_size * args.seq_len // args.tp,
            "warmup_steps": args.warmup_steps,
            "measured_steps": args.steps,
            "dtype": args.dtype,
            "attn_backend": args.attn_backend,
            "moe_comm_backend": args.moe_comm_backend,
            "tokens_per_step": tokens_per_step,
            "tokens_per_second": median_tps,
            "output_mean": float(checksum.item()),
            **latency,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")

    dist.destroy_process_group()


def _format_table(results: list[dict[str, Any]]) -> str:
    """Format block benchmark metrics for stdout."""
    by_variant = {item["variant"]: item for item in results}
    stock_latency = by_variant.get("stock", {}).get("median_ms")
    parallel_latency = by_variant.get("parallel", {}).get("median_ms")
    header = (
        "variant          median_ms  p90_ms   tokens/s  vs_stock  vs_parallel  "
        "output_mean"
    )
    rows = [header, "-" * len(header)]
    for item in results:
        versus_stock = (
            stock_latency / item["median_ms"]
            if stock_latency is not None
            else float("nan")
        )
        versus_parallel = (
            parallel_latency / item["median_ms"]
            if parallel_latency is not None
            else float("nan")
        )
        rows.append(
            f"{item['variant']:<16} "
            f"{item['median_ms']:>9.3f} "
            f"{item['p90_ms']:>7.3f} "
            f"{item['tokens_per_second']:>10.1f} "
            f"{versus_stock:>8.3f}x "
            f"{versus_parallel:>11.3f}x "
            f"{item['output_mean']:>11.6f}"
        )
    return "\n".join(rows)


def _run_driver(args: argparse.Namespace) -> None:
    """Launch one torchrun job per variant and aggregate the block results."""
    output_dir = args.output.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    for variant in args.variants:
        variant_output = output_dir / f"{args.output.stem}_{variant}.json"
        cmd = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            f"--nproc-per-node={args.nproc_per_node}",
            "-m",
            "torchtitan.experiments.moe_ladder.profile_block",
            "--worker",
            "--variant",
            variant,
            "--output",
            str(variant_output),
            "--batch-size",
            str(args.batch_size),
            "--seq-len",
            str(args.seq_len),
            "--warmup-steps",
            str(args.warmup_steps),
            "--steps",
            str(args.steps),
            "--seed",
            str(args.seed),
            "--tp",
            str(args.tp),
            "--ep",
            str(args.ep),
            "--layer-index",
            str(args.layer_index),
            "--dtype",
            args.dtype,
            "--attn-backend",
            args.attn_backend,
            "--moe-comm-backend",
            args.moe_comm_backend,
        ]
        print(f"Running {variant} block...", flush=True)
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
        if proc.returncode != 0:
            print(proc.stdout[-4000:])
            print(proc.stderr[-4000:])
            raise SystemExit(proc.returncode)
        results.append(json.loads(variant_output.read_text()))

    args.output.write_text(json.dumps(results, indent=2) + "\n")
    print(_format_table(results))
    print(f"\nWrote {args.output}")


def main() -> None:
    """Run the driver or distributed worker."""
    args = _parse_args()
    if args.worker:
        _run_worker(args)
    else:
        _run_driver(args)


if __name__ == "__main__":
    main()

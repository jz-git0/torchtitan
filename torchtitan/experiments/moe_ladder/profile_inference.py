# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Forward-only benchmark for MoE ladder schedules on local GPUs.

The default workload keeps 8 * 1024 input tokens in flight per data-parallel
group and reports the slowest rank's latency for each distributed step. This is
an end-to-end throughput benchmark; use Nsight Systems with the emitted
NVTX ranges to explain a latency difference after this script identifies one.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast, Literal

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed.tensor import DTensor

from torchtitan.config import CompileConfig, ParallelismConfig, TrainingConfig
from torchtitan.distributed import ParallelDims
from torchtitan.experiments.moe_ladder.models.qwen3 import (
    model_registry as ladder_model_registry,
)
from torchtitan.experiments.moe_ladder.nvtx import nvtx_range
from torchtitan.models.qwen3 import model_registry as qwen3_model_registry
from torchtitan.models.qwen3.model import Qwen3Model


Variant = Literal["stock", "parallel", "ladder", "hoisted_gateA", "hoisted_gateB"]
VARIANTS: tuple[Variant, ...] = (
    "stock",
    "parallel",
    "ladder",
    "hoisted_gateA",
    "hoisted_gateB",
)


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Input: process argv.
    Output: argparse namespace for driver or worker mode.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--variant", choices=VARIANTS)
    parser.add_argument(
        "--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS)
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/moe_ladder_inference_profile.json"),
    )
    parser.add_argument("--nproc-per-node", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--ep", type=int, default=2)
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


def _dtype(name: str) -> torch.dtype:
    """Resolve a dtype option.

    Input: dtype name from CLI.
    Output: torch dtype object.
    """
    return {"bfloat16": torch.bfloat16, "float32": torch.float32}[name]


def _parallel_configs(
    args: argparse.Namespace, world_size: int
) -> tuple[ParallelDims, ParallelismConfig, TrainingConfig]:
    """Build matching TorchTitan parallelism and training configs.

    Input: CLI args plus distributed world size.
    Output: ParallelDims, ParallelismConfig, and TrainingConfig for inference setup.
    """
    if world_size % args.tp != 0:
        raise ValueError(
            f"world_size ({world_size}) must be divisible by tp ({args.tp})"
        )
    if world_size % args.ep != 0:
        raise ValueError(
            f"world_size ({world_size}) must be divisible by ep ({args.ep})"
        )
    if args.steps < 10:
        raise ValueError("performance measurements require at least 10 steps")
    if args.ep > 1 and args.tp > 1 and args.seq_len % args.tp != 0:
        raise ValueError(
            "LadderMoE EP+TP profiling requires seq_len to be divisible by tp"
        )
    if args.moe_comm_backend == "deepep" and args.ep <= 1:
        raise ValueError(
            "moe_comm_backend='deepep' requires ep > 1: the DeepEP dispatcher "
            "needs an EP mesh and fails late in dispatch() without one"
        )
    dp_shard = world_size // args.tp
    parallel_dims = ParallelDims(
        dp_replicate=1,
        dp_shard=dp_shard,
        cp=1,
        tp=args.tp,
        pp=1,
        ep=args.ep,
        world_size=world_size,
    )
    parallel_dims.build_mesh()
    parallelism = ParallelismConfig(
        data_parallel_shard_degree=dp_shard,
        tensor_parallel_degree=args.tp,
        expert_parallel_degree=args.ep,
    )
    training = TrainingConfig(
        local_batch_size=args.batch_size,
        seq_len=args.seq_len,
        steps=args.steps,
        mixed_precision_param=args.dtype,
        mixed_precision_reduce="float32",
    )
    return parallel_dims, parallelism, training


def _model_spec(
    variant: Variant, attn_backend: str, moe_comm_backend: str = "standard"
):
    """Return the stock or ladder Qwen3 MoE ModelSpec.

    Input: variant name, attention backend, and MoE communication backend.
    Output: ModelSpec for debugmodel_moe.
    """
    if variant == "stock":
        return qwen3_model_registry(
            "debugmodel_moe",
            attn_backend=attn_backend,
            moe_comm_backend=moe_comm_backend,
        )
    return ladder_model_registry(
        "debugmodel_moe",
        schedule=variant,
        attn_backend=attn_backend,
        moe_comm_backend=moe_comm_backend,
    )


def _reference_state(seed: int, attn_backend: str) -> dict[str, torch.Tensor]:
    """Initialize stock Qwen3 MoE once and return CPU reference weights.

    Input: seed for parameter initialization and attention backend.
    Output: state_dict mapping names to detached CPU tensors.
    """
    torch.manual_seed(seed)
    model = _model_spec("stock", attn_backend).model.build()
    model.init_states(buffer_device=torch.device("cpu"))
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
    }


def _build_model(
    args: argparse.Namespace,
    parallel_dims: ParallelDims,
    parallelism: ParallelismConfig,
    training: TrainingConfig,
):
    """Build, load, move, and parallelize one benchmark model.

    Input: worker args, parallel dims, parallelism config, and training config.
    Output: eval-mode model with EP/TP applied and FSDP skipped.
    """
    assert args.variant is not None
    device = torch.device("cuda", torch.cuda.current_device())
    spec = _model_spec(args.variant, args.attn_backend, args.moe_comm_backend)
    model_config = spec.model
    model_config.update_from_config(config=SimpleNamespace(parallelism=parallelism))
    model = model_config.build()
    model.init_states(buffer_device=torch.device("cpu"))
    model.load_state_dict(_reference_state(args.seed, args.attn_backend), strict=True)
    model.to(device=device, dtype=_dtype(args.dtype))
    model = spec.parallelize_fn(
        model,
        parallel_dims=parallel_dims,
        training=training,
        parallelism=parallelism,
        compile_config=CompileConfig(enable=False),
        ac_config=None,
        dump_folder="/tmp",
        skip_dp=True,
    )
    model.eval()
    return model


def _make_batch(
    args: argparse.Namespace, rank: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create deterministic next-token inputs for one data-parallel shard.

    Input: worker args and distributed rank.
    Output: tokens [B, L], labels [B, L], and positions [B, L] on CUDA.
    """
    data_rank = rank // args.tp
    generator = torch.Generator(device="cpu")
    generator.manual_seed(args.seed + 1009 * data_rank)
    device = torch.device("cuda", torch.cuda.current_device())
    spec = _model_spec("stock", args.attn_backend)
    model_config = cast(Qwen3Model.Config, spec.model)
    vocab_size = model_config.vocab_size
    full = torch.randint(
        0,
        vocab_size,
        (args.batch_size, args.seq_len + 1),
        generator=generator,
        dtype=torch.long,
    ).to(device)
    tokens = full[:, :-1].contiguous()
    labels = full[:, 1:].contiguous()
    positions = (
        torch.arange(args.seq_len, device=device)
        .unsqueeze(0)
        .expand(args.batch_size, -1)
    )
    return tokens, labels, positions


def _to_local_logits(logits: torch.Tensor) -> torch.Tensor:
    """Materialize logits for loss computation.

    Input: Tensor or DTensor logits [B, L, V].
    Output: local dense Tensor [B, L, V].
    """
    return logits.full_tensor() if isinstance(logits, DTensor) else logits


def _summarize_ms(times_ms: list[float]) -> dict[str, float]:
    """Summarize per-step latency samples.

    Input: list of elapsed forward times in milliseconds.
    Output: mean, median, min, max, and p90 latency.
    """
    ordered = sorted(times_ms)
    p90_index = min(len(ordered) - 1, int(0.9 * (len(ordered) - 1)))
    return {
        "mean_ms": statistics.fmean(times_ms),
        "median_ms": statistics.median(times_ms),
        "min_ms": min(times_ms),
        "max_ms": max(times_ms),
        "p90_ms": ordered[p90_index],
    }


def _run_worker(args: argparse.Namespace) -> None:
    """Run one distributed worker for a single variant.

    Input: worker CLI args with variant and output path.
    Output: rank 0 writes a JSON metrics file.
    """
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    parallel_dims, parallelism, training = _parallel_configs(args, world_size)
    model = _build_model(args, parallel_dims, parallelism, training)
    tokens, labels, positions = _make_batch(args, rank)
    attention_masks = model.get_attention_masks(positions=positions)
    local_samples_ms: list[float] = []

    # DTensor-backed RoPE cache views currently require version counters, which
    # torch.inference_mode() disables. no_grad() retains the forward-only memory
    # behavior needed here without changing tensor semantics.
    with torch.no_grad():
        for _ in range(args.warmup_steps):
            with nvtx_range(f"moe_ladder/{args.variant}/warmup_step"):
                model(tokens, positions=positions, attention_masks=attention_masks)
        torch.cuda.synchronize(device)
        dist.barrier()

        logits = None
        for _ in range(args.steps):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            with nvtx_range(f"moe_ladder/{args.variant}/measure_step"):
                start.record()
                logits = model(
                    tokens, positions=positions, attention_masks=attention_masks
                )
                end.record()
                end.synchronize()
            local_samples_ms.append(start.elapsed_time(end))

        assert logits is not None
        logits = _to_local_logits(logits)
        loss = F.cross_entropy(
            logits.float().reshape(-1, logits.shape[-1]),
            labels.reshape(-1),
            reduction="mean",
        )
        dist.all_reduce(loss, op=dist.ReduceOp.AVG)

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
            "variant": args.variant,
            "world_size": world_size,
            "tp": args.tp,
            "ep": args.ep,
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
            "loss": float(loss.item()),
            **latency,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")

    dist.destroy_process_group()


def _format_table(results: list[dict[str, Any]]) -> str:
    """Format benchmark metrics for stdout.

    Input: per-variant result dictionaries.
    Output: plain-text table.
    """
    by_variant = {item["variant"]: item for item in results}
    stock_loss = by_variant.get("stock", {}).get("loss")
    parallel_loss = by_variant.get("parallel", {}).get("loss")
    stock_latency = by_variant.get("stock", {}).get("median_ms")
    parallel_latency = by_variant.get("parallel", {}).get("median_ms")
    header = (
        "variant          median_ms  p90_ms   tokens/s  vs_stock  vs_parallel  "
        "loss       delta_stock  delta_parallel"
    )
    rows = [header, "-" * len(header)]
    for item in results:
        loss = item["loss"]
        delta_stock = loss - stock_loss if stock_loss is not None else float("nan")
        delta_parallel = (
            loss - parallel_loss if parallel_loss is not None else float("nan")
        )
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
            f"{item['median_ms']:>9.2f} "
            f"{item['p90_ms']:>7.2f} "
            f"{item['tokens_per_second']:>10.1f} "
            f"{versus_stock:>8.3f}x "
            f"{versus_parallel:>11.3f}x "
            f"{loss:>10.6f} "
            f"{delta_stock:>12.6f} "
            f"{delta_parallel:>14.6f}"
        )
    return "\n".join(rows)


def _run_driver(args: argparse.Namespace) -> None:
    """Launch one torchrun job per variant and aggregate the results.

    Input: driver CLI args.
    Output: combined JSON plus stdout summary table.
    """
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
            "torchtitan.experiments.moe_ladder.profile_inference",
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
            "--dtype",
            args.dtype,
            "--attn-backend",
            args.attn_backend,
            "--moe-comm-backend",
            args.moe_comm_backend,
        ]
        print(f"Running {variant}...", flush=True)
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
    """Run the driver or distributed worker.

    Input: process argv.
    Output: benchmark files and stdout metrics.
    """
    args = _parse_args()
    if args.worker:
        _run_worker(args)
    else:
        _run_driver(args)


if __name__ == "__main__":
    main()

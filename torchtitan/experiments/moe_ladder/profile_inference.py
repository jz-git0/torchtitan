# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Forward-only benchmark for MoE ladder schedules on local GPUs.

The default workload keeps 8 * 1024 input tokens in flight per data-parallel
group and reports the slowest rank latency. Optional Nsight Systems NVTX ranges
explain end-to-end differences when `MOE_LADDER_NVTX=1`.
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
from typing import Any

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import torch.nn.functional as F
from torch.distributed.tensor import DTensor

from torchtitan.components.checkpoint import ModelWrapper
from torchtitan.config import CompileConfig, ParallelismConfig, TrainingConfig
from torchtitan.distributed import ParallelDims
from torchtitan.experiments.moe_ladder.models.qwen3 import (
    model_registry as ladder_model_registry,
    MOE_COMM_BACKEND,
)
from torchtitan.experiments.moe_ladder.nvtx import nvtx_range
from torchtitan.experiments.moe_ladder.schedules import SCHEDULE_NAMES
from torchtitan.models.qwen3 import model_registry as qwen3_model_registry
from torchtitan.protocols.model import BaseModel
from torchtitan.protocols.model_spec import ModelSpec
from torchtitan.protocols.state_dict_adapter import BaseStateDictAdapter
from torchtitan.tools.utils import get_peak_flops, set_default_dtype

VARIANTS = ("stock", *SCHEDULE_NAMES)


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
    parser.add_argument(
        "--model-flavor",
        default="debugmodel_moe",
        help=(
            "Qwen3 MoE flavor. The debug flavor is too small to saturate an "
            "H100 and shows no overlap; pair a production flavor with --n-layers."
        ),
    )
    parser.add_argument(
        "--n-layers",
        type=int,
        default=0,
        help=(
            "Keep only the first N decoder layers (0 keeps all). Cuts build "
            "cost while preserving per-layer geometry."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument(
        "--timeout",
        type=int,
        default=900,
        help=(
            "Per-variant subprocess timeout in seconds. Increase it for "
            "full-depth models and direct Hugging Face loading."
        ),
    )
    checkpoint_group = parser.add_mutually_exclusive_group()
    checkpoint_group.add_argument(
        "--checkpoint-path",
        type=Path,
        default=None,
        help=(
            "DCP directory of trained or converted weights. Without a DCP or "
            "--hf-model-path, the run uses seeded random weights, whose loss "
            "is not a quality signal."
        ),
    )
    checkpoint_group.add_argument(
        "--hf-model-path",
        type=Path,
        default=None,
        help=(
            "Hugging Face model directory containing safetensors and config. "
            "Weights are loaded directly through Qwen3's state-dict adapter. "
            "Also supplies the tokenizer unless --tokenizer-path is set."
        ),
    )
    parser.add_argument(
        "--tokenizer-path",
        type=Path,
        default=None,
        help=(
            "Directory holding tokenizer.json, used to encode --data-path into "
            "real tokens. Without it the batch is uniform random token ids, so "
            "loss is only a forward-pass smoke check, not a model-quality "
            "measurement."
        ),
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("tests/assets/c4_test/data.json"),
        help="JSONL with a 'text' field per line; used only with --tokenizer-path.",
    )
    parser.add_argument(
        "--loss-chunk-tokens",
        type=int,
        default=1024,
        help="Maximum tokens materialized in fp32 for cross-entropy.",
    )
    parser.add_argument(
        "--loss-only",
        action="store_true",
        help=(
            "Run one untimed forward per variant and report loss only. Use "
            "with pretrained weights, real text, and --n-layers 0."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--ep", type=int, default=2)
    parser.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    parser.add_argument(
        "--attn-backend",
        choices=("flex", "flex_flash", "varlen"),
        default="flex",
    )
    args = parser.parse_args()
    if args.worker and args.variant is None:
        parser.error("--worker requires --variant")
    if args.hf_model_path is not None and args.tokenizer_path is None:
        args.tokenizer_path = args.hf_model_path
    return args


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
    if not args.loss_only and args.steps < 10:
        raise ValueError("performance measurements require at least 10 steps")
    if args.loss_chunk_tokens <= 0:
        raise ValueError("loss_chunk_tokens must be positive")
    if args.ep > 1 and args.tp > 1 and args.seq_len % args.tp != 0:
        raise ValueError("MoE ladder EP+TP requires seq_len to be divisible by tp")
    if args.ep <= 1:
        raise ValueError("the DeepEP dispatcher needs an EP mesh, so ep must be > 1")
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
    variant: str,
    attn_backend: str,
    flavor: str = "debugmodel_moe",
    n_layers: int = 0,
) -> ModelSpec:
    """Return the stock or ladder Qwen3 MoE ModelSpec.

    Both use DeepEP, holding the dispatcher fixed. The ladder registry supports
    no other backend.

    Input: variant name, attention backend, Qwen3 flavor, and an optional layer
    truncation (0 keeps every layer).
    Output: ModelSpec for the requested flavor.
    """
    if variant == "stock":
        spec = qwen3_model_registry(
            flavor,
            attn_backend=attn_backend,
            moe_comm_backend=MOE_COMM_BACKEND,
        )
    else:
        spec = ladder_model_registry(
            flavor,
            schedule=variant,
            attn_backend=attn_backend,
        )
    if n_layers < 0:
        raise ValueError("n_layers must be non-negative")
    if n_layers:
        if n_layers > len(spec.model.layers):
            raise ValueError(
                f"n_layers ({n_layers}) exceeds the {len(spec.model.layers)} "
                f"layers of flavor {flavor!r}"
            )
        spec.model.layers = spec.model.layers[:n_layers]
    return spec


def _load_model_weights(
    model: torch.nn.Module,
    model_config: BaseModel.Config,
    state_dict_adapter: type[BaseStateDictAdapter] | None,
    *,
    checkpoint_path: Path | None,
    hf_model_path: Path | None,
) -> None:
    """Load DCP or Hugging Face weights into a materialized sharded model.

    Hugging Face weights are read directly; this function does not write DCP.
    """
    if checkpoint_path is None and hf_model_path is None:
        return

    model_wrapper = ModelWrapper(model)
    state_dict = model_wrapper.state_dict()
    if hf_model_path is not None:
        if state_dict_adapter is None:
            raise ValueError("the selected model does not support Hugging Face weights")
        adapter = state_dict_adapter(model_config, str(hf_model_path))
        hf_state_dict = adapter.to_hf(state_dict)
        dcp.load(
            hf_state_dict,
            storage_reader=adapter.get_hf_storage_reader(str(hf_model_path)),
        )
        state_dict = adapter.from_hf(hf_state_dict)
    else:
        assert checkpoint_path is not None
        dcp.load(state_dict, checkpoint_id=str(checkpoint_path))

    # DCP populates flattened tensors; load_state_dict then runs module hooks
    # such as fused-QKV merging and refreshes ModelWrapper's cache.
    model_wrapper.load_state_dict(state_dict)


def _build_model(
    args: argparse.Namespace,
    parallel_dims: ParallelDims,
    parallelism: ParallelismConfig,
    training: TrainingConfig,
):
    """Build and load an EP/TP-sharded inference model."""
    assert args.variant is not None
    device = torch.device("cuda", torch.cuda.current_device())
    spec = _model_spec(
        args.variant,
        args.attn_backend,
        flavor=args.model_flavor,
        n_layers=args.n_layers,
    )
    model_config = spec.model
    compile_config = CompileConfig(enable=False)
    model_config.update_from_config(
        config=SimpleNamespace(
            parallelism=parallelism,
            compile=compile_config,
            activation_checkpoint=None,
        )
    )
    with torch.device("meta"), set_default_dtype(_dtype(args.dtype)):
        model = model_config.build()

    # Count global FLOPs before parallelization changes parameter placements.
    _, flops_per_token = model_config.get_nparams_and_flops(model, args.seq_len)
    model = spec.parallelize_fn(
        model,
        parallel_dims=parallel_dims,
        training=training,
        parallelism=parallelism,
        compile_config=compile_config,
        ac_config=None,
        dump_folder="/tmp",
        skip_dp=True,
    )
    model.to_empty(device=device)
    torch.manual_seed(args.seed)
    with torch.no_grad():
        model.init_states()
    _load_model_weights(
        model,
        model_config,
        spec.state_dict_adapter,
        checkpoint_path=args.checkpoint_path,
        hf_model_path=args.hf_model_path,
    )
    model.eval()
    return model, flops_per_token, model_config.vocab_size


def _corpus_tokens(
    tokenizer_path: Path, data_path: Path, num_tokens: int, skip: int
) -> torch.Tensor:
    """Encode a JSONL text corpus into one flat token stream.

    Documents are concatenated with EOS between them, and the stream stops at
    num_tokens. Every rank tokenizes the same corpus and differs only by its
    skip offset. Input: tokenizer directory, JSONL path, requested token count,
    and number of leading tokens to skip.
    Output: [num_tokens] int64 tensor on CPU.
    """
    from torchtitan.components.tokenizer import HuggingFaceTokenizer

    tokenizer = HuggingFaceTokenizer(tokenizer_path=str(tokenizer_path))
    ids: list[int] = []
    needed = skip + num_tokens
    with open(data_path, encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            ids.extend(tokenizer.encode(json.loads(line)["text"], add_eos=True))
            if len(ids) >= needed:
                break
    if len(ids) < needed:
        raise ValueError(
            f"{data_path} yields {len(ids)} tokens but {needed} are required; "
            f"reduce --batch-size/--seq-len or supply a larger corpus"
        )
    return torch.tensor(ids[skip:needed], dtype=torch.long)


def _make_batch(
    args: argparse.Namespace, rank: int, vocab_size: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create deterministic next-token inputs for one data-parallel shard.

    With --tokenizer-path, the batch is real text, making the reported loss
    comparable across schedules. Otherwise, it is uniform random token ids; any model scores about
    ln(vocab_size) on that input.

    Input: worker args and distributed rank.
    Output: tokens [B, L], labels [B, L], and positions [B, L] on CUDA.
    """
    data_rank = rank // args.tp
    generator = torch.Generator(device="cpu")
    generator.manual_seed(args.seed + 1009 * data_rank)
    device = torch.device("cuda", torch.cuda.current_device())
    num_tokens = args.batch_size * (args.seq_len + 1)
    if args.tokenizer_path is not None:
        full = _corpus_tokens(
            args.tokenizer_path,
            args.data_path,
            num_tokens,
            skip=data_rank * num_tokens,
        ).view(args.batch_size, args.seq_len + 1)
        full = full.to(device)
    else:
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


def _chunked_cross_entropy(
    logits_BLV: torch.Tensor,
    labels_BL: torch.Tensor,
    chunk_tokens: int,
) -> torch.Tensor:
    """Compute mean cross-entropy without casting the full logits tensor."""
    logits_TV = logits_BLV.reshape(-1, logits_BLV.shape[-1])
    labels_T = labels_BL.reshape(-1)
    loss_sum = logits_TV.new_zeros((), dtype=torch.float32)
    for start in range(0, labels_T.numel(), chunk_tokens):
        end = min(start + chunk_tokens, labels_T.numel())
        loss_sum += F.cross_entropy(
            logits_TV[start:end].float(),
            labels_T[start:end],
            reduction="sum",
        )
    return loss_sum / labels_T.numel()


def _throughput_metrics(
    flops_per_token: int,
    tokens_per_step: int,
    median_ms: float,
    world_size: int,
) -> dict[str, float]:
    """Derive achieved TFLOP/s and MFU from a measured step latency.

    The shared FLOP helper counts forward and backward with a factor of 6.
    These profilers run forward only, so a third of that is the forward cost.

    Input: FLOPs per token, global tokens per step, median latency, and world size.
    Output: forward FLOPs per token, achieved TFLOP/s, and MFU percent.
    """
    forward_flops_per_token = flops_per_token / 3.0
    achieved = forward_flops_per_token * tokens_per_step / (median_ms / 1e3)
    peak = get_peak_flops(torch.cuda.get_device_name())
    return {
        "forward_flops_per_token": forward_flops_per_token,
        "achieved_tflops": achieved / 1e12,
        "peak_tflops_per_gpu": peak / 1e12,
        "mfu_pct": 100.0 * achieved / (world_size * peak),
    }


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
    model, flops_per_token, vocab_size = _build_model(
        args, parallel_dims, parallelism, training
    )
    tokens, labels, positions = _make_batch(args, rank, vocab_size)
    attention_masks = model.get_attention_masks(positions=positions)
    local_samples_ms: list[float] = []

    # DTensor-backed RoPE cache views currently require version counters.
    # torch.inference_mode() disables them; no_grad() retains the forward-only
    # memory behavior needed here without changing tensor semantics.
    with torch.no_grad():
        if args.loss_only:
            with nvtx_range(f"moe_ladder/{args.variant}/loss_step"):
                logits = model(
                    tokens, positions=positions, attention_masks=attention_masks
                )
            torch.cuda.synchronize(device)
        else:
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
                        tokens,
                        positions=positions,
                        attention_masks=attention_masks,
                    )
                    end.record()
                    end.synchronize()
                local_samples_ms.append(start.elapsed_time(end))

        assert logits is not None
        logits = _to_local_logits(logits)
        loss = _chunked_cross_entropy(
            logits,
            labels,
            args.loss_chunk_tokens,
        )
        dist.all_reduce(loss, op=dist.ReduceOp.AVG)

    samples_ms: list[float] = []
    if not args.loss_only:
        rank_samples: list[list[float] | None] = [None] * world_size
        dist.all_gather_object(rank_samples, local_samples_ms)
        if any(
            samples is None or len(samples) != args.steps for samples in rank_samples
        ):
            raise RuntimeError("ranks produced inconsistent latency sample counts")
        samples_ms = [
            max(samples[step] for samples in rank_samples if samples is not None)
            for step in range(args.steps)
        ]

    if rank == 0:
        num_data_groups = world_size // args.tp
        tokens_per_step = args.batch_size * args.seq_len * num_data_groups
        result: dict[str, Any] = {
            "variant": args.variant,
            "mode": "loss" if args.loss_only else "profile",
            "world_size": world_size,
            "tp": args.tp,
            "ep": args.ep,
            "batch_size": args.batch_size,
            "model_flavor": args.model_flavor,
            "n_layers": args.n_layers,
            "seq_len": args.seq_len,
            "local_tokens_per_rank": args.batch_size * args.seq_len // args.tp,
            "dtype": args.dtype,
            "attn_backend": args.attn_backend,
            "moe_comm_backend": MOE_COMM_BACKEND,
            "weights": (
                "huggingface"
                if args.hf_model_path is not None
                else "dcp"
                if args.checkpoint_path is not None
                else "random"
            ),
            "trained_weights": (
                args.checkpoint_path is not None or args.hf_model_path is not None
            ),
            "real_tokens": args.tokenizer_path is not None,
            "tokens_per_step": tokens_per_step,
            "loss_comparable_across_variants": (
                (args.checkpoint_path is not None or args.hf_model_path is not None)
                and args.tokenizer_path is not None
                and args.n_layers == 0
            ),
            "loss": float(loss.item()),
        }
        if not args.loss_only:
            latency = _summarize_ms(samples_ms)
            result.update(
                {
                    "warmup_steps": args.warmup_steps,
                    "measured_steps": args.steps,
                    "tokens_per_second": tokens_per_step
                    / (latency["median_ms"] / 1000.0),
                    "total_measured_s": sum(samples_ms) / 1e3,
                    **_throughput_metrics(
                        flops_per_token,
                        tokens_per_step,
                        latency["median_ms"],
                        world_size,
                    ),
                    **latency,
                }
            )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")

    dist.destroy_process_group()


def _loss_deltas(
    loss: float, stock_loss: float | None, parallel_loss: float | None
) -> tuple[float, float]:
    """Return loss deltas from the stock and parallel baselines."""
    return (
        loss - stock_loss if stock_loss is not None else float("nan"),
        loss - parallel_loss if parallel_loss is not None else float("nan"),
    )


def _format_table(results: list[dict[str, Any]]) -> str:
    """Format benchmark metrics for stdout.

    Input: per-variant result dictionaries.
    Output: plain-text table.
    """
    by_variant = {item["variant"]: item for item in results}
    stock_loss = by_variant.get("stock", {}).get("loss")
    parallel_loss = by_variant.get("parallel", {}).get("loss")
    if results and all(item.get("mode") == "loss" for item in results):
        header = "variant          loss       delta_stock  delta_parallel"
        rows = [header, "-" * len(header)]
        for item in results:
            loss = item["loss"]
            delta_stock, delta_parallel = _loss_deltas(loss, stock_loss, parallel_loss)
            rows.append(
                f"{item['variant']:<16} {loss:>10.6f} "
                f"{delta_stock:>12.6f} {delta_parallel:>14.6f}"
            )
        if not all(item["loss_comparable_across_variants"] for item in results):
            rows.append(
                "\nLoss comparison requires pretrained weights, real tokens, "
                "and the full model (--n-layers 0)."
            )
        return "\n".join(rows)

    stock_latency = by_variant.get("stock", {}).get("median_ms")
    parallel_latency = by_variant.get("parallel", {}).get("median_ms")
    header = (
        "variant          median_ms  p90_ms   tokens/s  mfu%   total_s  "
        "vs_stock  vs_parallel  loss       delta_stock  delta_parallel"
    )
    rows = [header, "-" * len(header)]
    for item in results:
        loss = item["loss"]
        delta_stock, delta_parallel = _loss_deltas(loss, stock_loss, parallel_loss)
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
            f"{item['mfu_pct']:>5.1f} "
            f"{item['total_measured_s']:>8.2f} "
            f"{versus_stock:>8.3f}x "
            f"{versus_parallel:>11.3f}x "
            f"{loss:>10.6f} "
            f"{delta_stock:>12.6f} "
            f"{delta_parallel:>14.6f}"
        )
    return "\n".join(rows)


_DRIVER_ONLY_ARGS = {
    "worker",
    "variant",
    "variants",
    "output",
    "nproc_per_node",
    "timeout",
}


def _worker_cli_args(args: argparse.Namespace, variant: str, output: Path) -> list[str]:
    """Build worker arguments from the parsed driver namespace."""
    command = ["--worker", "--variant", variant, "--output", str(output)]
    for name, value in vars(args).items():
        if name in _DRIVER_ONLY_ARGS or value is None or value is False:
            continue
        option = f"--{name.replace('_', '-')}"
        command.extend([option] if value is True else [option, str(value)])
    return command


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
            *_worker_cli_args(args, variant, variant_output),
        ]
        print(f"Running {variant}...", flush=True)
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=args.timeout)
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

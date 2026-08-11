# MoE Ladder

MoE Ladder compares four experimental Qwen3 MoE schedules with the stock block. Qwen3, [DeepEP v2](https://github.com/deepseek-ai/DeepEP), and expert parallelism greater than one are required.

## Compute and communication overlap

By default, ProcessGroupNCCL enqueues communication on the current CUDA stream. Splitting communication into phases therefore does not make it overlap compute on that stream.

DeepEP v2 launches token exchanges on an internal communication stream, allowing dispatch and combine communication to overlap compute on the default CUDA stream. MoE Ladder exposes the following MoE phases:

1. FFN normalization and gate computation produce top-k scores and expert IDs; no tokens move yet.
2. DeepEP dispatch applies those decisions and exchanges tokens.
3. Local experts run grouped GEMMs.
4. Combine preparation applies gate scores and restores local token order, then launches the inverse DeepEP exchange.
5. Combine completion waits for that exchange and restores the residual layout.

These boundaries let DeepEP dispatch/combine communication and local expert compute overlap attention instead of executing as one serial MoE call.

Attention runs on the current stream. Dispatch, expert compute, and both combine phases run on the MoE side stream so that DeepEP compute kernels do not queue behind attention or gating. Token exchanges remain on DeepEP's internal communication stream. Gate placement varies by schedule as detailed below.

Delayed schedules currently use conservative end-of-block stream joins. This limits overlap to each block; finer event-based synchronization could enable cross-block pipelining.

## Schedules

- `stock`: attention completes before the MoE starts; there is no attention/MoE overlap.
- `parallel`: attention and MoE read the same pre-attention residual. Gate computation runs on the current stream; dispatch, experts, and combine run on the MoE side stream and can overlap attention.
- `ladder`: the previous combine is launched on the MoE side stream while current attention runs on the pre-combine residual. The side stream then completes that combine and runs the current FFN normalization, gate, dispatch, and experts.
- `hoisted_gateA`: launches the previous combine before computing the gate on the pre-combine residual. DeepEP combine communication can overlap the gate and attention; dispatch later applies the stale gate decisions to post-combine activations.
- `hoisted_gateB`: computes the gate on the pre-combine residual before launching the previous combine. DeepEP combine communication overlaps attention, but not the gate; dispatch later applies the stale gate decisions to post-combine activations.

Both hoisted schedules run FFN normalization once for stale gating and again for the post-combine expert input.

All four experimental schedules change the stock architecture; the hoisted schedules also use stale gate decisions. Checkpoint keys and shapes are unchanged, but loss and convergence may change.

## Implementation

### Phase interface

`ladder_moe.py` splits `MoE.forward()` into the following calls:

- `prepare_input(x_BLD)`: applies the input placement expected by the MoE.
- `route(x_BLD)`: computes top-k scores and expert IDs without moving tokens.
- `dispatch(x_BLD, route_info)`: exchanges tokens through DeepEP and returns local expert inputs, token counts, and `DispatchState`.
- `experts_forward(routed_RD, counts_e)`: runs the local grouped expert MLPs.
- `begin_combine(routed_out_RD, state)`: starts the asynchronous inverse exchange.
- `finish_combine(handle, state)`: waits for that exchange and returns flattened token order. `finish_combine_bld()` also restores the residual shape and placements; `combine()` and `combine_bld()` are their synchronous forms.

`Pending` stores an expert result with its `DispatchState` so a delayed schedule can combine it in a later block.

Calling phases directly is what makes scheduling possible, but it bypasses the DTensor boundary normally applied around `MoE.forward()`. `_PhaseInput` therefore keeps both the placement-aware input and its local tensor. `_to_local()` and `_prepare_phase_input()` provide the local, dynamically sized tensors required by DeepEP; `_output_specs()`, `_wrap_output_src()`, and `_restore_output()` reconstruct the expected output placements after a delayed combine. `DispatchState` retains the corresponding local shape and placement metadata. These helpers reproduce the module-boundary semantics for the phase API; they do not define a second MoE path.

### Files

| File | Purpose |
| --- | --- |
| `ladder_moe.py` | Implements the phase-split MoE, DeepEP dispatch/combine, and explicit DTensor/local phase boundaries. |
| `schedule_runner.py` | Exposes `dispatch_experts()` and the `parallel_step()`, `ladder_step()`, `hoisted_gate_a_step()`, and `hoisted_gate_b_step()` stream/event schedules. |
| `models/qwen3.py` | Adapts Qwen3 blocks and the model loop to each schedule, including carrying and draining `Pending` state. |
| `models/common.py` | Builds `LadderMoE` and provides the shared delayed-combine drain helper. |
| `config_registry.py` | Registers the small validation config and Qwen3-30B-A3B schedule configs. |
| `schedules.py` | Defines schedule names and validates schedule selection. |
| `profile_inference.py` | Runs distributed forward/loss profiling with random, DCP, or Hugging Face weights. |
| `nvtx.py` | Adds NVTX ranges when `MOE_LADDER_NVTX=1`, as set by the Nsight script. |
| `jzh100_profile.slurm` | Launches Jean Zay H100 profiling runs and trace collection. |
| `jzh100_loss.slurm` | Compares full-model checkpoint losses on Jean Zay. |
| `analyze_overlap.py` | Computes overlap and idle-time summaries from `.nsys-rep` files using temporary, filtered SQLite exports. |
| `deepep_install.md` | Gives the supported DeepEP v2 installation procedure. |
| `deepep_v2.1_portable_build.patch` | Adjusts DeepEP v2.1 for the documented portable, single-node build. |
| `models/__init__.py` | Exposes the Qwen3 model adapters. |
| `README.md` | Documents the experiment, schedules, profiling workflow, and limitations. |
| `__init__.py` | Marks the experiment package. |

## Installation

Install TorchTitan, then build DeepEP v2 using [deepep_install.md](deepep_install.md). The supplied build targets one Hopper NVLink node and disables NVSHMEM; it cannot run DeepEP kernels on Ampere GPUs.

For this single-node build:

```bash
export EP_DISABLE_GIN=1
```

## Profile random weights

For a random-weight run, launch:

```bash
./torchtitan/experiments/moe_ladder/jzh100_profile.slurm
```

Defaults are 4 GPUs, 8 layers, batch size 16, and sequence length 2048. Set `GPUS` to change the allocation and `N_LAYERS` to change the number of layers (`0` keeps all 48). See the script header for examples. Random runs use deterministic token IDs and no tokenizer; layer subsets are for profiling only.

## Checkpoints

The profiler accepts either:

- `--checkpoint-path path/to/dcp`: a TorchTitan DCP checkpoint directory.
- `--hf-model-path path/to/hf-model`: local Hugging Face safetensors.

Paths may be relative to the checkout root. A DCP checkpoint does not contain tokenizer assets. Each Hugging Face model directory contains its tokenizer, so for a snapshot under `hf/assets/Qwen3-30B-A3B-Base`, use that directory as `TOKENIZER`. The HF path is loaded directly into the sharded model and does not create a DCP checkpoint. With neither option, the profiler uses seeded random weights.

The model is built on `meta`, sharded, materialized rank-locally, and then loaded. A full unsharded model is not instantiated on every GPU.

## Compare losses

Use full depth, identical weights, and the same token stream. With a DCP checkpoint directory available:

The script requests 4 GPUs by default (`EP=4`, `TP=1`) and evaluates all 48 layers with batch size 1 and sequence length 512.

```bash
CKPT=path/to/dcp TOKENIZER=path/to/Qwen3-30B-A3B-Base \
DATA=path/to/evaluation.jsonl \
RUN_NAME=qwen3_30b_checkpoint_losses \
./torchtitan/experiments/moe_ladder/jzh100_loss.slurm
```

The script writes `outputs/jz_h100_losses/<RUN_NAME>/losses.json`. For direct Hugging Face loading, replace `CKPT` and `TOKENIZER` with `HF_MODEL=path/to/Qwen3-30B-A3B-Base`; that directory supplies both weights and tokenizer. The JSONL input must contain a `text` field. Reported deltas measure the immediate architectural change at fixed weights.

## Retraining

Retraining has not been tested; the command below is an indicative TorchTitan setup.

Use separate output directories and identical data, seeds, optimizer settings, and parallelism. To initialize from an existing DCP:

```bash
EP_DISABLE_GIN=1 NGPU=4 MODULE=moe_ladder \
CONFIG=qwen3_30b_a3b_ladder ./run_train.sh \
  --dump_folder outputs/qwen3_30b_ladder \
  --hf_assets_path /absolute/path/to/hf-assets \
  --checkpoint.enable \
  --checkpoint.initial_load_path /absolute/path/to/dcp \
  --training.dtype bfloat16 \
  --training.local_batch_size 1 \
  --training.seq_len 1024
```

If only HF safetensors are available, set `--hf_assets_path /absolute/path/to/hf-model --checkpoint.initial_load_in_hf` and omit `--checkpoint.initial_load_path`. The initial load is direct; subsequent checkpoints written by the trainer are DCP.

Experiment configs disable activation checkpointing. Delayed schedules also reject pipeline parallelism because DeepEP state crosses block boundaries.

## Nsight Systems traces

On Jean Zay, run from the checkout root:

```bash
./torchtitan/experiments/moe_ladder/jzh100_profile.slurm
```

The script submits to `${IDRPROJ}@h100`, measures latency and MFU without NVTX instrumentation, and records one complete `.nsys-rep` per variant with NVTX enabled. Each trace contains CUDA API calls, GPU kernels, memory operations, stream IDs, GPU metrics, and schedule/step NVTX ranges.

`analyze_overlap.py` analyzes every complete `measure_step` range, including all layers, while excluding gaps between steps. Its temporary, filtered SQLite exports are removed after analysis.

## Memory

Full-model inference should fit within one 80 GB H100, but MoE Ladder requires at least two GPUs for expert parallelism. Retraining will likely require multiple GPUs; the supplied DeepEP build cannot scale beyond one node.

## Tests

Block tests exercise the router and schedule phases with local token permutation and lightweight attention, norm, and expert substitutes. They run on CPU and optionally one CUDA GPU; Hopper is not required.

```bash
CUDA_VISIBLE_DEVICES=0 pytest \
  tests/unit_tests/test_moe_ladder_blocks.py \
  tests/unit_tests/test_moe_ladder_moe.py \
  tests/unit_tests/test_moe_ladder_models.py \
  tests/unit_tests/test_moe_ladder_analysis.py
```

DeepEP integration and performance tests require Hopper GPUs.

## Limitations

- Shared experts and group-limited routing are not validated.
- EP+TP requires sequence length to be divisible by TP; phase-split sequence padding is not implemented.
- The profiler is forward-only and disables activation checkpointing, pipeline parallelism, and FSDP.
- The supplied DeepEP build is single-node only.
- Compact DeepEP dispatch host-synchronizes; the expanding inference path is not enabled.
- Model compilation is not supported.

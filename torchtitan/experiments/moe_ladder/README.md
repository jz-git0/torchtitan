# MoE Ladder

An experiment in hiding expert-parallel (EP) communication behind attention.
The common MoE forward is split into explicit phases -- route, counts
all-to-all, host split sync, token dispatch, expert GEMM, combine -- and a
decoder block reorders those phases around attention. Four schedules are
compared against stock Qwen3, over two communication backends: the standard
all-to-all dispatcher and DeepEP. Qwen3 is the only model wired up;
Llama3/MoLE and DeepSeek-V3 are scaffolds.

## Schedules

`schedule_runner.py` implements each schedule once; a model block
(`models/qwen3.py`) selects one through a class-level `step_fn`.

The token dispatch and combine all-to-alls move every routed token and are
the large collectives. The counts all-to-all is a tiny `[E]`-sized exchange,
and the host split sync (`sync_counts`) is a CPU-blocking device-to-host copy
needed before the variable-sized token dispatch can launch. The schedules
differ in which of these they try to hide under attention:

- `parallel`: attention and MoE read the same residual,
  `res_{t+1} = res_t + Attn(res_t) + MoE(res_t)`, so the MoE is independent
  of the attention output. No stale routing, no cross-block pending state.
  Under EP it syncs splits, launches the token dispatch, runs attention, then
  waits the dispatch; the combine stays exposed because one attention can
  hide only one large collective.
- `ladder`: the expert output is carried one block forward. The previous
  block's combine is launched, attention runs over it, then the combine is
  waited. Routing stays fresh (it reads the post-combine residual), which
  means the counts sync and token dispatch of the current block can only
  start after attention.
- `hoisted_gateA` / `hoisted_gateB`: routing is hoisted onto the stale
  residual `res_t`, so the previous block's combine can be launched before
  (A) or after (B) the routing gate. The difference that matters is when the
  host split sync resolves. A issues counts after the combine, so the sync
  queues behind the large collective and resolves late. B issues counts
  first: the tiny exchange completes immediately, the splits are on the host
  before attention starts, and once the combine finishes only the residual
  add, norm, and token gather separate it from the next dispatch launch.

This property is specific to B and motivates it: B is the only schedule
where the *layout* of the next dispatch (splits and routing metadata) is
fully known before the combine *data* arrives. `ladder` cannot reach this
state in principle -- its routing reads the post-combine residual, so the
layout depends on the combine -- and A forfeits it by ordering counts behind
the combine. B is therefore the candidate schedule for a fused
combine-recv -> dispatch-send kernel that pipelines the two exchanges at
chunk granularity (`backends/comet_fused.py` sketches the chunked MoE body).

Both hoisted schedules use stale routing, which changes the model. See
"Correctness vs convergence" below.

## Why there is no overlap yet (standard backend)

With the standard dispatcher every collective goes through
`dist.all_to_all(...)`, and PyTorch launches synchronous-style collectives on
the caller's current stream. Compute and NCCL kernels therefore share the
single default stream. A stream executes strictly in order, so an
"overlapped" collective simply queues in front of the compute meant to hide
it. Reordering phases changes where the queue drains, never whether
communication and computation run concurrently.

The phase split adds cost of its own. Each phase adds Python and launch
overhead (roughly 50 us of CPU per op), and the reordered schedules run
their small routing kernels right after a host sync, against an empty GPU
queue, where every CPU pause becomes GPU idle. Measured on 2x A5000 (EP=2,
one Qwen3 block, bf16): all variants execute the same ~9.5 ms of GPU work
per step and differ only in idle time -- about 0.6 ms for stock versus
1.2-2.7 ms for the reordered schedules. Under this backend stock is
therefore the fastest variant; this is expected, not a defect.

Overlap with plain NCCL requires launching the collectives on a separate
stream (`async_op=True` work objects, or an explicit side stream). This
comes with known precautions: events must order both directions
(producer -> collective, collective -> consumer), buffer lifetimes must be
extended across streams so the caching allocator does not recycle memory
mid-transfer, and the backward pass needs the same treatment. Instead of
rebuilding that machinery, the experiment uses DeepEP, which manages its own
communication stream.

## DeepEP backend

Both profilers accept `--moe-comm-backend {standard,deepep}` (default
`standard`). With `deepep`, every variant (including `stock`) swaps the
all-to-all dispatcher for `DeepEPTokenDispatcher`:

- There is no counts exchange and no host split sync: DeepEP computes its
  dispatch layout internally, so the schedules skip the `begin_counts` /
  `sync_counts` phases and use atomic dispatch. The counts-ordering trick
  that originally distinguished `hoisted_gateA` from `hoisted_gateB`
  therefore disappears. What still separates them: gateA launches the
  pending combine before the stale route (combine overlaps routing and
  attention), gateB launches it after (combine overlaps attention only),
  and both route on the stale pre-combine residual, unlike `ladder`'s
  fresh routing.
- Combine launches asynchronously on DeepEP's own stream:
  `begin_combine` starts the exchange, `finish_combine` waits the pending
  event. In `ladder` and both hoisted schedules the combine of block N-1
  therefore genuinely overlaps attention of block N -- unlike the standard
  backend. Only one DeepEP combine may be in flight per process (the
  deferred-sync event is process-global); the schedules respect this, and a
  second begin_combine before finish_combine raises.
- Dispatch is not phase-split under DeepEP, so `parallel` cannot overlap
  dispatch with attention. Instead it splits combine around attention
  within the block (begin before, finish after), which is legal because
  attention and the MoE read the same residual; dispatch stays exposed.

Requirements: an sm_90 GPU (H100/H800; the DeepEP v2 runtime JIT does not
support Ampere), `deep_ep >= 2.1` (ElasticBuffer API), NCCL >= 2.30 headers
and library, and `CUDA_HOME` pointing at a toolkit with `nvcc >= 12.3` so the
runtime JIT can compile kernels. On single-node runs without GPU-initiated
RDMA networking, set `EP_DISABLE_GIN=1`.
`jean_zay_profile.slurm` runs the full sweep (both backends plus per-variant
nsys captures) on a Jean Zay H100 node.

### Installing DeepEP on the Jean Zay login node (intranode-only)

The login node has internet access but no GPU; that is fine, because the
ahead-of-time build takes the target architecture explicitly and the kernel
JIT runs later on the compute node. The build uses the CUDA toolkit and NCCL
wheels already bundled inside the venv, so no CUDA module has to match.

`deepep_v2.1_portable_build.patch` is applied to the DeepEP source before
compiling and is not used anywhere else. It adds a `DISABLE_NVSHMEM=1` build
option (intranode-only, no rdma-core headers needed), fixes linking against
pip-wheel NCCL, and bypasses a compiler/header minor-version check inside
the bundled toolkit. Its remaining hunks only affect pre-Hopper paths and
are inert on H100.

```bash
REPO=$WORK/code/lqaif/torchtitan      # adjust to the checkout location
VENV=$REPO/titan-rl
NV=$VENV/lib/python3.12/site-packages/nvidia

# 1) DeepEP v2.1 needs NCCL >= 2.30 (GIN device API in the headers).
VIRTUAL_ENV=$VENV uv pip install "nvidia-nccl-cu13==2.30.7"

# 2) The cu13 wheel ships libcudart.so.13 without the unversioned symlink;
#    without it, -lcudart silently links a system CUDA 12.x runtime against
#    13.x headers, which corrupts cudaDeviceProp reads at runtime.
ln -sf libcudart.so.13 $NV/cu13/lib/libcudart.so

# 3) Fetch the source and apply the patch (based on upstream commit dd758ca).
git clone https://github.com/deepseek-ai/DeepEP.git $WORK/deepep-src
cd $WORK/deepep-src
git checkout dd758ca
git apply $REPO/torchtitan/experiments/moe_ladder/deepep_v2.1_portable_build.patch

# 4) Locate the CUDA driver stub. The extension links -lcuda (the driver
#    library), but a GPU-less login node has no NVIDIA driver, so libcuda.so
#    exists nowhere in the system paths. Every full CUDA toolkit ships a
#    link-time stub for this; borrow it from any CUDA module. The stub is
#    only used by the linker -- on compute nodes the real driver provides
#    libcuda.so.1 at runtime.
module load arch/h100 cuda   # any recent toolkit version works
STUBS=$(dirname $(which nvcc))/../lib64/stubs
ls $STUBS/libcuda.so         # must exist

# 5) Build. 9.0 = H100. Do NOT set DISABLE_SM90_FEATURES on H100.
env VIRTUAL_ENV=$VENV CUDA_HOME=$NV/cu13 PATH=$NV/cu13/bin:$PATH \
    LIBRARY_PATH=$STUBS \
    TORCH_CUDA_ARCH_LIST=9.0 DISABLE_NVSHMEM=1 \
    NCCL_DIR=$NV/nccl NVSHMEM_DIR=$NV/nvshmem MAX_JOBS=16 \
    uv pip install --no-build-isolation .

# 6) Verify. On the login node the import needs the stub at load time too
#    (there is no driver to provide libcuda.so.1); do NOT carry this
#    LD_LIBRARY_PATH into real runs on compute nodes.
LD_LIBRARY_PATH=$STUBS $VENV/bin/python \
  -c "import deep_ep; from deep_ep import ElasticBuffer; print('OK', deep_ep.__version__)"
```

If `uv` is missing, install it with
`curl -LsSf https://astral.sh/uv/install.sh | sh`. If the build stops on a
missing `Python.h`, the venv's base interpreter lacks dev headers; point
`CPATH` at a matching `include/python3.12` directory (uv-managed interpreters
ship one). Multi-node runs would need the full upstream NVSHMEM build (no
`DISABLE_NVSHMEM`) plus rdma-core headers; single-node profiling does not.

```bash
EP_DISABLE_GIN=1 python -m torchtitan.experiments.moe_ladder.profile_block \
  --nproc-per-node 2 --tp 1 --ep 2 --batch-size 8 --seq-len 1024 \
  --warmup-steps 20 --steps 50 --moe-comm-backend deepep
```

In an nsys report, DeepEP's dispatch/combine kernels appear on their own
stream; per-device comm/compute overlap during the `*/attention` ranges is
the signal that a schedule is working.

## Layout

```text
ladder_moe.py        LadderMoE: common MoE split into reorderable phases
schedule_runner.py   the four schedules as functions over those phases
schedules.py         schedule names and validation
config_registry.py   Qwen3 debug-model training configs (one per schedule)
models/qwen3.py      executable Qwen3 blocks, model, and registry
models/common.py     build/drain helpers
models/llama3_moe.py, models/deepseek_v3.py   scaffolds (NotImplementedError)
backends/comet_fused.py   chunk-pipelined fused-MoE sketch (NotImplementedError)
profile_inference.py end-to-end latency/throughput comparison
profile_block.py     single-Qwen3-block schedule microbenchmark
nvtx.py              no-op-safe NVTX range helper for Nsight Systems
jean_zay_profile.slurm          H100 profiling job (both backends + nsys)
deepep_v2.1_portable_build.patch  optional-NVSHMEM DeepEP build patch
```

The phase methods build on the dispatchers in
`torchtitan.models.common.token_dispatcher`: `AllToAllTokenDispatcher`
exposes begin/finish count exchange, token dispatch, and token combine;
`DeepEPTokenDispatcher` exposes begin/finish token combine over DeepEP's
async combine. `LadderMoE` owns the phase boundaries, the schedules own
ordering, and the dispatchers own token-layout mechanics.

## Run

```bash
./run_train.sh --module moe_ladder --config qwen3_debugmodel_moe_parallel
./run_train.sh --module moe_ladder --config qwen3_debugmodel_moe_ladder
./run_train.sh --module moe_ladder --config qwen3_debugmodel_moe_hoisted_gateA
./run_train.sh --module moe_ladder --config qwen3_debugmodel_moe_hoisted_gateB
```

Configs default to `expert_parallel_degree=2` and `training.steps=10`. Add TP
with an override such as `--parallelism.tensor_parallel_degree 2`.

## Test

```bash
./titan-rl/bin/python -m pytest \
  tests/unit_tests/test_moe_ladder_moe.py \
  tests/unit_tests/test_moe_ladder_models.py -q
```

`test_moe_ladder_moe.py` checks the phase split against a dense reference and
common-MoE TP/SP forward/backward parity; `test_moe_ladder_models.py` checks
the Qwen3 schedule blocks forward/backward.

## Profiling

The end-to-end benchmark compares stock Qwen3 with all four schedules under
identical weights and inputs:

```bash
./titan-rl/bin/python -m torchtitan.experiments.moe_ladder.profile_inference \
  --nproc-per-node 2 --tp 2 --ep 2 --batch-size 8 --seq-len 1024
```

`profile_block.py` takes the same flags and isolates one Qwen3 block. It
still builds and parallelizes the full model first, so EP/TP wiring stays
identical to the end-to-end path. Both report slowest-rank median/p90
latency and throughput.

For stream-level answers, use Nsight Systems. The schedules emit NVTX ranges
such as `moe_ladder/hoisted_gateB/sync_counts` and
`moe_ladder/ladder/attention`; the profilers wrap every step in
`warmup_step` / `measure_step` ranges under `moe_ladder/...` (full model) or
`moe_ladder_block/...` (single block):

```bash
nsys profile \
  --force-overwrite=true \
  --sample=none \
  --cuda-event-trace=false \
  --trace=cuda,nvtx,osrt,cublas,cudnn \
  -o outputs/nsys_moe_ladder_block_hoisted_gateB \
  ./titan-rl/bin/python -m torch.distributed.run --standalone --nproc-per-node=2 \
    -m torchtitan.experiments.moe_ladder.profile_block \
    --worker --variant hoisted_gateB --ep 2 --tp 1 \
    --batch-size 8 --seq-len 1024 --warmup-steps 20 --steps 50
```

## Correctness vs convergence

These are two different questions and they are validated differently.

Implementation correctness: one schedule must compute the same function no
matter how it is parallelized. Compare a schedule at EP>1 (and EP+TP)
against the same schedule at EP=1/TP=1, with the same seed and data order
and `--debug.seed=42 --debug.deterministic`. Expect tolerance-level
agreement, not bit-for-bit equality: changing the parallelism changes
reduction orders (bitwise reproducibility only holds between runs whose
parallelism is itself identical). The same applies across backends within
one schedule: DeepEP applies routing scores in fp32 before its combine
reduction, so standard-vs-deepep agreement is approximate by construction.

Model quality: `parallel`, `ladder`, and the hoisted schedules are
architecture changes -- a parallel attention/MoE block, a one-block-delayed
expert output, stale routing. They have no reason to reproduce stock's loss
or gradients, and small per-step deltas prove nothing in either direction.
The only meaningful comparison is convergence: train each variant and stock
on a representative dataset (e.g. C4) and compare loss curves. The `loss`
printed by `profile_inference.py` is a smoke check that weights loaded and
the forward is sane, not a quality metric.

## Support and limitations

- Qwen3 is the only model wired up; Llama3/MoLE and DeepSeek-V3 are
  scaffolds that raise `NotImplementedError`.
- EP runs through the standard all-to-all dispatcher or DeepEP. HybridEP and
  MinimalAsyncEP exist in core but are not validated with the phase-split
  schedules. The fused chunk-pipelined backend is a sketch
  (`backends/comet_fused.py`).
- TP/SP works at the `LadderMoE` boundary through the common MoE sharding
  contract. The atomic `LadderMoE.forward` path has 2-rank TP/SP forward and
  backward parity against common MoE, and the benchmark exercises all
  full-model schedules with TP+EP. Full-model distributed backward and EP+TP
  gradient parity are not covered yet.
- Under TP, the attention output reduce-scatter is issued at the `wo`
  boundary on the compute stream, so it serializes with the MoE tail even
  though nothing reads the attention output until the end-of-block residual
  add. Overlapping it (side stream or async-TP) is future work.
- EP+TP requires the global sequence length to be divisible by TP and at
  least TP. This is a phase-split limitation: common MoE pads uneven or
  short sequences, but that padding is not threaded through the separate
  route/dispatch/combine calls yet.
- Shared experts are rejected; group-limited routing is unvalidated.
- Qwen3 state-dict conversion is not provided.

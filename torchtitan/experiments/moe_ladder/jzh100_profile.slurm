#!/bin/bash
#SBATCH --job-name=moe_ladder_overlap
#SBATCH --constraint=h100
#SBATCH --qos=qos_gpu_h100-dev
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --hint=nomultithread
#SBATCH --time=02:00:00
#SBATCH --output=outputs/slurm/moe_ladder_overlap.%j.out
#
# Profile all schedules on one Jean Zay H100 node.
# Run from the checkout root:
#   ./torchtitan/experiments/moe_ladder/jzh100_profile.slurm
# Full model:
#   N_LAYERS=0 ./torchtitan/experiments/moe_ladder/jzh100_profile.slurm
# Example with two GPUs and four layers:
#   GPUS=2 N_LAYERS=4 ./torchtitan/experiments/moe_ladder/jzh100_profile.slurm
# Extra arguments are forwarded to sbatch.

set -euo pipefail

if [[ -z ${SLURM_JOB_ID:-} ]]; then
    : "${IDRPROJ:?IDRPROJ is not set}"
    mkdir -p outputs/slurm
    num_gpus=${GPUS:-4}
    exec sbatch --account="${IDRPROJ}@h100" --gres="gpu:$num_gpus" \
        --cpus-per-task="$((24 * num_gpus))" "$@" "$0"
fi

REPO=${SLURM_SUBMIT_DIR:-$PWD}

module purge
module load arch/h100
module load cuda/13.0.3

VENV=${VENV:-$REPO/titan-rl}
PY=$VENV/bin/python
[[ -x $PY ]] || { echo "No interpreter at $PY; set VENV." >&2; exit 1; }

# DeepEP JIT-compiles on first dispatch; keep nvcc and CCCL headers on one CTK.
export CUDA_HOME=$(dirname "$(dirname "$(command -v nvcc)")")
export PATH=$CUDA_HOME/bin:$PATH
export EP_DISABLE_GIN=1 # single node: no GPU-initiated RDMA
export EP_JIT_CACHE_DIR=$SCRATCH/deepep_jit_cache
export TRITON_CACHE_DIR=$SCRATCH/triton_cache
mkdir -p "$EP_JIT_CACHE_DIR" "$TRITON_CACHE_DIR"

cd "$REPO"

GPUS=${GPUS:-${SLURM_GPUS_ON_NODE:-4}}
EP=${EP:-$GPUS}
TP=${TP:-1}
BATCH=${BATCH:-16}
SEQ_LEN=${SEQ_LEN:-2048}
N_LAYERS=${N_LAYERS:-8}
MODEL_FLAVOR=${MODEL_FLAVOR:-30B-A3B}
WARMUP_STEPS=${WARMUP_STEPS:-10}
STEPS=${STEPS:-20}
MODEL_ARGS=(--model-flavor "$MODEL_FLAVOR" --n-layers "$N_LAYERS")
[[ -z ${CKPT:-} ]] || MODEL_ARGS+=(--checkpoint-path "$CKPT")
[[ -z ${HF_MODEL:-} ]] || MODEL_ARGS+=(--hf-model-path "$HF_MODEL")
[[ -z ${TOKENIZER:-} ]] || MODEL_ARGS+=(--tokenizer-path "$TOKENIZER")
[[ -z ${DATA:-} ]] || MODEL_ARGS+=(--data-path "$DATA")
COMMON_ARGS=(--ep "$EP" --tp "$TP" --batch-size "$BATCH" --seq-len "$SEQ_LEN"
             --warmup-steps "$WARMUP_STEPS" --steps "$STEPS")
read -r -a VARIANTS <<<"${VARIANTS:-stock parallel ladder hoisted_gateA hoisted_gateB}"
TIMEOUT=${TIMEOUT:-900}
RUN_NAME=${RUN_NAME:-${MODEL_FLAVOR}_g${GPUS}_ep${EP}_tp${TP}_b${BATCH}_s${SEQ_LEN}_L${N_LAYERS}}
OUT=outputs/jz_h100_overlap/$RUN_NAME
mkdir -p "$OUT/nsys"
echo "config: model=$MODEL_FLAVOR gpus=$GPUS ep=$EP tp=$TP batch=$BATCH seq_len=$SEQ_LEN layers=$N_LAYERS"
echo "variants: ${VARIANTS[*]}"
echo "output: $OUT"

# 1. Measure latency and MFU.
$PY -m torchtitan.experiments.moe_ladder.profile_inference \
  --nproc-per-node "$GPUS" "${MODEL_ARGS[@]}" "${COMMON_ARGS[@]}" \
  --variants "${VARIANTS[@]}" --timeout "$TIMEOUT" \
  --output "$OUT/sweep.json"

if [[ -n ${SWEEP_ONLY:-} ]]; then
  echo "Sweep complete: $OUT/sweep.json"
  exit 0
fi

# 2. Record one Nsight Systems trace per variant.
for v in "${VARIANTS[@]}"; do
  MOE_LADDER_NVTX=1 nsys profile \
    --force-overwrite=true --sample=none --cuda-event-trace=false \
    --trace=cuda,nvtx --gpu-metrics-device=all \
    -o "$OUT/nsys/${v}" \
    $PY -m torch.distributed.run --standalone --nproc-per-node="$GPUS" \
      -m torchtitan.experiments.moe_ladder.profile_inference \
      --worker --variant "$v" "${MODEL_ARGS[@]}" "${COMMON_ARGS[@]}" \
      --output "$OUT/nsys/${v}.json"
done

# 3. Generate the overlap report; temporary SQLite exports are removed.
$PY -m torchtitan.experiments.moe_ladder.analyze_overlap \
  "$OUT"/nsys/*.nsys-rep \
  --results "$OUT/sweep.json" | tee "$OUT/overlap_report.txt"

echo "Results in $OUT; nsys-ui $OUT/nsys/<variant>.nsys-rep"

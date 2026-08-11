#!/bin/bash
#SBATCH --job-name=moe_ladder_loss
#SBATCH --constraint=h100
#SBATCH --qos=qos_gpu_h100-dev
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --hint=nomultithread
#SBATCH --time=02:00:00
#SBATCH --output=outputs/slurm/moe_ladder_loss.%j.out
#
# Compare full-model schedule losses on one Jean Zay H100 node.
# Run from the checkout root; paths may be relative to it.
# DCP checkpoint directory with Hugging Face tokenizer assets:
#   CKPT=path/to/dcp TOKENIZER=path/to/Qwen3-30B-A3B-Base \
#       DATA=path/to/evaluation.jsonl \
#       ./torchtitan/experiments/moe_ladder/jzh100_loss.slurm
# Direct Hugging Face weights and tokenizer:
#   HF_MODEL=path/to/Qwen3-30B-A3B-Base DATA=path/to/evaluation.jsonl \
#       ./torchtitan/experiments/moe_ladder/jzh100_loss.slurm
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

export CUDA_HOME=$(dirname "$(dirname "$(command -v nvcc)")")
export PATH=$CUDA_HOME/bin:$PATH
export EP_DISABLE_GIN=1
export EP_JIT_CACHE_DIR=$SCRATCH/deepep_jit_cache
export TRITON_CACHE_DIR=$SCRATCH/triton_cache
mkdir -p "$EP_JIT_CACHE_DIR" "$TRITON_CACHE_DIR"

cd "$REPO"

GPUS=${GPUS:-${SLURM_GPUS_ON_NODE:-4}}
EP=${EP:-$GPUS}
TP=${TP:-1}
BATCH=${BATCH:-1}
SEQ_LEN=${SEQ_LEN:-512}
MODEL_FLAVOR=${MODEL_FLAVOR:-30B-A3B}
TIMEOUT=${TIMEOUT:-3600}

if [[ -n ${CKPT:-} && -n ${HF_MODEL:-} ]]; then
    echo "Set only CKPT or HF_MODEL." >&2
    exit 1
fi
if [[ -z ${CKPT:-} && -z ${HF_MODEL:-} ]]; then
    echo "Set CKPT or HF_MODEL." >&2
    exit 1
fi
if [[ -n ${CKPT:-} && -z ${TOKENIZER:-} ]]; then
    echo "DCP loss comparison requires TOKENIZER." >&2
    exit 1
fi
: "${DATA:?DATA is not set}"

MODEL_ARGS=(--model-flavor "$MODEL_FLAVOR" --n-layers 0 --data-path "$DATA")
[[ -z ${CKPT:-} ]] || MODEL_ARGS+=(--checkpoint-path "$CKPT")
[[ -z ${HF_MODEL:-} ]] || MODEL_ARGS+=(--hf-model-path "$HF_MODEL")
[[ -z ${TOKENIZER:-} ]] || MODEL_ARGS+=(--tokenizer-path "$TOKENIZER")
read -r -a VARIANTS <<<"${VARIANTS:-stock parallel ladder hoisted_gateA hoisted_gateB}"

RUN_NAME=${RUN_NAME:-${MODEL_FLAVOR}_g${GPUS}_ep${EP}_tp${TP}_b${BATCH}_s${SEQ_LEN}}
OUT=outputs/jz_h100_losses/$RUN_NAME
mkdir -p "$OUT"

echo "config: model=$MODEL_FLAVOR gpus=$GPUS ep=$EP tp=$TP batch=$BATCH seq_len=$SEQ_LEN"
echo "variants: ${VARIANTS[*]}"
echo "output: $OUT"

$PY -m torchtitan.experiments.moe_ladder.profile_inference \
  --nproc-per-node "$GPUS" "${MODEL_ARGS[@]}" \
  --ep "$EP" --tp "$TP" --batch-size "$BATCH" --seq-len "$SEQ_LEN" \
  --loss-only --variants "${VARIANTS[@]}" --timeout "$TIMEOUT" \
  --output "$OUT/losses.json"

echo "Losses written to $OUT/losses.json"

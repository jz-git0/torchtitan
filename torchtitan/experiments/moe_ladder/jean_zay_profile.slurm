#!/bin/bash
#SBATCH --job-name=moe_ladder_deepep
#SBATCH --account=CHANGEME@h100
#SBATCH --constraint=h100
#SBATCH --qos=qos_gpu_h100-dev
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=48
#SBATCH --hint=nomultithread
#SBATCH --time=01:00:00
#SBATCH --output=moe_ladder_deepep.%j.out

# MoE ladder profiling on Jean Zay H100: latency sweeps for the standard and
# DeepEP MoE comm backends, then per-variant nsys captures of the DeepEP runs.
# EP=2 on 2 GPUs matches the workstation traces; bump --gres and --ep together
# to try EP=4 on a full node.

set -euo pipefail

module purge
module load arch/h100
# Only needed for the nsys binary; pick any available version (module avail).
module load cuda/12.8.0

REPO=$WORK/code/lqaif/torchtitan
VENV=$REPO/titan-rl
PY=$VENV/bin/python
NV=$VENV/lib/python3.12/site-packages/nvidia

# DeepEP's runtime JIT compiles kernels on first dispatch; it needs a CUDA
# toolkit with nvcc >= 12.3. The venv's bundled cu13 wheel keeps it consistent
# with the torch build regardless of cluster modules.
export CUDA_HOME=$NV/cu13
# Single-node run: skip the GPU-initiated-RDMA (GIN) requirement.
export EP_DISABLE_GIN=1
export EP_JIT_CACHE_DIR=$SCRATCH/deepep_jit_cache
export TRITON_CACHE_DIR=$SCRATCH/triton_cache
mkdir -p "$EP_JIT_CACHE_DIR" "$TRITON_CACHE_DIR"

cd "$REPO"
OUT=outputs/jz_h100
mkdir -p "$OUT/nsys"

COMMON_ARGS=(--tp 1 --ep 2 --batch-size 8 --seq-len 1024
             --warmup-steps 20 --steps 50)

# 1) Latency sweeps over all variants (the driver spawns one torchrun each).
$PY -m torchtitan.experiments.moe_ladder.profile_block \
  --nproc-per-node 2 "${COMMON_ARGS[@]}" \
  --moe-comm-backend standard --output "$OUT/block_standard.json"

$PY -m torchtitan.experiments.moe_ladder.profile_block \
  --nproc-per-node 2 "${COMMON_ARGS[@]}" \
  --moe-comm-backend deepep --output "$OUT/block_deepep.json"

# 2) nsys captures of the DeepEP variants (one report per variant).
for v in stock parallel ladder hoisted_gateA hoisted_gateB; do
  nsys profile --force-overwrite=true --sample=none --cuda-event-trace=false \
    --trace=cuda,nvtx,osrt \
    -o "$OUT/nsys/${v}_deepep" \
    $PY -m torch.distributed.run --standalone --nproc-per-node=2 \
      -m torchtitan.experiments.moe_ladder.profile_block \
      --worker --variant "$v" "${COMMON_ARGS[@]}" \
      --moe-comm-backend deepep \
      --output "$OUT/nsys/${v}_deepep.json"
done

echo "Results in $OUT; open reports with nsys-ui $OUT/nsys/<variant>_deepep.nsys-rep"

# Installing DeepEP (single node)

This pinned recipe targets one Hopper NVLink node. DeepEP v2 requires an SM90 GPU (or newer hardware with SM90 PTX support), PyTorch 2.10 or later, CUDA 12.3 or later, and NCCL 2.30.4 or later. Ampere GPUs are not supported.

The commands below match a PyTorch cu13 environment with NVIDIA CUDA wheels under `site-packages/nvidia`; they expect `nvidia/cu13/bin/nvcc`. This is an intranode build: `DISABLE_NVSHMEM=1` disables the NVSHMEM/RDMA path.

## Build

```bash
REPO=/path/to/torchtitan
VENV=/path/to/virtualenv
BUILD=${TMPDIR:-/tmp}/deepep-src

# NVIDIA wheels installed in the venv. `nvidia` is a namespace package.
NV=$("$VENV/bin/python" -c "import nvidia; print(list(nvidia.__path__)[0])")

# Required: torch cu13/nightly plus CUDA 13 toolkit wheels.
"$VENV/bin/python" -c "import torch; print(torch.version.cuda); assert torch.version.cuda and torch.version.cuda.startswith('13')"
test -x "$NV/cu13/bin/nvcc"
test -f "$NV/cu13/lib/libcudart.so.13"

# DeepEP v2.1 needs NCCL >= 2.30 headers.
VIRTUAL_ENV=$VENV uv pip install "nvidia-nccl-cu13==2.30.7"

# The CUDA wheel has libcudart.so.13 but no unversioned linker name.
ln -sf libcudart.so.13 "$NV/cu13/lib/libcudart.so"

git clone https://github.com/deepseek-ai/DeepEP.git "$BUILD"
cd "$BUILD"
git checkout dd758ca
git apply "$REPO/torchtitan/experiments/moe_ladder/deepep_v2.1_portable_build.patch"

# DeepEP links -lcuda. On a GPU-less build host, use the toolkit stub only for
# linking; GPU jobs use the real driver-provided libcuda.so.1.
module load arch/h100 cuda/13.0.3  # or your site's CUDA module
CUDA_STUB_ROOT=$(dirname "$(dirname "$(command -v nvcc)")")
CUDA_STUB=$(find "$CUDA_STUB_ROOT" -path '*/stubs/libcuda.so' -print -quit)
test -n "$CUDA_STUB"
STUBS=$(dirname "$CUDA_STUB")

# Build with the venv's CUDA 13 toolkit, not the module toolkit.
env VIRTUAL_ENV=$VENV CUDA_HOME=$NV/cu13 PATH=$NV/cu13/bin:$PATH \
    LIBRARY_PATH=$STUBS \
    TORCH_CUDA_ARCH_LIST=9.0 DISABLE_NVSHMEM=1 \
    NCCL_DIR=$NV/nccl NVSHMEM_DIR=$NV/nvshmem MAX_JOBS=16 \
    uv pip install --no-build-isolation .
```

## Verify

Run the verification outside the DeepEP source checkout. Otherwise, Python
imports the source tree, which does not contain the wheel-built `_C` extension,
instead of the installed package.

On a GPU node:

```bash
cd "$REPO"
EP_DISABLE_GIN=1 "$VENV/bin/python" \
  -c "import deep_ep; from deep_ep import ElasticBuffer; print('OK', deep_ep.__version__)"
```

On a GPU-less login node, create a temporary `libcuda.so.1` stub for the import check only:

```bash
cd "$REPO"
mkdir -p "$BUILD/cuda-stub"
ln -sf "$STUBS/libcuda.so" "$BUILD/cuda-stub/libcuda.so.1"
LD_LIBRARY_PATH=$BUILD/cuda-stub "$VENV/bin/python" \
  -c "import deep_ep; from deep_ep import ElasticBuffer; print('OK', deep_ep.__version__)"
```

Do not use that `LD_LIBRARY_PATH` in GPU jobs.

## Smoke Test

```bash
EP_DISABLE_GIN=1 "$VENV/bin/python" \
  -m torchtitan.experiments.moe_ladder.profile_inference \
  --nproc-per-node 2 --ep 2 --batch-size 8 --seq-len 1024 \
  --warmup-steps 10 --steps 10
```

Notes:

- Do not set `DISABLE_SM90_FEATURES` on H100.
- If the build cannot find `Python.h`, use a Python/venv with development headers.

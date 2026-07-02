#!/usr/bin/env bash
set -euo pipefail

# Ascend/snt9b NanoVLM environment sync script.
# This script intentionally does NOT install torch, torchvision, flash-attn,
# triton, bitsandbytes, liger-kernel, or any nvidia-* CUDA wheels.

PIP_INDEX_URL="${PIP_INDEX_URL:-https://mirrors.aliyun.com/pypi/simple/}"
PIP_TRUSTED_HOST="${PIP_TRUSTED_HOST:-mirrors.aliyun.com}"
PYTHON_BIN="${PYTHON_BIN:-python}"

PIP_COMMON_ARGS=(
  --index-url "${PIP_INDEX_URL}"
  --trusted-host "${PIP_TRUSTED_HOST}"
)

echo "============================================================"
echo "[1/4] Checking bundled Ascend PyTorch environment"
echo "============================================================"
"${PYTHON_BIN}" - <<'PY'
import os
import sys

try:
    import torch
except Exception as exc:
    raise SystemExit(f"Failed to import torch from the base image: {exc}")

try:
    import torch_npu
except Exception as exc:
    raise SystemExit(f"Failed to import torch_npu from the base image: {exc}")

print("python:", sys.version.replace("\n", " "))
print("torch:", torch.__version__)
print("torch path:", torch.__file__)
print("torch_npu:", getattr(torch_npu, "__version__", "unknown"))
print("torch_npu path:", torch_npu.__file__)
print("ASCEND_HOME:", os.environ.get("ASCEND_HOME", ""))
print("ASCEND_TOOLKIT_HOME:", os.environ.get("ASCEND_TOOLKIT_HOME", ""))

if not hasattr(torch, "npu"):
    raise SystemExit("torch.npu is unavailable. Check whether this is an Ascend PyTorch image.")

print("npu available:", torch.npu.is_available())
print("npu count:", torch.npu.device_count())
if torch.npu.device_count() > 0:
    for idx in range(torch.npu.device_count()):
        try:
            print(f"npu {idx}:", torch.npu.get_device_name(idx))
        except Exception:
            print(f"npu {idx}: <name unavailable>")
PY

echo "============================================================"
echo "[2/4] Upgrading pip tooling with Aliyun mirror"
echo "============================================================"
"${PYTHON_BIN}" -m pip install \
  "${PIP_COMMON_ARGS[@]}" \
  --upgrade pip setuptools wheel

echo "============================================================"
echo "[3/4] Installing Ascend-safe NanoVLM dependencies"
echo "============================================================"
"${PYTHON_BIN}" -m pip install \
  "${PIP_COMMON_ARGS[@]}" \
  "accelerate==1.10.1" \
  "datasets==4.1.0" \
  "huggingface-hub==0.35.3" \
  "hf-transfer==0.1.9" \
  "safetensors==0.6.2" \
  "tokenizers==0.22.1" \
  "transformers>=4.57.1" \
  "pillow==12.0.0" \
  "numpy==2.2.0" \
  "pandas==2.3.2" \
  "pyarrow==21.0.0" \
  "pyyaml==6.0.3" \
  "jsonlines==4.0.0" \
  "pydantic==2.11.9" \
  "loguru==0.7.3" \
  "hydra-core==1.3.2" \
  "omegaconf==2.3.0" \
  "rich==14.1.0" \
  "tqdm==4.67.1" \
  "requests==2.32.5" \
  "einops==0.8.1" \
  "torchdata==0.11.0" \
  "qwen-vl-utils==0.0.14" \
  "opencv-python-headless==4.12.0.88" \
  "wandb==0.21.4"

echo "============================================================"
echo "[3.5/4] Installing DeepSpeed without building CUDA/NPU ops"
echo "============================================================"
DS_BUILD_OPS=0 "${PYTHON_BIN}" -m pip install \
  "${PIP_COMMON_ARGS[@]}" \
  "deepspeed==0.17.5" \
  --no-build-isolation

echo "============================================================"
echo "[4/4] Installing lmms-engine in editable mode without deps"
echo "============================================================"
"${PYTHON_BIN}" -m pip install \
  "${PIP_COMMON_ARGS[@]}" \
  --no-deps \
  -e .

echo "============================================================"
echo "Final import check"
echo "============================================================"
"${PYTHON_BIN}" - <<'PY'
import torch
import torch_npu
import transformers
import datasets
import lmms_engine
from lmms_engine.models.nanovlm import NanovlmConfig, NanovlmForConditionalGeneration
from lmms_engine.datasets.processor.nanovlm_processor import NanovlmDataProcessor

print("torch:", torch.__version__)
print("torch_npu:", getattr(torch_npu, "__version__", "unknown"))
print("npu available:", torch.npu.is_available())
print("npu count:", torch.npu.device_count())
print("transformers:", transformers.__version__)
print("datasets:", datasets.__version__)
print("lmms_engine:", getattr(lmms_engine, "__version__", "editable"))
print("NanoVLM imports ok")
PY

echo "============================================================"
echo "Ascend NanoVLM environment sync completed."
echo "Remember to run NanoVLM with:"
echo "  model_config.attn_implementation=sdpa"
echo "  trainer_args.use_liger_kernel=false"
echo "  trainer_args.use_rmpad=false"
echo "============================================================"

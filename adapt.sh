#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# adapt.sh
# Calibrate gate biases of the converted RFMoE checkpoint using the density
# auxiliary loss on a 1/10000 slice of OpenWebText.
#
# Only gate.gate_bias parameters are trained (one scalar per expert per layer).
# All FFN / attention weights remain frozen.
#
# Key choices:
#   --gate-threshold   1.0     must match conversion
#   --density-target   0.25    drive each expert to activate 25% of tokens
#   --quantization     int4    load frozen weights in NF4 (saves ~70% VRAM)
#   --per-device-batch-size       2  \
#   --gradient-accumulation-steps 8   → effective batch = 2×8 = 16 per GPU
#
# Output: Mixtral-8x7B-Instruct-v0.1-RFMoE-adapted/
# -----------------------------------------------------------------------------
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

MODEL_DIR="${SCRIPT_DIR}/Mixtral-8x7B-Instruct-v0.1-RFMoE-converted"
OUTPUT_DIR="${SCRIPT_DIR}/Mixtral-8x7B-Instruct-v0.1-RFMoE-adapted"

# ── Multi-GPU detection ──────────────────────────────────────────────────────
# Use torchrun when multiple GPUs are available, plain python otherwise.
N_GPUS=$(python -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo 0)

echo "================================================================"
echo "  RFMoE gate-bias adaptation"
echo "  model   : ${MODEL_DIR}"
echo "  output  : ${OUTPUT_DIR}"
echo "  GPUs    : ${N_GPUS}"
echo "  target density : 0.25"
echo "  quant   : int4 (frozen weights only; gate_bias stays float)"
echo "================================================================"

ADAPT_ARGS=(
    --model-dir      "${MODEL_DIR}"
    --output-dir     "${OUTPUT_DIR}"

    # RF aux-loss (must match conversion config)
    --gate-threshold            1.0
    --density-target            0.25
    --lambda-coef               1e-10
    --eta-coef                  0.02
    --per-expert-aux-loss-coef  0.5
    --per-token-aux-loss-coef   0.5

    # Dataset: 1/10000 of OWT (~800 documents)
    --dataset-fraction  1e-4
    --max-length        512

    # Batch / optimiser
    --per-device-batch-size        2
    --gradient-accumulation-steps  32
    --n-epochs                     1
    --learning-rate                1e-2
    --warmup-steps                 20
    --eval-steps                   50
    --save-steps                   200

    # Precision & memory
    --bf16
    --quantization  int4

    --n-workers  4
    --seed       42
)

if [ "${N_GPUS}" -gt 1 ]; then
    echo "Launching with torchrun (${N_GPUS} GPUs, DDP) …"
    torchrun --nproc_per_node="${N_GPUS}" adapt_gate_bias.py "${ADAPT_ARGS[@]}"
else
    echo "Launching single-process (device_map=auto) …"
    python adapt_gate_bias.py "${ADAPT_ARGS[@]}"
fi

echo ""
echo "Adaptation done → ${OUTPUT_DIR}/final_model"

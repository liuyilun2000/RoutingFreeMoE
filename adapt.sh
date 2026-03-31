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
#   --bf16                     use HF-native bf16 precision
#   --per-device-batch-size       2  \
#   --gradient-accumulation-steps 8   → effective batch = 2×8 = 16 per GPU
#
# Output: Mixtral-8x7B-Instruct-v0.1-RFMoE-adapted/
# -----------------------------------------------------------------------------
set -euo pipefail

# Optional: set W&B API key so runs log to wandb (no need for `wandb login` with long keys)
# export WANDB_API_KEY="wandb_v1_..."

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

MODEL_DIR="${SCRIPT_DIR}/OLMoE-1B-7B-Instruct-RFMoE-converted"
OUTPUT_DIR="${SCRIPT_DIR}/OLMoE-1B-7B-Instruct-RFMoE-adapted"

WANDB_PROJECT="${WANDB_PROJECT:-mixtral-baseline}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-gate_bias_adapt_OLMoE_1B_7B}"

# ── Multi-GPU detection ──────────────────────────────────────────────────────
# Use torchrun when multiple GPUs are available, plain python otherwise.
N_GPUS=$(python -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo 0)

echo "================================================================"
echo "  RFMoE gate-bias adaptation"
echo "  model   : ${MODEL_DIR}"
echo "  output  : ${OUTPUT_DIR}"
echo "  GPUs    : ${N_GPUS}"
echo "  target density : 0.25"
echo "  dtype   : bfloat16 (HF-native)"
echo "  wandb   : project=${WANDB_PROJECT} run=${WANDB_RUN_NAME}"
echo "================================================================"

ADAPT_ARGS=(
    --model-dir      "${MODEL_DIR}"
    --output-dir     "${OUTPUT_DIR}"

    # RF aux-loss (must match conversion config)
    --gate-threshold            1.0
    --density-target            0.25
    --lambda-coef               1e-1
    --eta-coef                  0.02
    --per-expert-aux-loss-coef  0.5
    --per-token-aux-loss-coef   0.5

    # Dataset: 1/10000 of OWT (~800 documents)
    --dataset-fraction  1e-3
    --max-length        512

    # Batch / optimiser
    --per-device-batch-size        2
    --gradient-accumulation-steps  8
    --n-epochs                     1
    --learning-rate                1e-1
    --warmup-steps                 10
    --eval-steps                   10
    --save-steps                   40

    # Precision & memory
    --bf16

    --n-workers  4
    --seed       42

    --wandb-project   "${WANDB_PROJECT}"
    --wandb-run-name  "${WANDB_RUN_NAME}"
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

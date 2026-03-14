#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# convert.sh
# Convert mistralai/Mixtral-8x7B-Instruct-v0.1 → RFMoE checkpoint
#
# Key choices:
#   --gate-proj-rank 128     low-rank gate projection (SVD of w1)
#   --gate-threshold 1.0     initial threshold (will be calibrated by adapt.sh)
#   --density-target 0.25    stored in config, used by adapt.sh aux loss
#   --quantization int4      load source model in NF4 to save ~70% VRAM
#   --dtype bfloat16         save converted checkpoint in bf16
#
# Output: Mixtral-8x7B-Instruct-v0.1-RFMoE-converted/
# -----------------------------------------------------------------------------
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

OUTPUT_DIR="${SCRIPT_DIR}/Mixtral-8x7B-Instruct-v0.1-RFMoE-converted"

echo "================================================================"
echo "  Mixtral-8x7B-Instruct-v0.1 → RFMoE conversion"
echo "  output  : ${OUTPUT_DIR}"
echo "  rank    : 128"
echo "  threshold: 1.0  (calibrated by adapt.sh)"
echo "  quant   : int4 (NF4, loads source model in ~24 GB VRAM)"
echo "================================================================"

python convert_mixtral_to_rfmoe.py \
    --source-model    "mistralai/Mixtral-8x7B-Instruct-v0.1" \
    --output-dir      "${OUTPUT_DIR}"               \
    --gate-proj-rank  128                           \
    --gate-threshold  1.0                           \
    --gate-norm       l2                            \
    --gate-act-fn     linear                        \
    --density-target  0.25                          \
    --lambda-coef     1e-10                          \
    --eta-coef        0.02                           \
    --per-expert-aux-loss-coef 0.5                  \
    --per-token-aux-loss-coef  0.5                  \
    --dtype           bfloat16                      \
    --quantization    int4

echo ""
echo "Conversion done → ${OUTPUT_DIR}"
echo "Run adapt.sh next to calibrate gate biases."

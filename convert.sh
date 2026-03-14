#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# convert.sh
# Convert allenai/OLMoE-1B-7B-0125-Instruct → RFMoE checkpoint
#
# Key choices:
#   --gate-proj-rank 128     low-rank gate projection (SVD of gate_proj / w_gate)
#   --gate-threshold 1.0     initial threshold (calibrated by adapt.sh)
#   --density-target 0.25    stored in config, enforced by adapt.sh aux loss
#   --dtype bfloat16         load & save in HF-native bf16
#
# Output: OLMoE-1B-7B-Instruct-RFMoE-converted/  (next to this script)
# -----------------------------------------------------------------------------
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

OUTPUT_DIR="${SCRIPT_DIR}/OLMoE-1B-7B-Instruct-RFMoE-converted"

echo "================================================================"
echo "  allenai/OLMoE-1B-7B-0125-Instruct → RFMoE conversion"
echo "  output   : ${OUTPUT_DIR}"
echo "  rank     : 128"
echo "  threshold: 1.0  (gate biases calibrated by adapt.sh)"
echo "  dtype    : bfloat16 (HF-native)"
echo "================================================================"

python convert_olmoe_to_rfmoe.py \
    --source-model    "allenai/OLMoE-1B-7B-0125-Instruct" \
    --output-dir      "${OUTPUT_DIR}"                      \
    --gate-proj-rank  128                                  \
    --gate-threshold  1.0                                  \
    --gate-norm       l2                                   \
    --gate-act-fn     linear                               \
    --density-target  0.25                                 \
    --lambda-coef     1e-10                                \
    --eta-coef        0.02                                 \
    --per-expert-aux-loss-coef 0.5                         \
    --per-token-aux-loss-coef  0.5                         \
    --dtype           bfloat16

echo ""
echo "Conversion done → ${OUTPUT_DIR}"
echo "Run adapt.sh next to calibrate gate biases."

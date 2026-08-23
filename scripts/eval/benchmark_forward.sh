#!/bin/bash
#SBATCH --partition=accelerated
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --mail-type=NONE

source /hkfs/home/project/hk-project-p0022189/hgf_mxv5488/miniconda3/bin/activate rfmoe

MODEL_DIR=${1:-output/mixtral_rf/S_rf_lr_1e-3_rank_32/final_model}
THRESHOLDS=${2:-0.8,0.9,1.0,1.1,1.2}
OUTPUT=${3:-results/threshold/timing.json}

echo "Model dir  : $MODEL_DIR"
echo "Thresholds : $THRESHOLDS"
echo "Output     : $OUTPUT"

python scripts/eval/eval_benchmarks.py \
    --model-dir "$MODEL_DIR" \
    --model-type routing_free \
    --benchmark-forward \
    --thresholds "$THRESHOLDS" \
    --output "$OUTPUT"

#!/bin/bash
#SBATCH --partition=accelerated
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --mail-type=NONE

source /hkfs/home/project/hk-project-p0022189/hgf_mxv5488/miniconda3/bin/activate py310

MODEL_DIR=${1:-./output_baseline_mixtral/1_mixtral_baseline_12L_128Dx12E_top2/final_model}
MAX_SAMPLES=${2:-}  # Empty for all samples
BATCH_SIZE=${3:-16}

echo "Evaluating model: $MODEL_DIR"
echo "Max samples: ${MAX_SAMPLES:-all}"
echo "Batch size: $BATCH_SIZE"

python eval_mixtral_baseline.py \
    --model-dir "$MODEL_DIR" \
    $(if [ -n "$MAX_SAMPLES" ]; then echo "--max-samples $MAX_SAMPLES"; fi) \
    --batch-size "$BATCH_SIZE"

#!/bin/bash
#SBATCH --partition=accelerated
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --mail-type=NONE

source /hkfs/home/project/hk-project-p0022189/hgf_mxv5488/miniconda3/bin/activate rfmoe

MODEL_DIR=${1:-output/mixtral_rf/S_rf_lr_2e-3_rank_16/final_model}
TASKS="piqa,hellaswag,winogrande,arc_easy,arc_challenge,openbookqa,qqp,qnli,sst2"
OUTDIR=${2:-results/threshold/timing_eval}

mkdir -p "$OUTDIR"

for THRESH in 0.1 1.9; do
    echo "===== Threshold: $THRESH ====="
    python scripts/eval/eval_benchmarks.py \
        --model-dir "$MODEL_DIR" \
        --model-type routing_free \
        --tasks "$TASKS" \
        --gate-threshold "$THRESH" \
        --output "$OUTDIR/thresh_${THRESH}.json"
done

echo "===== All done ====="
echo "Results in $OUTDIR"

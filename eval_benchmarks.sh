#!/bin/bash
#SBATCH --partition=accelerated
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --mail-type=NONE

source /hkfs/home/project/hk-project-p0022189/hgf_mxv5488/miniconda3/bin/activate py310

# Install lm-evaluation-harness if not present
pip show lm_eval &>/dev/null || pip install lm_eval

# ── Arguments ─────────────────────────────────────────────────────────────────
# $1: model directory (required)
# $2: model type — "baseline" or "routing_free" (default: baseline)
# $3: output JSON path (optional)
# $4: comma-separated task list (optional, default: all 8 benchmarks)
# $5: num few-shot (optional, default: 0)

#MODEL_DIR=${1:-output/mixtral_rf/S_rf_lr_1e-3_rank_32/final_model}
MODEL_DIR=${1:-output/mixtral_rf/mixtral_rf_12L_128Dx12E_temp_1.0_thres_1.0_density_0.25_lambda_1e-10_eta_0.02_aux_[E0.5_T0.5]_lr_2e-3/final_model}
MODEL_TYPE=${2:-routing_free}
OUTPUT_JSON=${3:-results/rf_2e-3.json}
TASKS=${4:-arc_easy,arc_challenge,piqa,winogrande,hellaswag,boolq,openbookqa,cola,sst2,mrpc,stsb,qqp,mnli,qnli,rte,wnli}
NUM_FEWSHOT=${5:-0}

echo "Model dir  : $MODEL_DIR"
echo "Model type : $MODEL_TYPE"
echo "Tasks      : $TASKS"
echo "Few-shot   : $NUM_FEWSHOT"
echo "Output     : ${OUTPUT_JSON:-<not saved>}"

python eval_benchmarks.py \
    --model-dir "$MODEL_DIR" \
    --model-type "$MODEL_TYPE" \
    --tasks "$TASKS" \
    --batch-size auto \
    --num-fewshot "$NUM_FEWSHOT" \
    $(if [ -n "$OUTPUT_JSON" ]; then echo "--output $OUTPUT_JSON"; fi)

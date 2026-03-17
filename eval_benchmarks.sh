#!/bin/bash
#SBATCH --partition=accelerated
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --mail-type=NONE

# Install lm-evaluation-harness if not present
pip show lm_eval &>/dev/null || pip install lm_eval

# ── Arguments ─────────────────────────────────────────────────────────────────
# $1: model directory (required)
# $2: model type — "baseline" or "routing_free" (default: routing_free)
# $3: output JSON path (optional)
# $4: comma-separated task list (optional, default: all benchmarks)
# $5: num few-shot (optional, default: 0)

MODEL_DIR=${1:?Usage: $0 <model_dir> [model_type] [output_json] [tasks] [num_fewshot]}
MODEL_TYPE=${2:-routing_free}
OUTPUT_JSON=${3:-results/eval_results.json}
TASKS=${4:-arc_easy,arc_challenge,piqa,winogrande,hellaswag,boolq,openbookqa,cola,sst2,mrpc,qqp,mnli,qnli,rte,wnli}
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

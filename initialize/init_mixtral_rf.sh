#!/bin/bash

# Default values based on init_baseline_mixtral.sh and init.sh


# L-size params: 32L 1024hs 24experts 256dim 32Q16KV
CONFIG_JSON=${1:-mixtral_rf_L.config.json}
OUTPUT_DIR=${2:-../config}
TOKENIZER_MODEL=${3:-EleutherAI/gpt-neo-125M}
GATE_PROJ_RANK=${4:-64}

num_hidden_layers=32
n_experts=24
intermediate_size_list=(256)

for intermediate_size in "${intermediate_size_list[@]}"; do
  model_name="RoutingFreeMixtral_${num_hidden_layers}L_${intermediate_size}D_rank${GATE_PROJ_RANK}"
  echo "Initializing $model_name ..."
  python init_mixtral_rf.py \
    --config-json "$CONFIG_JSON" \
    --num-hidden-layers $num_hidden_layers \
    --intermediate-size $intermediate_size \
    --n-experts $n_experts \
    --gate-proj-rank $GATE_PROJ_RANK \
    --output-dir "$OUTPUT_DIR" \
    --model-name "$model_name" \
    --tokenizer-model "$TOKENIZER_MODEL" \
    --bf16 
done

#!/bin/bash

# Default values based on init_baseline_mixtral.sh and init.sh

CONFIG_JSON=${1:-mixtral_rf.config.json} # Using baseline config as base
OUTPUT_DIR=${2:-../init_mixtral_rf}
TOKENIZER_MODEL=${3:-EleutherAI/gpt-neo-125M}

# Mixtral Baseline params
num_hidden_layers=12

# Routing Free params (from init.sh)
n_experts=12

# Intermediate size list (from baseline/RF intersection or just baseline)
# init_baseline_mixtral uses (128)
# init.sh uses (128)
intermediate_size_list=(128)


for intermediate_size in "${intermediate_size_list[@]}"; do
  model_name="RoutingFreeMixtral_${num_hidden_layers}L_${intermediate_size}D"
  echo "Initializing $model_name ..."
  python init_mixtral_rf.py \
    --config-json "$CONFIG_JSON" \
    --num-hidden-layers $num_hidden_layers \
    --intermediate-size $intermediate_size \
    --n-experts $n_experts \
    --output-dir "$OUTPUT_DIR" \
    --model-name "$model_name" \
    --tokenizer-model "$TOKENIZER_MODEL" \
    --bf16 
done

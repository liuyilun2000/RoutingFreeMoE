#!/bin/bash

CONFIG_JSON=${1:-baseline_mixtral.config.json}
OUTPUT_DIR=${2:-../init_baseline_mixtral}
TOKENIZER_MODEL=${3:-EleutherAI/gpt-neo-125M} #EleutherAI/gpt-neo-125M

num_hidden_layers=12
num_local_experts=12
num_experts_per_tok=2
intermediate_size_list=(128)


for intermediate_size in "${intermediate_size_list[@]}"; do
  model_name="Mixtral_${num_hidden_layers}L_${intermediate_size}D"
  echo "Initializing $model_name ..."
  python init_baseline_mixtral.py \
    --config-json "$CONFIG_JSON" \
    --num-hidden-layers $num_hidden_layers \
    --num-local-experts $num_local_experts \
    --num-experts-per-tok $num_experts_per_tok \
    --intermediate-size $intermediate_size \
    --output-dir "$OUTPUT_DIR" \
    --model-name "$model_name" \
    --tokenizer-model "$TOKENIZER_MODEL" \
    --bf16 
done
#!/bin/bash

CONFIG_JSON=${1:-120M_baseline.config.json}
OUTPUT_DIR=${2:-./init_baseline}
TOKENIZER_MODEL=${3:-EleutherAI/gpt-neo-125M}

n_hidden_layers=12
n_shared_experts=1
n_routed_experts=12
moe_intermediate_size_list=(128)


for moe_intermediate_size in "${moe_intermediate_size_list[@]}"; do
  model_name="DeepSeekV3_${n_hidden_layers}L_${moe_intermediate_size}D"
  echo "Initializing $model_name ..."
  python init_baseline.py \
    --config-json "$CONFIG_JSON" \
    --n-hidden-layers $n_hidden_layers \
    --n-shared-experts $n_shared_experts \
    --n-routed-experts $n_routed_experts \
    --moe-intermediate-size $moe_intermediate_size \
    --output-dir "$OUTPUT_DIR" \
    --model-name "$model_name" \
    --tokenizer-model "$TOKENIZER_MODEL" \
    --bf16 
done
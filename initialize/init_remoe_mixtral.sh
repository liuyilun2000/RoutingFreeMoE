#!/bin/bash

CONFIG_JSON=${1:-remoe_mixtral.config.json}
OUTPUT_DIR=${2:-../config}
TOKENIZER_MODEL=${3:-EleutherAI/gpt-neo-125M}

# S-size params: 12L 512hs 12experts 128dim
num_hidden_layers=12
num_local_experts=12
L1_COEF=0.01
intermediate_size_list=(128)

for intermediate_size in "${intermediate_size_list[@]}"; do
  model_name="ReMoE_${num_hidden_layers}L_${intermediate_size}Dx${num_local_experts}E"
  echo "Initializing $model_name ..."
  python init_remoe_mixtral.py \
    --config-json "$CONFIG_JSON" \
    --num-hidden-layers $num_hidden_layers \
    --num-local-experts $num_local_experts \
    --intermediate-size $intermediate_size \
    --l1-coef $L1_COEF \
    --output-dir "$OUTPUT_DIR" \
    --model-name "$model_name" \
    --tokenizer-model "$TOKENIZER_MODEL" \
    --bf16
done

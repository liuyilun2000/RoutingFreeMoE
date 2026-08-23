#!/bin/bash

CONFIG_JSON=${1:-aoe_mixtral.config.json}
OUTPUT_DIR=${2:-../config}
TOKENIZER_MODEL=${3:-EleutherAI/gpt-neo-125M}
GATE_PROJ_RANK=${4:-32}

# S-size params: 12L 512hs 12experts 128dim top-3
num_hidden_layers=12
num_local_experts=12
num_experts_per_tok=3
intermediate_size_list=(128)

for intermediate_size in "${intermediate_size_list[@]}"; do
  model_name="AoE_${num_hidden_layers}L_${intermediate_size}Dx${num_local_experts}E"
  echo "Initializing $model_name ..."
  python init_aoe_mixtral.py \
    --config-json "$CONFIG_JSON" \
    --num-hidden-layers $num_hidden_layers \
    --num-local-experts $num_local_experts \
    --num-experts-per-tok $num_experts_per_tok \
    --intermediate-size $intermediate_size \
    --gate-proj-rank $GATE_PROJ_RANK \
    --output-dir "$OUTPUT_DIR" \
    --model-name "$model_name" \
    --tokenizer-model "$TOKENIZER_MODEL" \
    --bf16
done

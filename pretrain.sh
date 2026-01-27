#!/bin/bash
#SBATCH --partition=accelerated
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --time=3:59:00
#SBATCH --mail-type=ALL

export WANDB_API_KEY="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"

source /hkfs/home/project/hk-project-p0022189/hgf_mxv5488/miniconda3/bin/activate py310

num_hidden_layers=12
num_attention_heads=16
num_key_value_heads=16
n_experts=12
moe_intermediate_size=128
GATE_TEMPERATURE=1.0
GATE_THRESHOLD=1.0
DENSITY_TARGET=0.25
LAMBDA_COEF=1e-10
#ETA_COEF=0.05
ETA_COEF=0.1
PER_EXPERT_AUX_LOSS_COEF=0.5
PER_TOKEN_AUX_LOSS_COEF=0.5
ORTHOGONALITY_LOSS_COEF=1e-5

config="${num_hidden_layers}L_${moe_intermediate_size}D"
RUN_NAME="test_${config}x${n_experts}E_temp_${GATE_TEMPERATURE}_thres_${GATE_THRESHOLD}_density_${DENSITY_TARGET}_lambda_${LAMBDA_COEF}_eta_${ETA_COEF}_aux_[E${PER_EXPERT_AUX_LOSS_COEF}_T${PER_TOKEN_AUX_LOSS_COEF}]"

MODEL_DIR=${1:-./initNew/RoutingFreeDeepseekV3_${config}}
OUTPUT_DIR=${4:-./output/new/${RUN_NAME}}
DATASET_NAME=${5:-roneneldan/TinyStories}
#DATASET_NAME=${5:-cerebras/SlimPajama-627B}
#PREPROCESSING_CACHE_DIR=${12:-/hkfs/work/workspace/scratch/hgf_mxv5488-slimpajama}
#HF_CACHE_DIR=${13:-/hkfs/work/workspace/scratch/hgf_mxv5488-slimpajama}
EPOCHS=${6:-1}
LR=${7:-5e-4} # 5e-4
BATCH_SIZE=${8:-32} # 32
GRAD_ACCUM=${9:-2} # 2
WANDB_PROJECT=${10:-routing-free-deepseek-v3}
WANDB_RUN=${11:-${RUN_NAME}}

echo "Running with the following parameters:"
echo "RUN_NAME: $RUN_NAME"
echo "MODEL_DIR: $MODEL_DIR"
echo "OUTPUT_DIR: $OUTPUT_DIR"
echo "N_HIDDEN_LAYERS: $num_hidden_layers"
echo "N_EXPERTS: $n_experts"
echo "MOE_INTERMEDIATE_SIZE: $moe_intermediate_size"
echo "GATE_TEMPERATURE: $GATE_TEMPERATURE"
echo "GATE_THRESHOLD: $GATE_THRESHOLD"
echo "DENSITY_TARGET: $DENSITY_TARGET"
echo "LAMBDA_COEF: $LAMBDA_COEF"
echo "ETA_COEF: $ETA_COEF"
echo "PER_EXPERT_AUX_LOSS_COEF: $PER_EXPERT_AUX_LOSS_COEF"
echo "PER_TOKEN_AUX_LOSS_COEF: $PER_TOKEN_AUX_LOSS_COEF"
echo "DATASET_NAME: $DATASET_NAME"
echo "EPOCHS: $EPOCHS"
echo "LR: $LR"
echo "BATCH_SIZE: $BATCH_SIZE"
echo "GRAD_ACCUM: $GRAD_ACCUM"
echo "WANDB_PROJECT: $WANDB_PROJECT"
echo "WANDB_RUN: $WANDB_RUN"
echo "PREPROCESSING_CACHE_DIR: $PREPROCESSING_CACHE_DIR"
echo "HF_CACHE_DIR: $HF_CACHE_DIR"

torchrun --nproc_per_node 1 pretrain.py \
  --model-dir "$MODEL_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --dataset-name "$DATASET_NAME" \
  --num-hidden-layers "$num_hidden_layers" \
  --num-attention-heads "$num_attention_heads" \
  --num-key-value-heads "$num_key_value_heads" \
  --n-experts "$n_experts" \
  --moe-intermediate-size "$moe_intermediate_size" \
  --gate-temperature "$GATE_TEMPERATURE" \
  --gate-threshold "$GATE_THRESHOLD" \
  --density-target "$DENSITY_TARGET" \
  --lambda-coef "$LAMBDA_COEF" \
  --eta-coef "$ETA_COEF" \
  --per-expert-aux-loss-coef "$PER_EXPERT_AUX_LOSS_COEF" \
  --per-token-aux-loss-coef "$PER_TOKEN_AUX_LOSS_COEF" \
  --epochs "$EPOCHS" \
  --lr "$LR" \
  --per_device_batch_size "$BATCH_SIZE" \
  --gradient_accumulation_steps "$GRAD_ACCUM" \
  --wandb-project "$WANDB_PROJECT" \
  --wandb-run "$WANDB_RUN" \
  --bf16 \
  --preprocessing_cache_dir "$PREPROCESSING_CACHE_DIR" \
  $(if [ -n "$HF_CACHE_DIR" ]; then echo "--hf-cache-dir $HF_CACHE_DIR"; fi) \
  --orthogonality-loss-coef "$ORTHOGONALITY_LOSS_COEF"
#!/bin/bash
#SBATCH --partition=accelerated
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --time=12:59:00
#SBATCH --mail-type=NONE

# Set your WANDB_API_KEY as an environment variable before running
# export WANDB_API_KEY="your_key_here"
export WANDB_LOG_MODEL="false"
export WANDB_SILENT="true"

# ── Model size configurations ────────────────────────────────────────────────
# Uncomment the desired size block (S, M, or L)

# S (Small): 12L, 128D, 12 experts, rank 32
num_hidden_layers=12
num_attention_heads=16
num_key_value_heads=16
n_experts=12
intermediate_size=128
GATE_PROJ_RANK=32
LEARNING_RATE=1e-3
BATCH_SIZE=32
GRAD_ACCUM=2

# M (Medium): 24L, 192D, 16 experts, rank 48
# num_hidden_layers=24
# num_attention_heads=24
# num_key_value_heads=12
# n_experts=16
# intermediate_size=192
# GATE_PROJ_RANK=48
# LEARNING_RATE=1e-3
# BATCH_SIZE=16
# GRAD_ACCUM=4

# L (Large): 32L, 256D, 24 experts, rank 64
# num_hidden_layers=32
# num_attention_heads=32
# num_key_value_heads=16
# n_experts=24
# intermediate_size=256
# GATE_PROJ_RANK=64
# LEARNING_RATE=8e-4
# BATCH_SIZE=16
# GRAD_ACCUM=4

# ── Routing-free hyperparameters ─────────────────────────────────────────────
GATE_TEMPERATURE=1.0
GATE_THRESHOLD=1.0
DENSITY_TARGET=0.25
LAMBDA_COEF=1e-10
ETA_COEF=0.02
PER_EXPERT_AUX_LOSS_COEF=0.5
PER_TOKEN_AUX_LOSS_COEF=0.5

# ── Paths ────────────────────────────────────────────────────────────────────
config="${num_hidden_layers}L_${intermediate_size}D_rank${GATE_PROJ_RANK}"
RUN_NAME="rf_${config}_lr_${LEARNING_RATE}"

MODEL_DIR=${1:-./config/RoutingFreeMixtral_${config}}
OUTPUT_DIR=${2:-./output/mixtral_rf/${RUN_NAME}}
DATASET_NAME=${3:-Skylion007/openwebtext}
PREPROCESSING_CACHE_DIR=${4:-./mapped_datasets}
HF_CACHE_DIR=${5:-./hf_cache}

EPOCHS=1
WANDB_PROJECT="routing-free-mixtral"
WANDB_RUN="${RUN_NAME}"

echo "Running with the following parameters:"
echo "RUN_NAME: $RUN_NAME"
echo "MODEL_DIR: $MODEL_DIR"
echo "OUTPUT_DIR: $OUTPUT_DIR"
echo "N_HIDDEN_LAYERS: $num_hidden_layers"
echo "N_EXPERTS: $n_experts"
echo "INTERMEDIATE_SIZE: $intermediate_size"
echo "GATE_TEMPERATURE: $GATE_TEMPERATURE"
echo "GATE_THRESHOLD: $GATE_THRESHOLD"
echo "DENSITY_TARGET: $DENSITY_TARGET"
echo "LAMBDA_COEF: $LAMBDA_COEF"
echo "ETA_COEF: $ETA_COEF"
echo "PER_EXPERT_AUX_LOSS_COEF: $PER_EXPERT_AUX_LOSS_COEF"
echo "PER_TOKEN_AUX_LOSS_COEF: $PER_TOKEN_AUX_LOSS_COEF"
echo "DATASET_NAME: $DATASET_NAME"
echo "EPOCHS: $EPOCHS"
echo "LR: $LEARNING_RATE"
echo "BATCH_SIZE: $BATCH_SIZE"
echo "GRAD_ACCUM: $GRAD_ACCUM"
echo "WANDB_PROJECT: $WANDB_PROJECT"
echo "WANDB_RUN: $WANDB_RUN"
echo "PREPROCESSING_CACHE_DIR: $PREPROCESSING_CACHE_DIR"
echo "HF_CACHE_DIR: $HF_CACHE_DIR"

torchrun --nproc_per_node 4 pretrain_mixtral_rf.py \
  --model-dir "$MODEL_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --dataset-name "$DATASET_NAME" \
  --num-hidden-layers "$num_hidden_layers" \
  --num-attention-heads "$num_attention_heads" \
  --num-key-value-heads "$num_key_value_heads" \
  --n-experts "$n_experts" \
  --intermediate-size "$intermediate_size" \
  --gate-temperature "$GATE_TEMPERATURE" \
  --gate-threshold "$GATE_THRESHOLD" \
  --density-target "$DENSITY_TARGET" \
  --lambda-coef "$LAMBDA_COEF" \
  --eta-coef "$ETA_COEF" \
  --per-expert-aux-loss-coef "$PER_EXPERT_AUX_LOSS_COEF" \
  --per-token-aux-loss-coef "$PER_TOKEN_AUX_LOSS_COEF" \
  --epochs "$EPOCHS" \
  --lr "$LEARNING_RATE" \
  --per_device_batch_size "$BATCH_SIZE" \
  --gradient_accumulation_steps "$GRAD_ACCUM" \
  --wandb-project "$WANDB_PROJECT" \
  --wandb-run "$WANDB_RUN" \
  --bf16 \
  --preprocessing_cache_dir "$PREPROCESSING_CACHE_DIR" \
  $(if [ -n "$HF_CACHE_DIR" ]; then echo "--hf-cache-dir $HF_CACHE_DIR"; fi)

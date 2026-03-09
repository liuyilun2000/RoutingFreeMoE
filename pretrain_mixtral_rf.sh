#!/bin/bash
#SBATCH --partition=accelerated-h100
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --time=24:59:00
#SBATCH --mail-type=NONE

export WANDB_API_KEY="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"
#export WANDB_WATCH="false"
export WANDB_LOG_MODEL="false"
export WANDB_SILENT="true"
#export WANDB_RUN_ID="edw0sl27"
#export WANDB_RESUME="must"

source /hkfs/home/project/hk-project-p0022189/hgf_mxv5488/miniconda3/bin/activate py310

num_hidden_layers=32
num_attention_heads=32
num_key_value_heads=16
n_experts=24
intermediate_size=256

GATE_TEMPERATURE=1.0
GATE_THRESHOLD=1.0
DENSITY_TARGET=0.25
LAMBDA_COEF=1e-10
ETA_COEF=0.02
PER_EXPERT_AUX_LOSS_COEF=0.5
PER_TOKEN_AUX_LOSS_COEF=0.5
LEARNING_RATE=1e-3
GATE_PROJ_RANK=64

config="${num_hidden_layers}L_${intermediate_size}D_rank${GATE_PROJ_RANK}"
RUN_NAME="L_rf_lr_${LEARNING_RATE}_rank_${GATE_PROJ_RANK}"

# Point to initialized Mixtral RF model
#MODEL_DIR=${1:-./init_mixtral_rf/RoutingFreeMixtral_${config}}
MODEL_DIR=${1:-./config/RoutingFreeMixtral_${config}}
OUTPUT_DIR=${4:-./output/mixtral_rf/${RUN_NAME}}
DATASET_NAME=${5:-Skylion007/openwebtext}
#DATASET_NAME=${5:-roneneldan/TinyStories}
#DATASET_NAME=${5:-cerebras/SlimPajama-627B}

# Define Workspace Path
WORKSPACE_DIR="/hkfs/work/workspace/scratch/hgf_mxv5488-myspace"
PREPROCESSING_CACHE_DIR=${12:-${WORKSPACE_DIR}/mapped_datasets}
HF_CACHE_DIR=${13:-${WORKSPACE_DIR}/hf_cache}
EPOCHS=${6:-1}
LR=${7:-$LEARNING_RATE}
BATCH_SIZE=${8:-16} # 32
GRAD_ACCUM=${9:-4} # 2
WANDB_PROJECT=${10:-mixtral-baseline} # routing-free-mixtral
WANDB_RUN=${11:-${RUN_NAME}}

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
echo "LR: $LR"
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
  --lr "$LR" \
  --per_device_batch_size "$BATCH_SIZE" \
  --gradient_accumulation_steps "$GRAD_ACCUM" \
  --wandb-project "$WANDB_PROJECT" \
  --wandb-run "$WANDB_RUN" \
  --bf16 \
  --preprocessing_cache_dir "$PREPROCESSING_CACHE_DIR" \
  $(if [ -n "$HF_CACHE_DIR" ]; then echo "--hf-cache-dir $HF_CACHE_DIR"; fi)

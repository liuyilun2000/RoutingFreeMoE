#!/bin/bash
#SBATCH --partition=accelerated
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --time=12:59:00
#SBATCH --mail-type=NONE

export WANDB_API_KEY="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"
export WANDB_LOG_MODEL="false"
export WANDB_SILENT="true"

source /hkfs/home/project/hk-project-p0022189/hgf_mxv5488/miniconda3/bin/activate py310

num_hidden_layers=12
num_local_experts=12
num_experts_per_tok=3
intermediate_size=128
LEARNING_RATE=1e-3

config="${num_hidden_layers}L_${intermediate_size}Dx${num_local_experts}E"
RUN_NAME="S_aoe_${config}_lr_${LEARNING_RATE}"

MODEL_DIR=${1:-./config/AoE_${config}}
OUTPUT_DIR=${4:-./output_aoe_mixtral/${RUN_NAME}}
DATASET_NAME=${5:-Skylion007/openwebtext}

# Define Workspace Path
WORKSPACE_DIR="/hkfs/work/workspace/scratch/hgf_mxv5488-myspace"
PREPROCESSING_CACHE_DIR=${12:-${WORKSPACE_DIR}/mapped_datasets}
HF_CACHE_DIR=${13:-${WORKSPACE_DIR}/hf_cache}
EPOCHS=${6:-1}
LR=${7:-$LEARNING_RATE}
BATCH_SIZE=${8:-32}
GRAD_ACCUM=${9:-2}
WANDB_PROJECT=${10:-mixtral-baseline}
WANDB_RUN=${11:-${RUN_NAME}}

echo "Running with the following parameters:"
echo "RUN_NAME: $RUN_NAME"
echo "MODEL_DIR: $MODEL_DIR"
echo "OUTPUT_DIR: $OUTPUT_DIR"
echo "NUM_HIDDEN_LAYERS: $num_hidden_layers"
echo "NUM_LOCAL_EXPERTS: $num_local_experts"
echo "INTERMEDIATE_SIZE: $intermediate_size"
echo "DATASET_NAME: $DATASET_NAME"
echo "EPOCHS: $EPOCHS"
echo "LR: $LR"
echo "BATCH_SIZE: $BATCH_SIZE"
echo "GRAD_ACCUM: $GRAD_ACCUM"
echo "WANDB_PROJECT: $WANDB_PROJECT"
echo "WANDB_RUN: $WANDB_RUN"
echo "PREPROCESSING_CACHE_DIR: $PREPROCESSING_CACHE_DIR"
echo "HF_CACHE_DIR: $HF_CACHE_DIR"

torchrun --nproc_per_node 4 pretrain_aoe_mixtral.py \
  --model-dir "$MODEL_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --dataset-name "$DATASET_NAME" \
  --preprocessing_cache_dir "$PREPROCESSING_CACHE_DIR" \
  $(if [ -n "$HF_CACHE_DIR" ]; then echo "--hf-cache-dir $HF_CACHE_DIR"; fi) \
  --num-hidden-layers "$num_hidden_layers" \
  --num-local-experts "$num_local_experts" \
  --num-experts-per-tok "$num_experts_per_tok" \
  --intermediate-size "$intermediate_size" \
  --epochs "$EPOCHS" \
  --lr "$LR" \
  --per_device_batch_size "$BATCH_SIZE" \
  --gradient_accumulation_steps "$GRAD_ACCUM" \
  --wandb-project "$WANDB_PROJECT" \
  --wandb-run "$WANDB_RUN" \
  --bf16

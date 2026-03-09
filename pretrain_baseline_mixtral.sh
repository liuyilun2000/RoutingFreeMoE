#!/bin/bash
#SBATCH --partition=accelerated-h100
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --time=5:59:00
#SBATCH --mail-type=NONE

export WANDB_API_KEY="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"
export WANDB_LOG_MODEL="false"
export WANDB_SILENT="true"
export WANDB_RUN_ID="684t48la"
export WANDB_RESUME="must"

source /hkfs/home/project/hk-project-p0022189/hgf_mxv5488/miniconda3/bin/activate py310

num_hidden_layers=32
num_local_experts=24
num_experts_per_tok=6
intermediate_size=256
LEARNING_RATE=5e-4

config="${num_hidden_layers}L_${intermediate_size}D"
RUN_NAME="L_1_mixtral_baseline_${config}x${num_local_experts}E_top${num_experts_per_tok}_lr_${LEARNING_RATE}"

#MODEL_DIR=${1:-./init_baseline_mixtral/Mixtral_${config}}
MODEL_DIR=${1:-./config/Mixtral_${config}}
OUTPUT_DIR=${4:-./output_baseline_mixtral/${RUN_NAME}}
DATASET_NAME=${5:-Skylion007/openwebtext}
#DATASET_NAME=${5:-roneneldan/TinyStories}
#DATASET_NAME=${5:-cerebras/SlimPajama-627B}

# Define Workspace Path
WORKSPACE_DIR="/hkfs/work/workspace/scratch/hgf_mxv5488-myspace"
PREPROCESSING_CACHE_DIR=${12:-${WORKSPACE_DIR}/mapped_datasets}
HF_CACHE_DIR=${13:-${WORKSPACE_DIR}/hf_cache}
EPOCHS=${6:-1}
LR=${7:-$LEARNING_RATE}
BATCH_SIZE=${8:-16}
GRAD_ACCUM=${9:-4}
WANDB_PROJECT=${10:-mixtral-baseline}
WANDB_RUN=${11:-${RUN_NAME}}

echo "Running with the following parameters:"
echo "RUN_NAME: $RUN_NAME"
echo "MODEL_DIR: $MODEL_DIR"
echo "OUTPUT_DIR: $OUTPUT_DIR"
echo "NUM_HIDDEN_LAYERS: $num_hidden_layers"
echo "NUM_LOCAL_EXPERTS: $num_local_experts"
echo "NUM_EXPERTS_PER_TOK: $num_experts_per_tok"
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

torchrun --nproc_per_node 4 pretrain_baseline_mixtral.py \
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
  --bf16 \
  --resume_from_checkpoint ./output_baseline_mixtral/L_1_mixtral_baseline_32L_256Dx24E_top6_lr_5e-4/checkpoint-29000

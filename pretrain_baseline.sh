#!/bin/bash
#SBATCH --partition=accelerated
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --time=10:00:00
#SBATCH --mail-type=ALL

source /hkfs/home/project/hk-project-p0022189/hgf_mxv5488/miniconda3/bin/activate py310

n_hidden_layers=12
n_shared_experts=0
n_routed_experts=12
n_experts=$((n_shared_experts + n_routed_experts))
moe_intermediate_size=128

config="${n_hidden_layers}L_${moe_intermediate_size}D"
RUN_NAME="1_baseline_${config}x${n_experts}E[${n_shared_experts}S_${n_routed_experts}R]"

MODEL_DIR=${1:-./init_baseline/DeepSeekV3_${config}}
OUTPUT_DIR=${4:-./output_baseline/${RUN_NAME}}
DATASET_NAME=${5:-roneneldan/TinyStories}
#DATASET_NAME=${5:-cerebras/SlimPajama-627B}
#PREPROCESSING_CACHE_DIR=${12:-/hkfs/work/workspace/scratch/hgf_mxv5488-slimpajama}
#HF_CACHE_DIR=${13:-/hkfs/work/workspace/scratch/hgf_mxv5488-slimpajama}
EPOCHS=${6:-1}
LR=${7:-5e-4}
BATCH_SIZE=${8:-32}
GRAD_ACCUM=${9:-1}
WANDB_PROJECT=${10:-routing-free-deepseek-v3}
WANDB_RUN=${11:-${RUN_NAME}}

echo "Running with the following parameters:"
echo "RUN_NAME: $RUN_NAME"
echo "MODEL_DIR: $MODEL_DIR"
echo "OUTPUT_DIR: $OUTPUT_DIR"
echo "N_HIDDEN_LAYERS: $n_hidden_layers"
echo "N_EXPERTS: $n_experts"
echo "MOE_INTERMEDIATE_SIZE: $moe_intermediate_size"
echo "DATASET_NAME: $DATASET_NAME"
echo "EPOCHS: $EPOCHS"
echo "LR: $LR"
echo "BATCH_SIZE: $BATCH_SIZE"
echo "GRAD_ACCUM: $GRAD_ACCUM"
echo "WANDB_PROJECT: $WANDB_PROJECT"
echo "WANDB_RUN: $WANDB_RUN"
echo "PREPROCESSING_CACHE_DIR: $PREPROCESSING_CACHE_DIR"
echo "HF_CACHE_DIR: $HF_CACHE_DIR"

torchrun --nproc_per_node 4 pretrain_baseline.py \
  --model-dir "$MODEL_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --dataset-name "$DATASET_NAME" \
  --preprocessing_cache_dir "$PREPROCESSING_CACHE_DIR" \
  $(if [ -n "$HF_CACHE_DIR" ]; then echo "--hf-cache-dir $HF_CACHE_DIR"; fi) \
  --n-hidden-layers "$n_hidden_layers" \
  --n-shared-experts "$n_shared_experts" \
  --n-routed-experts "$n_routed_experts" \
  --moe-intermediate-size "$moe_intermediate_size" \
  --epochs "$EPOCHS" \
  --lr "$LR" \
  --per_device_batch_size "$BATCH_SIZE" \
  --gradient_accumulation_steps "$GRAD_ACCUM" \
  --wandb-project "$WANDB_PROJECT" \
  --wandb-run "$WANDB_RUN" \
  --bf16 
  #--mlp-iter-layers "${mlp_iter_layers[@]}" \
  #--mlp-iter-times "$mlp_iter_times" \
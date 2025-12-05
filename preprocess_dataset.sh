#!/bin/bash
#SBATCH --partition=large
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=24:00:00
#SBATCH --job-name=preprocess_slimpajama
#SBATCH --output=preprocess_slimpajama.out
#SBATCH --error=preprocess_slimpajama.err
#SBATCH --mail-type=ALL

# Default parameters (can be overridden via command line arguments)
DATASET_NAME=${1:-cerebras/SlimPajama-627B}
TOKENIZER_MODEL=${2:-EleutherAI/gpt-neo-125M}
MAX_LENGTH=${3:-2048}
N_WORKERS=${4:-32}
FORCE_REPROCESS=${5:-false}

# Workspace directories (from workspace_config.py)
RAW_DATASET_DIR=${6:-}
PREPROCESSING_CACHE_DIR=${7:-}
HF_CACHE_DIR=${8:-/hkfs/work/workspace/scratch/hgf_mxv5488-slimpajama/hf_cache}

# Set HuggingFace cache environment variables EARLY (before activating conda/Python)
# This ensures they're available when Python imports transformers/datasets libraries
# CRITICAL: These override default ~/.cache/huggingface behavior
if [ -n "$HF_CACHE_DIR" ]; then
    export TRANSFORMERS_CACHE="$HF_CACHE_DIR"
    export HF_HOME="$HF_CACHE_DIR"
    export HF_HUB_CACHE="$HF_CACHE_DIR"
    export HF_DATASETS_CACHE="$HF_CACHE_DIR"
fi

# Activate conda environment
source /hkfs/home/project/hk-project-p0022189/hgf_mxv5488/miniconda3/bin/activate py310

# Set working directory
cd /home/hk-project-p0022189/hgf_mxv5488/RoutingFreeMoE

echo "=========================================="
echo "Dataset Preprocessing Job Started"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo ""
echo "Configuration:"
echo "  Dataset: $DATASET_NAME"
echo "  Tokenizer: $TOKENIZER_MODEL"
echo "  Max length: $MAX_LENGTH"
echo "  Workers: $N_WORKERS"
echo "  Force reprocess: $FORCE_REPROCESS"
echo "  Raw dataset dir: ${RAW_DATASET_DIR:-<auto>}"
echo "  Preprocessing cache: ${PREPROCESSING_CACHE_DIR:-<auto>}"
echo "  HF cache dir: $HF_CACHE_DIR"
if [ -n "$HF_CACHE_DIR" ]; then
    echo "  Using TRANSFORMERS_CACHE=$TRANSFORMERS_CACHE"
    echo "  Using HF_HOME=$HF_HOME"
    echo "  Using HF_HUB_CACHE=$HF_HUB_CACHE"
    echo "  Using HF_DATASETS_CACHE=$HF_DATASETS_CACHE"
fi
echo "=========================================="
echo ""

# Build command arguments
CMD_ARGS=(
    --dataset-name "$DATASET_NAME"
    --tokenizer-model "$TOKENIZER_MODEL"
    --max-length "$MAX_LENGTH"
    --n-workers "$N_WORKERS"
)

if [ "$FORCE_REPROCESS" = "true" ] || [ "$FORCE_REPROCESS" = "True" ] || [ "$FORCE_REPROCESS" = "1" ]; then
    CMD_ARGS+=(--force-reprocess)
fi

if [ -n "$RAW_DATASET_DIR" ]; then
    CMD_ARGS+=(--raw-dataset-dir "$RAW_DATASET_DIR")
fi

if [ -n "$PREPROCESSING_CACHE_DIR" ]; then
    CMD_ARGS+=(--preprocessing-cache-dir "$PREPROCESSING_CACHE_DIR")
fi

if [ -n "$HF_CACHE_DIR" ]; then
    CMD_ARGS+=(--hf-cache-dir "$HF_CACHE_DIR")
fi

# Run the preprocessing script
echo "Running preprocess_dataset.py..."
echo "Command: python preprocess_dataset.py ${CMD_ARGS[*]}"
echo ""

python preprocess_dataset.py "${CMD_ARGS[@]}"

EXIT_CODE=$?

echo ""
echo "=========================================="
echo "Job Completed"
echo "=========================================="
echo "End time: $(date)"
echo "Exit code: $EXIT_CODE"

if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ Dataset preprocessing completed successfully!"
    echo ""
    echo "Verifying cache..."
    python -c "
import os
from workspace_config import get_preprocessing_cache_dir
cache_dir = get_preprocessing_cache_dir('$DATASET_NAME')
print(f'Cache directory: {cache_dir}')
if os.path.exists(cache_dir):
    contents = os.listdir(cache_dir)
    if contents:
        print(f'Found {len(contents)} cached split(s):')
        for item in contents:
            item_path = os.path.join(cache_dir, item)
            if os.path.isdir(item_path):
                try:
                    from datasets import Dataset
                    dataset = Dataset.load_from_disk(item_path)
                    print(f'  ✓ {item}: {len(dataset):,} samples')
                except Exception as e:
                    print(f'  ✗ {item}: Error loading ({str(e)[:50]})')
    else:
        print('⚠️  Cache directory is empty!')
else:
    print('⚠️  Cache directory does not exist!')
"
else
    echo "✗ Dataset preprocessing failed with exit code $EXIT_CODE"
fi

exit $EXIT_CODE


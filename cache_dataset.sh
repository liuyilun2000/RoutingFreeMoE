#!/bin/bash
#SBATCH --partition=large
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=24:00:00
#SBATCH --job-name=cache_slimpajama
#SBATCH --output=cache_slimpajama.out
#SBATCH --error=cache_slimpajama.err
#SBATCH --mail-type=ALL

# Activate conda environment
source /hkfs/home/project/hk-project-p0022189/hgf_mxv5488/miniconda3/bin/activate py310

# Set working directory
cd /home/hk-project-p0022189/hgf_mxv5488/RoutingFreeMoE

# Default parameters (can be overridden via command line arguments)
#DATASET_NAME=${1:-roneneldan/TinyStories}
DATASET_NAME=${1:-cerebras/SlimPajama-627B}
TOKENIZER_MODEL=${2:-EleutherAI/gpt-neo-125M}
MAX_LENGTH=${3:-2048}
N_WORKERS=${4:-32}
FORCE_REPROCESS=${5:-false}

# Workspace directories (from workspace_config.py)
PREPROCESSING_CACHE_DIR=${6:-/hkfs/work/workspace/scratch/hgf_mxv5488-slimpajama}
HF_CACHE_DIR=${7:-/hkfs/work/workspace/scratch/hgf_mxv5488-slimpajama}

#HF_CACHE_DIR=${7:-/hkfs/work/workspace/scratch/hgf_mxv5488-slimpajama/hf_cache}

echo "=========================================="
echo "Dataset Caching Job Started"
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
echo "  Preprocessing cache: $PREPROCESSING_CACHE_DIR"
echo "  HF cache: $HF_CACHE_DIR"
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

# Only add preprocessing-cache-dir if explicitly provided
# Otherwise, let cache_dataset.py use workspace_config default
if [ -n "$PREPROCESSING_CACHE_DIR" ]; then
    CMD_ARGS+=(--preprocessing-cache-dir "$PREPROCESSING_CACHE_DIR")
fi

if [ -n "$HF_CACHE_DIR" ]; then
    CMD_ARGS+=(--hf-cache-dir "$HF_CACHE_DIR")
fi

# Run the caching script
echo "Running cache_dataset.py..."
echo "Command: python cache_dataset.py ${CMD_ARGS[*]}"
echo ""

python cache_dataset.py "${CMD_ARGS[@]}"

EXIT_CODE=$?

echo ""
echo "=========================================="
echo "Job Completed"
echo "=========================================="
echo "End time: $(date)"
echo "Exit code: $EXIT_CODE"

if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ Dataset caching completed successfully!"
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
    echo "✗ Dataset caching failed with exit code $EXIT_CODE"
fi

exit $EXIT_CODE


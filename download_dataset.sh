#!/bin/bash
# Script to download dataset on login node (better internet)
# Usage: bash download_dataset.sh [dataset_name] [hf_cache_dir] [raw_dataset_dir]

# Default parameters (can be overridden via command line arguments)
DATASET_NAME=${1:-cerebras/SlimPajama-627B}
HF_CACHE_DIR=${2:-/hkfs/work/workspace/scratch/hgf_mxv5488-slimpajama/hf_cache}

# Set HuggingFace cache environment variables EARLY (before activating conda/Python)
# This ensures they're available when Python imports the datasets library
if [ -n "$HF_CACHE_DIR" ]; then
    export HF_DATASETS_CACHE="$HF_CACHE_DIR"
    export HF_HUB_CACHE="$HF_CACHE_DIR"
    export HF_HOME="$HF_CACHE_DIR"
fi

# Activate conda environment
source /hkfs/home/project/hk-project-p0022189/hgf_mxv5488/miniconda3/bin/activate py310

# Set working directory
cd /home/hk-project-p0022189/hgf_mxv5488/RoutingFreeMoE
RAW_DATASET_DIR=${3:-}
FORCE_REDOWNLOAD=${4:-false}
HF_TOKEN=${5:-${HF_TOKEN:-}}  # Use argument or environment variable

# Load .env file if it exists (in same directory as script)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${SCRIPT_DIR}/.env"
if [ -f "$ENV_FILE" ]; then
    echo "Loading .env file from: $ENV_FILE"
    # Source the .env file to load variables (only if HF_TOKEN not already set)
    if [ -z "$HF_TOKEN" ]; then
        set -a  # Automatically export all variables
        source "$ENV_FILE" 2>/dev/null || true
        set +a  # Stop automatically exporting
        # Get HF_TOKEN from environment if it was loaded
        HF_TOKEN="${HF_TOKEN:-}"
    fi
fi

echo "=========================================="
echo "Dataset Download (Login Node)"
echo "=========================================="
echo "Start time: $(date)"
echo ""
echo "Configuration:"
echo "  Dataset: $DATASET_NAME"
echo "  HF cache: $HF_CACHE_DIR"
echo "  Raw dataset dir: ${RAW_DATASET_DIR:-<auto>}"
echo "  Force redownload: $FORCE_REDOWNLOAD"
if [ -n "$HF_TOKEN" ]; then
    echo "  HF token: [SET]"
else
    echo "  HF token: [NOT SET - consider setting for higher rate limits]"
fi
echo "=========================================="
echo ""

# Build command arguments
CMD_ARGS=(
    --dataset-name "$DATASET_NAME"
)

if [ -n "$HF_CACHE_DIR" ]; then
    CMD_ARGS+=(--hf-cache-dir "$HF_CACHE_DIR")
    # Environment variables are already set above (before conda activation)
    # This ensures they're available when Python imports the datasets library
    echo "  Using HF_DATASETS_CACHE=$HF_DATASETS_CACHE"
    echo "  Using HF_HUB_CACHE=$HF_HUB_CACHE"
    echo "  Using HF_HOME=$HF_HOME"
fi

if [ -n "$RAW_DATASET_DIR" ]; then
    CMD_ARGS+=(--raw-dataset-dir "$RAW_DATASET_DIR")
fi

if [ "$FORCE_REDOWNLOAD" = "true" ] || [ "$FORCE_REDOWNLOAD" = "True" ] || [ "$FORCE_REDOWNLOAD" = "1" ]; then
    CMD_ARGS+=(--force-redownload)
fi

if [ -n "$HF_TOKEN" ]; then
    CMD_ARGS+=(--hf-token "$HF_TOKEN")
fi

# Run the download script
echo "Running download_dataset.py..."
echo "Command: python download_dataset.py ${CMD_ARGS[*]}"
echo ""

python download_dataset.py "${CMD_ARGS[@]}"

EXIT_CODE=$?

echo ""
echo "=========================================="
echo "Download Completed"
echo "=========================================="
echo "End time: $(date)"
echo "Exit code: $EXIT_CODE"

if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ Dataset download completed successfully!"
    echo ""
    echo "Next step: Run preprocess_dataset.sh on CPU nodes to preprocess the dataset"
else
    echo "✗ Dataset download failed with exit code $EXIT_CODE"
fi

exit $EXIT_CODE


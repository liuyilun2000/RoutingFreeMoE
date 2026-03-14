"""
Workspace configuration file for RoutingFreeMoE project.
Centralizes all workspace and cache directory paths.
"""

import os
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent

# Cache directories (all within project root)
PREPROCESSING_CACHE_DIR = os.path.join(PROJECT_ROOT, "mapped_datasets")

# Model output directories
MODEL_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
MODEL_BASELINE_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output_baseline")

# Initialize directories if they don't exist
os.makedirs(PREPROCESSING_CACHE_DIR, exist_ok=True)
os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_BASELINE_OUTPUT_DIR, exist_ok=True)


def get_preprocessing_cache_dir(dataset_name: str = None) -> str:
    """
    Get the preprocessing cache directory for a dataset.
    
    Args:
        dataset_name: Name of the dataset (optional)
        
    Returns:
        Path to the preprocessing cache directory
    """
    if dataset_name:
        # Normalize dataset name (replace / with _)
        dataset_name = dataset_name.replace("/", "_")
        return os.path.join(PREPROCESSING_CACHE_DIR, dataset_name)
    return PREPROCESSING_CACHE_DIR


def print_workspace_config():
    """Print current workspace configuration."""
    print("=== Workspace Configuration ===")
    print(f"PROJECT_ROOT: {PROJECT_ROOT}")
    print(f"PREPROCESSING_CACHE_DIR: {PREPROCESSING_CACHE_DIR}")
    print(f"MODEL_OUTPUT_DIR: {MODEL_OUTPUT_DIR}")
    print(f"MODEL_BASELINE_OUTPUT_DIR: {MODEL_BASELINE_OUTPUT_DIR}")
    print("===============================")


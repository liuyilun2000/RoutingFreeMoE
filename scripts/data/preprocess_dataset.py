#!/usr/bin/env python3
"""
Preprocess downloaded dataset (for CPU nodes with better CPU and memory).
This script loads a previously downloaded dataset and preprocesses it.
"""

import argparse
import os
from multiprocessing import cpu_count

from datasets import load_dataset, Dataset
from transformers import AutoTokenizer

# Import unified functions from train_utils
from train_utils import preprocess_function_factory, preprocess_and_cache_dataset

# Import workspace config
try:
    from workspace_config import (
        WORKSPACE_DIR,
        HF_CACHE_DIR,
        PREPROCESSING_CACHE_DIR,
        get_preprocessing_cache_dir,
        print_workspace_config
    )
except ImportError:
    # Fallback if workspace_config not available
    WORKSPACE_DIR = "/hkfs/work/workspace/scratch/hgf_mxv5488-slimpajama"
    HF_CACHE_DIR = os.path.join(WORKSPACE_DIR, "hf_cache")
    PREPROCESSING_CACHE_DIR = os.path.join(WORKSPACE_DIR, "mapped_datasets")
    
    def get_preprocessing_cache_dir(dataset_name: str = None) -> str:
        if dataset_name:
            dataset_name = dataset_name.replace("/", "_")
            return os.path.join(PREPROCESSING_CACHE_DIR, dataset_name)
        return PREPROCESSING_CACHE_DIR
    
    def print_workspace_config():
        print("Using fallback workspace config")


def preprocess_downloaded_dataset(
    dataset_name: str,
    raw_dataset_dir: str = None,
    preprocessing_cache_dir: str = None,
    tokenizer_model: str = "EleutherAI/gpt-neo-125M",
    max_length: int = 2048,
    n_workers: int = None,
    force_reprocess: bool = False,
    hf_cache_dir: str = None,
):
    """
    Preprocess a previously downloaded dataset.
    
    Args:
        dataset_name (str): The name of the dataset (for determining cache paths).
        raw_dataset_dir (str): Directory containing the raw downloaded dataset.
                              If None, uses workspace/mapped_datasets/{dataset_name}/raw
        preprocessing_cache_dir (str): Directory to cache the preprocessed dataset.
                                      If None, uses workspace config default.
        tokenizer_model (str): The Hugging Face model name for the tokenizer.
        max_length (int): The maximum sequence length for tokenization.
        n_workers (int): The number of processes to use for multiprocessing.
        force_reprocess (bool): Force reprocessing even if cache exists.
        hf_cache_dir (str): Directory for HuggingFace caches (tokenizers, models).
                           If None, uses workspace config default.
    """
    # Print workspace configuration
    print_workspace_config()
    
    # Set environment variables EARLY to redirect ALL HuggingFace caching to workspace
    # This ensures transformers library (tokenizers, models) cache to workspace, not ~
    if hf_cache_dir is None:
        hf_cache_dir = HF_CACHE_DIR
    
    # CRITICAL: Set these BEFORE importing or using any transformers/HF libraries
    # TRANSFORMERS_CACHE: Controls where transformers library caches models/tokenizers
    # HF_HOME: Root directory for all HuggingFace caches (fallback)
    # HF_HUB_CACHE: Controls where huggingface_hub caches models/tokenizers
    # These override default ~/.cache/huggingface behavior
    os.environ["TRANSFORMERS_CACHE"] = hf_cache_dir
    os.environ["HF_HOME"] = hf_cache_dir
    os.environ["HF_HUB_CACHE"] = hf_cache_dir
    # Also set datasets cache (already set, but ensure consistency)
    os.environ["HF_DATASETS_CACHE"] = hf_cache_dir
    
    print(f"\nSet caching environment variables to use workspace:")
    print(f"  TRANSFORMERS_CACHE={hf_cache_dir}")
    print(f"  HF_HOME={hf_cache_dir}")
    print(f"  HF_HUB_CACHE={hf_cache_dir}")
    print(f"  HF_DATASETS_CACHE={hf_cache_dir}")
    
    # Determine raw dataset directory
    if raw_dataset_dir is None:
        dataset_cache_dir = get_preprocessing_cache_dir(dataset_name)
        raw_dataset_dir = os.path.join(dataset_cache_dir, "raw")
    
    # Use workspace config defaults if not provided
    if preprocessing_cache_dir is None:
        preprocessing_cache_dir = get_preprocessing_cache_dir(dataset_name)
    else:
        preprocessing_cache_dir = os.path.join(preprocessing_cache_dir, dataset_name.replace("/", "_"))
    
    print(f"\nUsing raw dataset: {raw_dataset_dir}")
    print(f"Using preprocessing cache: {preprocessing_cache_dir}")
    
    # Check if raw dataset exists
    if not os.path.exists(raw_dataset_dir):
        raise ValueError(f"Raw dataset directory does not exist: {raw_dataset_dir}\n"
                        f"Please run download_dataset.py first to download the dataset.")
    
    # Find available splits in raw dataset
    raw_contents = os.listdir(raw_dataset_dir)
    available_splits = [item for item in raw_contents if os.path.isdir(os.path.join(raw_dataset_dir, item))]
    
    if not available_splits:
        raise ValueError(f"No dataset splits found in {raw_dataset_dir}")
    
    print(f"\nFound {len(available_splits)} split(s): {available_splits}")
    
    # Get tokenizer - explicitly use cache_dir to ensure workspace caching
    print(f"\nLoading tokenizer: {tokenizer_model}")
    print(f"Tokenizer cache directory: {hf_cache_dir}")
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_model,
        cache_dir=hf_cache_dir  # Explicitly cache tokenizer to workspace
    )
    tokenizer.pad_token = tokenizer.eos_token
    
    # Create preprocessing function
    preprocess_function = preprocess_function_factory(tokenizer, max_length)
    
    # Set number of workers
    if n_workers is None:
        n_workers = max(1, int(cpu_count()))
    
    # Process each available split
    processed_datasets = {}
    
    for split_name in available_splits:
        print(f"\n{'='*60}")
        print(f"Processing {split_name} split...")
        print(f"{'='*60}")
        
        # Load raw dataset from disk
        split_path = os.path.join(raw_dataset_dir, split_name)
        print(f"Loading raw dataset from: {split_path}")
        split_dataset = Dataset.load_from_disk(split_path)
        
        print(f"Loaded {len(split_dataset):,} samples")
        
        # Preprocess and cache
        processed_split = preprocess_and_cache_dataset(
            dataset=split_dataset,
            cache_dir=preprocessing_cache_dir,
            split_name=split_name,
            preprocess_fn=preprocess_function,
            num_proc=n_workers,
            force_reprocess=force_reprocess
        )
        
        processed_datasets[split_name] = processed_split
        print(f"\n{split_name.capitalize()} dataset size: {len(processed_split):,}")
        print(f"Cached at: {os.path.join(preprocessing_cache_dir, f'processed_{split_name}')}")
    
    print(f"\n{'='*60}")
    print(f"Dataset preprocessing complete!")
    print(f"{'='*60}")
    print(f"All splits cached at: {preprocessing_cache_dir}")
    
    # Print summary
    print(f"\nSummary:")
    for split_name, split_data in processed_datasets.items():
        print(f"  {split_name}: {len(split_data):,} samples")
    
    return processed_datasets


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess a previously downloaded dataset (for CPU nodes)."
    )
    parser.add_argument(
        "--dataset-name", 
        type=str, 
        default="cerebras/SlimPajama-627B",
        help="Dataset name from Hugging Face Hub. Default: cerebras/SlimPajama-627B"
    )
    parser.add_argument(
        "--raw-dataset-dir", 
        type=str, 
        default=None,
        help="Directory containing the raw downloaded dataset. "
             "If not provided, uses workspace/mapped_datasets/{dataset_name}/raw"
    )
    parser.add_argument(
        "--preprocessing-cache-dir", 
        type=str, 
        default=None,
        help="Directory to cache the preprocessed dataset. If not provided, uses workspace config."
    )
    parser.add_argument(
        "--tokenizer-model", 
        type=str, 
        default="EleutherAI/gpt-neo-125M",
        help="Hugging Face model name for the tokenizer."
    )
    parser.add_argument(
        "--max-length", 
        type=int, 
        default=2048,
        help="Maximum sequence length for tokenization."
    )
    parser.add_argument(
        "--n-workers", 
        type=int, 
        default=None,
        help="Number of workers for dataset processing. If not provided, uses CPU count."
    )
    parser.add_argument(
        "--force-reprocess", 
        action="store_true",
        help="Force reprocessing even if cache exists."
    )
    parser.add_argument(
        "--hf-cache-dir", 
        type=str, 
        default=None,
        help="Directory for HuggingFace caches (tokenizers, models). If not provided, uses workspace config."
    )
    
    args = parser.parse_args()
    
    preprocess_downloaded_dataset(
        dataset_name=args.dataset_name,
        raw_dataset_dir=args.raw_dataset_dir,
        preprocessing_cache_dir=args.preprocessing_cache_dir,
        tokenizer_model=args.tokenizer_model,
        max_length=args.max_length,
        n_workers=args.n_workers,
        force_reprocess=args.force_reprocess,
        hf_cache_dir=args.hf_cache_dir,
    )


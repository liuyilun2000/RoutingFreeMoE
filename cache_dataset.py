import argparse
import os
os.environ["HF_HUB_ETAG_TIMEOUT"] = "500"

from os.path import join
from typing import Optional
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

def download_and_cache_dataset(
    dataset_name: str = None,
    preprocessing_cache_dir: str = None,
    hf_cache_dir: str = None,
    tokenizer_model: str = "EleutherAI/gpt-neo-125M",
    max_length: int = 2048,
    n_workers: int = None,
    force_reprocess: bool = False,
):
    """
    Downloads, preprocesses, and caches a specified dataset using original splits.
    Now uses unified cache format compatible with training scripts.

    Args:
        dataset_name (str): The name of the dataset to download.
        preprocessing_cache_dir (str): The directory to cache the preprocessed dataset.
                                      If None, uses workspace config default.
        hf_cache_dir (str): Directory for Hugging Face dataset cache.
                           If None, uses workspace config default.
        tokenizer_model (str): The Hugging Face model name for the tokenizer.
        max_length (int): The maximum sequence length for tokenization.
        n_workers (int): The number of processes to use for multiprocessing.
        force_reprocess (bool): Force reprocessing even if cache exists.
    """
    # Use default dataset name if not provided
    if dataset_name is None:
        dataset_name = "roneneldan/TinyStories"  # Default to TinyStories for testing
    
    # Use workspace config defaults if not provided
    if preprocessing_cache_dir is None:
        preprocessing_cache_dir = get_preprocessing_cache_dir(dataset_name)
    else:
        preprocessing_cache_dir = os.path.join(preprocessing_cache_dir, dataset_name.replace("/", "_"))
    
    if hf_cache_dir is None:
        hf_cache_dir = HF_CACHE_DIR
    
    # Print workspace configuration
    print_workspace_config()
    print(f"\nUsing preprocessing cache: {preprocessing_cache_dir}")
    print(f"Using HF cache: {hf_cache_dir}")
    
    # Get tokenizer
    print(f"\nLoading tokenizer: {tokenizer_model}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_model)
    tokenizer.pad_token = tokenizer.eos_token

    # Load dataset
    print(f"\nLoading dataset: {dataset_name}")
    load_dataset_kwargs = {
        "streaming": True,
        "download_mode": "reuse_cache_if_exists"
    }
    if hf_cache_dir is not None:
        load_dataset_kwargs["cache_dir"] = hf_cache_dir
    dataset = load_dataset(dataset_name, **load_dataset_kwargs)
    
    # Print available splits
    available_splits = list(dataset.keys())
    print(f"Available splits: {available_splits}")
    
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
        split_dataset = dataset[split_name]
        
        processed_split = preprocess_and_cache_dataset(
            dataset=split_dataset,
            cache_dir=preprocessing_cache_dir,
            split_name=split_name,
            preprocess_fn=preprocess_function,
            num_proc=n_workers,
            force_reprocess=force_reprocess
        )
        
        processed_datasets[split_name] = processed_split
        print(f"\n{split_name.capitalize()} dataset size: {len(processed_split)}")
        print(f"Cached at: {os.path.join(preprocessing_cache_dir, f'processed_{split_name}')}")
    
    print(f"\n{'='*60}")
    print(f"Dataset download and caching complete!")
    print(f"{'='*60}")
    print(f"All splits cached at: {preprocessing_cache_dir}")
    
    # Print summary
    print(f"\nSummary:")
    for split_name, split_data in processed_datasets.items():
        print(f"  {split_name}: {len(split_data):,} samples")
    
    return processed_datasets

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and cache a dataset for training.")
    parser.add_argument("--dataset-name", type=str, default="roneneldan/TinyStories",
                        help="Dataset name from Hugging Face Hub. Default: roneneldan/TinyStories")
    parser.add_argument("--preprocessing-cache-dir", type=str, default=None,
                        help="Directory to cache the preprocessed dataset. If not provided, uses workspace config.")
    parser.add_argument("--hf-cache-dir", type=str, default=None,
                        help="Directory for Hugging Face dataset cache. If not provided, uses workspace config.")
    parser.add_argument("--tokenizer-model", type=str, default="EleutherAI/gpt-neo-125M",
                        help="Hugging Face model name for the tokenizer.")
    parser.add_argument("--max-length", type=int, default=2048,
                        help="Maximum sequence length for tokenization.")
    parser.add_argument("--n-workers", type=int, default=None,
                        help="Number of workers for dataset processing. If not provided, uses CPU count.")
    parser.add_argument("--force-reprocess", action="store_true",
                        help="Force reprocessing even if cache exists.")
    
    args = parser.parse_args()
    
    download_and_cache_dataset(
        dataset_name=args.dataset_name,
        preprocessing_cache_dir=args.preprocessing_cache_dir,
        hf_cache_dir=args.hf_cache_dir,
        tokenizer_model=args.tokenizer_model,
        max_length=args.max_length,
        n_workers=args.n_workers,
        force_reprocess=args.force_reprocess,
    )
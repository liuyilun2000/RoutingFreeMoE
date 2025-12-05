#!/usr/bin/env python3
"""
Download dataset to workspace (for login node with better internet).
This script only downloads the dataset without preprocessing.
Preprocessing should be done separately on CPU nodes.
"""

import argparse
import os
import sys
import time
import re
from pathlib import Path
os.environ["HF_HUB_ETAG_TIMEOUT"] = "500"

# Note: HF_HUB_CACHE will be set in download_dataset() function
# before using HuggingFace libraries to redirect all caching

from datasets import load_dataset, DownloadMode, Dataset

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


def load_env_file(env_path: str = None) -> dict:
    """
    Load environment variables from .env file.
    
    Args:
        env_path: Path to .env file. If None, looks for .env in current directory.
    
    Returns:
        Dictionary of environment variables
    """
    if env_path is None:
        # Look for .env in the script's directory
        script_dir = Path(__file__).parent
        env_path = script_dir / ".env"
    else:
        env_path = Path(env_path)
    
    env_vars = {}
    if env_path.exists():
        try:
            with open(env_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    # Skip empty lines and comments
                    if not line or line.startswith('#'):
                        continue
                    # Match KEY="VALUE" or KEY='VALUE' or KEY=VALUE
                    # Handle quoted values (with or without quotes)
                    if '=' in line:
                        parts = line.split('=', 1)
                        key = parts[0].strip()
                        value = parts[1].strip()
                        # Remove surrounding quotes if present
                        if (value.startswith('"') and value.endswith('"')) or \
                           (value.startswith("'") and value.endswith("'")):
                            value = value[1:-1]
                        env_vars[key] = value
        except Exception as e:
            print(f"Warning: Could not load .env file: {e}", file=sys.stderr)
    
    return env_vars


def download_dataset(
    dataset_name: str,
    hf_cache_dir: str = None,
    raw_dataset_dir: str = None,
    force_redownload: bool = False,
    hf_token: str = None,
):
    """
    Download dataset to disk without preprocessing.
    
    Args:
        dataset_name (str): The name of the dataset to download.
        hf_cache_dir (str): Directory for Hugging Face dataset cache.
        raw_dataset_dir (str): Directory to save the raw downloaded dataset.
                              If None, uses workspace/mapped_datasets/{dataset_name}/raw
        force_redownload (bool): Force re-download even if dataset exists.
    """
    # Print workspace configuration
    print_workspace_config()
    
    if hf_cache_dir is None:
        hf_cache_dir = HF_CACHE_DIR
    
    # Set environment variables to redirect ALL HuggingFace caching to workspace
    # HF_DATASETS_CACHE: Controls where datasets library caches dataset files (CRITICAL for datasets)
    # HF_HUB_CACHE: Controls where huggingface_hub caches models/tokenizers
    # HF_HOME: Root directory for all HuggingFace caches (fallback)
    # Note: These should ideally be set before importing, but setting here ensures they're used
    os.environ["HF_DATASETS_CACHE"] = hf_cache_dir
    os.environ["HF_HUB_CACHE"] = hf_cache_dir
    os.environ["HF_HOME"] = hf_cache_dir
    
    print(f"Set HF_DATASETS_CACHE={hf_cache_dir}")
    print(f"Set HF_HUB_CACHE={hf_cache_dir}")
    print(f"Set HF_HOME={hf_cache_dir}")
    
    # Determine raw dataset directory
    if raw_dataset_dir is None:
        dataset_cache_dir = get_preprocessing_cache_dir(dataset_name)
        raw_dataset_dir = os.path.join(dataset_cache_dir, "raw")
    
    os.makedirs(raw_dataset_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Downloading Dataset: {dataset_name}")
    print(f"{'='*60}")
    print(f"HF Cache Directory: {hf_cache_dir}")
    print(f"Raw Dataset Directory: {raw_dataset_dir}")
    print(f"Force Redownload: {force_redownload}")
    print()
    
    # Check if dataset already exists
    if not force_redownload and os.path.exists(raw_dataset_dir):
        contents = os.listdir(raw_dataset_dir)
        if contents:
            print(f"Dataset already exists at {raw_dataset_dir}")
            print(f"Found {len(contents)} split(s): {contents}")
            print("Use --force-redownload to re-download")
            return raw_dataset_dir
    
    # Load dataset in non-streaming mode to download it
    print(f"Loading dataset (non-streaming mode)...")
    
    # Use valid DownloadMode enum values
    download_mode = DownloadMode.FORCE_REDOWNLOAD if force_redownload else DownloadMode.REUSE_CACHE_IF_EXISTS
    
    load_dataset_kwargs = {
        "streaming": False,  # Non-streaming to download full dataset
        "download_mode": download_mode
    }
    if hf_cache_dir is not None:
        load_dataset_kwargs["cache_dir"] = hf_cache_dir
    if hf_token is not None:
        load_dataset_kwargs["token"] = hf_token
        os.environ["HF_TOKEN"] = hf_token
    
    # Retry logic for rate limiting - unlimited retries with 180 second wait
    retry_delay = 180  # Wait 180 seconds between retries
    attempt = 0
    
    while True:
        try:
            dataset = load_dataset(dataset_name, **load_dataset_kwargs)
            break  # Success, exit retry loop
        except Exception as e:
            error_str = str(e)
            if "429" in error_str or "Too Many Requests" in error_str or "rate limit" in error_str.lower():
                attempt += 1
                print(f"\n⚠️  Rate limit hit (attempt {attempt})")
                print(f"Waiting {retry_delay} seconds before retry...")
                print("(Unlimited retries - will continue until download completes)")
                time.sleep(retry_delay)
                continue
            else:
                # Other errors - don't retry
                print(f"Error loading dataset: {e}")
                raise
    
    # Print available splits
    available_splits = list(dataset.keys())
    print(f"Available splits: {available_splits}")
    
    # Save each split to disk
    downloaded_splits = {}
    for split_name in available_splits:
        print(f"\n{'='*60}")
        print(f"Downloading {split_name} split...")
        print(f"{'='*60}")
        
        split_dataset = dataset[split_name]
        split_dir = os.path.join(raw_dataset_dir, split_name)
        
        print(f"Saving {split_name} split to: {split_dir}")
        print(f"Dataset size: {len(split_dataset):,} samples")
        
        # Save to disk
        split_dataset.save_to_disk(split_dir)
        
        downloaded_splits[split_name] = split_dir
        print(f"✓ {split_name.capitalize()} split saved successfully")
    
    print(f"\n{'='*60}")
    print(f"Dataset download complete!")
    print(f"{'='*60}")
    print(f"Raw dataset saved at: {raw_dataset_dir}")
    
    # Print summary
    print(f"\nSummary:")
    for split_name, split_path in downloaded_splits.items():
        # Get size info
        try:
            split_dataset = Dataset.load_from_disk(split_path)
            print(f"  {split_name}: {len(split_dataset):,} samples")
        except:
            print(f"  {split_name}: saved successfully")
        print(f"    Location: {split_path}")
    
    return raw_dataset_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download dataset to workspace (for login node with better internet). "
                    "This only downloads, preprocessing should be done separately."
    )
    parser.add_argument(
        "--dataset-name", 
        type=str, 
        default="cerebras/SlimPajama-627B",
        help="Dataset name from Hugging Face Hub. Default: cerebras/SlimPajama-627B"
    )
    parser.add_argument(
        "--hf-cache-dir", 
        type=str, 
        default=None,
        help="Directory for Hugging Face dataset cache. If not provided, uses workspace config."
    )
    parser.add_argument(
        "--raw-dataset-dir", 
        type=str, 
        default=None,
        help="Directory to save the raw downloaded dataset. "
             "If not provided, uses workspace/mapped_datasets/{dataset_name}/raw"
    )
    parser.add_argument(
        "--force-redownload", 
        action="store_true",
        help="Force re-download even if dataset exists."
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default=None,
        help="Hugging Face token for higher rate limits. Can also set HF_TOKEN environment variable."
    )
    
    args = parser.parse_args()
    
    # Load .env file first
    env_vars = load_env_file()
    if env_vars:
        # Update environment variables from .env file
        for key, value in env_vars.items():
            if key not in os.environ:
                os.environ[key] = value
    
    # Get token from argument, environment variable, or .env file
    hf_token = args.hf_token or os.environ.get("HF_TOKEN")
    
    download_dataset(
        dataset_name=args.dataset_name,
        hf_cache_dir=args.hf_cache_dir,
        raw_dataset_dir=args.raw_dataset_dir,
        force_redownload=args.force_redownload,
        hf_token=hf_token,
    )


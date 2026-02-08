import os
from datasets import load_dataset
import argparse

def download_data(dataset_name, cache_dir):
    print(f"Downloading {dataset_name} to {cache_dir}...")
    os.makedirs(cache_dir, exist_ok=True)
    
    # helper to force download
    load_dataset(dataset_name, cache_dir=cache_dir)
    print("Download complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="Skylion007/openwebtext")
    # Default to the workspace you specified
    parser.add_argument("--cache_dir", type=str, default="/hkfs/work/workspace/scratch/hgf_mxv5488-myspace/hf_cache")
    args = parser.parse_args()
    
    download_data(args.dataset_name, args.cache_dir)

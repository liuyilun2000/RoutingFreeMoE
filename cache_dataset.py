import argparse
import os
os.environ["HF_HUB_ETAG_TIMEOUT"] = "500"

from os.path import join
from typing import Optional

from datasets import load_dataset
from transformers import AutoTokenizer




def preprocess_function_factory(tokenizer, max_length):
    """
    Factory to create a preprocessing function for the dataset.
    """
    def preprocess_function(examples):
        outputs = tokenizer(
            examples['text'],
            truncation=True,
            max_length=max_length,
            padding='max_length',
            return_tensors='pt'
        )
        return {
            'input_ids': [x.tolist() for x in outputs['input_ids']],
            'attention_mask': [x.tolist() for x in outputs['attention_mask']],
            'labels': [x.tolist() for x in outputs['input_ids']],
        }
    return preprocess_function

def preprocess_and_cache_dataset(dataset, cache_dir, split_name, preprocess_fn, num_proc: int = 32):
    """
    Preprocesses a dataset split and caches it to a specified directory.
    If the cached dataset exists, it loads it instead of re-processing.
    """
    cache_path = os.path.join(cache_dir, split_name)
    if os.path.exists(cache_path):
        print(f"Loading cached dataset from {cache_path}")
        return dataset.load_from_disk(cache_path)
    else:
        print(f"Preprocessing and caching dataset to {cache_path}")
        processed_dataset = dataset.map(
            preprocess_fn,
            batched=True,
            remove_columns=['text']
        )
        processed_dataset.save_to_disk(cache_path)
        return processed_dataset

def download_and_cache_dataset(
    dataset_name: str = "cerebras/SlimPajama-627B",
    preprocessing_cache_dir: str = "/hkfs/work/workspace/scratch/hgf_mxv5488-myspace/mapped_datasets",  # Updated to use workspace
    tokenizer_model: str = "meta-llama/Meta-Llama-3-8B",
    max_length: int = 2048,
    n_workers: int = 4,
):
    """
    Downloads, preprocesses, and caches a specified dataset using original splits.

    Args:
        dataset_name (str): The name of the dataset to download.
        preprocessing_cache_dir (str): The directory to cache the preprocessed dataset.
        tokenizer_model (str): The Hugging Face model name for the tokenizer.
        max_length (int): The maximum sequence length for tokenization.
        n_workers (int): The number of processes to use for multiprocessing.
    """
    # Get tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_model)
    tokenizer.pad_token = tokenizer.eos_token

    # Load dataset
    print(f"Loading dataset: {dataset_name}")
    dataset = load_dataset(dataset_name, #num_proc=n_workers, 
                           streaming=True,
                           download_mode="reuse_cache_if_exists")
    
    # Print available splits
    available_splits = list(dataset.keys())
    print(f"Available splits: {available_splits}")
    
    # Prepare preprocessing cache directory
    preprocessing_cache = os.path.join(preprocessing_cache_dir, dataset_name.replace("/", "_"))
    preprocess_function = preprocess_function_factory(tokenizer, max_length)
    
    # Process each available split
    processed_datasets = {}
    
    for split_name in available_splits:
        print(f"\nProcessing {split_name} split...")
        split_dataset = dataset[split_name]
        
        processed_split = preprocess_and_cache_dataset(
            dataset=split_dataset,
            cache_dir=preprocessing_cache,
            split_name=split_name,
            preprocess_fn=preprocess_function,
            num_proc=n_workers
        )
        
        processed_datasets[split_name] = processed_split
        print(f"{split_name.capitalize()} dataset size: {len(processed_split)}")
    
    print(f"\nDataset download and caching complete.")
    print(f"All splits cached at: {preprocessing_cache}")
    
    # Print summary
    print(f"\nSummary:")
    for split_name, split_data in processed_datasets.items():
        print(f"  {split_name}: {len(split_data)} samples")
    
    return processed_datasets

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and cache a dataset for training.")
    parser.add_argument("--dataset-name", type=str, default="cerebras/SlimPajama-627B",
                        help="Dataset name from Hugging Face Hub.")
    parser.add_argument("--preprocessing-cache-dir", type=str, 
                        default="/hkfs/work/workspace/scratch/hgf_mxv5488-myspace/mapped_datasets",  # Updated default
                        help="Directory to cache the preprocessed dataset.")
    parser.add_argument("--tokenizer-model", type=str, default="meta-llama/Meta-Llama-3-8B",
                        help="Hugging Face model name for the tokenizer.")
    parser.add_argument("--max-length", type=int, default=2048,
                        help="Maximum sequence length for tokenization.")
    parser.add_argument("--n-workers", type=int, default=4,
                        help="Number of workers for dataset processing.")
    
    args = parser.parse_args()
    
    download_and_cache_dataset(
        dataset_name=args.dataset_name,
        preprocessing_cache_dir=args.preprocessing_cache_dir,
        tokenizer_model=args.tokenizer_model,
        max_length=args.max_length,
        n_workers=args.n_workers,
    )
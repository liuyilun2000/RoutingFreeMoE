import os
import numpy as np
import torch
from datasets import Dataset, load_dataset, IterableDataset
from multiprocessing import cpu_count
from transformers import Trainer, AutoTokenizer
import wandb


def preprocess_function_factory(tokenizer, max_length):
    """
    Factory to create a preprocessing function for the dataset.
    This function is used consistently across cache_dataset.py and training scripts.
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


def create_splits(dataset_name: str, cache_dir: str, val_size: int = 10000):
    print(f"Loading dataset {dataset_name}...")
    
    splits_cache_dir = os.path.join(cache_dir, "splits")
    train_cache_path = os.path.join(splits_cache_dir, "train")
    val_cache_path = os.path.join(splits_cache_dir, "validation")

    if os.path.exists(train_cache_path) and os.path.exists(val_cache_path):
        print("Loading splits from cache...")
        try:
            train_dataset = Dataset.load_from_disk(train_cache_path)
            val_dataset = Dataset.load_from_disk(val_cache_path)
            print(f"Loaded cached splits - Training: {len(train_dataset)}, Validation: {len(val_dataset)}")
            return train_dataset, val_dataset
        except Exception as e:
            print(f"Failed to load cached splits: {e}")
            print("Falling back to creating new splits...")

    print(f"Creating new validation split with {val_size} examples...")
    
    full_dataset = load_dataset(
        dataset_name,
        split="train",
        streaming=False,
        cache_dir=cache_dir
    )
    
    full_dataset = full_dataset.shuffle(seed=42)
    splits = full_dataset.train_test_split(
        test_size=val_size,
        shuffle=False
    )
    
    train_dataset = splits['train']
    val_dataset = splits['test']

    print("Saving splits to cache...")
    os.makedirs(splits_cache_dir, exist_ok=True)
    train_dataset.save_to_disk(train_cache_path)
    val_dataset.save_to_disk(val_cache_path)
    
    print(f"Created splits - Training: {len(train_dataset)}, Validation: {len(val_dataset)}")
    return train_dataset, val_dataset

def preprocess_and_cache_dataset(
    dataset: Dataset,
    cache_dir: str,
    split_name: str,
    preprocess_fn,
    num_proc: int = None,
    force_reprocess: bool = False
):
    """Preprocess dataset with multiprocessing and caching support."""
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"processed_{split_name}")
    
    if not force_reprocess and os.path.exists(cache_path):
        print(f"Loading preprocessed {split_name} dataset from cache...")
        try:
            return Dataset.load_from_disk(cache_path)
        except Exception as e:
            print(f"Failed to load cache: {e}")
            print("Falling back to preprocessing...")
    
    # Handle streaming datasets (IterableDataset)
    is_streaming = isinstance(dataset, IterableDataset)
    if is_streaming:
        print(f"Converting streaming dataset to regular dataset for {split_name}...")
        # For streaming datasets, we need to convert to list first
        # This is memory-intensive but necessary for preprocessing and caching
        print("Collecting all examples from streaming dataset...")
        dataset_list = []
        for example in dataset:
            dataset_list.append(example)
        print(f"Collected {len(dataset_list)} examples")
        # Convert to regular Dataset
        from datasets import Dataset as RegularDataset
        dataset = RegularDataset.from_list(dataset_list)
    
    if num_proc is None:
        num_proc = max(1, int(cpu_count()))
    
    print(f"Preprocessing {split_name} dataset using {num_proc} processes...")
    map_kwargs = {
        "function": preprocess_fn,
        "batched": True,
        "desc": f"Preprocessing {split_name} split"
    }
    
    # Add num_proc and remove_columns only for regular (non-streaming) datasets
    # After conversion above, dataset is no longer IterableDataset
    if not is_streaming or isinstance(dataset, Dataset):
        map_kwargs["num_proc"] = num_proc
        map_kwargs["remove_columns"] = dataset.column_names
    
    processed_dataset = dataset.map(**map_kwargs)
    
    print(f"Saving preprocessed {split_name} dataset to cache...")
    processed_dataset.save_to_disk(cache_path)
    
    return processed_dataset

def custom_data_collator(features):
    """Collate examples into batches."""
    return {
        'input_ids': torch.stack([torch.tensor(f['input_ids']) for f in features]),
        'attention_mask': torch.stack([torch.tensor(f['attention_mask']) for f in features]),
        'labels': torch.stack([torch.tensor(f['labels']) for f in features])
    }


class AuxLossTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._last_logged_step = -1
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Override compute_loss to extract and log auxiliary losses
        """
        outputs = model(**inputs)
        loss = outputs.loss if isinstance(outputs, dict) else outputs[0]

        # Only log on main process and only once per step
        if getattr(self.args, "local_rank", 0) <= 0:
            current_step = self.state.global_step
            # Only log at the frequency specified by logging_steps to prevent massive I/O overhead
            if current_step > self._last_logged_step and current_step > 0 and current_step % self.args.logging_steps == 0:
                self._last_logged_step = current_step
                
                aux_loss = getattr(outputs, "aux_dict", None)
                lambda_coef = getattr(outputs, "lambda_coef", None)
                
                if wandb.run is not None:
                    def format_value(val):
                        if isinstance(val, torch.Tensor):
                            if val.numel() == 1:
                                val = val.item()
                            else:
                                return val
                        if isinstance(val, float):
                            return float(f"{val:.4g}")
                        if isinstance(val, list):
                            return [format_value(x) for x in val]
                        return val

                    log_dict = {}
                    if lambda_coef:
                        log_dict["lambda_coef"] = lambda_coef
                    if aux_loss:
                        lm_loss = getattr(outputs, "lm_loss", None)
                        log_dict.update({k: format_value(v) for k, v in aux_loss.items()})
                        if lm_loss is not None:
                            log_dict["lm_loss"] = format_value(lm_loss)
                    if log_dict:
                        wandb.log(log_dict, step=current_step)

        return (loss, outputs) if return_outputs else loss
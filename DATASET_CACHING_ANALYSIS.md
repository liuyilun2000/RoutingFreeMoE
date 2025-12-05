# Dataset Caching & Loading Architecture Analysis

## Overview

This document analyzes the current dataset downloading, caching, and loading infrastructure in the RoutingFreeMoE project.

## Current Architecture

### 1. **Dataset Downloading & Caching** (`cache_dataset.py`)

**Purpose**: Standalone script for downloading and preprocessing datasets ahead of training.

**Key Functions**:
- `download_and_cache_dataset()`: Downloads dataset from HuggingFace Hub and caches all splits
- `preprocess_and_cache_dataset()`: Preprocesses a single dataset split and caches it

**Features**:
- ✅ Downloads datasets from HuggingFace Hub using `load_dataset()` with `streaming=True`
- ✅ Supports custom HF cache directory via `hf_cache_dir` parameter
- ✅ Processes all available splits automatically
- ✅ Caches preprocessed datasets to disk
- ✅ Loads from cache if it exists

**Cache Structure**:
```
preprocessing_cache_dir/
└── dataset_name/          # e.g., "cerebras_SlimPajama-627B"
    ├── train/
    ├── validation/
    └── test/
```

**Usage**:
```bash
python cache_dataset.py \
  --dataset-name cerebras/SlimPajama-627B \
  --preprocessing-cache-dir /path/to/cache \
  --hf-cache-dir /path/to/hf_cache \
  --tokenizer-model meta-llama/Meta-Llama-3-8B \
  --max-length 2048
```

### 2. **Training Script Dataset Loading** (`pretrain.py`, `pretrain_baseline.py`)

**Current Implementation**:
- Loads datasets using `load_dataset()` with optional HF cache directory
- Creates train/val splits using `train_test_split()`
- Uses `preprocess_and_cache_dataset()` from `train_utils.py` to cache preprocessed datasets

**Issues Identified**:

⚠️ **CRITICAL INCOMPATIBILITY**: There are **two different implementations** of `preprocess_and_cache_dataset()`:

1. **`cache_dataset.py` version**:
   - Cache path: `{cache_dir}/{split_name}` (e.g., `cache_dir/train`)
   - Removes only `['text']` column
   - No multiprocessing in map operation

2. **`train_utils.py` version** (used by training scripts):
   - Cache path: `{cache_dir}/processed_{split_name}` (e.g., `cache_dir/processed_train`)
   - Removes all columns: `remove_columns=dataset.column_names`
   - Supports multiprocessing
   - Has error handling and `force_reprocess` option

**This means datasets cached by `cache_dataset.py` cannot be loaded by training scripts!**

### 3. **Loading from Specified Directory**

**Current Support**:
- ✅ **HF Cache Directory**: Supported via `hf_cache_dir` parameter
  - Used for raw dataset downloads from HuggingFace Hub
  - Passed to `load_dataset(cache_dir=hf_cache_dir)`

- ⚠️ **Preprocessed Cache Directory**: Partially supported
  - Training scripts accept `preprocessing_cache_dir` parameter
  - But cache structure/format incompatibility means standalone cached datasets won't load

- ❌ **Loading from Local Directory**: NOT directly supported
  - No explicit support for loading already-downloaded datasets from local paths
  - Would need to use `load_dataset()` with path directly or modify code

## Workflow Analysis

### Current Training Workflow:
```
1. Training script runs
2. Loads raw dataset from HF Hub (or cache)
3. Creates train/val splits
4. Preprocesses each split (tokenization)
5. Caches preprocessed splits to disk
6. Uses cached splits for training
```

### Intended Workflow (with `cache_dataset.py`):
```
1. Run cache_dataset.py separately (before training)
2. Downloads dataset from HF Hub
3. Preprocesses all splits
4. Caches to disk
5. Training script should load from cache
```

**Problem**: These workflows are incompatible due to different cache formats!

## Recommendations

### Option 1: Unify Cache Format (Recommended)

1. **Consolidate to single implementation**:
   - Remove duplicate `preprocess_and_cache_dataset()` from one location
   - Use consistent cache path format
   - Use consistent column removal strategy

2. **Suggested unified approach**:
   - Use `train_utils.py` version as base (more features)
   - Update `cache_dataset.py` to use same format
   - Or create shared utility module

### Option 2: Support Loading from Local Directory

Add explicit support for loading datasets from local paths:

```python
def load_dataset_from_path(
    dataset_path: str,
    preprocessing_cache_dir: str,
    split_name: str = "train",
    preprocess_fn = None,
    num_proc: int = 32
):
    """Load dataset from local directory path."""
    if os.path.exists(preprocessing_cache_dir):
        # Load from preprocessed cache
        cache_path = os.path.join(preprocessing_cache_dir, split_name)
        if os.path.exists(cache_path):
            return Dataset.load_from_disk(cache_path)
    
    # Load raw dataset from path
    dataset = load_dataset("arrow", data_files=f"{dataset_path}/*.arrow")
    
    # Preprocess and cache if needed
    if preprocess_fn:
        return preprocess_and_cache_dataset(...)
    return dataset
```

### Option 3: Support Both Formats

Modify training scripts to check for both cache formats and handle accordingly.

## Current Configuration

### Parameters in Training Scripts:

| Parameter | Purpose | Default |
|-----------|---------|---------|
| `dataset_name` | HuggingFace dataset identifier | `"roneneldan/TinyStories"` |
| `hf_cache_dir` | Raw HF dataset cache location | `None` (uses default) |
| `preprocessing_cache_dir` | Preprocessed dataset cache | `"../mapped_datasets"` |
| `max_length` | Tokenization max length | `512` |

### Shell Script Configuration:

- `pretrain.sh` / `pretrain_baseline.sh`:
  - `PREPROCESSING_CACHE_DIR`: Defaults to workspace scratch directory
  - `HF_CACHE_DIR`: Defaults to workspace scratch directory

## Summary

### ✅ What Works:
- Downloading datasets from HuggingFace Hub
- Caching raw datasets via HF cache directory
- Preprocessing and caching datasets during training
- Loading preprocessed datasets from cache during training

### ⚠️ What's Incompatible:
- Standalone `cache_dataset.py` cached datasets cannot be loaded by training scripts
- Different cache path formats and column removal strategies

### ❌ What's Missing:
- Direct loading of pre-cached datasets from `cache_dataset.py`
- Loading datasets from arbitrary local directory paths
- Unified cache format across all scripts

## Next Steps

1. **Unify cache format**: Choose one implementation and update both locations
2. **Test end-to-end**: Verify `cache_dataset.py` → training script workflow
3. **Add documentation**: Document cache directory structure and expected formats
4. **Consider adding**: Support for loading from local paths (not just HF Hub)



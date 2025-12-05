# Code Organization Summary

## Project Structure

### Core Components

#### 1. **Model Initialization**
- **`init.py`** + **`init.sh`**: Initialize RoutingFreeDeepseekV3 models
- **`init_baseline.py`** + **`init_baseline.sh`**: Initialize baseline DeepseekV3 models

#### 2. **Training Scripts**
- **`pretrain.py`** + **`pretrain.sh`**: Train RoutingFreeDeepseekV3 models
- **`pretrain_baseline.py`** + **`pretrain_baseline.sh`**: Train baseline DeepseekV3 models

#### 3. **Dataset Management**
- **`cache_dataset.py`**: Standalone script for downloading and caching datasets
- **`train_utils.py`**: Training utilities including dataset preprocessing

#### 4. **Utilities**
- **`utils.py`**: Model parameter management utilities
- **`test.py`**: Model testing script

## Dataset Flow

### Current Training Flow (pretrain.py/pretrain_baseline.py):

```
1. Load raw dataset from HuggingFace Hub
   └── load_dataset(dataset_name, cache_dir=hf_cache_dir)

2. Create train/val splits
   └── dataset["train"].train_test_split(test_size=0.001)

3. Preprocess and cache each split
   └── preprocess_and_cache_dataset() from train_utils.py
       └── Cache location: {preprocessing_cache_dir}/{dataset_name}/processed_{split_name}

4. Train with cached preprocessed datasets
```

### Standalone Caching Flow (cache_dataset.py):

```
1. Download dataset from HuggingFace Hub
   └── load_dataset(dataset_name, streaming=True, cache_dir=hf_cache_dir)

2. Process all available splits
   └── preprocess_and_cache_dataset() from cache_dataset.py
       └── Cache location: {preprocessing_cache_dir}/{dataset_name}/{split_name}

3. Cache to disk for later use
   └── ⚠️ INCOMPATIBLE with training script cache format
```

## Key Incompatibility

### Cache Path Mismatch:

| Script | Cache Path Format | Example |
|--------|------------------|---------|
| `cache_dataset.py` | `{cache_dir}/{split_name}` | `cache/train` |
| `train_utils.py` | `{cache_dir}/processed_{split_name}` | `cache/processed_train` |

### Column Removal Mismatch:

| Script | Columns Removed |
|--------|----------------|
| `cache_dataset.py` | Only `['text']` |
| `train_utils.py` | All columns (`dataset.column_names`) |

**Result**: Datasets cached by `cache_dataset.py` cannot be loaded by training scripts!

## Configuration Parameters

### Dataset Loading Parameters:

```python
# In pretrain.py / pretrain_baseline.py
dataset_name: str = "roneneldan/TinyStories"  # HF dataset name
hf_cache_dir: str = None                       # Raw HF cache location
preprocessing_cache_dir: str = "../mapped_datasets"  # Preprocessed cache
max_length: int = 512                          # Tokenization length
n_workers: int = 32                            # Processing workers
```

### Shell Script Defaults (pretrain.sh):

```bash
DATASET_NAME="cerebras/SlimPajama-627B"
PREPROCESSING_CACHE_DIR="/hkfs/work/workspace/scratch/hgf_mxv5488-slimpajama"
HF_CACHE_DIR="/hkfs/work/workspace/scratch/hgf_mxv5488-slimpajama"
```

## Function Usage Map

### Dataset Functions:

| Function | Location | Used By | Purpose |
|----------|----------|---------|---------|
| `download_and_cache_dataset()` | `cache_dataset.py` | Standalone script | Download & cache all splits |
| `preprocess_and_cache_dataset()` | `cache_dataset.py` | `cache_dataset.py` | Cache single split (old format) |
| `preprocess_and_cache_dataset()` | `train_utils.py` | `pretrain.py`, `pretrain_baseline.py` | Cache single split (new format) |
| `preprocess_function_factory()` | `cache_dataset.py` | `cache_dataset.py` | Create preprocessing function |
| `preprocess_function_factory()` | `pretrain.py` | `pretrain.py` | Create preprocessing function |
| `preprocess_function_factory()` | `pretrain_baseline.py` | `pretrain_baseline.py` | Create preprocessing function |

### Model Utilities:

| Function | Location | Used By |
|----------|----------|---------|
| `print_trainable_parameters()` | `utils.py` | `pretrain.py`, `pretrain_baseline.py` |
| `print_filtered_model_size()` | `utils.py` | `init.py`, `init_baseline.py`, `test.py` |

### Training Utilities:

| Function | Location | Used By |
|----------|----------|---------|
| `AuxLossTrainer` | `train_utils.py` | `pretrain.py` |
| `create_splits()` | `train_utils.py` | Not currently used |
| `custom_data_collator()` | `train_utils.py` | Not currently used |

## Recommendations for Organization

### 1. Consolidate Duplicate Code

**Preprocessing Functions**:
- `preprocess_function_factory()` appears in 3 places (cache_dataset.py, pretrain.py, pretrain_baseline.py)
- **Solution**: Move to `train_utils.py` or create `dataset_utils.py`

**Caching Functions**:
- Two incompatible `preprocess_and_cache_dataset()` implementations
- **Solution**: Keep only one (recommend `train_utils.py` version) and update `cache_dataset.py`

### 2. Create Shared Dataset Utilities Module

```python
# dataset_utils.py (suggested structure)
- preprocess_function_factory()
- preprocess_and_cache_dataset()  # Unified version
- download_and_cache_dataset()
- load_dataset_from_cache()
- load_dataset_from_path()  # New: support local paths
```

### 3. Update Training Scripts

- Import preprocessing functions from shared module
- Remove duplicate implementations
- Support loading from both cache formats (backward compatibility)

### 4. Improve Documentation

- Document cache directory structure
- Document expected dataset formats
- Provide examples for each workflow



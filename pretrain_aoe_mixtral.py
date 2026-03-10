import argparse
import copy
import json
import math
import os
import re
import sys
from os.path import join
from pathlib import Path
from typing import List, Optional, Union

import fire
import requests
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential

from tqdm import tqdm

from safetensors import safe_open
from safetensors.torch import load_file, save_file

from datasets import load_dataset

import transformers
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
)
from transformers import Trainer
from transformers import DataCollatorForLanguageModeling

from routing_free.mixtral_aoe import AoEMixtralConfig, AoEMixtralForCausalLM, AoEMixtralModel

AutoConfig.register("aoe_mixtral", AoEMixtralConfig)
AutoModel.register(AoEMixtralConfig, AoEMixtralModel)
AutoModelForCausalLM.register(AoEMixtralConfig, AoEMixtralForCausalLM)

from utils import *
from train_utils import preprocess_function_factory, preprocess_and_cache_dataset

import numpy as np

tokenizer_model = "EleutherAI/gpt-neo-125M"


def train(
    # Model/data params
    model_dir: str,
    dataset_name: str = "roneneldan/TinyStories",
    output_dir: str = "./output",
    preprocessing_cache_dir: str = "../mapped_datasets",
    hf_cache_dir: str = None,
    # Model config params
    num_hidden_layers: int = 12,
    num_local_experts: int = 12,
    num_experts_per_tok: int = 3,
    intermediate_size: int = 128,
    # Training hyperparams
    per_device_batch_size: int = 16,
    gradient_accumulation_steps: int = 4,
    n_epochs: int = 1,
    learning_rate: float = 1e-3,
    weight_decay: float = 0.01,
    warmup_steps: int = 100,
    test_size: float = 0.001,
    eval_steps: int = 200,
    save_steps: int = 1000,
    max_length: int = 512,
    # Wandb params
    wandb_project: str = "mixtral-aoe",
    wandb_run_name: str = "test",
    # Additional params
    seed: int = 42,
    bf16: bool = True,
    n_workers: int = 32,
    resume_from_checkpoint: str = None,
):
    # Initialize DDP
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    if local_rank != -1:
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        torch.distributed.init_process_group(
            backend="nccl",
            init_method="env://"
        )
    # Initialize wandb
    if wandb_project is not None:
        os.environ["WANDB_PROJECT"] = wandb_project
    # Load model and tokenizer
    print(f"Rank {local_rank} / {world_size} : {torch.cuda.mem_get_info()}")
    config = AoEMixtralConfig.from_pretrained(model_dir)
    if config.num_hidden_layers != num_hidden_layers:
        raise ValueError(f"Number of hidden layers in config ({config.num_hidden_layers}) does not match the provided value ({num_hidden_layers})")
    if config.num_local_experts < num_local_experts:
        raise ValueError(f"Number of local experts in config ({config.num_local_experts}) is less than the provided value ({num_local_experts})")
    config.num_local_experts = num_local_experts
    config.num_experts_per_tok = num_experts_per_tok
    if config.intermediate_size != intermediate_size:
        raise ValueError(f"Intermediate size in config ({config.intermediate_size}) does not match the provided value ({intermediate_size})")
    if local_rank <= 0:
        print(config)
    model = AoEMixtralForCausalLM.from_pretrained(
        model_dir,
        config=config,
        torch_dtype=torch.bfloat16 if bf16 else torch.float32,
        ignore_mismatched_sizes=True,
    )
    model.config.use_cache = False

    print(local_rank, model)
    print_trainable_parameters(model)

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    tokenizer.pad_token = tokenizer.eos_token

    # Handle checkpoint resuming
    if resume_from_checkpoint:
        checkpoint_path = os.path.join(output_dir, resume_from_checkpoint)
        if os.path.isdir(checkpoint_path):
            if local_rank <= 0:
                print(f"Loading checkpoint from directory: {checkpoint_path}")
            model = AoEMixtralForCausalLM.from_pretrained(
                checkpoint_path,
                config=config,
                dtype=torch.bfloat16 if bf16 else torch.float32,
                device_map='cuda'
            )

    # Prepare dataset splits
    load_dataset_kwargs = {"num_proc": n_workers}
    if hf_cache_dir is not None:
        load_dataset_kwargs["cache_dir"] = hf_cache_dir
    dataset = None
    if dataset_name == "Skylion007/openwebtext":
        dataset = load_dataset(dataset_name, **load_dataset_kwargs)
        split_dataset = dataset["train"].train_test_split(test_size=test_size, seed=2357, shuffle=True)
        split_dataset['val'] = split_dataset.pop('test')
        train_dataset = split_dataset["train"]
        val_dataset = split_dataset["val"]
    else:
        if dataset_name == "roneneldan/TinyStories":
            dataset = load_dataset(
                "parquet",
                data_files={
                    "train": "hf://datasets/roneneldan/TinyStories/data/train-*.parquet",
                    "validation": "hf://datasets/roneneldan/TinyStories/data/validation-*.parquet",
                },
                **load_dataset_kwargs
            )
            split_dataset = dataset["train"].train_test_split(test_size=test_size, seed=2357, shuffle=True)
            split_dataset['val'] = split_dataset.pop('test')
            train_dataset = split_dataset["train"]
            val_dataset = split_dataset["val"]
        else:
            dataset = load_dataset(dataset_name, **load_dataset_kwargs)
            if "validation" not in dataset:
                split_dataset = dataset["train"].train_test_split(test_size=test_size, seed=2357, shuffle=True)
                split_dataset['val'] = split_dataset.pop('test')
                train_dataset = split_dataset["train"]
                val_dataset = split_dataset["val"]
            else:
                train_dataset = dataset["train"]
                val_dataset = dataset["validation"]

    # Preprocess datasets with caching
    preprocessing_cache = os.path.join(preprocessing_cache_dir, dataset_name)
    preprocess_function = preprocess_function_factory(tokenizer, max_length)

    train_dataset = preprocess_and_cache_dataset(
        dataset=train_dataset,
        cache_dir=preprocessing_cache,
        split_name="train",
        preprocess_fn=preprocess_function,
        num_proc=n_workers
    )

    val_dataset = preprocess_and_cache_dataset(
        dataset=val_dataset,
        cache_dir=preprocessing_cache,
        split_name="val",
        preprocess_fn=preprocess_function,
        num_proc=n_workers
    )

    # Calculate steps
    total_examples = len(train_dataset)
    examples_per_step = per_device_batch_size * gradient_accumulation_steps * world_size
    max_steps = int((total_examples * n_epochs) // examples_per_step)

    if local_rank <= 0:
        print(f"\n=== Training Configuration ===")
        print(f"World size (num GPUs): {world_size}")
        print(f"Per device batch size: {per_device_batch_size}")
        print(f"Gradient accumulation steps: {gradient_accumulation_steps}")
        print(f"Total batch size per step: {examples_per_step}")
        print(f"Total examples: {total_examples}")
        print(f"Number of epochs: {n_epochs}")
        print(f"Max steps: {max_steps} | not used in trainer")
        print("============================\n")

    # Prepare training arguments
    training_args = transformers.TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=n_epochs,
        per_device_train_batch_size=per_device_batch_size,
        per_device_eval_batch_size=per_device_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,

        local_rank=local_rank,
        ddp_backend="nccl",
        ddp_find_unused_parameters=True,
        dataloader_pin_memory=True,

        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_steps=warmup_steps,

        logging_dir=os.path.join(output_dir, "logs"),
        logging_steps=10,

        eval_steps=eval_steps,
        eval_strategy="steps",
        eval_on_start=True,
        save_steps=save_steps,
        save_strategy="steps",
        save_total_limit=2,

        load_best_model_at_end=True,
        label_names=["labels"],
        metric_for_best_model="eval_loss",
        greater_is_better=False,

        remove_unused_columns=False,
        bf16=bf16,
        dataloader_num_workers=n_workers,
        group_by_length=False,

        report_to="wandb" if wandb_project else None,
        run_name=wandb_run_name,
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )

    # Train the model
    trainer_output = trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    # Final evaluation
    if local_rank <= 0:
        print("\n" + "=" * 50)
        print("FINAL EVALUATION ON BEST MODEL")
        print("=" * 50)

    final_metrics = trainer.evaluate()

    if local_rank <= 0:
        print(f"Final Eval Loss: {final_metrics.get('eval_loss', 'N/A')}")
        import math
        if 'eval_loss' in final_metrics:
            perplexity = math.exp(final_metrics['eval_loss'])
            print(f"Final Perplexity: {perplexity:.4f}")
        print("=" * 50 + "\n")

        print("All final metrics:")
        for key, value in final_metrics.items():
            print(f"  {key}: {value}")

    # Save the best model
    if local_rank <= 0:
        final_model_path = os.path.join(output_dir, "final_model")
        print(f"\nSaving best model to {final_model_path}...")
        model_to_save = trainer.model
        if hasattr(model_to_save, 'module'):
            model_to_save = model_to_save.module
        model_to_save.save_pretrained(final_model_path)
        tokenizer.save_pretrained(final_model_path)
        print(f"Best model saved successfully!")

    return trainer_output, final_metrics

def main():
    parser = argparse.ArgumentParser(description="Train AoE Mixtral model")
    parser.add_argument("--model-dir", type=str, required=True,
                        help="Directory containing the initialized model")
    parser.add_argument("--dataset-name", type=str, default="roneneldan/TinyStories",
                        help="Dataset name")
    parser.add_argument("--output-dir", type=str, required=True, default="./output",
                        help="Output directory")
    parser.add_argument("--num-hidden-layers", type=int, default=12,
                        help="Number of hidden layers")
    parser.add_argument("--num-local-experts", type=int, default=12,
                        help="Number of local experts")
    parser.add_argument("--num-experts-per-tok", type=int, default=3,
                        help="Number of experts per token (top-K)")
    parser.add_argument("--intermediate-size", type=int, default=128,
                        help="Intermediate size")
    parser.add_argument("--epochs", type=int, default=1,
                        help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate")
    parser.add_argument("--wandb-project", type=str, default="mixtral-aoe",
                        help="Weights & Biases project name")
    parser.add_argument("--wandb-run", type=str, default=None,
                        help="Weights & Biases run name")
    parser.add_argument("--bf16", action="store_true",
                        help="Use bfloat16 precision")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                        help="Resume from checkpoint")
    parser.add_argument("--per_device_batch_size", type=int, default=16,
                        help="Per device batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4,
                        help="Number of gradient accumulation steps")
    parser.add_argument("--preprocessing_cache_dir", type=str, default="../mapped_datasets",
                        help="Directory for caching preprocessed datasets")
    parser.add_argument("--hf-cache-dir", type=str, default=None,
                        help="Directory for Hugging Face dataset cache")
    args = parser.parse_args()

    train(
        model_dir=args.model_dir,
        dataset_name=args.dataset_name,
        output_dir=args.output_dir,
        num_hidden_layers=args.num_hidden_layers,
        num_local_experts=args.num_local_experts,
        num_experts_per_tok=args.num_experts_per_tok,
        intermediate_size=args.intermediate_size,
        per_device_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        n_epochs=args.epochs,
        learning_rate=args.lr,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run,
        bf16=args.bf16,
        resume_from_checkpoint=args.resume_from_checkpoint,
        preprocessing_cache_dir=args.preprocessing_cache_dir,
        hf_cache_dir=args.hf_cache_dir,
    )

if __name__ == "__main__":
    main()

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
from transformers import DataCollatorForLanguageModeling

from routing_free.deepseek_v3.configuration_deepseek_v3_rf import RoutingFreeDeepseekV3Config
from routing_free.deepseek_v3_rf import RoutingFreeDeepseekV3Model, RoutingFreeDeepseekV3ForCausalLM

AutoConfig.register("routing_free_deepseek_v3", RoutingFreeDeepseekV3Config)
AutoModel.register(RoutingFreeDeepseekV3Config, RoutingFreeDeepseekV3Model)
AutoModelForCausalLM.register(RoutingFreeDeepseekV3Config, RoutingFreeDeepseekV3ForCausalLM)

from utils import *
from train_utils import preprocess_function_factory, preprocess_and_cache_dataset, AuxLossTrainer

import numpy as np

AutoConfig.register("routing_free_deepseek_v3", RoutingFreeDeepseekV3Config)
AutoModel.register(RoutingFreeDeepseekV3Config, RoutingFreeDeepseekV3Model)
AutoModelForCausalLM.register(RoutingFreeDeepseekV3Config, RoutingFreeDeepseekV3ForCausalLM)

base_model = "deepseek-ai/DeepSeek-V3"
tokenizer_model = "EleutherAI/gpt-neo-125M"




def train(
    # Model/data params
    model_dir: str,
    dataset_name: str = "roneneldan/TinyStories",
    output_dir: str = "./output",
    preprocessing_cache_dir: str = "../mapped_datasets",
    hf_cache_dir: str = None,
    # Model config params
    n_hidden_layers: int = 12,
    n_experts: int = 24,
    moe_intermediate_size: int = 128,
    gate_temperature: float = 1.0,
    gate_threshold: float = 0.5,
    density_target: float = 0.1,
    lambda_coef: float = 1e-5,
    eta_coef: float = 1.2,
    per_expert_aux_loss_coef: float = 0.5,
    per_token_aux_loss_coef: float = 0.5,
    # Training hyperparams
    per_device_batch_size: int = 32,
    gradient_accumulation_steps: int = 2,
    n_epochs: int = 1,
    learning_rate: float = 1e-3,
    weight_decay: float = 0.01,
    warmup_steps: int = 100,
    test_size: float = 0.001,
    eval_steps: int = 200,
    save_steps: int = 200,
    max_length: int = 512,
    # Wandb params
    wandb_project: str = "deepseek-v3-noae",
    wandb_run_name: str = "test",
    # Additional params
    seed: int = 42,
    bf16: bool = True,
    n_workers: int = 32,
    resume_from_checkpoint: str = None, 
    orthogonality_loss: bool = True,
    orthogonality_loss_coef: float = 1e-5,
):
    # Set debug environment variable for distributed training issues
    os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
    
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
        os.environ["WANDB_PROJECT"]=wandb_project
    # Load model and tokenizer
    print(f"Rank {local_rank} / {world_size} : {torch.cuda.mem_get_info()}")
    config = RoutingFreeDeepseekV3Config.from_pretrained(model_dir)
    config.output_gate_scores = True
    if config.n_hidden_layers != n_hidden_layers:
        raise ValueError(f"Number of hidden layers in config ({config.n_hidden_layers}) does not match the provided value ({n_hidden_layers})")
    if config.n_experts < n_experts:
        raise ValueError(f"Number of experts in config ({config.n_experts}) is less than the provided value ({n_experts})")
    config.n_experts = n_experts
    if config.moe_intermediate_size != moe_intermediate_size:
        raise ValueError(f"MOE intermediate size in config ({config.moe_intermediate_size}) does not match the provided value ({moe_intermediate_size})")
    config.gate_temperature = gate_temperature
    config.gate_threshold = gate_threshold
    config.density_target = density_target
    config.lambda_coef = lambda_coef
    config.eta_coef = eta_coef
    config.per_expert_aux_loss_coef = per_expert_aux_loss_coef
    config.per_token_aux_loss_coef = per_token_aux_loss_coef
    config.orthogonality_loss = orthogonality_loss
    config.orthogonality_loss_coef = orthogonality_loss_coef
    model = RoutingFreeDeepseekV3ForCausalLM.from_pretrained(
        model_dir,
        config=config,
        torch_dtype=torch.bfloat16 if bf16 else torch.float32,
        ignore_mismatched_sizes=True,
        #device_map='cuda'
    )
    model.config.use_cache = False
    if local_rank <= 0:
        print(model.config)
    
    if world_size > 1 and local_rank != -1:
        model = model.to(device)
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=True  # Enable to handle unused parameters
        )

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    tokenizer.pad_token = tokenizer.eos_token

    # Handle checkpoint resuming
    if resume_from_checkpoint:
        checkpoint_path = os.path.join(output_dir, resume_from_checkpoint)
        if os.path.isdir(checkpoint_path):
            if local_rank <= 0:
                print(f"Loading checkpoint from directory: {checkpoint_path}")
            model = RoutingFreeDeepseekV3ForCausalLM.from_pretrained(
                checkpoint_path,
                config=config,
                torch_dtype=torch.bfloat16 if bf16 else torch.float32,
                device_map='cuda'
            )

    # Prepare dataset splits
    load_dataset_kwargs = {"num_proc": n_workers}
    if hf_cache_dir is not None:
        load_dataset_kwargs["cache_dir"] = hf_cache_dir
    dataset = load_dataset(dataset_name, **load_dataset_kwargs)
    split_dataset = dataset["train"].train_test_split(test_size=test_size, seed=2357, shuffle=True)
    if local_rank <= 0:
        print(split_dataset)
    split_dataset['val'] = split_dataset.pop('test') 
    
    train_dataset = split_dataset["train"]
    val_dataset = split_dataset["val"]
    '''
    dataset_cache = os.path.join(dataset_cache_dir, dataset_name)
    train_dataset, val_dataset = create_splits(
        #dataset_name=dataset_name,
        #cache_dir=dataset_cache,
        dataset,
        val_size=val_size
    )
    '''    
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

    # Print training info only from main process
    if local_rank <= 0:  # Main process
        print(f"\n=== Training Configuration ===")
        print(f"World size (num GPUs): {world_size}")
        print(f"Per device batch size: {per_device_batch_size}")
        print(f"Gradient accumulation steps: {gradient_accumulation_steps}")
        print(f"Total batch size per step: {examples_per_step}")
        print(f"Total examples: {total_examples}")
        print(f"Number of epochs: {n_epochs}")
        print(f"Max steps: {max_steps} | not used in trainer")
        print(f"Model: {model}")
        print("============================\n")
    
    # Prepare training arguments
    training_args = transformers.TrainingArguments(
        output_dir=output_dir,
        #max_steps=max_steps,
        num_train_epochs=n_epochs, 
        per_device_train_batch_size=per_device_batch_size,
        per_device_eval_batch_size=per_device_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        
        # Add DDP-specific arguments
        local_rank=local_rank,
        ddp_backend="nccl",
        dataloader_pin_memory=True,  # Important for performance
        
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
    trainer = AuxLossTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )
    

    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)
    
    trainer_output = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    if local_rank <= 0:
        if hasattr(model, 'module'):
            # If model is wrapped in DDP
            model.module.save_pretrained(os.path.join(output_dir, "final_model"))
        else:
            model.save_pretrained(os.path.join(output_dir, "final_model"))
    
    return trainer_output

def main():
    parser = argparse.ArgumentParser(description="Train DroppedFFN model")
    parser.add_argument("--model-dir", type=str, required=True,
                      help="Directory containing the initialized model")
    parser.add_argument("--dataset-name", type=str, default="roneneldan/TinyStories",
                      help="Dataset name")
    #parser.add_argument("--dataset-config", type=str, default="zyda_crossdeduped-filtered",
    #                  help="Dataset configuration")
    parser.add_argument("--output-dir", type=str, required=True, default="./output",
                      help="Output directory")
    parser.add_argument("--num-hidden-layers", type=int, default=12,
                      help="Number of hidden layers")
    parser.add_argument("--n-experts", type=int, default=24,
                      help="Number of experts")
    parser.add_argument("--moe-intermediate-size", type=int, default=128,
                      help="MOE intermediate size")
    parser.add_argument("--gate-temperature", type=float, default=1.0,
                      help="Gate temperature")
    parser.add_argument("--gate-threshold", type=float, default=0.5,
                      help="Gate threshold")
    parser.add_argument("--density-target", type=float, default=0.1,
                      help="Density target")
    parser.add_argument("--lambda-coef", type=float, default=1e-5,
                      help="Lambda coefficient")
    parser.add_argument("--eta-coef", type=float, default=1.2,
                      help="Eta coefficient")
    parser.add_argument("--per-expert-aux-loss-coef", type=float, default=0.5,
                      help="Per expert auxiliary loss coefficient")
    parser.add_argument("--per-token-aux-loss-coef", type=float, default=0.5,
                      help="Per token auxiliary loss coefficient")
    parser.add_argument("--epochs", type=int, default=1,
                      help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-3,
                      help="Learning rate")
    parser.add_argument("--wandb-project", type=str, default="dropped-ffn",
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
    parser.add_argument("--orthogonality-loss", type=bool, default=True,
                      help="Use orthogonality loss")
    parser.add_argument("--orthogonality-loss-coef", type=float, default=1e-5,
                      help="Orthogonality loss coefficient")
    args = parser.parse_args()
    
    train(
        model_dir=args.model_dir,
        dataset_name=args.dataset_name,
        output_dir=args.output_dir,
        n_hidden_layers=args.n_hidden_layers,
        n_experts=args.n_experts,
        moe_intermediate_size=args.moe_intermediate_size,
        gate_temperature=args.gate_temperature,
        gate_threshold=args.gate_threshold,
        density_target=args.density_target,
        lambda_coef=args.lambda_coef,
        eta_coef=args.eta_coef,
        per_expert_aux_loss_coef=args.per_expert_aux_loss_coef,
        per_token_aux_loss_coef=args.per_token_aux_loss_coef,
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
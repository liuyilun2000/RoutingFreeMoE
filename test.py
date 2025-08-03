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

from routing_free.deepseek_v3.configuration_deepseek_v3 import RoutingFreeDeepseekV3Config
from routing_free.deepseek_v3.modeling_deepseek_v3 import (
    RoutingFreeDeepseekV3Model,
    RoutingFreeDeepseekV3ForCausalLM,
)

AutoConfig.register("routing_free_deepseek_v3", RoutingFreeDeepseekV3Config)
AutoModel.register(RoutingFreeDeepseekV3Config, RoutingFreeDeepseekV3Model)
AutoModelForCausalLM.register(RoutingFreeDeepseekV3Config, RoutingFreeDeepseekV3ForCausalLM)

from utils import *

base_model = "deepseek-ai/DeepSeek-V3"
tokenizer_model = "EleutherAI/gpt-neo-125M"

# Get tokenizer first to determine vocab size
tokenizer = AutoTokenizer.from_pretrained(tokenizer_model)
tokenizer.pad_token = tokenizer.eos_token if tokenizer.pad_token is None else tokenizer.pad_token
tokenizer.padding_side = 'right' if tokenizer.padding_side is None else tokenizer.padding_side

# Initialize model config with correct vocab size
print(f"\nTokenizer vocabulary size: {len(tokenizer)}")

#config = DeepseekV3NoAEConfig.from_pretrained("120M_1iter_12exp_256dim.config.json")
config = RoutingFreeDeepseekV3Config.from_pretrained("120M.config.json")


print("Initializing NoAE DeepseekV3 with config:", config)

model = RoutingFreeDeepseekV3ForCausalLM(
    config=config,
#    torch_dtype=torch.bfloat16,
#    device_map="auto"
)

print_filtered_model_size(model, ["lm_head", "embed_tokens", "position_embeddings", "token_type_embeddings", "LayerNorm"])

# Create a sample input
text = ["Once upon a time", "In a galaxy far, far away", "The quick brown fox jumps over"]
print(f"\nTesting with input text: {text}")

# Tokenize and create input tensors
inputs = tokenizer(
    text,
    return_tensors="pt",
    padding=True,
    truncation=True
)

'''
# Generate output
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=20,
        num_return_sequences=1,
        temperature=0.7,
        do_sample=True,
        output_gate_scores=True
    )

# Decode and print output
generated_text = [tokenizer.decode(outputs[i], skip_special_tokens=True) for i in range(len(outputs))]
print("\nGenerated text:")
print(generated_text)
'''
# check loss
output = model(
    **inputs,
    labels=inputs['input_ids'],
    output_gate_scores=True
)
print(f"loss: {output.loss}, aux_loss: {output.aux_loss}")   
import argparse
import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import AutoTokenizer
from transformers import MixtralForCausalLM, MixtralConfig

from utils import print_filtered_model_size

import random
import numpy as np
import torch

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)


def initialize_model(
    config_json: str,
    num_hidden_layers: int,
    num_local_experts: int,
    num_experts_per_tok: int,
    intermediate_size: int, 
    output_dir: str,
    model_name: str,
    tokenizer_model: str = "EleutherAI/gpt-neo-125M",
    bf16: bool = True
):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_model)
    tokenizer.pad_token = tokenizer.eos_token if tokenizer.pad_token is None else tokenizer.pad_token
    tokenizer.padding_side = 'right' if tokenizer.padding_side is None else tokenizer.padding_side

    # Load config
    config = MixtralConfig.from_pretrained(config_json)
    config.vocab_size = len(tokenizer)
    config.num_hidden_layers = num_hidden_layers
    config.num_local_experts = num_local_experts
    config.num_experts_per_tok = num_experts_per_tok
    config.intermediate_size = intermediate_size

    print(f"Initializing Mixtral baseline with config: {config}")
    model = MixtralForCausalLM(config=config)

    if bf16:
        model = model.to(torch.bfloat16)
        print("Model initialized as bfloat16")

    print_filtered_model_size(model, ["lm_head", "embed_tokens", "position_embeddings", "token_type_embeddings", "LayerNorm"])

    # Save model and config
    output_dir = os.path.join(output_dir, model_name)
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    config.save_pretrained(output_dir)
    print(f"Model and tokenizer successfully saved at {output_dir}")
    return model, tokenizer

def main():
    parser = argparse.ArgumentParser(description="Initialize a Mixtral baseline model")
    parser.add_argument("--config-json", type=str, required=True, help="Path to config.json file")
    parser.add_argument("--num-hidden-layers", type=int, required=True, help="Number of hidden layers")
    parser.add_argument("--num-local-experts", type=int, required=True, help="Number of local experts")
    parser.add_argument("--num-experts-per-tok", type=int, required=True, help="Number of experts per token")
    parser.add_argument("--intermediate-size", type=int, required=True, help="Intermediate size")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save the initialized model")
    parser.add_argument("--model-name", type=str, required=True, help="Model name")
    parser.add_argument("--tokenizer-model", type=str, default="EleutherAI/gpt-neo-125M", help="Tokenizer model name or path")
    parser.add_argument("--bf16", action="store_true", default=True, help="Convert model to bfloat16 precision")
    args = parser.parse_args()

    initialize_model(
        config_json=args.config_json,
        num_hidden_layers=args.num_hidden_layers,   
        num_local_experts=args.num_local_experts,
        num_experts_per_tok=args.num_experts_per_tok,
        intermediate_size=args.intermediate_size,
        output_dir=args.output_dir,
        model_name=args.model_name,
        tokenizer_model=args.tokenizer_model,
        bf16=args.bf16
    )

if __name__ == "__main__":
    main() 
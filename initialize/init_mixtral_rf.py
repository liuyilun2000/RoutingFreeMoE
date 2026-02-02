import argparse
import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import AutoTokenizer
from routing_free.mixtral_rf import RoutingFreeMixtralForCausalLM, RoutingFreeMixtralConfig

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
    intermediate_size: int, 
    # Routing Free specific params
    n_experts: int,
    output_dir: str,
    model_name: str,
    tokenizer_model: str = "EleutherAI/gpt-neo-125M",
    bf16: bool = True,
    # Other Mixtral params that might be useful to keep consistent/overridable
    num_attention_heads: int = 32,
    num_key_value_heads: int = 8,
    hidden_size: int = 4096,
):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_model)
    tokenizer.pad_token = tokenizer.eos_token if tokenizer.pad_token is None else tokenizer.pad_token
    tokenizer.padding_side = 'right' if tokenizer.padding_side is None else tokenizer.padding_side

    # Load config
    # We load base Mixtral config structure then override with our params
    config = RoutingFreeMixtralConfig.from_pretrained(config_json)
    config.vocab_size = len(tokenizer)
    
    # Mixtral structural params
    config.num_hidden_layers = num_hidden_layers
    config.intermediate_size = intermediate_size # This is likely 'moe_intermediate_size' in RF context context if decomposed, but MixtralConfig uses intermediate_size for the experts.
    config.hidden_size = hidden_size
    config.num_attention_heads = num_attention_heads
    config.num_key_value_heads = num_key_value_heads
    
    # Routing Free params
    config.n_experts = n_experts
    # We might want to expose other RF params like gate_threshold here if needed, but sticking to command line args requested.
    
    # Ensure num_local_experts matches n_experts for consistency if baseline code checks it (though RF ignores it usually)
    config.num_local_experts = n_experts 

    print(f"Initializing Routing Free Mixtral with config: {config}")
    model = RoutingFreeMixtralForCausalLM(config=config)

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
    parser = argparse.ArgumentParser(description="Initialize a Routing Free Mixtral model")
    parser.add_argument("--config-json", type=str, required=True, help="Path to config.json file")
    
    # Mixtral Baseline related
    parser.add_argument("--num-hidden-layers", type=int, required=True, help="Number of hidden layers")
    parser.add_argument("--intermediate-size", type=int, required=True, help="Intermediate size (expert size)")
    # Note: init_baseline_mixtral uses intermediate-size. init.sh uses moe-intermediate-size. We use intermediate-size to match baseline Mixtral as requested.
    
    # Routing Free related (from init.sh)
    parser.add_argument("--n-experts", type=int, required=True, help="Number of experts (n_experts)")
    
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save the initialized model")
    parser.add_argument("--model-name", type=str, required=True, help="Model name")
    parser.add_argument("--tokenizer-model", type=str, default="EleutherAI/gpt-neo-125M", help="Tokenizer model name or path")
    parser.add_argument("--bf16", action="store_true", default=True, help="Convert model to bfloat16 precision")
    
    args = parser.parse_args()

    initialize_model(
        config_json=args.config_json,
        num_hidden_layers=args.num_hidden_layers,   
        intermediate_size=args.intermediate_size,
        n_experts=args.n_experts,
        output_dir=args.output_dir,
        model_name=args.model_name,
        tokenizer_model=args.tokenizer_model,
        bf16=args.bf16
    )

if __name__ == "__main__":
    main() 

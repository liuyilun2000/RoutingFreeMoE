"""
Print trainable and filtered (non-embedding) parameter counts for baseline or RF Mixtral models.

Usage:
    python print_model_size.py --model-type baseline --model-dir ./init_baseline_mixtral/Mixtral_12L_128D
    python print_model_size.py --model-type rf       --model-dir ./init_mixtral_rf/RoutingFreeMixtral_12L_128D
"""
import argparse
import torch
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, MixtralForCausalLM, MixtralConfig

from utils import print_trainable_parameters, print_filtered_model_size

# Embedding-related parameter name substrings to exclude from "core model" size
EMBEDDING_PARAM_NAMES = ["embed_tokens", "lm_head"]


def load_baseline(model_dir: str) -> torch.nn.Module:
    config = MixtralConfig.from_pretrained(model_dir)
    model = MixtralForCausalLM.from_pretrained(
        model_dir,
        config=config,
        torch_dtype=torch.bfloat16,
        ignore_mismatched_sizes=True,
    )
    return model


def load_rf(model_dir: str) -> torch.nn.Module:
    from routing_free.mixtral_rf import (
        RoutingFreeMixtralConfig,
        RoutingFreeMixtralForCausalLM,
        RoutingFreeMixtralModel,
    )
    AutoConfig.register("routing_free_mixtral", RoutingFreeMixtralConfig)
    AutoModel.register(RoutingFreeMixtralConfig, RoutingFreeMixtralModel)
    AutoModelForCausalLM.register(RoutingFreeMixtralConfig, RoutingFreeMixtralForCausalLM)

    config = RoutingFreeMixtralConfig.from_pretrained(model_dir)
    model = RoutingFreeMixtralForCausalLM.from_pretrained(
        model_dir,
        config=config,
        torch_dtype=torch.bfloat16,
        ignore_mismatched_sizes=True,
    )
    return model


def main():
    parser = argparse.ArgumentParser(description="Print model size statistics")
    parser.add_argument("--model-type", choices=["baseline", "rf"], required=True,
                        help="Model type: 'baseline' (standard Mixtral) or 'rf' (Routing-Free Mixtral)")
    parser.add_argument("--model-dir", type=str, required=True,
                        help="Path to the model directory")
    args = parser.parse_args()

    print(f"Loading {args.model_type} model from: {args.model_dir}")

    if args.model_type == "baseline":
        model = load_baseline(args.model_dir)
    else:
        model = load_rf(args.model_dir)

    print(f"\n--- {args.model_type.upper()} model ---")
    print_trainable_parameters(model)
    print_filtered_model_size(model, param_names_to_exclude=EMBEDDING_PARAM_NAMES)


if __name__ == "__main__":
    main()

"""
Print trainable and filtered (non-embedding) parameter counts for baseline, RF, AoE, or ReMoE Mixtral models.

Usage:
    python print_model_size.py --model-type baseline --model-dir ./config/Mixtral_12L_128D
    python print_model_size.py --model-type rf       --model-dir ./config/RoutingFreeMixtral_12L_128D_rank32
    python print_model_size.py --model-type aoe      --model-dir ./config/AoE_12L_128D
    python print_model_size.py --model-type remoe    --model-dir ./config/ReMoE_12L_128D
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


def load_aoe(model_dir: str) -> torch.nn.Module:
    from routing_free.mixtral_aoe import (
        AoEMixtralConfig,
        AoEMixtralForCausalLM,
        AoEMixtralModel,
    )
    AutoConfig.register("aoe_mixtral", AoEMixtralConfig)
    AutoModel.register(AoEMixtralConfig, AoEMixtralModel)
    AutoModelForCausalLM.register(AoEMixtralConfig, AoEMixtralForCausalLM)

    config = AoEMixtralConfig.from_pretrained(model_dir)
    model = AoEMixtralForCausalLM.from_pretrained(
        model_dir,
        config=config,
        torch_dtype=torch.bfloat16,
        ignore_mismatched_sizes=True,
    )
    return model


def load_remoe(model_dir: str) -> torch.nn.Module:
    from routing_free.mixtral_remoe import (
        ReMoEMixtralConfig,
        ReMoEMixtralForCausalLM,
        ReMoEMixtralModel,
    )
    AutoConfig.register("remoe_mixtral", ReMoEMixtralConfig)
    AutoModel.register(ReMoEMixtralConfig, ReMoEMixtralModel)
    AutoModelForCausalLM.register(ReMoEMixtralConfig, ReMoEMixtralForCausalLM)

    config = ReMoEMixtralConfig.from_pretrained(model_dir)
    model = ReMoEMixtralForCausalLM.from_pretrained(
        model_dir,
        config=config,
        torch_dtype=torch.bfloat16,
        ignore_mismatched_sizes=True,
    )
    return model


LOADERS = {
    "baseline": load_baseline,
    "rf": load_rf,
    "aoe": load_aoe,
    "remoe": load_remoe,
}


def main():
    parser = argparse.ArgumentParser(description="Print model size statistics")
    parser.add_argument("--model-type", choices=list(LOADERS.keys()), required=True,
                        help="Model type: 'baseline', 'rf', 'aoe', or 'remoe'")
    parser.add_argument("--model-dir", type=str, required=True,
                        help="Path to the model directory")
    args = parser.parse_args()

    print(f"Loading {args.model_type} model from: {args.model_dir}")

    model = LOADERS[args.model_type](args.model_dir)

    print(f"\n--- {args.model_type.upper()} model ---")
    print_trainable_parameters(model)
    print_filtered_model_size(model, param_names_to_exclude=EMBEDDING_PARAM_NAMES)


if __name__ == "__main__":
    main()

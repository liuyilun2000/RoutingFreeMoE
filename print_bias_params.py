"""
Print the values of all parameters whose name contains "bias".

Usage (from RoutingFreeMoE directory):
  python print_bias_params.py
  python print_bias_params.py --model-dir ./OLMoE-1B-7B-Instruct-RFMoE-adapted-LR1/final_model
"""

import argparse
import os
import sys

import torch

_project_root = os.path.dirname(os.path.abspath(__file__))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from routing_free.olmoe_rf import (
    RoutingFreeOlmoeConfig,
    RoutingFreeOlmoeForCausalLM,
    RoutingFreeOlmoeModel,
)

AutoConfig.register("routing_free_olmoe", RoutingFreeOlmoeConfig)
AutoModel.register(RoutingFreeOlmoeConfig, RoutingFreeOlmoeModel)
AutoModelForCausalLM.register(RoutingFreeOlmoeConfig, RoutingFreeOlmoeForCausalLM)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-dir",
        type=str,
        default=os.path.join(_project_root, "OLMoE-1B-7B-Instruct-RFMoE-adapted-LR1", "final_model"),
        help="Path to the model directory",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.model_dir):
        raise FileNotFoundError(f"Model directory not found: {args.model_dir}")

    print(f"Loading model from {args.model_dir} …")
    model = RoutingFreeOlmoeForCausalLM.from_pretrained(
        args.model_dir,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
    )

    print("\nParameters whose name contains 'bias':\n")
    for name, param in model.named_parameters():
        if "bias" not in name.lower():
            continue
        p = param.detach()
        print(f"  {name}", end="")
        if p.numel() <= 10:
            print(f"    value: {p.tolist()}")
        else:
            print(f"    min: {p.min().item():.6g}  max: {p.max().item():.6g}  mean: {p.float().mean().item():.6g}")


if __name__ == "__main__":
    main()

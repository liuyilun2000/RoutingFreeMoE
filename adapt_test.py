"""
Load the final adapted OLMoE-RFMoE model and generate text with Hugging Face .generate().

Usage (from RoutingFreeMoE directory):
  python generate_adapted.py
  python generate_adapted.py --prompt "What is machine learning?"
  python generate_adapted.py --model-dir ./OLMoE-1B-7B-Instruct-RFMoE-adapted-LR1/final_model
"""

import argparse
import os
import sys

import torch

_project_root = os.path.dirname(os.path.abspath(__file__))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from transformers import AutoModelForCausalLM, AutoTokenizer

from routing_free.olmoe_rf import (
    RoutingFreeOlmoeConfig,
    RoutingFreeOlmoeForCausalLM,
    RoutingFreeOlmoeModel,
)
from transformers import AutoConfig, AutoModel

# Register custom OLMoE-RF classes so from_pretrained can load the config/model
AutoConfig.register("routing_free_olmoe", RoutingFreeOlmoeConfig)
AutoModel.register(RoutingFreeOlmoeConfig, RoutingFreeOlmoeModel)
AutoModelForCausalLM.register(RoutingFreeOlmoeConfig, RoutingFreeOlmoeForCausalLM)


def main():
    parser = argparse.ArgumentParser(description="Generate text with the adapted RFMoE model")
    parser.add_argument(
        "--model-dir",
        type=str,
        default=os.path.join(_project_root, "OLMoE-1B-7B-Instruct-RFMoE-adapted-LR1", "final_model"),
        help="Path to the final model directory (default: .../OLMoE-1B-7B-Instruct-RFMoE-adapted-LR1/final_model)",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="The meaning of life is",
        help="Input prompt for generation",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=20,
        help="Maximum number of new tokens to generate",
    )
    parser.add_argument(
        "--do-sample",
        action="store_true",
        help="Use sampling instead of greedy decoding",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (only used if --do-sample)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to load the model on (e.g. cuda:0). Default: auto (cuda if available)",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.model_dir):
        raise FileNotFoundError(f"Model directory not found: {args.model_dir}")

    # Reproducibility
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = args.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    use_cuda = device != "cpu" and (device.startswith("cuda") or device == "cuda")

    print(f"Loading tokenizer from {args.model_dir} …")
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print(f"Loading model from {args.model_dir} (dtype={dtype}, device={device}) …")
    load_kwargs = {"torch_dtype": dtype}
    if use_cuda:
        load_kwargs["device_map"] = device if device != "cuda" else "auto"
    model = RoutingFreeOlmoeForCausalLM.from_pretrained(args.model_dir, **load_kwargs)
    if not use_cuda:
        model = model.to(device)

    model.eval()

    model_device = next(model.parameters()).device
    inputs = tokenizer(args.prompt, return_tensors="pt").to(model_device)
    print(f"\nPrompt: {args.prompt!r}")
    print("Generating …")

    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=args.do_sample,
            temperature=args.temperature if args.do_sample else None,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # Decode only the generated part (skip input tokens)
    generated_ids = out[:, inputs["input_ids"].shape[1] :]
    response = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    print(f"\nGenerated ({generated_ids.shape[1]} new tokens):")
    print(response)
    print()
    return response


if __name__ == "__main__":
    main()

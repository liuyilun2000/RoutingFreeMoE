"""
Compute per-expert average activation rates across evaluation datasets.

For each (model_variant, dataset, layer), produces a [E] vector of activation
probabilities — the fraction of tokens that activate each expert.

Output: activation_data.pt containing:
    {
        "activation": {variant: {dataset: Tensor[L, E]}},
        "num_tokens": {variant: {dataset: int}},
        "layers": [0, 1, ..., L-1],
        "experts": [0, 1, ..., E-1],
        "datasets": ["qnli", "arc_easy", "openbookqa"],
        "variants": ["balanced", "token_only", "expert_only"],
    }

Usage:
    python extract_activation_by_dataset.py [--max-samples 500] [--device cpu]
"""

import argparse
import os
import sys
import torch
from pathlib import Path
from tqdm import tqdm

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from routing_free.mixtral_rf import (
    RoutingFreeMixtralForCausalLM,
    RoutingFreeMixtralConfig,
)

AutoConfig.register("routing_free_mixtral", RoutingFreeMixtralConfig)
AutoModelForCausalLM.register(RoutingFreeMixtralConfig, RoutingFreeMixtralForCausalLM)

# ── Checkpoints ──────────────────────────────────────────────────────
CHECKPOINTS = {
    "balanced": {
        "path": "output/mixtral_rf/S_rf_lr_1e-3_rank_32/final_model",
        "label": "E0.5T0.5",
    },
    "token_only": {
        "path": "output/mixtral_rf/S_rf_lr_1e-3_rank_32_E0.0_T1.0/final_model",
        "label": "E0.0T1.0",
    },
    "expert_only": {
        "path": "output/mixtral_rf/S_rf_lr_1e-3_rank_32_E1.0_T0.0/final_model",
        "label": "E1.0T0.0",
    },
}

# ── Datasets ─────────────────────────────────────────────────────────
DATASETS = {
    "qnli": {
        "hf_name": "nyu-mll/glue",
        "hf_subset": "qnli",
        "split": "validation",
        "text_fields": ["question", "sentence"],
    },
    "arc_easy": {
        "hf_name": "allenai/ai2_arc",
        "hf_subset": "ARC-Easy",
        "split": "test",
        "text_fields": ["question"],
    },
    "openbookqa": {
        "hf_name": "allenai/openbookqa",
        "hf_subset": "main",
        "split": "test",
        "text_fields": ["question_stem"],
    },
}


def load_model(model_path, device="cpu"):
    config = AutoConfig.from_pretrained(model_path)
    config.output_router_logits = True
    config.output_gate_scores = True
    model = AutoModelForCausalLM.from_pretrained(
        model_path, config=config, torch_dtype=torch.float32
    )
    model.to(device).eval()
    return model


def load_dataset_texts(ds_info, max_samples, hf_cache_dir=None):
    """Load text samples from a HuggingFace dataset."""
    from datasets import load_dataset

    kwargs = {}
    if hf_cache_dir:
        kwargs["cache_dir"] = hf_cache_dir

    ds = load_dataset(
        ds_info["hf_name"],
        ds_info.get("hf_subset"),
        split=ds_info["split"],
        **kwargs,
    )

    texts = []
    for sample in ds:
        parts = [sample.get(f, "") for f in ds_info["text_fields"]]
        text = " ".join(p for p in parts if p).strip()
        if text:
            texts.append(text)
        if len(texts) >= max_samples:
            break

    return texts


def compute_activation_rates(model, tokenizer, texts, device="cpu", max_len=512):
    """
    Compute per-expert activation rate across all tokens from texts.
    Returns: activation_rate [L, E], total_tokens int
    """
    # Get model dimensions from first pass
    dummy = tokenizer("hello", return_tensors="pt")
    with torch.no_grad():
        dummy_out = model(
            input_ids=dummy["input_ids"].to(device),
            attention_mask=dummy["attention_mask"].to(device),
            output_gate_scores=True,
        )
    L = len(dummy_out.router_logits)
    E = dummy_out.router_logits[0].shape[-1]

    # Accumulate counts
    active_count = torch.zeros(L, E, dtype=torch.float64)
    total_tokens = 0

    for text in tqdm(texts, desc="    Processing", leave=False):
        inputs = tokenizer(
            text, return_tensors="pt", truncation=True, max_length=max_len
        )
        input_ids = inputs["input_ids"].to(device)
        T = input_ids.shape[1]

        with torch.no_grad():
            out = model(
                input_ids=input_ids,
                attention_mask=inputs["attention_mask"].to(device),
                output_gate_scores=True,
            )

        for l in range(L):
            mask = (out.router_logits[l].squeeze(0) != float("-inf"))  # [T, E]
            active_count[l] += mask.cpu().float().sum(dim=0)  # [E]

        total_tokens += T

    activation_rate = active_count / total_tokens  # [L, E]
    return activation_rate.float(), total_tokens


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-samples", type=int, default=500)
    parser.add_argument("--max-len", type=int, default=512)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output", type=str, default="./gate_scores/activation_data.pt")
    parser.add_argument("--hf-cache-dir", type=str, default=None)
    args = parser.parse_args()

    root = Path(PROJECT_ROOT)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    # Load tokenizer
    tok_path = str(root / CHECKPOINTS["balanced"]["path"])
    print(f"Loading tokenizer from: {tok_path}")
    tokenizer = AutoTokenizer.from_pretrained(tok_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load all datasets
    all_texts = {}
    for ds_name, ds_info in DATASETS.items():
        print(f"\nLoading dataset: {ds_name}")
        texts = load_dataset_texts(ds_info, args.max_samples, args.hf_cache_dir)
        print(f"  Loaded {len(texts)} samples")
        all_texts[ds_name] = texts

    # Process each model variant
    activation = {}
    num_tokens = {}

    for var_name, var_info in CHECKPOINTS.items():
        model_path = str(root / var_info["path"])
        print(f"\n{'='*60}")
        print(f"Loading model: {var_info['label']} ({var_name})")
        print(f"  Path: {model_path}")
        model = load_model(model_path, args.device)

        activation[var_name] = {}
        num_tokens[var_name] = {}

        for ds_name, texts in all_texts.items():
            print(f"\n  Dataset: {ds_name} ({len(texts)} samples)")
            act_rate, n_tok = compute_activation_rates(
                model, tokenizer, texts, args.device, args.max_len
            )
            activation[var_name][ds_name] = act_rate  # [L, E]
            num_tokens[var_name][ds_name] = n_tok

            # Print summary
            print(f"    Total tokens: {n_tok}")
            print(f"    Activation rate: mean={act_rate.mean():.4f}, "
                  f"std={act_rate.std():.4f}, "
                  f"min={act_rate.min():.4f}, max={act_rate.max():.4f}")

        del model
        torch.cuda.empty_cache() if args.device != "cpu" else None

    # Save
    L = list(activation.values())[0][list(DATASETS.keys())[0]].shape[0]
    E = list(activation.values())[0][list(DATASETS.keys())[0]].shape[1]

    save_data = {
        "activation": activation,
        "num_tokens": num_tokens,
        "layers": list(range(L)),
        "experts": list(range(E)),
        "datasets": list(DATASETS.keys()),
        "variants": list(CHECKPOINTS.keys()),
        "variant_labels": {k: v["label"] for k, v in CHECKPOINTS.items()},
    }

    torch.save(save_data, args.output)
    print(f"\nSaved to {args.output}")
    print(f"Shape per (variant, dataset): [{L}, {E}]")


if __name__ == "__main__":
    main()

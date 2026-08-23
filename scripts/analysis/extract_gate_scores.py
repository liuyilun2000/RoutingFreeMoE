"""
Extract per-expert gate scores from Routing-Free MoE checkpoints.

Modes:
  1. --sentence "..."        : single sentence
  2. --use-dataset           : long sequence from OpenWebText
  3. --screen                : screen many sentences, pick the one showing
                               the biggest imbalance gap between configs

Usage:
    # Screen 200 sentences and save the best one
    python extract_gate_scores.py --screen --num-screen 200 --seq-len 128

    # Direct extraction
    python extract_gate_scores.py --use-dataset --seq-len 512
"""

import argparse
import os
import sys
import numpy as np
import torch
from pathlib import Path

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from routing_free.mixtral_rf import (
    RoutingFreeMixtralForCausalLM,
    RoutingFreeMixtralConfig,
)

# Register custom model
AutoConfig.register("routing_free_mixtral", RoutingFreeMixtralConfig)
AutoModelForCausalLM.register(RoutingFreeMixtralConfig, RoutingFreeMixtralForCausalLM)

# ── Checkpoints ──────────────────────────────────────────────────────
CHECKPOINTS = {
    "balanced": {
        "path": "output/mixtral_rf/S_rf_lr_1e-3_rank_32/final_model",
        "label": "Balanced (E=0.5, T=0.5)",
    },
    "token_only": {
        "path": "output/mixtral_rf/S_rf_lr_1e-3_rank_32_E0.0_T1.0/final_model",
        "label": "Token-only (E=0.0, T=1.0)",
    },
    "expert_only": {
        "path": "output/mixtral_rf/S_rf_lr_1e-3_rank_32_E1.0_T0.0/final_model",
        "label": "Expert-only (E=1.0, T=0.0)",
    },
}

DEFAULT_SENTENCE = (
    "The Mixture of Experts architecture enables efficient scaling "
    "of language models by activating only a subset of parameters for each token."
)


# ── Metrics ──────────────────────────────────────────────────────────

def compute_cv(values):
    """Coefficient of variation: std / mean."""
    m = values.mean()
    return float(values.std() / m) if m > 0 else 0.0


def compute_imbalance_score(gate_masks):
    """
    Compute a single imbalance score from gate_masks [L, T, E].
    Returns (cv_expert, cv_token):
      - cv_expert: avg CV of tokens-per-expert across layers (high = expert imbalance)
      - cv_token:  avg CV of experts-per-token across layers (high = token imbalance)
    """
    L, T, E = gate_masks.shape
    cv_experts = []
    cv_tokens = []
    for l in range(L):
        tpe = gate_masks[l].sum(dim=0).float().numpy()  # [E]
        ept = gate_masks[l].sum(dim=1).float().numpy()   # [T]
        cv_experts.append(compute_cv(tpe))
        cv_tokens.append(compute_cv(ept))
    return np.mean(cv_experts), np.mean(cv_tokens)


# ── Model helpers ────────────────────────────────────────────────────

def load_model(model_path, device="cpu"):
    """Load a single model."""
    config = AutoConfig.from_pretrained(model_path)
    config.output_router_logits = True
    config.output_gate_scores = True
    model = AutoModelForCausalLM.from_pretrained(
        model_path, config=config, torch_dtype=torch.float32
    )
    model.to(device).eval()
    return model


def forward_pass(model, input_ids, device="cpu"):
    """Run forward pass and return gate_masks [L, T, E]."""
    input_ids = input_ids.to(device)
    with torch.no_grad():
        out = model(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            output_gate_scores=True,
        )
    gate_scores = torch.stack(
        [gs.squeeze(0) for gs in out.router_logits], dim=0
    )  # [L, T, E]
    gate_masks = gate_scores != float("-inf")
    return gate_scores.cpu(), gate_masks.cpu()


def extract_full(model, tokenizer, input_ids, device="cpu"):
    """Full extraction with token decoding."""
    gate_scores, gate_masks = forward_pass(model, input_ids, device)
    tokens = [tokenizer.decode(tid) for tid in input_ids[0].tolist()]
    return {
        "input_ids": input_ids.cpu(),
        "gate_scores": gate_scores,
        "gate_masks": gate_masks,
        "tokens": tokens,
    }


# ── Dataset helpers ──────────────────────────────────────────────────

def load_long_text(dataset_name, seq_len, tokenizer, hf_cache_dir=None):
    """Load and concatenate texts from dataset until we have seq_len tokens."""
    from datasets import load_dataset
    kwargs = {"cache_dir": hf_cache_dir} if hf_cache_dir else {}
    print(f"Loading dataset: {dataset_name} ...")
    ds = load_dataset(dataset_name, split="train", streaming=True, **kwargs)
    all_ids = []
    for sample in ds:
        text = sample.get("text", "")
        if not text.strip():
            continue
        ids = tokenizer.encode(text, add_special_tokens=False)
        all_ids.extend(ids)
        if len(all_ids) >= seq_len:
            break
    return all_ids[:seq_len]


def load_sentences(dataset_name, num_sentences, min_tokens, max_tokens,
                   tokenizer, hf_cache_dir=None):
    """Load individual sentences from dataset, filtering by token length."""
    from datasets import load_dataset
    kwargs = {"cache_dir": hf_cache_dir} if hf_cache_dir else {}
    print(f"Loading sentences from {dataset_name} ...")
    ds = load_dataset(dataset_name, split="train", streaming=True, **kwargs)

    sentences = []
    seen = 0
    for sample in ds:
        text = sample.get("text", "").strip()
        if not text:
            continue
        ids = tokenizer.encode(text, add_special_tokens=False)
        if min_tokens <= len(ids) <= max_tokens:
            sentences.append((text, ids))
            if len(sentences) >= num_sentences:
                break
        seen += 1
        if seen > num_sentences * 20:  # don't scan forever
            break

    print(f"  Found {len(sentences)} sentences ({min_tokens}-{max_tokens} tokens)")
    return sentences


# ── Screen mode ──────────────────────────────────────────────────────

def screen_sentences(models, tokenizer, args):
    """
    Screen many sentences to find the one with the biggest imbalance
    gap between balanced and imbalanced configs.
    """
    root = Path(PROJECT_ROOT)

    sentences = load_sentences(
        args.dataset, args.num_screen, args.min_tokens, args.seq_len,
        tokenizer, args.hf_cache_dir,
    )

    print(f"\nScreening {len(sentences)} sentences across 3 models ...\n")

    results = []
    for idx, (text, ids) in enumerate(sentences):
        input_ids = torch.tensor([ids], dtype=torch.long)
        scores_per_config = {}

        for name, model in models.items():
            _, gate_masks = forward_pass(model, input_ids, args.device)
            cv_exp, cv_tok = compute_imbalance_score(gate_masks)
            scores_per_config[name] = {"cv_expert": cv_exp, "cv_token": cv_tok}

        # Compute gap: how much more imbalanced are the ablation configs
        # vs balanced. Higher = better for our story.
        bal = scores_per_config["balanced"]
        tok = scores_per_config["token_only"]
        exp = scores_per_config["expert_only"]

        # Token-only should have higher cv_expert (expert imbalance)
        gap_expert = tok["cv_expert"] - bal["cv_expert"]
        # Expert-only should have higher cv_token (token imbalance)
        gap_token = exp["cv_token"] - bal["cv_token"]
        # Combined score: sum of both gaps (both should be positive)
        combined_gap = gap_expert + gap_token

        results.append({
            "idx": idx,
            "text": text[:80],
            "n_tokens": len(ids),
            "scores": scores_per_config,
            "gap_expert": gap_expert,
            "gap_token": gap_token,
            "combined_gap": combined_gap,
            "input_ids": input_ids,
        })

        if (idx + 1) % 20 == 0:
            print(f"  {idx+1}/{len(sentences)} screened ...")

    # Sort by combined gap (descending)
    results.sort(key=lambda r: r["combined_gap"], reverse=True)

    # Print top 10
    print(f"\n{'='*80}")
    print(f"Top 10 sentences by imbalance gap (balanced is better):")
    print(f"{'='*80}")
    for i, r in enumerate(results[:10]):
        s = r["scores"]
        print(f"\n#{i+1} (gap={r['combined_gap']:.4f}, tokens={r['n_tokens']})")
        print(f"  Text: {r['text']}...")
        print(f"  Balanced:    cv_expert={s['balanced']['cv_expert']:.4f}, "
              f"cv_token={s['balanced']['cv_token']:.4f}")
        print(f"  Token-only:  cv_expert={s['token_only']['cv_expert']:.4f}, "
              f"cv_token={s['token_only']['cv_token']:.4f}  "
              f"(expert gap={r['gap_expert']:+.4f})")
        print(f"  Expert-only: cv_expert={s['expert_only']['cv_expert']:.4f}, "
              f"cv_token={s['expert_only']['cv_token']:.4f}  "
              f"(token gap={r['gap_token']:+.4f})")

    return results


# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Extract gate scores from RF-MoE checkpoints")
    parser.add_argument("--sentence", type=str, default=None)
    parser.add_argument("--use-dataset", action="store_true")
    parser.add_argument("--dataset", type=str, default="Skylion007/openwebtext")
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--min-tokens", type=int, default=64,
                        help="Min tokens per sentence in screen mode")
    parser.add_argument("--hf-cache-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="./gate_scores")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--tokenizer", type=str, default=None)

    # Screen mode
    parser.add_argument("--screen", action="store_true",
                        help="Screen many sentences to find the best one for visualization")
    parser.add_argument("--num-screen", type=int, default=200,
                        help="Number of sentences to screen")
    parser.add_argument("--save-top", type=int, default=3,
                        help="Save top N sentences from screening")
    args = parser.parse_args()

    root = Path(PROJECT_ROOT)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load tokenizer
    tok_path = args.tokenizer or str(root / CHECKPOINTS["balanced"]["path"])
    print(f"Loading tokenizer from: {tok_path}")
    tokenizer = AutoTokenizer.from_pretrained(tok_path)

    if args.screen:
        # ── Screen mode: load all models once, screen many sentences ──
        print("\nLoading all 3 models ...")
        models = {}
        for name, info in CHECKPOINTS.items():
            model_path = str(root / info["path"])
            print(f"  Loading {info['label']} from {model_path}")
            models[name] = load_model(model_path, args.device)

        ranked = screen_sentences(models, tokenizer, args)

        # Save top N
        print(f"\nSaving top {args.save_top} results ...")
        for rank, r in enumerate(ranked[:args.save_top]):
            input_ids = r["input_ids"]
            for name, model in models.items():
                result = extract_full(model, tokenizer, input_ids, args.device)
                result["config_name"] = name
                result["config_label"] = CHECKPOINTS[name]["label"]
                result["screen_rank"] = rank
                result["combined_gap"] = r["combined_gap"]

                save_path = out_dir / f"gate_scores_{name}_rank{rank}.pt"
                torch.save(result, str(save_path))

            print(f"  Rank {rank}: gap={r['combined_gap']:.4f}, "
                  f"tokens={r['n_tokens']}, text={r['text'][:60]}...")

        # Also save the best as the default files (for plot_gate_heatmap.py)
        if ranked:
            best = ranked[0]
            input_ids = best["input_ids"]
            for name, model in models.items():
                result = extract_full(model, tokenizer, input_ids, args.device)
                result["config_name"] = name
                result["config_label"] = CHECKPOINTS[name]["label"]
                save_path = out_dir / f"gate_scores_{name}.pt"
                torch.save(result, str(save_path))
            print(f"\nBest sentence saved as default gate_scores_*.pt files.")

    else:
        # ── Direct extraction mode ──
        if args.use_dataset:
            token_ids = load_long_text(args.dataset, args.seq_len, tokenizer,
                                       args.hf_cache_dir)
            input_ids = torch.tensor([token_ids], dtype=torch.long)
            print(f"Input shape: {input_ids.shape}")
        else:
            sentence = args.sentence or DEFAULT_SENTENCE
            print(f"Sentence: {sentence!r}")
            inputs = tokenizer(sentence, return_tensors="pt")
            input_ids = inputs["input_ids"]

        T = input_ids.shape[1]
        print(f"Total tokens: {T}\n")

        for name, info in CHECKPOINTS.items():
            model_path = str(root / info["path"])
            print(f"[{info['label']}]  Loading from {model_path} ...")

            model = load_model(model_path, args.device)
            result = extract_full(model, tokenizer, input_ids, args.device)
            result["config_name"] = name
            result["config_label"] = info["label"]

            L, T, E = result["gate_scores"].shape
            n_active = result["gate_masks"].sum().item()
            total = L * T * E
            density = n_active / total if total > 0 else 0

            print(f"  Shape: L={L}, T={T}, E={E}")
            print(f"  Active: {n_active}/{total} ({density:.2%})")

            active_scores = result["gate_scores"][result["gate_masks"]]
            if len(active_scores) > 0:
                print(f"  Score range: {active_scores.min():.4f} ~ {active_scores.max():.4f}")

            save_path = out_dir / f"gate_scores_{name}.pt"
            torch.save(result, str(save_path))
            print(f"  Saved to {save_path}\n")

            del model  # free memory

    print("Done. Use plot_gate_heatmap.py to visualize.")


if __name__ == "__main__":
    main()

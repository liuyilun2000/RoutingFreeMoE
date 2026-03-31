#!/usr/bin/env python3
"""
Benchmark prefill and decode latency for baseline and RF-MoE models.

This script uses HuggingFace loading with `device_map="auto"` so model
partitions are handled by HF/accelerate across visible GPUs.
"""

import argparse
import os
import random
import sys
import time
from typing import Dict

import numpy as np
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark prefill/decode speed for baseline and RoutingFree models."
    )
    parser.add_argument(
        "--baseline-model-dir", type=str, required=True, help="Path to baseline model directory."
    )
    parser.add_argument(
        "--rf-model-dir", type=str, required=True, help="Path to routing-free model directory."
    )
    parser.add_argument(
        "--baseline-model-type",
        type=str,
        default="auto",
        choices=["auto", "mixtral", "olmoe"],
        help="Baseline model family for registration logic.",
    )
    parser.add_argument(
        "--rf-model-type",
        type=str,
        default="routing_free_mixtral",
        choices=["routing_free_mixtral", "routing_free_olmoe"],
        help="Routing-free model family for registration logic.",
    )
    parser.add_argument(
        "--cuda-devices",
        type=str,
        default="1,2,4",
        help="Comma-separated physical CUDA IDs to use, e.g. '1,2,4'.",
    )
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size.")
    parser.add_argument(
        "--prefill-length", type=int, default=1024, help="Input sequence length for prefill."
    )
    parser.add_argument(
        "--decode-steps", type=int, default=128, help="Number of decode steps (1 token/step)."
    )
    parser.add_argument(
        "--warmup-iters", type=int, default=3, help="Warmup iterations for each benchmark."
    )
    parser.add_argument("--repeats", type=int, default=10, help="Timed iterations for each benchmark.")
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float16", "bfloat16", "float32"],
        help="Torch dtype for loading model.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--trust-remote-code", action="store_true", help="Enable trust_remote_code for HF loading."
    )
    return parser.parse_args()


def register_routing_free_models(project_root: str, rf_model_type: str) -> None:
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    if rf_model_type == "routing_free_mixtral":
        from routing_free.mixtral_rf import RoutingFreeMixtralConfig, RoutingFreeMixtralForCausalLM

        AutoConfig.register("routing_free_mixtral", RoutingFreeMixtralConfig)
        AutoModelForCausalLM.register(RoutingFreeMixtralConfig, RoutingFreeMixtralForCausalLM)
    elif rf_model_type == "routing_free_olmoe":
        from routing_free.configuration_olmoe_rf import RoutingFreeOlmoeConfig
        from routing_free.olmoe_rf import RoutingFreeOlmoeForCausalLM

        AutoConfig.register("routing_free_olmoe", RoutingFreeOlmoeConfig)
        AutoModelForCausalLM.register(RoutingFreeOlmoeConfig, RoutingFreeOlmoeForCausalLM)
    else:
        raise ValueError(f"Unsupported RF model type: {rf_model_type}")


def to_torch_dtype(dtype: str) -> torch.dtype:
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    return mapping[dtype]


def choose_input_device(model: torch.nn.Module) -> torch.device:
    if hasattr(model, "hf_device_map"):
        for dev in model.hf_device_map.values():
            if isinstance(dev, int):
                return torch.device(f"cuda:{dev}")
            if isinstance(dev, str) and dev.startswith("cuda"):
                return torch.device(dev)
    return next(model.parameters()).device


def _disable_aux_loss_paths_for_inference(model: torch.nn.Module) -> None:
    # Some configs enable router/gate outputs which can trigger aux-loss codepaths
    # (e.g. Mixtral load-balancing loss). These are irrelevant for inference timing
    # and can crash under KV-cache decode depending on model/config.
    cfg = getattr(model, "config", None)
    if cfg is None:
        return

    if hasattr(cfg, "output_router_logits"):
        cfg.output_router_logits = False
    if hasattr(cfg, "output_gate_scores"):
        cfg.output_gate_scores = False
    if hasattr(cfg, "router_aux_loss_coef"):
        cfg.router_aux_loss_coef = 0.0

    # Runtime flags are sometimes copied from config into submodules at init.
    # Force-disable gate-score collection recursively for inference benchmarks.
    for module in model.modules():
        if hasattr(module, "output_gate_scores"):
            module.output_gate_scores = False


def load_model(model_dir: str, dtype: torch.dtype, trust_remote_code: bool) -> tuple:
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        dtype=dtype,
        device_map="auto",
        trust_remote_code=trust_remote_code,
    )
    model.eval()
    _disable_aux_loss_paths_for_inference(model)
    return model, tokenizer


def benchmark_prefill(
    model: torch.nn.Module,
    batch_size: int,
    prefill_length: int,
    vocab_size: int,
    warmup: int,
    repeats: int,
    seed: int,
) -> Dict[str, float]:
    input_device = choose_input_device(model)
    g = torch.Generator(device=input_device).manual_seed(seed)
    input_ids = torch.randint(0, vocab_size, (batch_size, prefill_length), generator=g, device=input_device)

    with torch.no_grad():
        for _ in range(warmup):
            _ = model(input_ids=input_ids, use_cache=True)
        torch.cuda.synchronize()

    times = []
    with torch.no_grad():
        for _ in range(repeats):
            start = time.perf_counter()
            _ = model(input_ids=input_ids, use_cache=True)
            torch.cuda.synchronize()
            times.append(time.perf_counter() - start)

    avg_s = float(np.mean(times))
    toks_per_s = (batch_size * prefill_length) / avg_s
    return {"prefill_avg_s": avg_s, "prefill_toks_per_s": toks_per_s}


def benchmark_decode(
    model: torch.nn.Module,
    batch_size: int,
    prefill_length: int,
    decode_steps: int,
    vocab_size: int,
    warmup: int,
    repeats: int,
    seed: int,
) -> Dict[str, float]:
    input_device = choose_input_device(model)
    g = torch.Generator(device=input_device).manual_seed(seed + 1)
    prompt_ids = torch.randint(0, vocab_size, (batch_size, prefill_length), generator=g, device=input_device)

    def run_one() -> float:
        attention_mask = torch.ones((batch_size, prefill_length), device=input_device, dtype=torch.long)
        with torch.no_grad():
            out = model(input_ids=prompt_ids, attention_mask=attention_mask, use_cache=True)
            past_key_values = out.past_key_values

            next_token = torch.randint(0, vocab_size, (batch_size, 1), generator=g, device=input_device)
            start = time.perf_counter()
            for _ in range(decode_steps):
                out = model(
                    input_ids=next_token,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
                past_key_values = out.past_key_values
                next_token = torch.randint(0, vocab_size, (batch_size, 1), generator=g, device=input_device)
                attention_mask = torch.cat(
                    [attention_mask, torch.ones((batch_size, 1), device=input_device, dtype=torch.long)],
                    dim=1,
                )
            torch.cuda.synchronize()
        return time.perf_counter() - start

    for _ in range(warmup):
        _ = run_one()

    times = [run_one() for _ in range(repeats)]
    avg_s = float(np.mean(times))
    toks_per_s = (batch_size * decode_steps) / avg_s
    ms_per_tok = (avg_s * 1000.0) / decode_steps
    return {
        "decode_avg_s": avg_s,
        "decode_toks_per_s": toks_per_s,
        "decode_ms_per_tok": ms_per_tok,
    }


def benchmark_model(
    label: str,
    model_dir: str,
    dtype: torch.dtype,
    batch_size: int,
    prefill_length: int,
    decode_steps: int,
    warmup: int,
    repeats: int,
    seed: int,
    trust_remote_code: bool,
) -> Dict[str, float]:
    print(f"\n=== Loading {label}: {model_dir}")
    model, _ = load_model(model_dir, dtype=dtype, trust_remote_code=trust_remote_code)
    input_device = choose_input_device(model)
    n_params = sum(p.numel() for p in model.parameters())
    vocab_size = int(model.config.vocab_size)

    print(f"{label} input device: {input_device}")
    print(f"{label} parameters: {n_params:,}")
    print(f"{label} vocab_size: {vocab_size}")

    prefill = benchmark_prefill(
        model=model,
        batch_size=batch_size,
        prefill_length=prefill_length,
        vocab_size=vocab_size,
        warmup=warmup,
        repeats=repeats,
        seed=seed,
    )
    decode = benchmark_decode(
        model=model,
        batch_size=batch_size,
        prefill_length=prefill_length,
        decode_steps=decode_steps,
        vocab_size=vocab_size,
        warmup=warmup,
        repeats=repeats,
        seed=seed,
    )
    metrics = {**prefill, **decode}
    print(
        f"{label} prefill: {metrics['prefill_avg_s']:.4f}s "
        f"({metrics['prefill_toks_per_s']:.2f} tok/s)"
    )
    print(
        f"{label} decode : {metrics['decode_avg_s']:.4f}s total, "
        f"{metrics['decode_ms_per_tok']:.3f} ms/token "
        f"({metrics['decode_toks_per_s']:.2f} tok/s)"
    )
    return metrics


def main() -> None:
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_devices

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. This benchmark is intended for GPU execution.")

    set_seed(args.seed)
    dtype = to_torch_dtype(args.dtype)
    project_root = os.path.dirname(os.path.abspath(__file__))

    register_routing_free_models(project_root, args.rf_model_type)

    print(f"CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}")
    print(f"Visible GPU count={torch.cuda.device_count()}")
    print(
        f"dtype={args.dtype}, seed={args.seed}, "
        f"baseline_model_type={args.baseline_model_type}, rf_model_type={args.rf_model_type}"
    )
    print(
        f"batch_size={args.batch_size}, prefill_length={args.prefill_length}, "
        f"decode_steps={args.decode_steps}, warmup={args.warmup_iters}, repeats={args.repeats}"
    )

    baseline_metrics = benchmark_model(
        label="Baseline",
        model_dir=args.baseline_model_dir,
        dtype=dtype,
        batch_size=args.batch_size,
        prefill_length=args.prefill_length,
        decode_steps=args.decode_steps,
        warmup=args.warmup_iters,
        repeats=args.repeats,
        seed=args.seed,
        trust_remote_code=args.trust_remote_code,
    )
    rf_metrics = benchmark_model(
        label="RoutingFree",
        model_dir=args.rf_model_dir,
        dtype=dtype,
        batch_size=args.batch_size,
        prefill_length=args.prefill_length,
        decode_steps=args.decode_steps,
        warmup=args.warmup_iters,
        repeats=args.repeats,
        seed=args.seed,
        trust_remote_code=args.trust_remote_code,
    )

    print("\n=== Summary ===")
    print(
        f"Prefill tok/s     Baseline={baseline_metrics['prefill_toks_per_s']:.2f} | "
        f"RoutingFree={rf_metrics['prefill_toks_per_s']:.2f}"
    )
    print(
        f"Decode ms/token   Baseline={baseline_metrics['decode_ms_per_tok']:.3f} | "
        f"RoutingFree={rf_metrics['decode_ms_per_tok']:.3f}"
    )
    print(
        f"Decode tok/s      Baseline={baseline_metrics['decode_toks_per_s']:.2f} | "
        f"RoutingFree={rf_metrics['decode_toks_per_s']:.2f}"
    )


if __name__ == "__main__":
    main()

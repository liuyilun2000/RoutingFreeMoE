#!/usr/bin/env python3
"""
Benchmark prefill and decode latency for baseline and RF-MoE models.

This script uses HuggingFace loading with `device_map="auto"` so model
partitions are handled by HF/accelerate across visible GPUs.
"""

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.distributed as dist
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
        "--baseline-model",
        "--baseline-model-dir",
        dest="baseline_model",
        type=str,
        required=True,
        help="Baseline model reference: local directory path or HF repo id.",
    )
    parser.add_argument(
        "--rf-model",
        "--rf-model-dir",
        dest="rf_model",
        type=str,
        required=True,
        help="Routing-free model reference: local directory path or HF repo id.",
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
        "--verbose-iter",
        action="store_true",
        help="Print every warmup and timed iteration latency.",
    )
    parser.add_argument(
        "--trust-remote-code", action="store_true", help="Enable trust_remote_code for HF loading."
    )
    parser.add_argument(
        "--use-real-text",
        action="store_true",
        help="Use real text prompts sampled from a cached dataset subset.",
    )
    parser.add_argument(
        "--text-dataset-name",
        type=str,
        default="Skylion007/openwebtext",
        help="HF dataset used for real-text benchmark mode.",
    )
    parser.add_argument(
        "--text-dataset-split",
        type=str,
        default="train",
        help="Dataset split used for real-text benchmark mode.",
    )
    parser.add_argument(
        "--text-sample-ratio",
        type=float,
        default=0.001,
        help="Fraction of split to cache locally (e.g. 0.001 = 0.1%%).",
    )
    parser.add_argument(
        "--text-cache-dir",
        type=str,
        default=None,
        help="Local cache dir for dataset subset and HF dataset cache.",
    )
    parser.add_argument(
        "--text-column",
        type=str,
        default="text",
        help="Column containing raw text.",
    )
    parser.add_argument(
        "--expert-parallel",
        action="store_true",
        help="Enable expert-parallel execution for RoutingFree model under torch.distributed.",
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


def load_hf_token() -> str | None:
    # Prefer explicit runtime env; fall back to .env next to project root.
    for key in ("HF_TOKEN", "HF_READ_TOKEN", "HUGGINGFACE_HUB_TOKEN"):
        token = os.environ.get(key)
        if token:
            return token

    env_path = Path(__file__).resolve().parent.parent / ".env"
    if not env_path.exists():
        return None

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key not in {"HF_TOKEN", "HF_READ_TOKEN", "HUGGINGFACE_HUB_TOKEN"}:
            continue
        token = value.strip().strip('"').strip("'")
        if token:
            return token
    return None


def setup_distributed() -> dict:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    is_distributed = world_size > 1
    if is_distributed and not dist.is_initialized():
        dist.init_process_group(backend="nccl", init_method="env://")
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    return {
        "is_distributed": is_distributed,
        "world_size": world_size,
        "rank": rank,
        "local_rank": local_rank,
        "is_main": rank == 0,
    }


def load_model(
    model_dir: str,
    dtype: torch.dtype,
    trust_remote_code: bool,
    hf_token: str | None,
    local_rank: int,
    use_distributed_model_replica: bool,
) -> tuple:
    tokenizer = AutoTokenizer.from_pretrained(
        model_dir,
        trust_remote_code=trust_remote_code,
        token=hf_token,
    )
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    if use_distributed_model_replica:
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            dtype=dtype,
            trust_remote_code=trust_remote_code,
            token=hf_token,
        )
        model.to(torch.device(f"cuda:{local_rank}"))
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            dtype=dtype,
            device_map="auto",
            trust_remote_code=trust_remote_code,
            token=hf_token,
        )
    model.eval()
    _disable_aux_loss_paths_for_inference(model)
    return model, tokenizer


def _safe_dataset_tag(name: str) -> str:
    return name.replace("/", "_").replace(":", "_")


def _build_or_load_text_subset(
    dataset_name: str,
    split: str,
    sample_ratio: float,
    cache_dir: str,
    text_column: str,
    seed: int,
) -> list[str]:
    from datasets import load_dataset, load_dataset_builder

    if sample_ratio <= 0.0 or sample_ratio > 1.0:
        raise ValueError(f"text sample ratio must be in (0, 1], got {sample_ratio}")

    os.makedirs(cache_dir, exist_ok=True)
    ratio_tag = f"{sample_ratio:.6f}".rstrip("0").rstrip(".")
    subset_file = os.path.join(
        cache_dir,
        f"text_subset_{_safe_dataset_tag(dataset_name)}_{split}_{ratio_tag}_seed{seed}.jsonl",
    )

    if os.path.exists(subset_file):
        texts: list[str] = []
        with open(subset_file, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                text = item.get("text", "")
                if isinstance(text, str) and text.strip():
                    texts.append(text)
        if texts:
            print(f"Loaded cached text subset: {subset_file} ({len(texts)} rows)")
            return texts

    builder = load_dataset_builder(dataset_name, cache_dir=cache_dir)
    split_info = builder.info.splits.get(split)
    if split_info is None or split_info.num_examples is None:
        raise ValueError(f"Could not determine num_examples for {dataset_name}:{split}")
    total_examples = int(split_info.num_examples)
    target_examples = max(1, int(total_examples * sample_ratio))
    print(
        f"Building local text subset from {dataset_name}:{split} "
        f"(target={target_examples}/{total_examples}, ratio={sample_ratio})"
    )

    stream = load_dataset(
        dataset_name,
        split=split,
        streaming=True,
        cache_dir=cache_dir,
    ).shuffle(seed=seed, buffer_size=100_000)

    texts = []
    for row in stream.take(target_examples):
        text = row.get(text_column, None)
        if isinstance(text, str) and text.strip():
            texts.append(text)

    if not texts:
        raise RuntimeError(
            f"No valid text rows collected from {dataset_name}:{split} using column '{text_column}'"
        )

    with open(subset_file, "w", encoding="utf-8") as f:
        for text in texts:
            f.write(json.dumps({"text": text}, ensure_ascii=True) + "\n")
    print(f"Saved text subset cache: {subset_file} ({len(texts)} rows)")
    return texts


def _build_real_text_token_pool(
    tokenizer,
    prefill_length: int,
    decode_steps: int,
    cache_dir: str,
    dataset_name: str,
    dataset_split: str,
    sample_ratio: float,
    text_column: str,
    seed: int,
) -> torch.Tensor:
    texts = _build_or_load_text_subset(
        dataset_name=dataset_name,
        split=dataset_split,
        sample_ratio=sample_ratio,
        cache_dir=cache_dir,
        text_column=text_column,
        seed=seed,
    )
    total_len = prefill_length + decode_steps
    token_seqs = []
    for t in texts:
        enc = tokenizer(
            t,
            add_special_tokens=False,
            truncation=True,
            max_length=total_len,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        ids = enc["input_ids"]
        if len(ids) == total_len:
            token_seqs.append(ids)

    if not token_seqs:
        raise RuntimeError(
            f"No tokenized text reached length >= {total_len}. "
            "Try lower prefill/decode lengths or increase text sample ratio."
        )

    pool = torch.tensor(token_seqs, dtype=torch.long)
    print(f"Real-text token pool ready: {pool.shape[0]} sequences x {pool.shape[1]} tokens")
    return pool


def benchmark_prefill(
    model: torch.nn.Module,
    batch_size: int,
    prefill_length: int,
    vocab_size: int,
    warmup: int,
    repeats: int,
    seed: int,
    label: str,
    verbose_iter: bool = False,
    real_text_pool: torch.Tensor | None = None,
    is_main: bool = True,
) -> Dict[str, float]:
    input_device = choose_input_device(model)
    g = torch.Generator(device=input_device).manual_seed(seed)
    rng = np.random.default_rng(seed)

    if real_text_pool is not None:
        def sample_prefill_batch() -> torch.Tensor:
            idx = rng.integers(0, real_text_pool.shape[0], size=batch_size)
            return real_text_pool[idx, :prefill_length].to(device=input_device)
    else:
        input_ids = torch.randint(
            0, vocab_size, (batch_size, prefill_length), generator=g, device=input_device
        )

        def sample_prefill_batch() -> torch.Tensor:
            return input_ids

    with torch.no_grad():
        for i in range(warmup):
            input_ids = sample_prefill_batch()
            start = time.perf_counter()
            _ = model(
                input_ids=input_ids,
                use_cache=True,
                output_router_logits=False,
                output_gate_scores=False,
            )
            torch.cuda.synchronize()
            if verbose_iter and is_main:
                elapsed = time.perf_counter() - start
                print(f"[{label}] prefill warmup {i + 1}/{warmup}: {elapsed:.6f}s")
        torch.cuda.synchronize()

    times = []
    with torch.no_grad():
        for i in range(repeats):
            input_ids = sample_prefill_batch()
            start = time.perf_counter()
            _ = model(
                input_ids=input_ids,
                use_cache=True,
                output_router_logits=False,
                output_gate_scores=False,
            )
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            times.append(elapsed)
            if verbose_iter and is_main:
                print(f"[{label}] prefill repeat {i + 1}/{repeats}: {elapsed:.6f}s")

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
    label: str,
    verbose_iter: bool = False,
    real_text_pool: torch.Tensor | None = None,
    is_main: bool = True,
) -> Dict[str, float]:
    input_device = choose_input_device(model)
    g = torch.Generator(device=input_device).manual_seed(seed + 1)
    rng = np.random.default_rng(seed + 1)

    prompt_ids_fallback = torch.randint(
        0, vocab_size, (batch_size, prefill_length), generator=g, device=input_device
    )
    next_token_fallback = torch.zeros((batch_size, 1), dtype=torch.long, device=input_device)
    # Preallocate max-length mask and use cheap slicing instead of per-step torch.cat.
    full_attention_mask = torch.ones(
        (batch_size, prefill_length + decode_steps), device=input_device, dtype=torch.long
    )

    def run_one() -> float:
        if real_text_pool is not None:
            idx = rng.integers(0, real_text_pool.shape[0], size=batch_size)
            seq_batch = real_text_pool[idx].to(device=input_device)
        # Decode benchmark uses the same sampled sequence for both:
        # prompt = first prefill_length tokens, decode stream = next decode_steps tokens.
            prompt_ids = seq_batch[:, :prefill_length]
        else:
            seq_batch = None
            prompt_ids = prompt_ids_fallback

        with torch.no_grad():
            out = model(
                input_ids=prompt_ids,
                attention_mask=full_attention_mask[:, :prefill_length],
                use_cache=True,
                output_router_logits=False,
                output_gate_scores=False,
            )
            past_key_values = out.past_key_values
            start = time.perf_counter()
            for step in range(decode_steps):
                if seq_batch is not None:
                    next_token = seq_batch[:, prefill_length + step].unsqueeze(1)
                else:
                    next_token = next_token_fallback
                out = model(
                    input_ids=next_token,
                    attention_mask=full_attention_mask[:, : prefill_length + step + 1],
                    past_key_values=past_key_values,
                    use_cache=True,
                    output_router_logits=False,
                    output_gate_scores=False,
                )
                past_key_values = out.past_key_values
            torch.cuda.synchronize()
        return time.perf_counter() - start

    for i in range(warmup):
        elapsed = run_one()
        if verbose_iter and is_main:
            print(f"[{label}] decode warmup {i + 1}/{warmup}: {elapsed:.6f}s total")

    times = []
    for i in range(repeats):
        elapsed = run_one()
        times.append(elapsed)
        if verbose_iter and is_main:
            print(
                f"[{label}] decode repeat {i + 1}/{repeats}: "
                f"{elapsed:.6f}s total, {(elapsed * 1000.0) / decode_steps:.3f} ms/token"
            )
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
    model_ref: str,
    dtype: torch.dtype,
    batch_size: int,
    prefill_length: int,
    decode_steps: int,
    warmup: int,
    repeats: int,
    seed: int,
    trust_remote_code: bool,
    hf_token: str | None,
    verbose_iter: bool,
    real_text_pool: torch.Tensor | None,
    local_rank: int,
    use_distributed_model_replica: bool,
    is_main: bool,
) -> Dict[str, float]:
    if is_main:
        print(f"\n=== Loading {label}: {model_ref}")
    model, _ = load_model(
        model_ref,
        dtype=dtype,
        trust_remote_code=trust_remote_code,
        hf_token=hf_token,
        local_rank=local_rank,
        use_distributed_model_replica=use_distributed_model_replica,
    )
    input_device = choose_input_device(model)
    n_params = sum(p.numel() for p in model.parameters())
    vocab_size = int(model.config.vocab_size)

    if is_main:
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
        label=label,
        verbose_iter=verbose_iter,
        real_text_pool=real_text_pool,
        is_main=is_main,
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
        label=label,
        verbose_iter=verbose_iter,
        real_text_pool=real_text_pool,
        is_main=is_main,
    )
    metrics = {**prefill, **decode}
    if is_main:
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
    dist_ctx = setup_distributed()
    is_main = dist_ctx["is_main"]

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. This benchmark is intended for GPU execution.")

    set_seed(args.seed)
    dtype = to_torch_dtype(args.dtype)
    project_root = os.path.dirname(os.path.abspath(__file__))
    hf_token = load_hf_token()
    if hf_token:
        os.environ.setdefault("HF_TOKEN", hf_token)

    register_routing_free_models(project_root, args.rf_model_type)
    use_distributed_model_replica = dist_ctx["is_distributed"]
    if args.expert_parallel:
        os.environ["RF_EXPERT_PARALLEL"] = "1"
    else:
        os.environ.setdefault("RF_EXPERT_PARALLEL", "0")
    text_cache_dir = args.text_cache_dir or os.path.join(project_root, "benchmark_text_cache")
    real_text_pool = None
    if args.use_real_text:
        if is_main:
            print(
                f"Real-text mode enabled: dataset={args.text_dataset_name}:{args.text_dataset_split}, "
                f"ratio={args.text_sample_ratio}, cache_dir={text_cache_dir}, text_column={args.text_column}"
            )
        real_text_pool = _build_real_text_token_pool(
            tokenizer=AutoTokenizer.from_pretrained(
                args.baseline_model,
                trust_remote_code=args.trust_remote_code,
                token=hf_token,
            ),
            prefill_length=args.prefill_length,
            decode_steps=args.decode_steps,
            cache_dir=text_cache_dir,
            dataset_name=args.text_dataset_name,
            dataset_split=args.text_dataset_split,
            sample_ratio=args.text_sample_ratio,
            text_column=args.text_column,
            seed=args.seed,
        )

    if is_main:
        print(f"CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}")
        print(f"Visible GPU count={torch.cuda.device_count()}")
        print(
            f"dtype={args.dtype}, seed={args.seed}, "
            f"baseline_model_type={args.baseline_model_type}, rf_model_type={args.rf_model_type}"
        )
        print(
            f"batch_size={args.batch_size}, prefill_length={args.prefill_length}, "
            f"decode_steps={args.decode_steps}, warmup={args.warmup_iters}, repeats={args.repeats}, "
            f"verbose_iter={args.verbose_iter}, expert_parallel={args.expert_parallel}, "
            f"world_size={dist_ctx['world_size']}"
        )

    baseline_metrics = benchmark_model(
        label="Baseline",
        model_ref=args.baseline_model,
        dtype=dtype,
        batch_size=args.batch_size,
        prefill_length=args.prefill_length,
        decode_steps=args.decode_steps,
        warmup=args.warmup_iters,
        repeats=args.repeats,
        seed=args.seed,
        trust_remote_code=args.trust_remote_code,
        hf_token=hf_token,
        verbose_iter=args.verbose_iter,
        real_text_pool=real_text_pool,
        local_rank=dist_ctx["local_rank"],
        use_distributed_model_replica=use_distributed_model_replica,
        is_main=is_main,
    )
    if dist_ctx["is_distributed"]:
        dist.barrier()
    rf_metrics = benchmark_model(
        label="RoutingFree",
        model_ref=args.rf_model,
        dtype=dtype,
        batch_size=args.batch_size,
        prefill_length=args.prefill_length,
        decode_steps=args.decode_steps,
        warmup=args.warmup_iters,
        repeats=args.repeats,
        seed=args.seed,
        trust_remote_code=args.trust_remote_code,
        hf_token=hf_token,
        verbose_iter=args.verbose_iter,
        real_text_pool=real_text_pool,
        local_rank=dist_ctx["local_rank"],
        use_distributed_model_replica=use_distributed_model_replica,
        is_main=is_main,
    )
    if dist_ctx["is_distributed"]:
        dist.barrier()

    if is_main:
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
    if dist_ctx["is_distributed"] and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

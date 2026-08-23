"""
Evaluate models on downstream benchmarks using lm-evaluation-harness.

Supported benchmarks (all confirmed available in lm-eval):
  Commonsense reasoning (LLM-Adapters):
    ARC-E     -> arc_easy
    ARC-C     -> arc_challenge
    PIQA      -> piqa
    WINO      -> winogrande
    HELLA     -> hellaswag
    BoolQ     -> boolq
    OBQA      -> openbookqa
    SIQA      -> (dropped — social_iqa uses old dataset script, incompatible with datasets>=4.x)

  NLU / GLUE:
    MNLI      -> mnli      (uses validation set)
    QNLI      -> qnli      (uses validation set)
    SST2      -> sst2      (uses validation set)

  Math reasoning (LLM-Adapters) — temporarily removed:
    GSM8K     -> gsm8k
    (AQuA, SVAMP, MAWPS not present in this lm-eval install)

Usage:
  # Baseline (standard MixtralForCausalLM):
  python eval_benchmarks.py --model-dir ./output_baseline_mixtral/.../final_model

  # Routing-free model (registers custom class before lm-eval loads it):
  python eval_benchmarks.py --model-dir ./output/mixtral_rf/.../checkpoint-XXXXX --model-type routing_free

  # Subset of tasks:
  python eval_benchmarks.py --model-dir <path> --tasks arc_easy,piqa,hellaswag
"""

import argparse
import json
import math
import sys
import os
import time

# ── Task list ─────────────────────────────────────────────────────────────────
ALL_TASKS = [
    # Commonsense reasoning (LLM-Adapters 8)
    "arc_easy",
    "arc_challenge",
    "piqa",
    # "social_iqa",  # incompatible with datasets>=4.x (uses old dataset script)
    "winogrande",
    "hellaswag",
    "boolq",
    "openbookqa",
    # GLUE (all 9 tasks)
    "cola",
    "sst2",
    "mrpc",
    "stsb",
    "qqp",
    "mnli",
    "qnli",
    "rte",
    "wnli",
    # Math reasoning (LLM-Adapters) — temporarily removed
    # "gsm8k",
]


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))


def _ensure_project_root():
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)


def _register_custom_model(model_type):
    """Register custom model classes with HuggingFace AutoModel."""
    from transformers import AutoConfig, AutoModelForCausalLM
    _ensure_project_root()

    if model_type == "routing_free":
        from routing_free.mixtral_rf import RoutingFreeMixtralConfig, RoutingFreeMixtralForCausalLM
        AutoConfig.register("routing_free_mixtral", RoutingFreeMixtralConfig)
        AutoModelForCausalLM.register(RoutingFreeMixtralConfig, RoutingFreeMixtralForCausalLM)
    elif model_type == "aoe":
        from routing_free.mixtral_aoe import AoEMixtralConfig, AoEMixtralForCausalLM
        AutoConfig.register("aoe_mixtral", AoEMixtralConfig)
        AutoModelForCausalLM.register(AoEMixtralConfig, AoEMixtralForCausalLM)
    elif model_type == "remoe":
        from routing_free.mixtral_remoe import ReMoEMixtralConfig, ReMoEMixtralForCausalLM
        AutoConfig.register("remoe_mixtral", ReMoEMixtralConfig)
        AutoModelForCausalLM.register(ReMoEMixtralConfig, ReMoEMixtralForCausalLM)
    print(f"Registered {model_type} model.")


def run_evaluation(
    model_dir: str,
    model_type: str,
    tasks: list[str],
    batch_size: str,
    device: str,
    num_fewshot: int,
    output_path: str | None,
    gate_threshold: float | None = None,
):
    import lm_eval

    # For custom models, load tokenizer from a baseline dir to avoid AutoTokenizer registration issues
    if model_type != "baseline":
        # Use the baseline S model's tokenizer (GPT2, shared across all variants)
        baseline_tok = os.path.join(PROJECT_ROOT,
                                    "output_baseline_mixtral/S_1_mixtral_baseline_12L_128Dx12E_top3_lr_1e-3/final_model")
        model_args = f"pretrained={model_dir},dtype=bfloat16,tokenizer={baseline_tok}"
    else:
        model_args = f"pretrained={model_dir},dtype=bfloat16"

    print(f"\nModel dir  : {model_dir}")
    print(f"Model type : {model_type}")
    print(f"Tasks      : {tasks}")
    print(f"Batch size : {batch_size}")
    print(f"Device     : {device}")
    print(f"Few-shot   : {num_fewshot}")
    if gate_threshold is not None:
        print(f"Gate thresh: {gate_threshold}")
    print()

    # If gate_threshold override is requested, patch config.json in-place and restore after
    config_path = os.path.join(model_dir, "config.json")
    original_config = None
    if gate_threshold is not None:
        with open(config_path) as f:
            original_config = f.read()
        config_data = json.loads(original_config)
        old_thresh = config_data.get("gate_threshold")
        config_data["gate_threshold"] = gate_threshold
        with open(config_path, "w") as f:
            json.dump(config_data, f, indent=2)
        print(f"Overriding gate_threshold: {old_thresh} -> {gate_threshold}")

    try:
        start_time = time.time()
        results = lm_eval.simple_evaluate(
            model="hf",
            model_args=model_args,
            tasks=tasks,
            num_fewshot=num_fewshot,
            batch_size=batch_size,
            device=device,
            log_samples=False,
        )
        elapsed = time.time() - start_time
        print(f"\nInference time: {elapsed:.2f}s ({elapsed/60:.2f}min)")
    finally:
        # Restore original config.json
        if original_config is not None:
            with open(config_path, "w") as f:
                f.write(original_config)

    # ── Print results table ────────────────────────────────────────────────────
    table_lines = []
    def log_print(msg):
        print(msg)
        table_lines.append(msg)

    log_print("\n" + "=" * 70)
    log_print("BENCHMARK RESULTS")
    log_print("=" * 70)
    log_print(f"{'Task':<20} {'Size':<10} {'Metric':<20} {'Value':>8}")
    log_print("-" * 70)

    task_results = results.get("results", {})
    n_samples = results.get("n-samples", {})
    
    total_samples = 0
    total_acc = 0.0
    total_weighted_acc = 0.0
    valid_tasks = 0

    for task in tasks:
        if task not in task_results:
            log_print(f"{task:<20} {'N/A':<10} {'N/A':<20} {'N/A':>8}")
            continue
            
        tr = task_results[task]
        size = n_samples.get(task, {}).get("effective", 0)
        
        # Pick the main metric: acc_norm > acc > exact_match > first numeric
        if "acc_norm,none" in tr:
            metric, value = "acc_norm", tr["acc_norm,none"]
        elif "acc,none" in tr:
            metric, value = "acc", tr["acc,none"]
        elif "exact_match,strict-match" in tr:
            metric, value = "exact_match", tr["exact_match,strict-match"]
        else:
            # fallback: first numeric value
            metric, value = next(
                ((k, v) for k, v in tr.items() if isinstance(v, float)), ("?", float("nan"))
            )
            
        value_str = f"{value:>8.4f}" if not math.isnan(value) else f"{'nan':>8}"
        log_print(f"{task:<20} {size:<10} {metric:<20} {value_str}")

        if not math.isnan(value):
            total_samples += size
            total_acc += value
            total_weighted_acc += value * size
            valid_tasks += 1

    log_print("=" * 70)
    avg_acc = 0.0
    weighted_avg_acc = 0.0
    if valid_tasks > 0:
        avg_acc = total_acc / valid_tasks
        weighted_avg_acc = total_weighted_acc / total_samples if total_samples > 0 else 0
        log_print(f"{'Avg Acc':<52} {avg_acc:>8.4f}")
        log_print(f"{'Weighted Avg Acc':<52} {weighted_avg_acc:>8.4f}")
        log_print(f"{'Inference Time':<52} {elapsed:>7.2f}s")
        log_print("=" * 70)

    table_str = "\n".join(table_lines)

    # ── Save results ───────────────────────────────────────────────────────────
    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        # 1. Plain-text table (human-readable)
        base = output_path[:-5] if output_path.endswith(".json") else output_path
        txt_path = base + ".txt"
        with open(txt_path, "w") as f:
            f.write(table_str + "\n")
        print(f"Table saved to  : {txt_path}")

        # 2. JSON with per-task metrics and summary (machine-readable)
        summary = {
            "model_dir": model_dir,
            "model_type": model_type,
            "num_fewshot": num_fewshot,
            "tasks": tasks,
            "per_task": {},
            "avg_acc": avg_acc,
            "weighted_avg_acc": weighted_avg_acc,
            "total_samples": total_samples,
            "valid_tasks": valid_tasks,
            "inference_time_s": round(elapsed, 2),
        }
        for task in tasks:
            if task not in task_results:
                continue
            tr = task_results[task]
            size = n_samples.get(task, {}).get("effective", 0)
            if "acc_norm,none" in tr:
                metric, value = "acc_norm", tr["acc_norm,none"]
            elif "acc,none" in tr:
                metric, value = "acc", tr["acc,none"]
            elif "exact_match,strict-match" in tr:
                metric, value = "exact_match", tr["exact_match,strict-match"]
            else:
                metric, value = next(
                    ((k, v) for k, v in tr.items() if isinstance(v, float)),
                    ("?", float("nan")),
                )
            summary["per_task"][task] = {"metric": metric, "value": value, "size": size}

        with open(output_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"JSON saved to   : {output_path}")

    return results


def benchmark_forward_time(
    model_dir: str,
    model_type: str,
    gate_thresholds: list[float],
    seq_len: int = 512,
    batch_size: int = 8,
    warmup: int = 5,
    repeats: int = 20,
    device: str = "cuda",
):
    """Measure pure forward-pass time for different gate thresholds."""
    import torch
    from transformers import AutoModelForCausalLM, AutoConfig

    config_path = os.path.join(model_dir, "config.json")
    with open(config_path) as f:
        original_config = f.read()

    try:
        # Load model once
        config = AutoConfig.from_pretrained(model_dir)
        model = AutoModelForCausalLM.from_pretrained(
            model_dir, config=config, torch_dtype=torch.bfloat16
        ).to(device).eval()

        # Fixed random input
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device)

        results = []
        for thresh in gate_thresholds:
            # Patch threshold on all MoE layers
            for module in model.modules():
                if hasattr(module, "config") and hasattr(module.config, "gate_threshold"):
                    module.config.gate_threshold = thresh

            # Warmup
            with torch.no_grad():
                for _ in range(warmup):
                    model(input_ids)
            torch.cuda.synchronize()

            # Timed runs with CUDA events
            start_events = [torch.cuda.Event(enable_timing=True) for _ in range(repeats)]
            end_events = [torch.cuda.Event(enable_timing=True) for _ in range(repeats)]

            with torch.no_grad():
                for i in range(repeats):
                    start_events[i].record()
                    model(input_ids)
                    end_events[i].record()
            torch.cuda.synchronize()

            times_ms = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
            avg_ms = sum(times_ms) / len(times_ms)
            std_ms = (sum((t - avg_ms) ** 2 for t in times_ms) / len(times_ms)) ** 0.5

            results.append({
                "threshold": thresh,
                "avg_ms": round(avg_ms, 2),
                "std_ms": round(std_ms, 2),
                "times_ms": [round(t, 2) for t in times_ms],
            })
            print(f"  threshold={thresh:.1f}  avg={avg_ms:.2f}ms  std={std_ms:.2f}ms")

    finally:
        # Restore original config
        with open(config_path, "w") as f:
            f.write(original_config)

    # Print summary table
    print("\n" + "=" * 50)
    print("FORWARD-PASS TIMING (per batch)")
    print("=" * 50)
    print(f"  batch_size={batch_size}, seq_len={seq_len}, repeats={repeats}")
    print(f"{'Threshold':>10} {'Avg (ms)':>10} {'Std (ms)':>10}")
    print("-" * 50)
    for r in results:
        print(f"{r['threshold']:>10.1f} {r['avg_ms']:>10.2f} {r['std_ms']:>10.2f}")
    print("=" * 50)

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate model on downstream benchmarks")
    parser.add_argument(
        "--model-dir",
        type=str,
        required=True,
        help="Path to the model checkpoint directory",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["baseline", "routing_free", "aoe", "remoe"],
        default="baseline",
        help="Model type: 'baseline', 'routing_free', 'aoe', or 'remoe'",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        default=",".join(ALL_TASKS),
        help=f"Comma-separated list of tasks (default: all). Available: {ALL_TASKS}",
    )
    parser.add_argument(
        "--batch-size",
        type=str,
        default="auto",
        help="Batch size or 'auto' for automatic selection (default: auto)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device (default: cuda)",
    )
    parser.add_argument(
        "--num-fewshot",
        type=int,
        default=0,
        help="Number of few-shot examples (default: 0)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save JSON results (optional)",
    )
    parser.add_argument(
        "--gate-threshold",
        type=float,
        default=None,
        help="Override gate_threshold in model config (for ablation study)",
    )
    parser.add_argument(
        "--benchmark-forward",
        action="store_true",
        help="Run forward-pass timing benchmark instead of evaluation",
    )
    parser.add_argument(
        "--thresholds",
        type=str,
        default="0.8,0.9,1.0,1.1,1.2",
        help="Comma-separated thresholds for --benchmark-forward (default: 0.8,0.9,1.0,1.1,1.2)",
    )
    args = parser.parse_args()

    # Register custom model BEFORE lm-eval imports/loads anything
    if args.model_type != "baseline":
        _register_custom_model(args.model_type)

    if args.benchmark_forward:
        thresholds = [float(t) for t in args.thresholds.split(",")]
        results = benchmark_forward_time(
            model_dir=args.model_dir,
            model_type=args.model_type,
            gate_thresholds=thresholds,
        )
        if args.output:
            os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
            with open(args.output, "w") as f:
                json.dump(results, f, indent=2)
            print(f"Timing saved to: {args.output}")
        return

    tasks = [t.strip() for t in args.tasks.split(",")]

    run_evaluation(
        model_dir=args.model_dir,
        model_type=args.model_type,
        tasks=tasks,
        batch_size=args.batch_size,
        device=args.device,
        num_fewshot=args.num_fewshot,
        output_path=args.output,
        gate_threshold=args.gate_threshold,
    )


if __name__ == "__main__":
    main()

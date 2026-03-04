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


def register_routing_free_model():
    """Register custom RoutingFreeMixtral classes with HuggingFace AutoModel."""
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    from transformers import GPT2Tokenizer

    # Add project root to path so routing_free package is importable
    project_root = os.path.dirname(os.path.abspath(__file__))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    from routing_free.mixtral_rf import (
        RoutingFreeMixtralConfig,
        RoutingFreeMixtralForCausalLM,
    )

    AutoConfig.register("routing_free_mixtral", RoutingFreeMixtralConfig)
    AutoModelForCausalLM.register(RoutingFreeMixtralConfig, RoutingFreeMixtralForCausalLM)
    # Register only the slow tokenizer — the fast variant triggers a companion-slow-tokenizer
    # creation path that fails to resolve vocab_file from the directory.
    AutoTokenizer.register(RoutingFreeMixtralConfig, slow_tokenizer_class=GPT2Tokenizer)
    print("Registered RoutingFreeMixtral with AutoModel and AutoTokenizer.")


def run_evaluation(
    model_dir: str,
    model_type: str,
    tasks: list[str],
    batch_size: str,
    device: str,
    num_fewshot: int,
    output_path: str | None,
):
    import lm_eval

    model_args = f"pretrained={model_dir},dtype=bfloat16"

    print(f"\nModel dir  : {model_dir}")
    print(f"Model type : {model_type}")
    print(f"Tasks      : {tasks}")
    print(f"Batch size : {batch_size}")
    print(f"Device     : {device}")
    print(f"Few-shot   : {num_fewshot}")
    print()

    results = lm_eval.simple_evaluate(
        model="hf",
        model_args=model_args,
        tasks=tasks,
        num_fewshot=num_fewshot,
        batch_size=batch_size,
        device=device,
        log_samples=False,
    )

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
        choices=["baseline", "routing_free"],
        default="baseline",
        help="Model type: 'baseline' for MixtralForCausalLM, 'routing_free' for custom model",
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
    args = parser.parse_args()

    tasks = [t.strip() for t in args.tasks.split(",")]

    # Register custom model BEFORE lm-eval imports/loads anything
    if args.model_type == "routing_free":
        register_routing_free_model()

    run_evaluation(
        model_dir=args.model_dir,
        model_type=args.model_type,
        tasks=tasks,
        batch_size=args.batch_size,
        device=args.device,
        num_fewshot=args.num_fewshot,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()

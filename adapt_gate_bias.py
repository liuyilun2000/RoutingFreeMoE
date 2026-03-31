"""
Gate-bias adaptation for a converted RFMoE checkpoint.

Only the per-expert gate.gate_bias parameters are trained; every other weight
is frozen.  The same routing-free auxiliary loss (density + balancing) that is
used in pretrain_mixtral_rf.py drives the biases toward the density_target.

Dataset: Skylion007/openwebtext loaded from the workspace HF cache directory.
Only 1/10000 of the train split is used (≈800 documents), which is enough to
calibrate the threshold offsets without touching the model's knowledge.

Usage
-----
  # Single GPU:
  python adapt_gate_bias.py \\
      --model-dir  ./rfmoe_converted \\
      --output-dir ./rfmoe_adapted

  # With explicit RF loss params (should match conversion config):
  python adapt_gate_bias.py \\
      --model-dir        ./rfmoe_converted \\
      --output-dir       ./rfmoe_adapted \\
      --gate-threshold   0.05 \\
      --density-target   0.25 \\
      --lambda-coef      1e-5 \\
      --learning-rate    1e-2 \\
      --dataset-fraction 1e-3

  # Multi-GPU with torchrun:
  torchrun --nproc_per_node=4 adapt_gate_bias.py \\
      --model-dir ./rfmoe_converted --output-dir ./rfmoe_adapted
"""

import argparse
import os
import sys

import torch

_project_root = os.path.dirname(os.path.abspath(__file__))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import transformers
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
)

from routing_free.olmoe_rf import (
    RoutingFreeOlmoeConfig,
    RoutingFreeOlmoeForCausalLM,
    RoutingFreeOlmoeModel,
)
from train_utils import preprocess_function_factory, preprocess_and_cache_dataset, AuxLossTrainer
from workspace_config import PREPROCESSING_CACHE_DIR

# Register custom classes so AutoModel can load the checkpoint
AutoConfig.register("routing_free_olmoe", RoutingFreeOlmoeConfig)
AutoModel.register(RoutingFreeOlmoeConfig, RoutingFreeOlmoeModel)
AutoModelForCausalLM.register(RoutingFreeOlmoeConfig, RoutingFreeOlmoeForCausalLM)


# ---------------------------------------------------------------------------
# Quantization helper
# ---------------------------------------------------------------------------

def _make_bnb_config(quantization: str, compute_dtype: torch.dtype):
    """
    Build a BitsAndBytesConfig for the requested quantization level.

    quantization:
        "none"  – no quantization
        "int8"  – bitsandbytes LLM.int8() (frozen weights only)
        "int4"  – bitsandbytes QLoRA NF4 double-quant (frozen weights only)

    The trainable gate_bias parameters are plain nn.Parameter scalars, not
    Linear weights, so bitsandbytes never quantises them.
    """
    if quantization == "none":
        return None
    try:
        from transformers import BitsAndBytesConfig
    except ImportError:
        raise ImportError(
            "BitsAndBytesConfig not found.  Install bitsandbytes:\n"
            "  pip install bitsandbytes"
        )
    if quantization == "int8":
        return BitsAndBytesConfig(load_in_8bit=True)
    if quantization == "int4":
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    raise ValueError(f"Unknown quantization {quantization!r}. Choose 'none', 'int8', or 'int4'.")


# ---------------------------------------------------------------------------
# Core adaptation routine
# ---------------------------------------------------------------------------

def adapt(
    model_dir: str,
    output_dir: str,
    # Dataset
    dataset_fraction: float = 1e-3,    # 1/10000 of OWT train split
    max_length: int = 512,
    # RF aux-loss params (should match the values used during conversion)
    gate_temperature: float = 1.0,
    gate_threshold: float = 0.05,
    density_target: float = 0.25,
    lambda_coef: float = 1e-5,
    eta_coef: float = 0.2,
    per_expert_aux_loss_coef: float = 0.5,
    per_token_aux_loss_coef: float = 0.5,
    # Training hyper-params
    per_device_batch_size: int = 4,
    gradient_accumulation_steps: int = 4,
    n_epochs: int = 1,
    learning_rate: float = 1e-2,   # biases only → higher lr is fine
    warmup_steps: int = 20,
    eval_steps: int = 50,
    save_steps: int = 200,
    n_workers: int = 4,
    bf16: bool = True,
    quantization: str = "none",    # "none" | "int8" | "int4"
    seed: int = 42,
    # Logging
    wandb_project: str = None,
    wandb_run_name: str = "gate_bias_adapt",
    # Paths (fall back to workspace_config defaults)
    hf_cache_dir: str = None,
    preprocessing_cache_dir: str = None,
):
    from datasets import load_dataset

    preprocessing_cache_dir = preprocessing_cache_dir or PREPROCESSING_CACHE_DIR

    compute_dtype = torch.bfloat16 if bf16 else torch.float32

    # ── GPU / DDP detection ────────────────────────────────────────────────
    # Respect torchrun (LOCAL_RANK set) → DDP.
    # Otherwise auto-detect all visible GPUs via device_map="auto".
    local_rank  = int(os.environ.get("LOCAL_RANK", -1))
    world_size  = int(os.environ.get("WORLD_SIZE", 1))
    n_gpus      = torch.cuda.device_count()
    in_torchrun = local_rank != -1

    if in_torchrun:
        torch.cuda.set_device(local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        device_map = None   # Trainer / DDP handles placement
    elif n_gpus > 0:
        device_map = "auto"  # spread model across all visible GPUs automatically
    else:
        device_map = "cpu"

    is_main = (not in_torchrun) or (local_rank <= 0)

    if wandb_project:
        os.environ["WANDB_PROJECT"] = wandb_project

    # ── Load model ─────────────────────────────────────────────────────────
    if is_main:
        print(f"Loading model from {model_dir} …")
        print(f"  GPUs visible : {n_gpus}  |  in torchrun : {in_torchrun}  "
              f"|  device_map : {device_map!r}  |  quantization : {quantization}")

    config = RoutingFreeOlmoeConfig.from_pretrained(model_dir)
    config.output_gate_scores       = True  # required for aux loss
    config.gate_temperature         = gate_temperature
    config.gate_threshold           = gate_threshold
    config.density_target           = density_target
    config.lambda_coef              = lambda_coef
    config.eta_coef                 = eta_coef
    config.per_expert_aux_loss_coef = per_expert_aux_loss_coef
    config.per_token_aux_loss_coef  = per_token_aux_loss_coef

    bnb_config = _make_bnb_config(quantization, compute_dtype)

    load_kwargs = dict(config=config)
    if bnb_config is not None:
        load_kwargs["quantization_config"] = bnb_config
        load_kwargs["device_map"] = device_map or "auto"
    else:
        load_kwargs["dtype"] = compute_dtype
        if device_map is not None:
            load_kwargs["device_map"] = device_map

    model = RoutingFreeOlmoeForCausalLM.from_pretrained(model_dir, **load_kwargs)

    # ── Freeze everything except gate.gate_bias ────────────────────────────
    n_trainable = 0
    n_total     = 0
    for name, param in model.named_parameters():
        if name.endswith(".gate.gate_bias"):
            param.requires_grad = True
            n_trainable += param.numel()
        else:
            param.requires_grad = False
        n_total += param.numel()

    if is_main:
        print(f"\nParameter budget:")
        print(f"  Trainable : {n_trainable:,}  (gate.gate_bias only)")
        print(f"  Frozen    : {n_total - n_trainable:,}")
        print(f"  Total     : {n_total:,}  ({100*n_trainable/n_total:.5f}% trainable)\n")
        trainable_names = [n for n, p in model.named_parameters() if p.requires_grad]
        for nm in trainable_names[:6]:
            print(f"  ✓ {nm}")
        if len(trainable_names) > 6:
            print(f"  … and {len(trainable_names)-6} more")
        print()

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    tokenizer.pad_token = tokenizer.eos_token

    # ── Dataset: 1/10000 of OpenWebText ────────────────────────────────────
    if is_main:
        print(f"Loading OpenWebText from cache: {hf_cache_dir}")

    full_dataset = load_dataset(
        "Skylion007/openwebtext",
        split="train",
        cache_dir=hf_cache_dir,
    )

    total = len(full_dataset)
    n_adapt = max(16, int(total * dataset_fraction))   # at least 16 docs
    n_val   = max(4,  n_adapt // 10)

    full_dataset = full_dataset.shuffle(seed=seed)
    train_raw = full_dataset.select(range(n_adapt))
    val_raw   = full_dataset.select(range(n_adapt, n_adapt + n_val))

    if is_main:
        print(f"  OWT total  : {total:,}")
        print(f"  Train slice: {n_adapt:,}  ({dataset_fraction*100:.4f}% of full)")
        print(f"  Val   slice: {n_val:,}")

    preprocess_fn = preprocess_function_factory(tokenizer, max_length)
    cache_base    = os.path.join(preprocessing_cache_dir, "owt_gate_bias_adapt")

    train_dataset = preprocess_and_cache_dataset(
        dataset=train_raw,
        cache_dir=cache_base,
        split_name="train",
        preprocess_fn=preprocess_fn,
        num_proc=n_workers,
    )
    val_dataset = preprocess_and_cache_dataset(
        dataset=val_raw,
        cache_dir=cache_base,
        split_name="val",
        preprocess_fn=preprocess_fn,
        num_proc=n_workers,
    )

    # ── TrainingArguments ──────────────────────────────────────────────────
    training_args = transformers.TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=n_epochs,
        per_device_train_batch_size=per_device_batch_size,
        per_device_eval_batch_size=per_device_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,

        learning_rate=learning_rate,
        weight_decay=0.0,        # no weight decay for bias-only training
        warmup_steps=warmup_steps,

        local_rank=local_rank if in_torchrun else -1,
        ddp_backend="nccl" if in_torchrun else None,
        ddp_find_unused_parameters=True,
        dataloader_pin_memory=(n_gpus > 0),

        logging_dir=os.path.join(output_dir, "logs"),
        logging_steps=10,

        eval_steps=eval_steps,
        eval_strategy="steps",
        eval_on_start=True,
        save_steps=save_steps,
        save_strategy="steps",
        save_total_limit=2,

        load_best_model_at_end=True,
        label_names=["labels"],
        metric_for_best_model="eval_loss",
        greater_is_better=False,

        remove_unused_columns=False,
        bf16=bf16,
        dataloader_num_workers=n_workers,

        report_to="wandb" if wandb_project else None,
        run_name=wandb_run_name,
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = AuxLossTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )

    if is_main:
        print("\n=== Gate-bias adaptation ===")
        print(f"  density_target = {density_target}")
        print(f"  gate_threshold = {gate_threshold}")
        print(f"  lambda_coef    = {lambda_coef}")
        print(f"  eta_coef       = {eta_coef}")
        print(f"  learning_rate  = {learning_rate}")
        print(f"  quantization   = {quantization}")
        print(f"  train samples  = {len(train_dataset)}")
        print(f"  val   samples  = {len(val_dataset)}")
        print("============================\n")

    trainer.train()

    # ── Save ───────────────────────────────────────────────────────────────
    final_dir = os.path.join(output_dir, "final_model")
    if is_main:
        target = model.module if hasattr(model, "module") else model
        target.save_pretrained(final_dir)
        tokenizer.save_pretrained(final_dir)
        print(f"\nAdapted model saved to {final_dir}")

    return trainer


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Adapt gate biases of a converted RFMoE checkpoint using the density aux loss.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Paths
    parser.add_argument("--model-dir",  required=True,
                        help="Path to the converted RFMoE checkpoint "
                             "(output of convert_mixtral_to_rfmoe.py)")
    parser.add_argument("--output-dir", required=True,
                        help="Directory to write training checkpoints and final model")

    # Dataset
    parser.add_argument("--dataset-fraction", type=float, default=1e-3,
                        help="Fraction of the OWT train split to use (default: 1/10000)")
    parser.add_argument("--max-length", type=int, default=512,
                        help="Tokenisation max length")

    # RF aux-loss (should match conversion config)
    parser.add_argument("--gate-temperature", type=float, default=1.0)
    parser.add_argument("--gate-threshold",   type=float, default=0.05,
                        help="Same threshold used during conversion / intended inference")
    parser.add_argument("--density-target",   type=float, default=0.25,
                        help="Target fraction of tokens that activate each expert")
    parser.add_argument("--lambda-coef",      type=float, default=1e-5,
                        help="Initial aux-loss coefficient (adaptive during training)")
    parser.add_argument("--eta-coef",         type=float, default=0.2,
                        help="Step size for adaptive lambda update")
    parser.add_argument("--per-expert-aux-loss-coef", type=float, default=0.5)
    parser.add_argument("--per-token-aux-loss-coef",  type=float, default=0.5)

    # Training
    parser.add_argument("--per-device-batch-size",       type=int,   default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int,   default=4)
    parser.add_argument("--n-epochs",                    type=int,   default=1)
    parser.add_argument("--learning-rate",               type=float, default=1e-2,
                        help="Learning rate for gate_bias params (bias-only, can be higher)")
    parser.add_argument("--warmup-steps",  type=int, default=20)
    parser.add_argument("--eval-steps",    type=int, default=50)
    parser.add_argument("--save-steps",    type=int, default=200)
    parser.add_argument("--n-workers",     type=int, default=4)
    parser.add_argument("--bf16",          action="store_true", default=True)
    parser.add_argument("--quantization",  default="none",
                        choices=["none", "int8", "int4"],
                        help="Load the RFMoE model in quantized form to reduce GPU memory. "
                             "Only Linear weights are quantised; gate_bias scalars stay float. "
                             "Requires bitsandbytes. 'int4' uses NF4 double-quant (QLoRA-style).")
    parser.add_argument("--seed",          type=int, default=42)

    # Logging
    parser.add_argument("--wandb-project",  default="rfmoe",
                        help="W&B project name (default: rfmoe; set to 'none' to disable)")
    parser.add_argument("--wandb-run-name", default="gate_bias_adapt")

    # Paths
    parser.add_argument("--hf-cache-dir",             default=None,
                        help="HF dataset cache dir (default: from workspace_config.py)")
    parser.add_argument("--preprocessing-cache-dir",  default=None,
                        help="Preprocessing cache dir (default: from workspace_config.py)")

    args = parser.parse_args()
    if args.wandb_project and args.wandb_project.lower() == "none":
        args.wandb_project = None

    adapt(
        model_dir=args.model_dir,
        output_dir=args.output_dir,
        dataset_fraction=args.dataset_fraction,
        max_length=args.max_length,
        gate_temperature=args.gate_temperature,
        gate_threshold=args.gate_threshold,
        density_target=args.density_target,
        lambda_coef=args.lambda_coef,
        eta_coef=args.eta_coef,
        per_expert_aux_loss_coef=args.per_expert_aux_loss_coef,
        per_token_aux_loss_coef=args.per_token_aux_loss_coef,
        per_device_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        n_epochs=args.n_epochs,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        n_workers=args.n_workers,
        bf16=args.bf16,
        quantization=args.quantization,
        seed=args.seed,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        hf_cache_dir=args.hf_cache_dir,
        preprocessing_cache_dir=args.preprocessing_cache_dir,
    )


if __name__ == "__main__":
    main()

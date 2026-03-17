# Routing-Free Mixture of Experts

Official implementation of **Routing-Free Mixture of Experts (RF-MoE)**, a novel MoE architecture that eliminates the traditional router network by replacing it with a low-rank gating mechanism integrated directly into each expert's feed-forward network.

## Overview

In standard MoE architectures, a learned router network assigns tokens to experts, introducing training instability and load-balancing challenges. RF-MoE removes the router entirely: each expert autonomously decides whether to activate for a given token using a lightweight, low-rank gate projection followed by a norm-based threshold.

Key features:
- **No router network** — each expert self-determines activation via low-rank gate projections
- **Threshold-based sparse activation** — experts activate only when gate score exceeds a learned threshold
- **Adaptive auxiliary loss** — balances expert utilization with an automatically-tuned lambda coefficient
- **Drop-in replacement** — built on top of HuggingFace Transformers' Mixtral architecture

## Requirements

- Python 3.10+
- CUDA 12.4
- PyTorch 2.6+

```bash
pip install -r requirements.txt
```

## Model Configurations

Three model sizes are provided (S / M / L):

| Config | Layers | Hidden | Experts | Expert Dim | Gate Rank | Params (core) |
|--------|--------|--------|---------|------------|-----------|---------------|
| **S**  | 12     | 512    | 12      | 128        | 32        | ~42M          |
| **M**  | 24     | 768    | 16      | 192        | 48        | ~230M         |
| **L**  | 32     | 1024   | 24      | 256        | 64        | ~310M         |

Pre-built configs are in `config/RoutingFreeMixtral_*/`.

## Quick Start

### 1. Initialize a model

```bash
cd initialize

# Small model
python init_mixtral_rf.py \
  --config-json mixtral_rf.config.json \
  --num-hidden-layers 12 \
  --intermediate-size 128 \
  --n-experts 12 \
  --gate-proj-rank 32 \
  --output-dir ../config \
  --model-name RoutingFreeMixtral_12L_128D_rank32 \
  --bf16
```

Or use the provided script:
```bash
bash init_mixtral_rf.sh
```

### 2. Train

```bash
# Train with 4 GPUs (adjust pretrain_mixtral_rf.sh for S/M/L)
sbatch pretrain_mixtral_rf.sh

# Or run directly:
torchrun --nproc_per_node 4 pretrain_mixtral_rf.py \
  --model-dir ./config/RoutingFreeMixtral_12L_128D_rank32 \
  --output-dir ./output/rf_small \
  --dataset-name Skylion007/openwebtext \
  --num-hidden-layers 12 \
  --num-attention-heads 16 \
  --num-key-value-heads 16 \
  --n-experts 12 \
  --intermediate-size 128 \
  --lr 1e-3 \
  --bf16
```

### 3. Evaluate

```bash
# Evaluate on downstream benchmarks
python eval_benchmarks.py \
  --model-dir ./output/rf_small/final_model \
  --model-type routing_free \
  --output results/rf_small.json

# Or via SLURM:
sbatch eval_benchmarks.sh ./output/rf_small/final_model routing_free results/rf_small.json
```

## Project Structure

```
.
├── routing_free/
│   ├── __init__.py
│   ├── modules.py              # Core: RoutingFreeGate, RoutingFreeFFNWrapper, RoutingFreeMaskedMoE
│   ├── mixtral_rf.py           # RoutingFreeMixtral{Config,Model,ForCausalLM}
│   └── mixtral/                # Reference: HuggingFace Mixtral source
│       ├── configuration_mixtral.py
│       └── modeling_mixtral.py
├── config/                     # Pre-built model configs (S/M/L)
├── initialize/                 # Model initialization scripts & base configs
├── pretrain_mixtral_rf.py      # Training script
├── pretrain_mixtral_rf.sh      # SLURM training launcher
├── eval_benchmarks.py          # Downstream evaluation (lm-evaluation-harness)
├── eval_benchmarks.sh          # SLURM eval launcher
├── train_utils.py              # Dataset preprocessing & AuxLossTrainer
├── utils.py                    # Parameter counting utilities
├── print_model_size.py         # Print model parameter statistics
└── requirements.txt
```

## Citation

```bibtex
@article{routingfreemoe2025,
  title={Routing-Free Mixture of Experts},
  author={},
  year={2025}
}
```

## License

This project is released under the Apache License 2.0.

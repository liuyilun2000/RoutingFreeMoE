# Routing-Free Mixture-of-Experts

<p align="center">
  <a href="https://arxiv.org/abs/2604.00801"><img src="https://img.shields.io/badge/arXiv-2604.00801-b31b1b.svg" alt="arXiv"></a>
  <a href="https://liuyilun2000.github.io/RoutingFreeMoE/"><img src="https://img.shields.io/badge/Project-Page-2E8B57.svg" alt="Project Page"></a>
  <a href="https://github.com/liuyilun2000/RoutingFreeMoE"><img src="https://img.shields.io/badge/GitHub-Code-181717.svg?logo=github&logoColor=white" alt="Code"></a>
</p>

Official repository of **Routing-Free Mixture-of-Experts**, accepted at **EMNLP 2026 Main Conference**.

Standard MoE relies on a top-down pipeline: an external router scores experts, Softmax normalizes the distribution, and Top-K fixes the activation pattern. **Routing-Free MoE (RFMoE)** removes this centralized routing entirely. Each expert decides whether to activate on its own, and the full stack is trained end-to-end through continuous gradient flow—without an external router, Softmax, or hard-coded Top-K. A unified adaptive load-balancing objective jointly optimizes expert- and token-level balance through a configurable interpolation.

## Repository layout

```
routing_free/           # model code: RFMoE + AoE, ReMoE, Mixtral baseline
  mixtral_rf.py         # Routing-Free Mixtral (this paper)
  mixtral_aoe.py        # Autonomy-of-Experts baseline
  mixtral_remoe.py      # ReMoE baseline
  modules.py            # RF gate + masked MoE modules
initialize/             # build fresh model dirs from *.config.json
config/                 # initialized configs + tokenizer (small model set)
scripts/
  data/                 # download / preprocess / cache pretraining corpora
  train/                # torchrun launchers for the four variants (S/M/L)
  eval/                 # lm-eval-harness runner + threshold / timing sweeps
  analysis/             # FLOPs, activation heatmaps, gate-score extraction
Latex/                  # EMNLP26 camera-ready sources
markdown/, results/     # working notes and evaluation outputs (gitignored)
```

## Environment

```bash
conda create -n rfmoe python=3.10 -y
conda activate rfmoe
pip install -r requirements.txt
```

Set `HF_TOKEN` in `.env` (or export it) for gated datasets.

## Data pipeline

Runs on login node → GPU node in three stages:

```bash
bash scripts/data/download_dataset.sh   cerebras/SlimPajama-627B
bash scripts/data/preprocess_dataset.sh cerebras/SlimPajama-627B
bash scripts/data/cache_dataset.sh      cerebras/SlimPajama-627B
```

## Initialize model configs

```bash
cd initialize
bash init_baseline_mixtral.sh   # Mixtral baseline
bash init_mixtral_rf.sh         # Routing-Free MoE
bash init_aoe_mixtral.sh        # AoE baseline
bash init_remoe_mixtral.sh      # ReMoE baseline
```

Outputs land under `config/<Family>_<Layers>L_<Dim>D[...]/`. Small-scale
configs are checked in (weight file `model.safetensors` is regenerated
locally and gitignored).

## Training

Each `.sh` under `scripts/train/` is a Slurm sbatch script:

```bash
sbatch scripts/train/pretrain_baseline_mixtral.sh
sbatch scripts/train/pretrain_mixtral_rf.sh
sbatch scripts/train/pretrain_aoe_mixtral.sh
sbatch scripts/train/pretrain_remoe_mixtral.sh
```

Scales S / M / L are selected by editing `num_hidden_layers`,
`num_local_experts`, `intermediate_size`, and `LEARNING_RATE` at the top of the
script.

## Evaluation

Zero-shot downstream benchmarks via `lm-evaluation-harness`:

```bash
sbatch scripts/eval/eval_benchmarks.sh \
  <model_dir> \
  <baseline|routing_free|aoe|remoe> \
  results/<name>.json
```

Threshold ablation for RFMoE (sweeps `gate_threshold` and patches
`config.json` in place per run):

```bash
sbatch scripts/eval/benchmark_eval_time.sh <rf_model_dir>
```

Pure forward-pass timing (CUDA-event synchronized):

```bash
sbatch scripts/eval/benchmark_forward.sh <rf_model_dir> "0.8,0.9,1.0,1.1,1.2"
```

## Analysis

```bash
python scripts/analysis/compute_flops.py         # per-threshold FLOPs
python scripts/analysis/extract_gate_scores.py   # gate scores per token/layer
python scripts/analysis/plot_activation_heatmap.py
python scripts/analysis/print_model_size.py
```

## Citation

```bibtex
@misc{liu2026routingfreemixtureofexperts,
  title={Routing-Free Mixture-of-Experts},
  author={Yilun Liu and Jinru Han and Sikuan Yan and Volker Tresp and Yunpu Ma},
  year={2026},
  eprint={2604.00801},
  archivePrefix={arXiv},
  primaryClass={cs.LG},
  url={https://arxiv.org/abs/2604.00801}
}
```

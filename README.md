# Routing-Free Mixture-of-Experts

[![arXiv](https://img.shields.io/badge/arXiv-2604.00801-b31b1b.svg)](https://arxiv.org/abs/2604.00801)

**Routing-Free Mixture-of-Experts**
Yilun Liu\*, Jinru Han\*, Sikuan Yan, Volker Tresp, Yunpu Ma
(\* Equal contribution)

This is the official repository for **Routing-Free Mixture-of-Experts** 
\[To be updated\].

> *Standard Mixture-of-Experts models rely on centralized routing mechanisms that introduce rigid inductive biases.
> Routing-Free MoE eliminates any hard-coded centralized designs — including external routers, Softmax, TopK, and load balancing — and encapsulates all activation functionalities within individual experts, enabling each expert to determine its own activation entirely on its own.*

---

## Overview

Routing-Free MoE is a bottom-up MoE architecture where each expert independently determines its own activation through an internal confidence score, without any external router, TopK selection, or Softmax normalization.
It is accompanied by a unified adaptive load-balancing framework that jointly optimizes token-balancing and expert-balancing objectives via a configurable interpolation parameter.

Key properties:
- Each expert activates itself when its internal score surpasses a learnable per-expert bias threshold.
- A global post-activation threshold `θ` provides lightweight inference-time control over overall sparsity.
- The adaptive auxiliary loss coefficient `λ` automatically drives activation density toward a target `ρ∞` without manual tuning.
- A configurable interpolation parameter `μ` unifies token-balancing and expert-balancing in a single framework.

---

## Results

Routing-Free MoE consistently outperforms standard MoE, AoE, and ReMoE baselines across all three model scales (S/M/L, up to 0.8B parameters) in both language modeling perplexity and average downstream accuracy across 9 benchmarks.

| Scale | Arch. | PPL ↓ | Avg. Acc. ↑ |
|-------|-------|--------|-------------|
| S     | MoE   | 31.22  | 38.96       |
| S     | RFMoE | **27.42**  | **39.77**   |
| M     | MoE   | 25.00  | 39.64       |
| M     | RFMoE | **22.08**  | **40.40**   |
| L     | MoE   | 24.58  | 40.00       |
| L     | RFMoE | **19.97**  | **40.76**   |

All models trained on [OpenWebText](http://Skylion007.github.io/OpenWebTextCorpus) for one epoch under iso-compute conditions.
Zero-shot evaluation on PIQA, HellaSwag, WinoGrande, ARC-Easy, ARC-Challenge, OpenBookQA, QQP, QNLI, and SST-2.

---

## Installation

```bash
git clone https://github.com/liuyilun2000/RoutingFreeMoE.git
cd RoutingFreeMoE
pip install -r requirements.txt
```

The implementation is built on top of the [HuggingFace Transformers](https://github.com/huggingface/transformers) Mixtral architecture (`v4.57.6`).

---

## Usage

\[To be updated\]

---

## Citation

If you find this work useful, please cite:

```bibtex
@article{liu2026routingfree,
  title   = {Routing-Free Mixture-of-Experts},
  author  = {Liu, Yilun and Han, Jinru and Yan, Sikuan and Tresp, Volker and Ma, Yunpu},
  journal = {arXiv preprint arXiv:2604.00801},
  year    = {2026}
}
```

---

## License

This project is licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

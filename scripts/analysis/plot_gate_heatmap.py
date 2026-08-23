"""
Visualize gate scores as heatmaps comparing different aux loss configurations.

Reads .pt files produced by extract_gate_scores.py and generates:
  1. Per-config heatmaps: one figure per checkpoint, one subplot per layer
  2. Comparison figure: side-by-side heatmaps for a selected layer across all configs

Usage:
    python plot_gate_heatmap.py [--input-dir ./gate_scores] [--layer 0 5 11] [--output-dir ./figures]
"""

import argparse
import os
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path


def load_results(input_dir: str):
    """Load all gate_scores_*.pt files from input_dir."""
    results = {}
    for f in sorted(Path(input_dir).glob("gate_scores_*.pt")):
        data = torch.load(str(f), map_location="cpu", weights_only=False)
        name = data.get("config_name", f.stem)
        results[name] = data
    return results


def make_heatmap_data(gate_scores: torch.Tensor, gate_masks: torch.Tensor):
    """Convert gate scores to plottable array. Inactive = NaN for white cells."""
    arr = gate_scores.float().numpy().copy()
    mask = gate_masks.numpy()
    arr[~mask] = np.nan
    return arr


def plot_layer_with_bars(fig, gs_slot, scores_2d, masks_2d, tokens,
                         full_masks_2d=None,
                         title=None, vmin=None, vmax=None):
    """
    Plot a [T_win, E] heatmap with marginal bar charts.
    - Top bar: experts activated per token (windowed, aligns with heatmap columns)
    - Right bar: tokens activated per expert (full sequence if full_masks_2d given)

    Args:
        scores_2d:     [T_win, E] numpy array (NaN = inactive) — windowed heatmap data
        masks_2d:      [T_win, E] bool numpy array — windowed masks
        tokens:        list[str] of length T_win — windowed token labels
        full_masks_2d: [T_full, E] bool numpy array — full-sequence masks for right bar.
                       If None, uses masks_2d (same as windowed).
    Returns (im, ax_main) for colorbar attachment.
    """
    from matplotlib.gridspec import GridSpecFromSubplotSpec

    T_win, E = scores_2d.shape
    if full_masks_2d is None:
        full_masks_2d = masks_2d

    inner = GridSpecFromSubplotSpec(
        2, 2, subplot_spec=gs_slot,
        height_ratios=[1, 5], width_ratios=[5, 1],
        hspace=0.05, wspace=0.05,
    )

    ax_top = fig.add_subplot(inner[0, 0])
    ax_main = fig.add_subplot(inner[1, 0])
    ax_right = fig.add_subplot(inner[1, 1])
    ax_corner = fig.add_subplot(inner[0, 1])
    ax_corner.axis("off")

    # ── Main heatmap ──
    cmap = plt.cm.YlOrRd.copy()
    cmap.set_bad(color="white")
    im = ax_main.imshow(
        scores_2d.T,  # [E, T_win]
        aspect="auto",
        cmap=cmap,
        vmin=vmin, vmax=vmax,
        interpolation="nearest",
    )
    ax_main.set_yticks(range(E))
    ax_main.set_yticklabels([f"E{i}" for i in range(E)], fontsize=6)

    # Token labels — skip if too many
    if T_win <= 60:
        short_tokens = [t[:8] for t in tokens]
        ax_main.set_xticks(range(T_win))
        ax_main.set_xticklabels(short_tokens, fontsize=5, rotation=60, ha="right")
    else:
        # Show every Nth token
        step = max(T_win // 20, 1)
        ticks = list(range(0, T_win, step))
        ax_main.set_xticks(ticks)
        ax_main.set_xticklabels([tokens[i][:6] for i in ticks], fontsize=5, rotation=60, ha="right")

    # ── Top bar: experts activated per token (windowed) ──
    experts_per_tok = masks_2d.sum(axis=1)  # [T_win]
    ax_top.bar(range(T_win), experts_per_tok, color="steelblue", alpha=0.7, width=1.0,
               edgecolor="none")
    ax_top.set_xlim(-0.5, T_win - 0.5)
    ax_top.set_ylim(0, E + 0.5)
    ax_top.set_ylabel("#E/tok", fontsize=6, rotation=0, labelpad=25, va="center")
    ax_top.set_xticks([])
    ax_top.tick_params(labelsize=5)
    if title:
        ax_top.set_title(title, fontsize=9)

    # ── Right bar: tokens activated per expert (FULL sequence) ──
    tokens_per_exp = full_masks_2d.sum(axis=0)  # [E]
    T_full = full_masks_2d.shape[0]
    ax_right.barh(range(E), tokens_per_exp, color="coral", alpha=0.7, height=0.8)
    ax_right.set_ylim(-0.5, E - 0.5)
    ax_right.set_xlim(0, tokens_per_exp.max() * 1.1 + 1)
    ax_right.set_xlabel(f"#T/exp\n(of {T_full})", fontsize=6)
    ax_right.set_yticks([])
    ax_right.tick_params(labelsize=5)
    ax_right.invert_yaxis()

    return im, ax_main


def get_window(T, window_size, window_start):
    """Compute window slice. Returns (start, end)."""
    if window_size is None or window_size >= T:
        return 0, T
    start = window_start if window_start is not None else max((T - window_size) // 2, 0)
    end = min(start + window_size, T)
    return start, end


def plot_per_config(results, layers, output_dir, window_size=None, window_start=None):
    """One figure per config: subplots for each selected layer with marginal bars."""
    from matplotlib.gridspec import GridSpec
    output_dir = Path(output_dir)

    for name, data in results.items():
        gate_scores = data["gate_scores"]  # [L, T, E]
        gate_masks = data["gate_masks"]
        tokens = data["tokens"]
        label = data.get("config_label", name)
        L, T, E = gate_scores.shape

        ws, we = get_window(T, window_size, window_start)
        T_win = we - ws

        arr = make_heatmap_data(gate_scores, gate_masks)
        masks_np = gate_masks.numpy()
        active_vals = arr[~np.isnan(arr)]
        vmin = float(active_vals.min()) if len(active_vals) > 0 else 0
        vmax = float(active_vals.max()) if len(active_vals) > 0 else 1

        win_tokens = tokens[ws:we]

        plot_layers = [l for l in layers if l < L]
        n = len(plot_layers)

        fig_w = max(T_win * 0.22, 5) + 1.5
        fig_h = n * (E * 0.28 + 1.2) + 0.8
        fig = plt.figure(figsize=(fig_w, fig_h))
        win_label = f" (tokens {ws}-{we-1})" if T_win < T else ""
        fig.suptitle(f"Gate Scores — {label}{win_label}", fontsize=12)
        outer = GridSpec(n, 1, figure=fig, hspace=0.45)

        for i, l in enumerate(plot_layers):
            im, _ = plot_layer_with_bars(
                fig, outer[i],
                arr[l, ws:we, :], masks_np[l, ws:we, :], win_tokens,
                full_masks_2d=masks_np[l],  # full sequence for right bar
                title=f"Layer {l}", vmin=vmin, vmax=vmax,
            )

        fig.colorbar(im, ax=fig.axes, shrink=0.4, pad=0.02, label="Gate Score")
        save_path = output_dir / f"heatmap_{name}.pdf"
        fig.savefig(str(save_path), bbox_inches="tight", dpi=150)
        plt.close(fig)
        print(f"  Saved {save_path}")


def plot_comparison(results, layer_idx, output_dir, window_size=None, window_start=None):
    """Stacked comparison of all configs for a single layer with marginal bars."""
    from matplotlib.gridspec import GridSpec
    output_dir = Path(output_dir)
    names = list(results.keys())
    n = len(names)

    tokens = results[names[0]]["tokens"]
    T = len(tokens)
    E = results[names[0]]["gate_scores"].shape[2]

    ws, we = get_window(T, window_size, window_start)
    T_win = we - ws
    win_tokens = tokens[ws:we]

    # Compute global vmin/vmax across all configs for this layer
    all_active = []
    for name in names:
        gs = results[name]["gate_scores"][layer_idx]
        gm = results[name]["gate_masks"][layer_idx]
        active = gs[gm].float().numpy()
        if len(active) > 0:
            all_active.append(active)
    if all_active:
        all_active = np.concatenate(all_active)
        vmin, vmax = float(all_active.min()), float(all_active.max())
    else:
        vmin, vmax = 0, 1

    panel_w = max(T_win * 0.22, 5) + 1.5
    panel_h = E * 0.28 + 1.5
    fig = plt.figure(figsize=(panel_w, n * panel_h + 0.8))
    win_label = f" (tokens {ws}-{we-1})" if T_win < T else ""
    fig.suptitle(f"Gate Score Comparison — Layer {layer_idx}{win_label}", fontsize=13)
    outer = GridSpec(n, 1, figure=fig, hspace=0.5)

    for i, name in enumerate(names):
        data = results[name]
        arr = make_heatmap_data(data["gate_scores"], data["gate_masks"])
        masks_np = data["gate_masks"].numpy()
        label = data.get("config_label", name)
        im, _ = plot_layer_with_bars(
            fig, outer[i],
            arr[layer_idx, ws:we, :], masks_np[layer_idx, ws:we, :], win_tokens,
            full_masks_2d=masks_np[layer_idx],  # full sequence for right bar
            title=label, vmin=vmin, vmax=vmax,
        )

    fig.colorbar(im, ax=fig.axes, shrink=0.3, pad=0.02, label="Gate Score")
    save_path = output_dir / f"heatmap_comparison_L{layer_idx}.pdf"
    fig.savefig(str(save_path), bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved {save_path}")


def plot_comparison_avg(results, output_dir, window_size=None, window_start=None):
    """
    Stacked comparison using layer-averaged activation density.
    Heatmap: mean activation rate per (token, expert) across all layers [0,1].
    Top bar: mean experts/token across layers.
    Right bar: mean tokens/expert across layers.
    This matches the screening metric (mean CV across layers).
    """
    from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
    output_dir = Path(output_dir)
    names = list(results.keys())
    n = len(names)

    tokens = results[names[0]]["tokens"]
    T = len(tokens)
    L = results[names[0]]["gate_masks"].shape[0]
    E = results[names[0]]["gate_masks"].shape[2]

    ws, we = get_window(T, window_size, window_start)
    T_win = we - ws
    win_tokens = tokens[ws:we]

    panel_w = max(T_win * 0.22, 5) + 1.5
    panel_h = E * 0.28 + 1.5
    fig = plt.figure(figsize=(panel_w, n * panel_h + 0.8))
    win_label = f" (tokens {ws}-{we-1})" if T_win < T else ""
    fig.suptitle(f"Gate Activation Density — Averaged over {L} layers{win_label}",
                 fontsize=13)
    outer = GridSpec(n, 1, figure=fig, hspace=0.5)

    for i, name in enumerate(names):
        data = results[name]
        masks = data["gate_masks"].float()  # [L, T, E]

        # Average activation rate across layers: [T, E] in [0, 1]
        avg_density = masks.mean(dim=0).numpy()          # [T, E]
        avg_density_win = avg_density[ws:we, :]           # [T_win, E]

        # For marginal bars: average across layers
        # experts/token: for each token, mean # experts active across layers
        avg_ept_win = masks[:, ws:we, :].sum(dim=2).mean(dim=0).numpy()  # [T_win]
        # tokens/expert: for each expert, mean # tokens active across layers
        avg_tpe_full = masks.sum(dim=1).mean(dim=0).numpy()              # [E]

        label = data.get("config_label", name)

        # Build subplot with marginal bars
        inner = GridSpecFromSubplotSpec(
            2, 2, subplot_spec=outer[i],
            height_ratios=[1, 5], width_ratios=[5, 1],
            hspace=0.05, wspace=0.05,
        )
        ax_top = fig.add_subplot(inner[0, 0])
        ax_main = fig.add_subplot(inner[1, 0])
        ax_right = fig.add_subplot(inner[1, 1])
        ax_corner = fig.add_subplot(inner[0, 1])
        ax_corner.axis("off")

        # Heatmap: activation density [0, 1]
        cmap = plt.cm.YlOrRd.copy()
        cmap.set_bad(color="white")
        im = ax_main.imshow(
            avg_density_win.T,  # [E, T_win]
            aspect="auto", cmap=cmap, vmin=0, vmax=1,
            interpolation="nearest",
        )
        ax_main.set_yticks(range(E))
        ax_main.set_yticklabels([f"E{i}" for i in range(E)], fontsize=6)
        if T_win <= 60:
            ax_main.set_xticks(range(T_win))
            ax_main.set_xticklabels([t[:8] for t in win_tokens],
                                     fontsize=5, rotation=60, ha="right")
        else:
            step = max(T_win // 20, 1)
            ticks = list(range(0, T_win, step))
            ax_main.set_xticks(ticks)
            ax_main.set_xticklabels([win_tokens[j][:6] for j in ticks],
                                     fontsize=5, rotation=60, ha="right")

        # Top bar: mean experts/token
        ax_top.bar(range(T_win), avg_ept_win, color="steelblue", alpha=0.7,
                   width=1.0, edgecolor="none")
        ax_top.set_xlim(-0.5, T_win - 0.5)
        ax_top.set_ylim(0, E + 0.5)
        ax_top.set_ylabel("avg\n#E/tok", fontsize=6, rotation=0, labelpad=28, va="center")
        ax_top.set_xticks([])
        ax_top.tick_params(labelsize=5)
        ax_top.set_title(label, fontsize=9)

        # Right bar: mean tokens/expert
        ax_right.barh(range(E), avg_tpe_full, color="coral", alpha=0.7, height=0.8)
        ax_right.set_ylim(-0.5, E - 0.5)
        ax_right.set_xlim(0, avg_tpe_full.max() * 1.1 + 1)
        ax_right.set_xlabel(f"avg #T/exp", fontsize=6)
        ax_right.set_yticks([])
        ax_right.tick_params(labelsize=5)
        ax_right.invert_yaxis()

    fig.colorbar(im, ax=fig.axes, shrink=0.3, pad=0.02, label="Activation Rate")
    save_path = output_dir / "heatmap_comparison_avg.pdf"
    fig.savefig(str(save_path), bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved {save_path}")


def plot_density_comparison(results, output_dir):
    """Bar chart showing per-layer activation density for each config."""
    output_dir = Path(output_dir)
    names = list(results.keys())
    L = results[names[0]]["gate_masks"].shape[0]

    fig, ax = plt.subplots(figsize=(max(L * 0.6, 6), 3.5))
    x = np.arange(L)
    width = 0.8 / len(names)

    for i, name in enumerate(names):
        masks = results[name]["gate_masks"].float()  # [L, T, E]
        density_per_layer = masks.mean(dim=(1, 2)).numpy()  # [L]
        label = results[name].get("config_label", name)
        ax.bar(x + i * width, density_per_layer, width, label=label, alpha=0.85)

    ax.set_xlabel("Layer")
    ax.set_ylabel("Activation Density")
    ax.set_title("Per-Layer Expert Activation Density")
    ax.set_xticks(x + width * (len(names) - 1) / 2)
    ax.set_xticklabels([str(i) for i in range(L)])
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1)
    fig.tight_layout()

    save_path = output_dir / "density_comparison.pdf"
    fig.savefig(str(save_path), bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved {save_path}")


def plot_experts_per_token(results, output_dir):
    """Per-layer line plot: avg number of experts activated per token."""
    output_dir = Path(output_dir)
    names = list(results.keys())
    L = results[names[0]]["gate_masks"].shape[0]

    fig, ax = plt.subplots(figsize=(max(L * 0.5, 6), 3.5))
    x = np.arange(L)

    for name in names:
        masks = results[name]["gate_masks"]  # [L, T, E]
        # Sum over experts for each (layer, token), then average over tokens
        experts_per_tok = masks.sum(dim=2).float()  # [L, T]
        mean_per_layer = experts_per_tok.mean(dim=1).numpy()  # [L]
        std_per_layer = experts_per_tok.std(dim=1).numpy()    # [L]
        label = results[name].get("config_label", name)
        ax.plot(x, mean_per_layer, marker="o", markersize=4, label=label)
        ax.fill_between(x, mean_per_layer - std_per_layer,
                        mean_per_layer + std_per_layer, alpha=0.15)

    E = results[names[0]]["gate_masks"].shape[2]
    ax.set_xlabel("Layer")
    ax.set_ylabel("# Experts Activated per Token")
    ax.set_title("Experts Activated per Token (mean ± std)")
    ax.set_xticks(x)
    ax.set_xticklabels([str(i) for i in range(L)])
    ax.set_ylim(0, E)
    ax.legend(fontsize=8)
    fig.tight_layout()

    save_path = output_dir / "experts_per_token.pdf"
    fig.savefig(str(save_path), bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved {save_path}")


def plot_tokens_per_expert(results, output_dir):
    """Per-layer line plot: avg number of tokens activated per expert."""
    output_dir = Path(output_dir)
    names = list(results.keys())
    L = results[names[0]]["gate_masks"].shape[0]
    E = results[names[0]]["gate_masks"].shape[2]

    fig, ax = plt.subplots(figsize=(max(L * 0.5, 6), 3.5))
    x = np.arange(L)

    for name in names:
        masks = results[name]["gate_masks"]  # [L, T, E]
        # Sum over tokens for each (layer, expert), then average over experts
        tokens_per_exp = masks.sum(dim=1).float()  # [L, E]
        mean_per_layer = tokens_per_exp.mean(dim=1).numpy()  # [L]
        std_per_layer = tokens_per_exp.std(dim=1).numpy()    # [L]
        label = results[name].get("config_label", name)
        ax.plot(x, mean_per_layer, marker="s", markersize=4, label=label)
        ax.fill_between(x, mean_per_layer - std_per_layer,
                        mean_per_layer + std_per_layer, alpha=0.15)

    T = results[names[0]]["gate_masks"].shape[1]
    ax.set_xlabel("Layer")
    ax.set_ylabel("# Tokens Activated per Expert")
    ax.set_title("Tokens Activated per Expert (mean ± std)")
    ax.set_xticks(x)
    ax.set_xticklabels([str(i) for i in range(L)])
    ax.set_ylim(0, T)
    ax.legend(fontsize=8)
    fig.tight_layout()

    save_path = output_dir / "tokens_per_expert.pdf"
    fig.savefig(str(save_path), bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved {save_path}")


def plot_load_profile(results, output_dir):
    """
    Sorted load profile: for each config, sort experts by their token count
    (descending) and plot as a line. Flat = balanced; steep = imbalanced.
    One subplot per layer, averaged across layers for a summary.
    """
    output_dir = Path(output_dir)
    names = list(results.keys())
    L = results[names[0]]["gate_masks"].shape[0]
    E = results[names[0]]["gate_masks"].shape[2]

    # ── Per-layer load profiles ──
    # Pick a few representative layers
    show_layers = [0, L // 4, L // 2, 3 * L // 4, L - 1]
    show_layers = sorted(set(show_layers))
    n = len(show_layers)

    fig, axes = plt.subplots(1, n, figsize=(n * 2.8, 3), sharey=True, squeeze=False)
    fig.suptitle("Expert Load Profile (sorted, per layer)", fontsize=11)

    for j, l in enumerate(show_layers):
        ax = axes[0, j]
        for name in names:
            masks = results[name]["gate_masks"]  # [L, T, E]
            tokens_per_exp = masks[l].sum(dim=0).float().numpy()  # [E]
            sorted_load = np.sort(tokens_per_exp)[::-1]
            label = results[name].get("config_label", name)
            ax.plot(range(E), sorted_load, marker="o", markersize=3, label=label)
        ax.set_title(f"Layer {l}", fontsize=9)
        ax.set_xlabel("Expert rank", fontsize=8)
        if j == 0:
            ax.set_ylabel("# Tokens assigned", fontsize=8)
        ax.set_xticks(range(E))
        ax.tick_params(labelsize=6)
    axes[0, -1].legend(fontsize=6, loc="upper right")
    fig.tight_layout()
    save_path = output_dir / "load_profile_per_layer.pdf"
    fig.savefig(str(save_path), bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved {save_path}")

    # ── Aggregated across all layers ──
    fig, ax = plt.subplots(figsize=(5, 3.5))
    for name in names:
        masks = results[name]["gate_masks"]  # [L, T, E]
        # Sum tokens across all layers for each expert, then sort
        total_per_exp = masks.sum(dim=(0, 1)).float().numpy()  # [E]
        sorted_load = np.sort(total_per_exp)[::-1]
        label = results[name].get("config_label", name)
        ax.bar(range(E), sorted_load, alpha=0.6, label=label, width=0.25,
               align="edge")
    ax.set_xlabel("Expert (sorted by load)")
    ax.set_ylabel("Total tokens assigned (all layers)")
    ax.set_title("Expert Load Profile — Aggregated")
    ax.set_xticks(range(E))
    ax.legend(fontsize=8)
    fig.tight_layout()
    save_path = output_dir / "load_profile_aggregated.pdf"
    fig.savefig(str(save_path), bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved {save_path}")


def compute_cv(values):
    """Coefficient of variation: std / mean. Returns inf if mean=0."""
    m = values.mean()
    if m == 0:
        return 0.0
    return float(values.std() / m)


def compute_gini(values):
    """Gini coefficient: 0 = perfect equality, 1 = max inequality."""
    v = np.sort(values)
    n = len(v)
    if v.sum() == 0:
        return 0.0
    index = np.arange(1, n + 1)
    return float((2.0 * (index * v).sum() / (n * v.sum())) - (n + 1) / n)


def plot_cv_comparison(results, output_dir):
    """
    Coefficient of Variation per layer for:
      - tokens-per-expert (expert load balance)
      - experts-per-token (token coverage balance)
    Higher CV = more imbalance.
    """
    output_dir = Path(output_dir)
    names = list(results.keys())
    L = results[names[0]]["gate_masks"].shape[0]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.5))

    x = np.arange(L)
    for name in names:
        masks = results[name]["gate_masks"]
        label = results[name].get("config_label", name)

        # CV of tokens-per-expert (expert load balance)
        cv_expert = []
        for l in range(L):
            tpe = masks[l].sum(dim=0).float().numpy()  # [E]
            cv_expert.append(compute_cv(tpe))
        ax1.plot(x, cv_expert, marker="o", markersize=4, label=label)

        # CV of experts-per-token (token coverage balance)
        cv_token = []
        for l in range(L):
            ept = masks[l].sum(dim=1).float().numpy()  # [T]
            cv_token.append(compute_cv(ept))
        ax2.plot(x, cv_token, marker="s", markersize=4, label=label)

    ax1.set_xlabel("Layer")
    ax1.set_ylabel("CV (std/mean)")
    ax1.set_title("Expert Load Imbalance\n(CV of tokens-per-expert)")
    ax1.set_xticks(x)
    ax1.legend(fontsize=7)

    ax2.set_xlabel("Layer")
    ax2.set_ylabel("CV (std/mean)")
    ax2.set_title("Token Coverage Imbalance\n(CV of experts-per-token)")
    ax2.set_xticks(x)
    ax2.legend(fontsize=7)

    fig.tight_layout()
    save_path = output_dir / "cv_imbalance.pdf"
    fig.savefig(str(save_path), bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved {save_path}")


def plot_gini_comparison(results, output_dir):
    """
    Gini coefficient per layer — like income inequality.
    0 = perfect balance, 1 = all load on one expert/token.
    """
    output_dir = Path(output_dir)
    names = list(results.keys())
    L = results[names[0]]["gate_masks"].shape[0]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.5))

    x = np.arange(L)
    for name in names:
        masks = results[name]["gate_masks"]
        label = results[name].get("config_label", name)

        gini_expert = []
        for l in range(L):
            tpe = masks[l].sum(dim=0).float().numpy()
            gini_expert.append(compute_gini(tpe))
        ax1.plot(x, gini_expert, marker="o", markersize=4, label=label)

        gini_token = []
        for l in range(L):
            ept = masks[l].sum(dim=1).float().numpy()
            gini_token.append(compute_gini(ept))
        ax2.plot(x, gini_token, marker="s", markersize=4, label=label)

    ax1.set_xlabel("Layer")
    ax1.set_ylabel("Gini Coefficient")
    ax1.set_title("Expert Load Inequality\n(Gini of tokens-per-expert)")
    ax1.set_xticks(x)
    ax1.set_ylim(0, 1)
    ax1.legend(fontsize=7)

    ax2.set_xlabel("Layer")
    ax2.set_ylabel("Gini Coefficient")
    ax2.set_title("Token Coverage Inequality\n(Gini of experts-per-token)")
    ax2.set_xticks(x)
    ax2.set_ylim(0, 1)
    ax2.legend(fontsize=7)

    fig.tight_layout()
    save_path = output_dir / "gini_imbalance.pdf"
    fig.savefig(str(save_path), bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved {save_path}")


def plot_lorenz_curve(results, layer_idx, output_dir):
    """
    Lorenz curves for a selected layer.
    Diagonal = perfect balance; curve below = inequality.
    Left: expert load (tokens-per-expert). Right: token coverage (experts-per-token).
    """
    output_dir = Path(output_dir)
    names = list(results.keys())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))

    for name in names:
        masks = results[name]["gate_masks"]
        label = results[name].get("config_label", name)

        # Expert load Lorenz
        tpe = masks[layer_idx].sum(dim=0).float().numpy()  # [E]
        tpe_sorted = np.sort(tpe)
        cum = np.concatenate([[0], np.cumsum(tpe_sorted)])
        cum = cum / cum[-1] if cum[-1] > 0 else cum
        x_exp = np.linspace(0, 1, len(cum))
        ax1.plot(x_exp, cum, marker="o", markersize=3, label=label)

        # Token coverage Lorenz
        ept = masks[layer_idx].sum(dim=1).float().numpy()  # [T]
        ept_sorted = np.sort(ept)
        cum = np.concatenate([[0], np.cumsum(ept_sorted)])
        cum = cum / cum[-1] if cum[-1] > 0 else cum
        x_tok = np.linspace(0, 1, len(cum))
        ax2.plot(x_tok, cum, linewidth=1.5, label=label)

    # Diagonal (perfect equality)
    ax1.plot([0, 1], [0, 1], "k--", alpha=0.3, label="Perfect balance")
    ax2.plot([0, 1], [0, 1], "k--", alpha=0.3, label="Perfect balance")

    ax1.set_xlabel("Cumulative share of experts")
    ax1.set_ylabel("Cumulative share of tokens")
    ax1.set_title(f"Expert Load — Layer {layer_idx}")
    ax1.legend(fontsize=7)
    ax1.set_aspect("equal")

    ax2.set_xlabel("Cumulative share of tokens")
    ax2.set_ylabel("Cumulative share of experts activated")
    ax2.set_title(f"Token Coverage — Layer {layer_idx}")
    ax2.legend(fontsize=7)
    ax2.set_aspect("equal")

    fig.tight_layout()
    save_path = output_dir / f"lorenz_L{layer_idx}.pdf"
    fig.savefig(str(save_path), bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved {save_path}")


def plot_lorenz_curve_avg(results, output_dir):
    """
    Lorenz curves averaged across ALL layers.
    More robust than picking a single layer — matches the screening metric.
    """
    output_dir = Path(output_dir)
    names = list(results.keys())
    L = results[names[0]]["gate_masks"].shape[0]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))

    for name in names:
        masks = results[name]["gate_masks"]
        label = results[name].get("config_label", name)

        # Average tokens-per-expert across layers
        avg_tpe = masks.sum(dim=1).float().mean(dim=0).numpy()  # [E]
        tpe_sorted = np.sort(avg_tpe)
        cum = np.concatenate([[0], np.cumsum(tpe_sorted)])
        cum = cum / cum[-1] if cum[-1] > 0 else cum
        x_exp = np.linspace(0, 1, len(cum))
        ax1.plot(x_exp, cum, marker="o", markersize=3, label=label)

        # Average experts-per-token across layers
        avg_ept = masks.sum(dim=2).float().mean(dim=0).numpy()  # [T]
        ept_sorted = np.sort(avg_ept)
        cum = np.concatenate([[0], np.cumsum(ept_sorted)])
        cum = cum / cum[-1] if cum[-1] > 0 else cum
        x_tok = np.linspace(0, 1, len(cum))
        ax2.plot(x_tok, cum, linewidth=1.5, label=label)

    ax1.plot([0, 1], [0, 1], "k--", alpha=0.3, label="Perfect balance")
    ax2.plot([0, 1], [0, 1], "k--", alpha=0.3, label="Perfect balance")

    ax1.set_xlabel("Cumulative share of experts")
    ax1.set_ylabel("Cumulative share of tokens")
    ax1.set_title(f"Expert Load (avg over {L} layers)")
    ax1.legend(fontsize=7)
    ax1.set_aspect("equal")

    ax2.set_xlabel("Cumulative share of tokens")
    ax2.set_ylabel("Cumulative share of experts activated")
    ax2.set_title(f"Token Coverage (avg over {L} layers)")
    ax2.legend(fontsize=7)
    ax2.set_aspect("equal")

    fig.tight_layout()
    save_path = output_dir / "lorenz_avg.pdf"
    fig.savefig(str(save_path), bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved {save_path}")


def plot_load_profile_avg(results, output_dir):
    """
    Sorted load profile averaged across all layers.
    For each config, sort experts by their avg token count and plot.
    """
    output_dir = Path(output_dir)
    names = list(results.keys())
    E = results[names[0]]["gate_masks"].shape[2]

    fig, ax = plt.subplots(figsize=(5, 3.5))
    for name in names:
        masks = results[name]["gate_masks"]
        avg_tpe = masks.sum(dim=1).float().mean(dim=0).numpy()  # [E]
        sorted_load = np.sort(avg_tpe)[::-1]
        label = results[name].get("config_label", name)
        ax.plot(range(E), sorted_load, marker="o", markersize=5, label=label,
                linewidth=2)

    ax.set_xlabel("Expert (sorted by load)")
    ax.set_ylabel("Avg tokens per expert per layer")
    ax.set_title("Expert Load Profile — Averaged over all layers")
    ax.set_xticks(range(E))
    ax.legend(fontsize=8)
    fig.tight_layout()
    save_path = output_dir / "load_profile_avg.pdf"
    fig.savefig(str(save_path), bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved {save_path}")


def print_activation_stats(results):
    """Print per-config activation statistics."""
    for name, data in results.items():
        masks = data["gate_masks"]  # [L, T, E]
        L, T, E = masks.shape
        label = data.get("config_label", name)

        experts_per_tok = masks.sum(dim=2).float()  # [L, T]
        tokens_per_exp = masks.sum(dim=1).float()   # [L, E]

        print(f"\n  [{label}]")
        print(f"    Experts/token:  mean={experts_per_tok.mean():.2f}, "
              f"std={experts_per_tok.std():.2f}, "
              f"min={experts_per_tok.min().item()}, "
              f"max={experts_per_tok.max().item()}")
        print(f"    Tokens/expert:  mean={tokens_per_exp.mean():.2f}, "
              f"std={tokens_per_exp.std():.2f}, "
              f"min={tokens_per_exp.min().item()}, "
              f"max={tokens_per_exp.max().item()}")

        # Per-layer breakdown
        for l in range(L):
            ept = experts_per_tok[l]  # [T]
            tpe = tokens_per_exp[l]   # [E]
            print(f"    L{l:2d}: experts/tok={ept.mean():.1f}±{ept.std():.1f}, "
                  f"tokens/exp={tpe.mean():.1f}±{tpe.std():.1f}")


def main():
    parser = argparse.ArgumentParser(description="Plot gate score heatmaps")
    parser.add_argument("--input-dir", type=str, default="./gate_scores")
    parser.add_argument("--output-dir", type=str, default="./figures")
    parser.add_argument("--layers", type=int, nargs="+", default=[0, 6, 11],
                        help="Layers to plot in per-config figures")
    parser.add_argument("--compare-layer", type=int, default=6,
                        help="Layer for side-by-side comparison")
    parser.add_argument("--window-size", type=int, default=50,
                        help="Number of tokens to show in heatmap (None=all). "
                             "Right bar still aggregates over full sequence.")
    parser.add_argument("--window-start", type=int, default=None,
                        help="Start position of token window (default: center of sequence)")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading gate scores from {args.input_dir} ...")
    results = load_results(args.input_dir)

    if not results:
        print("No gate_scores_*.pt files found. Run extract_gate_scores.py first.")
        return

    for name, data in results.items():
        L, T, E = data["gate_scores"].shape
        density = data["gate_masks"].float().mean().item()
        print(f"  {data.get('config_label', name)}: L={L}, T={T}, E={E}, density={density:.2%}")

    print("\nActivation statistics:")
    print_activation_stats(results)

    print("\nGenerating per-config heatmaps ...")
    plot_per_config(results, args.layers, out_dir, args.window_size, args.window_start)

    print("\nGenerating layer-averaged comparison heatmap ...")
    plot_comparison_avg(results, out_dir, args.window_size, args.window_start)

    print("\nGenerating layer-averaged load profile ...")
    plot_load_profile_avg(results, out_dir)

    print("\nGenerating layer-averaged Lorenz curves ...")
    plot_lorenz_curve_avg(results, out_dir)

    print("\nGenerating CV imbalance plot ...")
    plot_cv_comparison(results, out_dir)

    print("\nGenerating Gini imbalance plot ...")
    plot_gini_comparison(results, out_dir)

    print("\nGenerating per-layer comparison heatmap ...")
    plot_comparison(results, args.compare_layer, out_dir, args.window_size, args.window_start)

    print("\nGenerating per-config heatmaps ...")
    plot_per_config(results, args.layers, out_dir, args.window_size, args.window_start)

    print("\nGenerating density comparison ...")
    plot_density_comparison(results, out_dir)

    print("\nGenerating experts-per-token plot ...")
    plot_experts_per_token(results, out_dir)

    print("\nGenerating tokens-per-expert plot ...")
    plot_tokens_per_expert(results, out_dir)

    print("\nGenerating per-layer load profiles ...")
    plot_load_profile(results, out_dir)

    print("\nGenerating per-layer Lorenz curves ...")
    plot_lorenz_curve(results, args.compare_layer, out_dir)

    print("\nAll figures saved to", out_dir)


if __name__ == "__main__":
    main()

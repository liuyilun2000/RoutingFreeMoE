"""
Multi-panel expert activation heatmap.

Layout: 3 separate figures, one per model variant.
  Within each figure: 3 layer sub-panels (e.g., Layer 0, 5, 11) top to bottom.
  Within each layer: 3 rows (one per dataset: QNLI, ARC-Easy, OBQA).
  Each row is a [1 x E] heatmap of average expert activation probability.

Output: heatmap_E0.5T0.5.pdf, heatmap_E0.0T1.0.pdf, heatmap_E1.0T0.0.pdf

Usage:
    python plot_activation_heatmap.py [--input gate_scores/activation_data.pt]
                                      [--layers 0 5 11]
                                      [--output-dir figures/]
"""

import argparse
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path


# ── Global font: STIX (matches Times New Roman, built into matplotlib) ──
matplotlib.rcParams["font.family"] = "serif"
matplotlib.rcParams["font.serif"] = ["STIX", "STIXGeneral", "Times New Roman", "Times"]
matplotlib.rcParams["mathtext.fontset"] = "stix"


# Custom colormap: pale cream -> orange -> dark red -> near-black
CMAP_COLORS = [
    (1.0, 0.97, 0.92),   # pale cream
    (0.99, 0.85, 0.65),   # light peach
    (0.95, 0.55, 0.25),   # orange
    (0.75, 0.20, 0.10),   # dark red
    (0.35, 0.08, 0.08),   # very dark
    (0.10, 0.02, 0.02),   # near black
]
ACTIVATION_CMAP = LinearSegmentedColormap.from_list("activation", CMAP_COLORS, N=256)


DATASET_DISPLAY = {
    "qnli": "QNLI",
    "arc_easy": "ARCe",
    "openbookqa": "OBQA",
}


def fmt_val(val):
    """Format cell value: no leading zero, 1.00 -> '1.0', else 2 decimal."""
    if abs(val - 1.0) < 0.005:
        return "1.0"
    s = f"{val:.2f}"
    # Remove leading zero: "0.55" -> ".55"
    if s.startswith("0."):
        s = s[1:]
    return s


def plot_variant_figure(data, var, layers, output_dir, vmin, vmax):
    """
    Create one figure for a single model variant.
    3 layer sub-panels (top to bottom), each with 3 dataset rows.
    """
    datasets = data["datasets"]
    variant_labels = data["variant_labels"]
    E = len(data["experts"])
    n_layers = len(layers)
    n_datasets = len(datasets)

    var_label = variant_labels.get(var, var)

    # ── Square cells: cell_size determines both width and height ──
    cell_size = 0.55  # inches per cell side
    heatmap_w = E * cell_size
    heatmap_h = 1 * cell_size  # each row is 1 cell tall

    # Figure dimensions
    left_margin = 1.0   # space for dataset labels
    right_margin = 0.3
    top_margin = 0.6
    bottom_margin = 0.6
    layer_title_h = 0.25  # space for layer title above each sub-panel
    layer_gap = 0.25       # gap between layer sections
    row_gap = 0.05        # tiny gap between dataset rows within a layer

    panel_h = n_datasets * heatmap_h + (n_datasets - 1) * row_gap + layer_title_h
    fig_w = left_margin + heatmap_w + right_margin
    fig_h = top_margin + n_layers * panel_h + (n_layers - 1) * layer_gap + bottom_margin

    fig = plt.figure(figsize=(fig_w, fig_h))

    for li, layer_idx in enumerate(layers):
        for di, ds in enumerate(datasets):
            act = data["activation"][var][ds][layer_idx].numpy()  # [E]
            act_2d = act.reshape(1, -1)  # [1, E]

            # Compute axes position in figure coordinates
            x0 = left_margin / fig_w
            y_panel_top = 1.0 - (top_margin + li * (panel_h + layer_gap)) / fig_h
            y0 = y_panel_top - (layer_title_h + (di + 1) * heatmap_h + di * row_gap) / fig_h
            w = heatmap_w / fig_w
            h = heatmap_h / fig_h

            ax = fig.add_axes([x0, y0, w, h])

            im = ax.imshow(
                act_2d,
                aspect="equal",
                cmap=ACTIVATION_CMAP,
                vmin=vmin, vmax=vmax,
                interpolation="nearest",
            )

            # Remove all cell borders/gridlines
            ax.tick_params(
                left=False, right=False, top=False, bottom=False,
                labelleft=True, labelbottom=False,
            )
            for spine in ax.spines.values():
                spine.set_visible(False)

            # Y-axis: dataset label
            ds_label = DATASET_DISPLAY.get(ds, ds)
            ax.set_yticks([0])
            ax.set_yticklabels([ds_label], fontsize=20, fontweight="bold")

            # X-axis: only at bottom of the last layer section
            is_bottom = (di == n_datasets - 1)
            is_last_layer = (li == n_layers - 1)
            if is_bottom and is_last_layer:
                ax.tick_params(labelbottom=True, bottom=False)
                ax.set_xticks(range(E))
                ax.set_xticklabels(range(E), fontsize=20, fontweight="bold")
                ax.set_xlabel("Expert ID", fontsize=20, fontweight="bold")
            else:
                ax.set_xticks([])

            # Annotate values in cells
            for e in range(E):
                val = act[e]
                text_color = "white" if val > vmax * 0.5 else "black"
                ax.text(e, 0, fmt_val(val), ha="center", va="center",
                        #fontsize=12, fontweight="bold", color=text_color)
                        fontsize=18,  color=text_color)

        # Layer title
        title_y = y_panel_top - (layer_title_h * 0.3) / fig_h
        fig.text(
            x0 + w / 2, title_y,
            f"Layer {layer_idx}",
            ha="center", va="bottom",
            fontsize=20, fontweight="bold",
            transform=fig.transFigure,
        )

    # No colorbar (removed per requirement)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_path = output_dir / f"heatmap_{var_label}.pdf"
    fig.savefig(str(save_path), bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"  Saved {save_path}")


def print_summary(data, layers):
    """Print activation summary table."""
    variants = data["variants"]
    datasets = data["datasets"]
    variant_labels = data["variant_labels"]

    print(f"\n{'Variant':<12} {'Dataset':<12} {'Layer':<6} "
          f"{'Mean':>6} {'Std':>6} {'Min':>6} {'Max':>6} {'CV':>6}")
    print("-" * 66)

    for var in variants:
        for ds in datasets:
            act = data["activation"][var][ds]  # [L, E]
            for l in layers:
                row = act[l].numpy()
                m, s = row.mean(), row.std()
                cv = s / m if m > 0 else 0
                print(f"{variant_labels[var]:<12} {ds:<12} L{l:<4} "
                      f"{m:6.4f} {s:6.4f} {row.min():6.4f} {row.max():6.4f} {cv:6.3f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="./gate_scores/activation_data.pt")
    parser.add_argument("--output-dir", type=str, default="./figures")
    parser.add_argument("--layers", type=int, nargs="+", default=[0, 5, 11])
    args = parser.parse_args()

    print(f"Loading data from {args.input} ...")
    data = torch.load(args.input, map_location="cpu", weights_only=False)

    print(f"Variants: {data['variants']}")
    print(f"Datasets: {data['datasets']}")
    print(f"Layers available: {data['layers']}")
    print(f"Experts: {len(data['experts'])}")

    # Validate layer indices
    valid_layers = [l for l in args.layers if l in data["layers"]]
    if not valid_layers:
        print(f"No valid layers in {args.layers}. Available: {data['layers']}")
        return

    print_summary(data, valid_layers)

    # Compute global vmin/vmax across ALL variants
    all_vals = []
    for var in data["variants"]:
        for ds in data["datasets"]:
            act = data["activation"][var][ds]
            for l in valid_layers:
                all_vals.append(act[l].numpy())
    all_vals = np.concatenate(all_vals)
    vmin = 0.0
    vmax = float(np.percentile(all_vals, 99))

    # Generate one figure per variant
    print(f"\nGenerating heatmaps for layers {valid_layers} ...")
    for var in data["variants"]:
        plot_variant_figure(data, var, valid_layers, args.output_dir, vmin, vmax)

    print("Done.")


if __name__ == "__main__":
    main()

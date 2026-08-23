"""
Detailed FLOPs-per-token calculation for all model variants.

Computes forward-pass FLOPs per token broken down by module:
  - RMSNorm, Attention projections, Attention scores, Residuals
  - MoE routing (router/gate_proj_A, softmax, top-K, L2 norm, threshold)
  - MoE FFN (gate/gate_proj_B, SiLU, up_proj, elem mul, down_proj)

Usage:
    python compute_flops.py
    python compute_flops.py --seq-len 1024
    python compute_flops.py --size M
"""
import argparse
import math


# ── Size presets ──────────────────────────────────────────────────────
SIZES = {
    "S": dict(d=512, D=128, L=12, E=12, K=3, nh=16, nkv=16, hd=32),
    "M": dict(d=768, D=192, L=24, E=16, K=4, nh=24, nkv=12, hd=32),
    "L": dict(d=1024, D=256, L=32, E=24, K=6, nh=32, nkv=16, hd=32),
}

# ── Model variants to compute ────────────────────────────────────────
# (display_name, model_type, rank)
VARIANTS_S = [
    ("Baseline MoE", "baseline", 0),
    ("AoE (r=16)",   "aoe",      16),
    ("AoE (r=32)",   "aoe",      32),
    ("ReMoE",        "remoe",    0),
    ("RFMoE (r=8)",  "rf",       8),
    ("RFMoE (r=16)", "rf",       16),
    ("RFMoE (r=32)", "rf",       32),
    ("RFMoE (r=64)", "rf",       64),
]

VARIANTS_M = [
    ("Baseline MoE (Top4/16)", "baseline", 0),
    ("RFMoE (r=48)",           "rf",       48),
]


def compute_shared(d, nkv, hd, nh, T):
    """Shared per-layer FLOPs (same for all model types)."""
    d_kv = nkv * hd
    items = {}

    # RMSNorm ×2 (input_layernorm + post_attention_layernorm)
    # Per norm: d (square) + d (sum) + 1 (div+sqrt) + d (scale) ≈ 3d
    items["RMSNorm (×2)"] = {
        "formula": "2 × 3×d",
        "values":  f"2 × 3×{d}",
        "flops":   2 * 3 * d,
    }

    # Attention Q/K/V/O projections
    q = 2 * d * d
    k = 2 * d * d_kv
    v = 2 * d * d_kv
    o = 2 * d * d
    items["Q_proj"] = {"formula": "2×d×d",    "values": f"2×{d}×{d}",      "flops": q}
    items["K_proj"] = {"formula": "2×d×d_kv", "values": f"2×{d}×{d_kv}",   "flops": k}
    items["V_proj"] = {"formula": "2×d×d_kv", "values": f"2×{d}×{d_kv}",   "flops": v}
    items["O_proj"] = {"formula": "2×d×d",    "values": f"2×{d}×{d}",      "flops": o}

    # Attention score computation
    qk_flops = 2 * nh * hd * T
    sm_flops = 3 * T * nh
    av_flops = 2 * nh * T * hd
    items["Q·Kᵀ"]       = {"formula": "2×nh×hd×T",  "values": f"2×{nh}×{hd}×{T}",  "flops": qk_flops}
    items["Softmax(T)"]  = {"formula": "3×T×nh",     "values": f"3×{T}×{nh}",        "flops": sm_flops}
    items["Attn·V"]      = {"formula": "2×nh×T×hd",  "values": f"2×{nh}×{T}×{hd}",  "flops": av_flops}

    # Residual additions ×2
    items["Residuals (×2)"] = {"formula": "2×d", "values": f"2×{d}", "flops": 2 * d}

    total = sum(v["flops"] for v in items.values())
    return items, total


def compute_moe(d, D, E, K, r, T, mtype):
    """MoE FFN per-layer FLOPs for a given model type."""
    routing_items = {}
    expert_items = {}

    if mtype == "baseline":
        # ── Routing ──
        routing_items["Router proj"]  = {"formula": "2×d×E",       "values": f"2×{d}×{E}",     "flops": 2*d*E}
        routing_items["Softmax(E)"]   = {"formula": "3×E",         "values": f"3×{E}",          "flops": 3*E}
        routing_items["Top-K select"] = {"formula": "E×log₂(K)",   "values": f"{E}×{max(1,int(math.log2(K)))}",  "flops": E*max(1,int(math.log2(K)))}
        # ── Per active expert (standard SwiGLU) ──
        expert_items["gate_proj"] = {"formula": "2×d×D", "values": f"2×{d}×{D}", "flops": 2*d*D}
        expert_items["SiLU"]      = {"formula": "3×D",   "values": f"3×{D}",     "flops": 3*D}
        expert_items["up_proj"]   = {"formula": "2×d×D", "values": f"2×{d}×{D}", "flops": 2*d*D}
        expert_items["elem mul"]  = {"formula": "D",     "values": f"{D}",       "flops": D}
        expert_items["down_proj"] = {"formula": "2×D×d", "values": f"2×{D}×{d}", "flops": 2*D*d}
        K_act = K

    elif mtype == "aoe":
        # ── Routing (self-evaluation) ──
        routing_items["gate_proj_A (batched)"] = {"formula": "2×d×E×r",  "values": f"2×{d}×{E}×{r}",   "flops": 2*d*E*r}
        routing_items["L2 norm (×E)"]          = {"formula": "E×(2r+1)", "values": f"{E}×{2*r+1}",      "flops": E*(2*r+1)}
        routing_items["Top-K select"]          = {"formula": "E×log₂(K)","values": f"{E}×{max(1,int(math.log2(K)))}",  "flops": E*max(1,int(math.log2(K)))}
        routing_items["Softmax(K)"]            = {"formula": "3×K",      "values": f"3×{K}",             "flops": 3*K}
        # ── Per active expert (cached gate_hidden → gate_proj_B) ──
        expert_items["gate_proj_B"] = {"formula": "2×r×D", "values": f"2×{r}×{D}", "flops": 2*r*D}
        expert_items["SiLU"]        = {"formula": "3×D",   "values": f"3×{D}",     "flops": 3*D}
        expert_items["up_proj"]     = {"formula": "2×d×D", "values": f"2×{d}×{D}", "flops": 2*d*D}
        expert_items["elem mul"]    = {"formula": "D",     "values": f"{D}",       "flops": D}
        expert_items["down_proj"]   = {"formula": "2×D×d", "values": f"2×{D}×{d}", "flops": 2*D*d}
        K_act = K

    elif mtype == "remoe":
        # ── Routing (ReLU router) ──
        routing_items["Router proj"] = {"formula": "2×d×E", "values": f"2×{d}×{E}", "flops": 2*d*E}
        routing_items["ReLU"]        = {"formula": "E",     "values": f"{E}",        "flops": E}
        # No softmax, no top-K
        # ── Per active expert (standard SwiGLU) ──
        expert_items["gate_proj"] = {"formula": "2×d×D", "values": f"2×{d}×{D}", "flops": 2*d*D}
        expert_items["SiLU"]      = {"formula": "3×D",   "values": f"3×{D}",     "flops": 3*D}
        expert_items["up_proj"]   = {"formula": "2×d×D", "values": f"2×{d}×{D}", "flops": 2*d*D}
        expert_items["elem mul"]  = {"formula": "D",     "values": f"{D}",       "flops": D}
        expert_items["down_proj"] = {"formula": "2×D×d", "values": f"2×{D}×{d}", "flops": 2*D*d}
        K_act = max(1, int(E * 0.25))

    elif mtype == "rf":
        # ── Routing (gate_proj_A reused for FFN) ──
        routing_items["gate_proj_A (reused)"] = {"formula": "2×d×E×r",  "values": f"2×{d}×{E}×{r}",  "flops": 2*d*E*r}
        routing_items["L2 norm (×E)"]         = {"formula": "E×(2r+1)", "values": f"{E}×{2*r+1}",     "flops": E*(2*r+1)}
        routing_items["Scale+Bias (×E)"]      = {"formula": "E×2",      "values": f"{E}×2",            "flops": E*2}
        routing_items["Threshold"]             = {"formula": "E",        "values": f"{E}",              "flops": E}
        # No softmax, no top-K, no separate router
        # ── Per active expert (low-rank gate) ──
        expert_items["gate_proj_B"] = {"formula": "2×r×D", "values": f"2×{r}×{D}", "flops": 2*r*D}
        expert_items["SiLU"]        = {"formula": "3×D",   "values": f"3×{D}",     "flops": 3*D}
        expert_items["up_proj"]     = {"formula": "2×d×D", "values": f"2×{d}×{D}", "flops": 2*d*D}
        expert_items["elem mul"]    = {"formula": "D",     "values": f"{D}",       "flops": D}
        expert_items["down_proj"]   = {"formula": "2×D×d", "values": f"2×{D}×{d}", "flops": 2*D*d}
        K_act = max(1, int(E * 0.25))

    route_total = sum(v["flops"] for v in routing_items.values())
    per_expert  = sum(v["flops"] for v in expert_items.values())
    moe_total   = route_total + K_act * per_expert

    return routing_items, expert_items, route_total, per_expert, K_act, moe_total


def print_model(name, mtype, r, cfg, T):
    d, D, L, E, K = cfg["d"], cfg["D"], cfg["L"], cfg["E"], cfg["K"]
    nh, nkv, hd = cfg["nh"], cfg["nkv"], cfg["hd"]

    print("=" * 70)
    print(f"  {name}")
    print("=" * 70)

    # ── Shared ──
    shared_items, shared_total = compute_shared(d, nkv, hd, nh, T)
    print()
    print("  [Shared modules — per layer, per token]")
    for k, v in shared_items.items():
        print(f"    {k:<20s}  {v['formula']:<16s} = {v['values']:<20s} = {v['flops']:>12,}")
    print(f"    {'SUBTOTAL':<20s}  {'':16s}   {'':20s}   {shared_total:>12,}")

    # ── MoE ──
    routing_items, expert_items, route_total, per_expert, K_act, moe_total = \
        compute_moe(d, D, E, K, r, T, mtype)

    print()
    print("  [Routing — per layer, per token]")
    for k, v in routing_items.items():
        print(f"    {k:<28s}  {v['formula']:<16s} = {v['values']:<20s} = {v['flops']:>12,}")
    print(f"    {'SUBTOTAL':<28s}  {'':16s}   {'':20s}   {route_total:>12,}")

    print()
    print(f"  [Per active expert FFN]")
    for k, v in expert_items.items():
        print(f"    {k:<20s}  {v['formula']:<16s} = {v['values']:<20s} = {v['flops']:>12,}")
    print(f"    {'SUBTOTAL':<20s}  {'':16s}   {'':20s}   {per_expert:>12,}")
    print(f"    × K_act={K_act:<14d} {'':16s}   {'':20s}   {K_act*per_expert:>12,}")

    print()
    print(f"  [Summary]")
    layer_total = shared_total + moe_total
    model_total = L * layer_total + 3 * d  # + final RMSNorm
    print(f"    MoE FFN / layer:    {route_total:,} (routing) + {K_act}×{per_expert:,} (experts) = {moe_total:,}")
    print(f"    Total / layer:      {shared_total:,} (shared) + {moe_total:,} (MoE) = {layer_total:,}")
    print(f"    Total model:        {L} × {layer_total:,} + {3*d} (final norm) = {model_total:,}")
    print(f"    GFLOPs/token:       {model_total / 1e9:.4f}")
    print()

    return model_total


def main():
    parser = argparse.ArgumentParser(description="Compute per-token forward FLOPs")
    parser.add_argument("--seq-len", type=int, default=512, help="Sequence length T (default: 512)")
    parser.add_argument("--size", choices=["S", "M", "all"], default="all", help="Model size (default: all)")
    args = parser.parse_args()

    T = args.seq_len

    results = []

    if args.size in ("S", "all"):
        cfg = SIZES["S"]
        print(f"\n{'#'*70}")
        print(f"  S-size: d={cfg['d']}, D={cfg['D']}, L={cfg['L']}, E={cfg['E']}, K={cfg['K']}, T={T}")
        print(f"{'#'*70}")
        for name, mtype, r in VARIANTS_S:
            flops = print_model(name, mtype, r, cfg, T)
            results.append((name, flops))

    if args.size in ("M", "all"):
        cfg = SIZES["M"]
        print(f"\n{'#'*70}")
        print(f"  M-size: d={cfg['d']}, D={cfg['D']}, L={cfg['L']}, E={cfg['E']}, K={cfg['K']}, T={T}")
        print(f"{'#'*70}")
        for name, mtype, r in VARIANTS_M:
            flops = print_model(name, mtype, r, cfg, T)
            results.append((name, flops))

    # ── Summary table ──
    print("\n" + "=" * 55)
    print(f"  {'Model':<30s}  {'FLOPs':>12s}  {'GFLOPs/tok':>10s}")
    print("-" * 55)
    for name, flops in results:
        print(f"  {name:<30s}  {flops:>12,}  {flops/1e9:>10.4f}")
    print("=" * 55)


if __name__ == "__main__":
    main()

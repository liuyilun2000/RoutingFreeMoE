"""
Convert a pretrained Mixtral 8x7B (or any Mixtral-family MoE) into a
RoutingFreeMixtral (RFMoE) checkpoint that can be loaded and run via
RoutingFreeMixtralForCausalLM.

Structural changes
------------------
  COPIED (key-identical):
    model.embed_tokens, model.norm, lm_head
    model.layers.*.self_attn.*
    model.layers.*.input_layernorm, post_attention_layernorm

  COPIED (key-renamed):
    experts[i].w1.weight  ->  experts[i].gate_proj.weight
    experts[i].w3.weight  ->  experts[i].up_proj.weight
    experts[i].w2.weight  ->  experts[i].down_proj.weight

  DISCARDED:
    block_sparse_moe.gate.weight  (original top-k router, not used in RFMoE)

  DERIVED via SVD of gate_proj.weight (w1):
    experts[i].gate_proj_A.weight   shape [rank, hidden]
    experts[i].gate_proj_B.weight   shape [intermediate, rank]

    gate_proj.weight  =  U @ diag(S) @ Vh   (full SVD)
    gate_proj_A       =  diag(S[:r]^0.5) @ Vh[:r, :]     [rank, hidden]
    gate_proj_B       =  U[:, :r] @ diag(S[:r]^0.5)      [intermediate, rank]
    → gate_proj_B @ gate_proj_A  ≈  gate_proj.weight  (best rank-r approximation)

    The gate score is norm(gate_proj_A(x)), so seeding from the top singular
    directions of w1 makes the initial gating sensitive to the same input
    subspace that the pretrained expert uses for its gate projection.

  KEPT as default init:
    experts[i].gate.gate_scale
    experts[i].gate.gate_bias

Usage
-----
  # From a local checkpoint directory:
  python convert_mixtral_to_rfmoe.py \\
      --source-model /path/to/Mixtral-8x7B-v0.1 \\
      --output-dir   ./rfmoe_converted

  # With explicit rank and RF gating hyper-params:
  python convert_mixtral_to_rfmoe.py \\
      --source-model mistralai/Mixtral-8x7B-v0.1 \\
      --output-dir   ./rfmoe_converted \\
      --gate-proj-rank 512 \\
      --gate-threshold 0.05 \\
      --density-target 0.25 \\
      --dtype bfloat16

  # The resulting checkpoint can be loaded with:
  #   register_routing_free_model()
  #   model = AutoModelForCausalLM.from_pretrained("./rfmoe_converted")
  # (see eval_benchmarks.py for the registration pattern)
"""

import argparse
import os
import sys
from pathlib import Path

import torch

# ---------------------------------------------------------------------------
# Make project imports work regardless of cwd
# ---------------------------------------------------------------------------
_project_root = os.path.dirname(os.path.abspath(__file__))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)



# ---------------------------------------------------------------------------
# Key-mapping helpers
# ---------------------------------------------------------------------------

def _is_direct_copy_key(key: str) -> bool:
    """
    Returns True for keys that exist with the same name in both models.
    Handles: embed_tokens, norm, lm_head, all self_attn.*, layernorms.
    """
    # Top-level non-layer tensors
    if key.startswith(("model.embed_tokens.", "model.norm.", "lm_head.")):
        return True
    # Layer tensors that are NOT inside block_sparse_moe
    if key.startswith("model.layers.") and "block_sparse_moe" not in key:
        return True
    return False


def _remap_expert_key(key: str) -> str | None:
    """
    Remap Mixtral expert weight names to RFMoE names.

    Mixtral (HF ModuleList):
        ...block_sparse_moe.experts.{i}.w1.weight  (gate)
        ...block_sparse_moe.experts.{i}.w2.weight  (down)
        ...block_sparse_moe.experts.{i}.w3.weight  (up)

    RFMoE:
        ...block_sparse_moe.experts.{i}.gate_proj.weight
        ...block_sparse_moe.experts.{i}.down_proj.weight
        ...block_sparse_moe.experts.{i}.up_proj.weight

    Returns the remapped key, or None if the key is not a recognised expert
    weight (so the caller can decide to skip / discard it).
    """
    if "block_sparse_moe.experts." not in key:
        return None

    original = key
    key = key.replace(".w1.weight", ".gate_proj.weight")
    key = key.replace(".w2.weight", ".down_proj.weight")
    key = key.replace(".w3.weight", ".up_proj.weight")

    # If nothing changed the key was something else (e.g. bias, unknown sub-module)
    return key if key != original else None


def _report_svd_reconstruction(new_sd: dict, rank: int):
    """
    Print the relative reconstruction error  ||W - B@A|| / ||W||  for the
    gate_proj of layer 0 / expert 0 as a quick sanity check.
    """
    key_w = "model.layers.0.block_sparse_moe.experts.0.gate_proj.weight"
    key_a = "model.layers.0.block_sparse_moe.experts.0.gate_proj_A.weight"
    key_b = "model.layers.0.block_sparse_moe.experts.0.gate_proj_B.weight"
    if not all(k in new_sd for k in (key_w, key_a, key_b)):
        return
    W = new_sd[key_w].float()
    A = new_sd[key_a].float()
    B = new_sd[key_b].float()
    err = (W - B @ A).norm() / W.norm()
    var_explained = 1.0 - err ** 2
    print(f"    Reconstruction check (layer 0 / expert 0, rank={rank}):")
    print(f"      ||W - B@A|| / ||W||  = {err:.6f}")
    print(f"      Variance explained   ≈ {var_explained * 100:.2f}%")


def _build_fused_expert_tensors(src_sd: dict, num_layers: int, num_experts: int):
    """
    Handle the *fused* MixtralExperts format used by the local modeling_mixtral.py:
        block_sparse_moe.experts.gate_up_proj  shape [E, 2*I, H]
        block_sparse_moe.experts.down_proj     shape [E, H, I]

    Returns a dict of {rfmoe_key: tensor} for all expert weights, or an empty
    dict if the checkpoint does not use the fused format.
    """
    result = {}
    for layer_idx in range(num_layers):
        pfx = f"model.layers.{layer_idx}.block_sparse_moe.experts"
        gate_up_key = f"{pfx}.gate_up_proj"
        down_key    = f"{pfx}.down_proj"

        if gate_up_key not in src_sd:
            continue  # not fused format for this layer

        gate_up = src_sd[gate_up_key]   # [E, 2*I, H]
        down    = src_sd[down_key]      # [E, H, I]  (note: Linear stores [out, in])
        inter_dim = gate_up.shape[1] // 2

        for e in range(num_experts):
            ep = f"model.layers.{layer_idx}.block_sparse_moe.experts.{e}"
            result[f"{ep}.gate_proj.weight"] = gate_up[e, :inter_dim, :]
            result[f"{ep}.up_proj.weight"]   = gate_up[e, inter_dim:, :]
            result[f"{ep}.down_proj.weight"] = down[e]

    return result


# ---------------------------------------------------------------------------
# Quantization helpers (shared logic)
# ---------------------------------------------------------------------------

def _make_bnb_config(quantization: str, compute_dtype: torch.dtype):
    """
    Build a BitsAndBytesConfig for the requested quantization level.

    quantization:
        "none"  – no quantization (plain dtype load)
        "int8"  – bitsandbytes LLM.int8()
        "int4"  – bitsandbytes QLoRA NF4 double-quant
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


def _extract_src_sd_dequant(src_model) -> dict:
    """
    Extract a plain float32 state dict from src_model.

    Calling .to(torch.float32) on bitsandbytes quantized parameter tensors
    triggers automatic dequantization via bitsandbytes' tensor protocol, so
    this works transparently for both plain and quantized source models.
    SVD is always computed in float32 for numerical stability.
    """
    return {k: v.to(torch.float32) for k, v in src_model.state_dict().items()}


# ---------------------------------------------------------------------------
# Main conversion routine
# ---------------------------------------------------------------------------

def _init_gate_scale_bias(
    new_sd: dict,
    num_layers: int,
    num_experts: int,
    dtype: torch.dtype,
    gate_bias_init: float = -1e-6,
):
    """
    Explicitly set gate_scale = 1.0 and gate_bias = gate_bias_init for every
    expert so the init is deterministic regardless of the RF model's default.

    gate_scale = 1   → gate score = ||gate_proj_A(x)||, no rescaling at start
    gate_bias  = -ε  → threshold-crossing is purely norm-based at init
    """
    for layer_idx in range(num_layers):
        for expert_idx in range(num_experts):
            pfx = (
                f"model.layers.{layer_idx}"
                f".block_sparse_moe.experts.{expert_idx}.gate"
            )
            scale_key = f"{pfx}.gate_scale"
            bias_key  = f"{pfx}.gate_bias"
            if scale_key in new_sd:
                new_sd[scale_key] = torch.ones(1, dtype=dtype)
            if bias_key in new_sd:
                new_sd[bias_key] = torch.full((1,), gate_bias_init, dtype=dtype)


def _svd_init_gate_proj(
    new_sd: dict,
    num_layers: int,
    num_experts: int,
    rank: int,
    dtype: torch.dtype,
):
    """
    For every expert in every layer, decompose gate_proj.weight via truncated SVD
    and store the factors into gate_proj_A.weight and gate_proj_B.weight.

    gate_proj.weight  W  has shape  [intermediate, hidden].
    Full SVD:  W = U S Vᵀ
      gate_proj_A.weight  =  diag(S[:r]^0.5) @ Vh[:r, :]   shape [rank, hidden]
      gate_proj_B.weight  =  U[:, :r] @ diag(S[:r]^0.5)    shape [intermediate, rank]
    so that  gate_proj_B @ gate_proj_A  ≈  W  (best rank-r approximation).

    SVD is computed in float32 for numerical stability, then cast back to dtype.
    """
    total = num_layers * num_experts
    done  = 0
    for layer_idx in range(num_layers):
        for expert_idx in range(num_experts):
            pfx = (
                f"model.layers.{layer_idx}"
                f".block_sparse_moe.experts.{expert_idx}"
            )
            w_key = f"{pfx}.gate_proj.weight"
            a_key = f"{pfx}.gate_proj_A.weight"
            b_key = f"{pfx}.gate_proj_B.weight"

            if w_key not in new_sd:
                continue  # shouldn't happen, but be safe

            W = new_sd[w_key].float()  # [intermediate, hidden]

            # Truncated SVD via torch.linalg.svd (economy mode)
            U, S, Vh = torch.linalg.svd(W, full_matrices=False)
            # U:  [intermediate, min(I,H)]
            # S:  [min(I,H)]
            # Vh: [min(I,H), hidden]

            r = min(rank, S.shape[0])
            S_sqrt = S[:r].sqrt()                           # [r]

            A = S_sqrt.unsqueeze(1) * Vh[:r, :]            # [r, hidden]
            B = U[:, :r] * S_sqrt.unsqueeze(0)             # [intermediate, r]

            new_sd[a_key] = A.to(dtype)
            new_sd[b_key] = B.to(dtype)

            done += 1
            print(f"\r    SVD progress: {done}/{total} experts", end="", flush=True)

    print()  # newline after progress line


def convert_mixtral_to_rfmoe(
    source_model_path: str,
    output_dir: str,
    # RF gating hyper-params
    gate_proj_rank: int | None = None,
    gate_norm: str = "l2",
    gate_bias: bool = True,
    gate_scale: bool = True,
    gate_act_fn: str = "linear",
    gate_threshold: float = 0.05,
    gate_temperature: float = 1.0,
    output_gate_scores: bool = False,
    density_target: float = 0.25,
    lambda_coef: float = 1e-5,
    eta_coef: float = 0.2,
    per_expert_aux_loss_coef: float = 0.5,
    per_token_aux_loss_coef: float = 0.5,
    # Loading options
    dtype: str = "bfloat16",
    quantization: str = "none",   # "none" | "int8" | "int4"
    hf_cache_dir: str = None,     # defaults to HuggingFace default cache
):
    """
    Load a pretrained Mixtral model and produce an RFMoE checkpoint.

    FFN expert weights are preserved exactly; the original top-k router is
    discarded; gate_proj_A / gate_proj_B are initialised from a truncated SVD
    of each expert's gate_proj (w1) matrix.

    quantization controls how the *source* model is loaded to save host memory:
      "none"  – full dtype (bf16/fp16/fp32)
      "int8"  – bitsandbytes LLM.int8() (≈ half the bf16 memory)
      "int4"  – bitsandbytes NF4 double-quant (≈ quarter the bf16 memory)
    Weights are always dequantised to float32 before SVD and saved in `dtype`.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from routing_free.mixtral_rf import (
        RoutingFreeMixtralConfig,
        RoutingFreeMixtralForCausalLM,
    )

    torch_dtype = getattr(torch, dtype)
    bnb_config   = _make_bnb_config(quantization, torch_dtype)

    # ------------------------------------------------------------------
    # 1. Load source model
    # ------------------------------------------------------------------
    n_gpus = torch.cuda.device_count()
    load_kwargs = dict(
        trust_remote_code=True,
        cache_dir=hf_cache_dir,
    )
    if bnb_config is not None:
        # quantization requires GPU; device_map="auto" shards across all GPUs
        load_kwargs["quantization_config"] = bnb_config
        load_kwargs["device_map"] = "auto"
        print(f"[1/6] Loading source model ({quantization}) from: {source_model_path}")
        print(f"      cache_dir={hf_cache_dir}, GPUs available={n_gpus}")
    else:
        # Without quantization: load in dtype, let the user choose device
        load_kwargs["torch_dtype"] = torch_dtype
        load_kwargs["device_map"] = "auto" if n_gpus > 0 else "cpu"
        print(f"[1/6] Loading source model ({dtype}) from: {source_model_path}")
        print(f"      cache_dir={hf_cache_dir}, device_map={load_kwargs['device_map']}")

    src_model = AutoModelForCausalLM.from_pretrained(source_model_path, **load_kwargs)
    src_cfg = src_model.config
    num_layers   = src_cfg.num_hidden_layers
    num_experts  = src_cfg.num_local_experts
    hidden_size  = src_cfg.hidden_size
    resolved_gate_proj_rank = gate_proj_rank or (hidden_size // 4)

    print(
        f"    model_type={src_cfg.model_type}, "
        f"layers={num_layers}, experts={num_experts}, "
        f"hidden={hidden_size}, intermediate={src_cfg.intermediate_size}"
    )

    # ------------------------------------------------------------------
    # 2. Build RF config (inherit all Mixtral fields, add RF fields)
    # ------------------------------------------------------------------
    print("[2/6] Building RoutingFreeMixtralConfig …")

    # Fields that live in RoutingFreeConfigMixin — we set them explicitly below.
    _rf_fields = {
        "gate_proj_rank", "gate_norm", "gate_bias", "gate_scale",
        "gate_act_fn", "gate_threshold", "gate_temperature",
        "output_gate_scores", "n_experts", "density_target",
        "lambda_coef", "eta_coef",
        "per_expert_aux_loss_coef", "per_token_aux_loss_coef",
    }
    # Fields that should not be forwarded from the source config
    _skip_src_fields = {"model_type", "_name_or_path", "architectures", "auto_map"} | _rf_fields

    src_dict = {k: v for k, v in src_cfg.to_dict().items() if k not in _skip_src_fields}

    rf_config = RoutingFreeMixtralConfig(
        **src_dict,
        # RF gating params
        gate_proj_rank=resolved_gate_proj_rank,
        gate_norm=gate_norm,
        gate_bias=gate_bias,
        gate_scale=gate_scale,
        gate_act_fn=gate_act_fn,
        gate_threshold=gate_threshold,
        gate_temperature=gate_temperature,
        output_gate_scores=output_gate_scores,
        n_experts=num_experts,
        density_target=density_target,
        lambda_coef=lambda_coef,
        eta_coef=eta_coef,
        per_expert_aux_loss_coef=per_expert_aux_loss_coef,
        per_token_aux_loss_coef=per_token_aux_loss_coef,
    )
    print(f"    gate_proj_rank={resolved_gate_proj_rank}, gate_threshold={gate_threshold}, "
          f"density_target={density_target}")

    # ------------------------------------------------------------------
    # 3. Instantiate the RF model (random weights — will be overwritten)
    # ------------------------------------------------------------------
    print("[3/6] Instantiating RoutingFreeMixtralForCausalLM …")
    rf_model = RoutingFreeMixtralForCausalLM(rf_config).to(torch_dtype)

    # ------------------------------------------------------------------
    # 4. Copy / remap weights from source into a new state dict
    # ------------------------------------------------------------------
    print("[4/6] Copying weights …")
    # _extract_src_sd_dequant casts all tensors to float32, triggering
    # bitsandbytes dequantisation automatically for int8/int4 models.
    src_sd = _extract_src_sd_dequant(src_model)
    dst_sd = rf_model.state_dict()

    # Handle fused-expert format (local modeling_mixtral.py)
    fused_expert_tensors = _build_fused_expert_tensors(src_sd, num_layers, num_experts)

    new_sd        = dict(dst_sd)   # start from RF model defaults (gate_scale/bias etc.)
    n_copied      = 0
    n_discarded   = 0
    shape_errors  = []
    unknown_keys  = []

    for src_key, src_val in src_sd.items():
        # ── Direct copy (same key) ──────────────────────────────────────
        if _is_direct_copy_key(src_key):
            if src_key not in dst_sd:
                unknown_keys.append(src_key)
                continue
            if src_val.shape != dst_sd[src_key].shape:
                shape_errors.append((src_key, src_val.shape, dst_sd[src_key].shape))
                continue
            new_sd[src_key] = src_val.to(torch_dtype)
            n_copied += 1
            continue

        # ── Expert weights (renaming) ───────────────────────────────────
        # Skip fused-format expert params here; they were pre-split above
        if "block_sparse_moe.experts" in src_key and not any(
            src_key.endswith(s) for s in (".w1.weight", ".w2.weight", ".w3.weight")
        ):
            continue  # handled via fused path or irrelevant

        dst_key = _remap_expert_key(src_key)
        if dst_key is not None:
            if dst_key not in dst_sd:
                unknown_keys.append(f"{src_key} -> {dst_key}")
                continue
            if src_val.shape != dst_sd[dst_key].shape:
                shape_errors.append((src_key, src_val.shape, dst_sd[dst_key].shape))
                continue
            new_sd[dst_key] = src_val.to(torch_dtype)
            n_copied += 1
            continue

        # ── Router: discard ─────────────────────────────────────────────
        if "block_sparse_moe.gate" in src_key:
            n_discarded += 1
            continue

        unknown_keys.append(src_key)

    # Apply pre-split fused expert tensors
    for dst_key, tensor in fused_expert_tensors.items():
        if dst_key not in dst_sd:
            unknown_keys.append(f"[fused] -> {dst_key}")
            continue
        if tensor.shape != dst_sd[dst_key].shape:
            shape_errors.append((f"[fused]->{dst_key}", tensor.shape, dst_sd[dst_key].shape))
            continue
        new_sd[dst_key] = tensor.to(torch_dtype)
        n_copied += 1

    print(f"\n    Weight copy summary:")
    print(f"      Copied (exact / renamed) : {n_copied}")
    print(f"      Discarded (top-k router) : {n_discarded}")
    if shape_errors:
        print(f"      Shape mismatches (skipped):")
        for key, src_sh, dst_sh in shape_errors:
            print(f"        {key}: src={src_sh} vs dst={dst_sh}")
    if unknown_keys:
        print(f"      Unknown / unmapped src keys:")
        for k in unknown_keys:
            print(f"        {k}")

    # ------------------------------------------------------------------
    # 5. Initialise gate_proj_A and gate_proj_B via SVD of gate_proj
    # ------------------------------------------------------------------
    print(f"[5/6] SVD-initialising gate_proj_A / gate_proj_B "
          f"(rank={resolved_gate_proj_rank}) …")
    _svd_init_gate_proj(new_sd, num_layers, num_experts, resolved_gate_proj_rank, torch_dtype)

    # Verify reconstruction quality on first expert of first layer
    _report_svd_reconstruction(new_sd, rank=resolved_gate_proj_rank)

    # Explicitly set gate_scale=1 and gate_bias=-epsilon (deterministic, dtype-safe)
    _init_gate_scale_bias(new_sd, num_layers, num_experts, torch_dtype, gate_bias_init=-1e-6)
    print(f"    gate_scale initialised to 1.0, gate_bias initialised to -1e-6 for all experts.")

    rf_model.load_state_dict(new_sd, strict=True)

    # ------------------------------------------------------------------
    # 6. Save
    # ------------------------------------------------------------------
    print(f"\n[6/6] Saving to {output_dir} …")
    os.makedirs(output_dir, exist_ok=True)
    rf_model.save_pretrained(output_dir)
    rf_config.save_pretrained(output_dir)

    # Try to copy the tokenizer
    try:
        tok = AutoTokenizer.from_pretrained(source_model_path)
        tok.save_pretrained(output_dir)
        print("    Tokenizer saved.")
    except Exception as exc:
        print(f"    Could not save tokenizer ({exc}); save it manually if needed.")

    print("\nDone.  To load the converted model:")
    print("    from routing_free.mixtral_rf import RoutingFreeMixtralConfig, RoutingFreeMixtralForCausalLM")
    print("    from transformers import AutoConfig, AutoModelForCausalLM")
    print(f'    AutoConfig.register("routing_free_mixtral", RoutingFreeMixtralConfig)')
    print(f'    AutoModelForCausalLM.register(RoutingFreeMixtralConfig, RoutingFreeMixtralForCausalLM)')
    print(f'    model = AutoModelForCausalLM.from_pretrained("{output_dir}")')

    return rf_model, rf_config


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Convert a pretrained Mixtral MoE checkpoint to RFMoE format.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Paths
    parser.add_argument("--source-model", required=True,
                        help="Local path or HF hub name of the source Mixtral model "
                             "(e.g. mistralai/Mixtral-8x7B-v0.1)")
    parser.add_argument("--output-dir", required=True,
                        help="Directory to write the converted RFMoE checkpoint")

    # RF gating hyper-params
    parser.add_argument("--gate-proj-rank", type=int, default=None,
                        help="Low-rank gate projection rank.  "
                             "Defaults to hidden_size // 4 (= 1024 for Mixtral 8x7B).")
    parser.add_argument("--gate-norm", default="l2", choices=["l1", "l2", "linf"],
                        help="Norm used to compute the gate score from the low-rank projection.")
    parser.add_argument("--gate-threshold", type=float, default=0.05,
                        help="Activation threshold: tokens with gate_score < threshold are masked.")
    parser.add_argument("--gate-temperature", type=float, default=1.0,
                        help="Temperature applied to the gate threshold.")
    parser.add_argument("--gate-act-fn", default="linear",
                        help="Activation for aux-loss gate proxy (e.g. linear, relu, sigmoid).")
    parser.add_argument("--density-target", type=float, default=0.25,
                        help="Target expert activation density for the aux loss.")
    parser.add_argument("--lambda-coef", type=float, default=1e-5,
                        help="Initial coefficient for the routing-free aux loss.")
    parser.add_argument("--eta-coef", type=float, default=0.2,
                        help="Step size for the adaptive lambda update.")
    parser.add_argument("--per-expert-aux-loss-coef", type=float, default=0.5)
    parser.add_argument("--per-token-aux-loss-coef", type=float, default=0.5)

    # Loading / precision
    parser.add_argument("--dtype", default="bfloat16",
                        choices=["float32", "float16", "bfloat16"],
                        help="Dtype for the saved RFMoE checkpoint. "
                             "Source weights are always dequantised to float32 "
                             "internally before being cast to this dtype.")
    parser.add_argument("--quantization", default="none",
                        choices=["none", "int8", "int4"],
                        help="Quantization to use when loading the *source* Mixtral model. "
                             "Reduces peak host/GPU memory during conversion. "
                             "Weights are dequantised before SVD and saved in --dtype. "
                             "'int8' ≈ half bf16 memory; 'int4' ≈ quarter. "
                             "Requires bitsandbytes and a CUDA GPU.")
    parser.add_argument("--hf-cache-dir", default=None,
                        help="HuggingFace model cache directory. "
                             "Defaults to HuggingFace default cache (~/.cache/huggingface).")

    args = parser.parse_args()

    convert_mixtral_to_rfmoe(
        source_model_path=args.source_model,
        output_dir=args.output_dir,
        gate_proj_rank=args.gate_proj_rank,
        gate_norm=args.gate_norm,
        gate_bias=True,
        gate_scale=True,
        gate_act_fn=args.gate_act_fn,
        gate_threshold=args.gate_threshold,
        gate_temperature=args.gate_temperature,
        output_gate_scores=False,
        density_target=args.density_target,
        lambda_coef=args.lambda_coef,
        eta_coef=args.eta_coef,
        per_expert_aux_loss_coef=args.per_expert_aux_loss_coef,
        per_token_aux_loss_coef=args.per_token_aux_loss_coef,
        dtype=args.dtype,
        quantization=args.quantization,
        hf_cache_dir=args.hf_cache_dir,
    )


if __name__ == "__main__":
    main()

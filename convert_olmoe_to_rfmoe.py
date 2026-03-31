"""
Convert a pretrained OLMoE model (allenai/OLMoE-1B-7B-*) into a
RoutingFreeOlmoe (RFMoE) checkpoint that can be loaded and run via
RoutingFreeOlmoeForCausalLM.

Structural changes vs the original OLMoE checkpoint
-----------------------------------------------------
  COPIED (key-identical):
    model.embed_tokens, model.norm, lm_head
    model.layers.*.self_attn.*
    model.layers.*.input_layernorm, post_attention_layernorm
    model.layers.*.mlp.experts.{i}.gate_proj.weight   (w_gate, already named)
    model.layers.*.mlp.experts.{i}.up_proj.weight
    model.layers.*.mlp.experts.{i}.down_proj.weight

  DISCARDED:
    model.layers.*.mlp.gate.weight   (original top-k router)

  DERIVED via SVD of gate_proj.weight:
    mlp.experts.{i}.gate_proj_A.weight  [rank, hidden]
    mlp.experts.{i}.gate_proj_B.weight  [intermediate, rank]
    gate_proj_B @ gate_proj_A  ≈  gate_proj.weight  (best rank-r approx)

  INITIALISED explicitly:
    mlp.experts.{i}.gate.gate_scale = 1.0
    mlp.experts.{i}.gate.gate_bias  = -1e-6

  Note: Unlike Mixtral, OLMoE experts are always a plain ModuleList and
  already use gate/up/down_proj naming — no key renaming or fused-tensor
  handling is required.

Usage
-----
  python convert_olmoe_to_rfmoe.py \\
      --source-model  allenai/OLMoE-1B-7B-0125-Instruct \\
      --output-dir    ./OLMoE-1B-7B-RFMoE-converted      \\
      --gate-proj-rank 128                                \\
      --gate-threshold 1.0                                \\
      --quantization   int4

  # Load the result:
  #   from routing_free.olmoe_rf import (
  #       RoutingFreeOlmoeConfig, RoutingFreeOlmoeModel,
  #       RoutingFreeOlmoeForCausalLM)
  #   AutoConfig.register("routing_free_olmoe", RoutingFreeOlmoeConfig)
  #   AutoModel.register(RoutingFreeOlmoeConfig, RoutingFreeOlmoeModel)
  #   AutoModelForCausalLM.register(RoutingFreeOlmoeConfig, RoutingFreeOlmoeForCausalLM)
  #   model = AutoModelForCausalLM.from_pretrained("./OLMoE-1B-7B-RFMoE-converted")
"""

import argparse
import os
import sys
from pathlib import Path

import torch

_project_root = os.path.dirname(os.path.abspath(__file__))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)


# ---------------------------------------------------------------------------
# Key-mapping helpers
# ---------------------------------------------------------------------------

def _is_direct_copy_key(key: str) -> bool:
    """
    True for keys that exist with the same name in both source and RF model.

    Covers:
      - embed_tokens, norm, lm_head
      - all self_attn.* and layernorm weights (anything in layers NOT under mlp)
      - mlp.experts.*.{gate,up,down}_proj.weight  (already named correctly)
    """
    if key.startswith(("model.embed_tokens.", "model.norm.", "lm_head.")):
        return True
    if "model.layers." in key and ".mlp." not in key:
        return True
    if ".mlp.experts." in key:
        return True
    return False


def _is_router_key(key: str) -> bool:
    """True for the original top-k router weight — discarded in RFMoE."""
    # e.g. model.layers.0.mlp.gate.weight
    return ".mlp.gate." in key


# ---------------------------------------------------------------------------
# SVD-init and reconstruction check
# ---------------------------------------------------------------------------

def _report_svd_truncation_precision(
    rel_errors: list,
    variance_explained: list,
    rank: int,
    num_experts: int,
    sample_layer: int = 0,
    sample_expert: int = 0,
):
    """
    Print summary of SVD truncation precision: distance between original W
    and reconstructed W_recon = B @ A (i.e. USV with top-rank only).
    """
    if not rel_errors:
        return
    rel = torch.tensor(rel_errors, dtype=torch.float32)
    var = torch.tensor(variance_explained, dtype=torch.float32)
    print(f"    SVD truncation precision (rank={rank}, {len(rel_errors)} experts):")
    print(f"      Relative error  ||W - B@A|| / ||W||   min={rel.min():.6f}  mean={rel.mean():.6f}  max={rel.max():.6f}")
    print(f"      Variance explained (1 - rel_err²)     min={var.min():.2f}%  mean={var.mean():.2f}%  max={var.max():.2f}%")
    idx = min(sample_layer * num_experts + sample_expert, len(rel_errors) - 1)
    print(f"      Example (layer {sample_layer} expert {sample_expert}): rel_err={rel[idx]:.6f}, var_explained={var[idx]:.2f}%")


def _svd_init_gate_proj(
    new_sd: dict,
    num_layers: int,
    num_experts: int,
    rank: int,
    dtype: torch.dtype,
):
    """
    For every expert in every layer, decompose gate_proj.weight via truncated
    SVD and store factors into gate_proj_A.weight / gate_proj_B.weight.

    W = U S Vᵀ   (W shape: [intermediate, hidden])
      gate_proj_A = diag(S[:r]^0.5) @ Vh[:r, :]   [rank, hidden]
      gate_proj_B = U[:, :r] @ diag(S[:r]^0.5)    [intermediate, rank]

    Returns (rel_errors, variance_explained) for reporting truncation precision.
    """
    total = num_layers * num_experts
    done  = 0
    rel_errors = []
    variance_explained = []
    for li in range(num_layers):
        for ei in range(num_experts):
            pfx   = f"model.layers.{li}.mlp.experts.{ei}"
            w_key = f"{pfx}.gate_proj.weight"
            a_key = f"{pfx}.gate_proj_A.weight"
            b_key = f"{pfx}.gate_proj_B.weight"

            if w_key not in new_sd:
                continue

            W = new_sd[w_key].float()          # [intermediate, hidden]
            U, S, Vh = torch.linalg.svd(W, full_matrices=False)
            r     = min(rank, S.shape[0])
            S_sqrt = S[:r].sqrt()

            A = (S_sqrt.unsqueeze(1) * Vh[:r, :]).to(dtype)   # [r, hidden]
            B = (U[:, :r] * S_sqrt.unsqueeze(0)).to(dtype)    # [intermediate, r]
            new_sd[a_key] = A
            new_sd[b_key] = B

            # Truncation precision: distance between original W and reconstructed B@A
            W_recon = B.float() @ A.float()
            err_fro = (W - W_recon).norm(p="fro")
            w_fro = W.norm(p="fro")
            rel_err = (err_fro / w_fro).item() if w_fro > 0 else 0.0
            var_exp = (1.0 - rel_err * rel_err) * 100.0  # variance explained %
            rel_errors.append(rel_err)
            variance_explained.append(var_exp)

            done += 1
            print(f"\r    SVD progress: {done}/{total} experts", end="", flush=True)
    print()
    return rel_errors, variance_explained


def _init_gate_scale_bias(
    new_sd: dict,
    num_layers: int,
    num_experts: int,
    dtype: torch.dtype,
    gate_bias_init: float = -1e-6,
):
    """
    Set gate_scale = 1.0 and gate_bias = gate_bias_init for every expert.
    gate_scale = 1  → no rescaling at init
    gate_bias  = -ε → bias is effectively zero; threshold controls activation
    """
    for li in range(num_layers):
        for ei in range(num_experts):
            pfx       = f"model.layers.{li}.mlp.experts.{ei}.gate"
            scale_key = f"{pfx}.gate_scale"
            bias_key  = f"{pfx}.gate_bias"
            if scale_key in new_sd:
                new_sd[scale_key] = torch.ones(1, dtype=dtype)
            if bias_key in new_sd:
                new_sd[bias_key]  = torch.full((1,), gate_bias_init, dtype=dtype)


# ---------------------------------------------------------------------------
# Quantization helper
# ---------------------------------------------------------------------------

def _make_bnb_config(quantization: str, compute_dtype: torch.dtype):
    if quantization == "none":
        return None
    try:
        from transformers import BitsAndBytesConfig
    except ImportError:
        raise ImportError("Install bitsandbytes:  pip install bitsandbytes")
    if quantization == "int8":
        return BitsAndBytesConfig(load_in_8bit=True)
    if quantization == "int4":
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    raise ValueError(f"Unknown quantization {quantization!r}. Use 'none', 'int8', or 'int4'.")


def _extract_src_sd_dequant(src_model) -> dict:
    """
    Extract a plain float32 state dict.  For bitsandbytes quantized models,
    calling .to(float32) on quantized parameter tensors triggers automatic
    dequantisation via the bitsandbytes tensor protocol.
    """
    return {k: v.to(torch.float32) for k, v in src_model.state_dict().items()}


# ---------------------------------------------------------------------------
# Main conversion routine
# ---------------------------------------------------------------------------

def convert_olmoe_to_rfmoe(
    source_model_path: str,
    output_dir: str,
    # RF gating hyper-params
    gate_proj_rank: int | None = None,
    gate_norm: str = "l2",
    gate_bias: bool = True,
    gate_scale: bool = True,
    gate_act_fn: str = "linear",
    gate_threshold: float = 1.0,
    gate_temperature: float = 1.0,
    output_gate_scores: bool = False,
    density_target: float = 0.25,
    lambda_coef: float = 1e-10,
    eta_coef: float = 0.02,
    per_expert_aux_loss_coef: float = 0.5,
    per_token_aux_loss_coef: float = 0.5,
    # Loading options
    dtype: str = "bfloat16",
    quantization: str = "none",   # "none" | "int8" | "int4"
    hf_cache_dir: str = None,
):
    """
    Load a pretrained OLMoE model and produce an RFMoE checkpoint.

    Expert FFN weights are copied exactly; the top-k router is discarded;
    gate_proj_A / gate_proj_B are derived from a truncated SVD of each
    expert's gate_proj (w_gate) weight matrix.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from routing_free.olmoe_rf import (
        RoutingFreeOlmoeConfig,
        RoutingFreeOlmoeForCausalLM,
    )

    torch_dtype = getattr(torch, dtype)
    bnb_config  = _make_bnb_config(quantization, torch_dtype)
    n_gpus      = torch.cuda.device_count()

    # ------------------------------------------------------------------
    # 1. Load source model
    # ------------------------------------------------------------------
    load_kwargs = dict(trust_remote_code=True, cache_dir=hf_cache_dir)
    if bnb_config is not None:
        load_kwargs["quantization_config"] = bnb_config
        load_kwargs["device_map"] = "auto"
        print(f"[1/6] Loading source model ({quantization}) from: {source_model_path}")
        print(f"      cache_dir={hf_cache_dir or 'HF default'}, GPUs={n_gpus}")
    else:
        load_kwargs["torch_dtype"] = torch_dtype
        load_kwargs["device_map"] = "auto" if n_gpus > 0 else "cpu"
        print(f"[1/6] Loading source model ({dtype}) from: {source_model_path}")
        print(f"      cache_dir={hf_cache_dir or 'HF default'}, device_map={load_kwargs['device_map']}")

    src_model = AutoModelForCausalLM.from_pretrained(source_model_path, **load_kwargs)
    src_cfg   = src_model.config

    num_layers  = src_cfg.num_hidden_layers
    num_experts = src_cfg.num_experts           # OLMoE uses num_experts (not num_local_experts)
    hidden_size = src_cfg.hidden_size
    resolved_rank = gate_proj_rank or max(64, hidden_size // 16)

    print(
        f"    model_type={src_cfg.model_type}, layers={num_layers}, "
        f"experts={num_experts}, hidden={hidden_size}, "
        f"intermediate={src_cfg.intermediate_size}"
    )

    # ------------------------------------------------------------------
    # 2. Build RF config
    # ------------------------------------------------------------------
    print("[2/6] Building RoutingFreeOlmoeConfig …")

    _rf_fields = {
        "gate_proj_rank", "gate_norm", "gate_bias", "gate_scale",
        "gate_act_fn", "gate_threshold", "gate_temperature",
        "output_gate_scores", "n_experts", "density_target",
        "lambda_coef", "eta_coef",
        "per_expert_aux_loss_coef", "per_token_aux_loss_coef",
    }
    # Drop fields that conflict with RF naming or come from the original MoE router
    _skip = {"model_type", "_name_or_path", "architectures", "auto_map",
             "router_aux_loss_coef"} | _rf_fields

    src_dict = {k: v for k, v in src_cfg.to_dict().items() if k not in _skip}

    rf_config = RoutingFreeOlmoeConfig(
        **src_dict,
        gate_proj_rank=resolved_rank,
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
    print(f"    gate_proj_rank={resolved_rank}, gate_threshold={gate_threshold}, "
          f"density_target={density_target}")

    # ------------------------------------------------------------------
    # 3. Instantiate RF model (random weights — overwritten below)
    # ------------------------------------------------------------------
    print("[3/6] Instantiating RoutingFreeOlmoeForCausalLM …")
    rf_model = RoutingFreeOlmoeForCausalLM(rf_config).to(torch_dtype)

    # ------------------------------------------------------------------
    # 4. Copy weights from source into a new state dict
    # ------------------------------------------------------------------
    print("[4/6] Copying weights …")
    src_sd = _extract_src_sd_dequant(src_model)   # float32, dequantised
    dst_sd = rf_model.state_dict()

    new_sd       = dict(dst_sd)   # start from RF defaults (gate_scale/bias etc.)
    n_copied     = 0
    n_discarded  = 0
    shape_errors = []
    unknown_keys = []

    for src_key, src_val in src_sd.items():
        if _is_router_key(src_key):
            n_discarded += 1
            continue

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

        unknown_keys.append(src_key)

    print(f"\n    Weight copy summary:")
    print(f"      Copied   (exact)  : {n_copied}")
    print(f"      Discarded (router): {n_discarded}")
    if shape_errors:
        print(f"      Shape mismatches:")
        for key, ss, ds in shape_errors:
            print(f"        {key}: src={ss} dst={ds}")
    if unknown_keys:
        print(f"      Unmapped src keys:")
        for k in unknown_keys:
            print(f"        {k}")

    # ------------------------------------------------------------------
    # 5. SVD-initialise gate_proj_A / gate_proj_B
    # ------------------------------------------------------------------
    print(f"[5/6] SVD-initialising gate_proj_A / gate_proj_B (rank={resolved_rank}) …")
    rel_errors, variance_explained = _svd_init_gate_proj(
        new_sd, num_layers, num_experts, resolved_rank, torch_dtype
    )
    _report_svd_truncation_precision(
        rel_errors, variance_explained, rank=resolved_rank,
        num_experts=num_experts, sample_layer=0, sample_expert=0,
    )

    _init_gate_scale_bias(new_sd, num_layers, num_experts, torch_dtype, gate_bias_init=-1e-6)
    print("    gate_scale=1.0 and gate_bias=-1e-6 set for all experts.")

    rf_model.load_state_dict(new_sd, strict=True)

    # ------------------------------------------------------------------
    # 6. Save
    # ------------------------------------------------------------------
    print(f"\n[6/6] Saving to {output_dir} …")
    os.makedirs(output_dir, exist_ok=True)
    rf_model.save_pretrained(output_dir)
    rf_config.save_pretrained(output_dir)

    try:
        tok = AutoTokenizer.from_pretrained(source_model_path, cache_dir=hf_cache_dir)
        tok.save_pretrained(output_dir)
        print("    Tokenizer saved.")
    except Exception as exc:
        print(f"    Could not save tokenizer ({exc}); save manually if needed.")

    print("\nDone.  To load:")
    print("    from routing_free.olmoe_rf import (")
    print("        RoutingFreeOlmoeConfig, RoutingFreeOlmoeModel, RoutingFreeOlmoeForCausalLM)")
    print("    from transformers import AutoConfig, AutoModel, AutoModelForCausalLM")
    print('    AutoConfig.register("routing_free_olmoe", RoutingFreeOlmoeConfig)')
    print("    AutoModel.register(RoutingFreeOlmoeConfig, RoutingFreeOlmoeModel)")
    print("    AutoModelForCausalLM.register(RoutingFreeOlmoeConfig, RoutingFreeOlmoeForCausalLM)")
    print(f'    model = AutoModelForCausalLM.from_pretrained("{output_dir}")')

    return rf_model, rf_config


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Convert a pretrained OLMoE checkpoint to RFMoE format.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--source-model", required=True,
                        help="HF hub name or local path of the OLMoE model "
                             "(e.g. allenai/OLMoE-1B-7B-0125-Instruct)")
    parser.add_argument("--output-dir", required=True,
                        help="Directory to write the converted RFMoE checkpoint")

    parser.add_argument("--gate-proj-rank", type=int, default=None,
                        help="Low-rank gate projection rank. "
                             "Defaults to max(64, hidden_size//16).")
    parser.add_argument("--gate-norm", default="l2", choices=["l1", "l2", "linf"])
    parser.add_argument("--gate-threshold", type=float, default=1.0,
                        help="Initial gate threshold (calibrated by adapt script).")
    parser.add_argument("--gate-temperature", type=float, default=1.0)
    parser.add_argument("--gate-act-fn", default="linear")
    parser.add_argument("--density-target", type=float, default=0.25)
    parser.add_argument("--lambda-coef", type=float, default=1e-10)
    parser.add_argument("--eta-coef", type=float, default=0.02)
    parser.add_argument("--per-expert-aux-loss-coef", type=float, default=0.5)
    parser.add_argument("--per-token-aux-loss-coef",  type=float, default=0.5)

    parser.add_argument("--dtype", default="bfloat16",
                        choices=["float32", "float16", "bfloat16"],
                        help="Dtype for the saved RFMoE checkpoint.")
    parser.add_argument("--quantization", default="none",
                        choices=["none", "int8", "int4"],
                        help="Quantization for loading the *source* OLMoE model "
                             "to reduce peak memory. Weights are dequantised "
                             "before SVD and saved in --dtype. Requires bitsandbytes.")
    parser.add_argument("--hf-cache-dir", default=None,
                        help="HuggingFace model cache directory. "
                             "Defaults to the HuggingFace default (~/.cache/huggingface).")

    args = parser.parse_args()
    convert_olmoe_to_rfmoe(
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

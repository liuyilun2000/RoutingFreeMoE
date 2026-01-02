import torch
from torch import nn
from transformers.activations import ACT2FN


class RoutingFreeGate(nn.Module):
    """
    Basic gating module:
    Calculate gate_score and generate gate_mask.
    Only responsible for scoring and masking, not for specific MLP calculation, which is reusable for different MoE/MLP models.
    """

    def __init__(self, hidden_size: int, cfg):
        super().__init__()
        self.hidden_size = hidden_size
        self.gate_proj_rank = getattr(cfg, "gate_proj_rank", hidden_size // 4)

        # Low-rank projection A: hidden -> gate_proj_rank
        self.gate_proj_A = nn.Linear(self.hidden_size, self.gate_proj_rank, bias=False)

        # Normalization type
        self._gate_norm = self._build_gate_norm(getattr(cfg, "gate_norm", "l2"))

        # Optional scaling/bias
        self.gate_scale = nn.Parameter(torch.ones(1)) if getattr(cfg, "gate_scale", None) else None
        # Initialize gate_bias to the negative expected norm of the gate_hidden vector -E[||gate_hidden||],
        # so that the gating threshold is meaningful at initialization.
        self.gate_bias = nn.Parameter(torch.full((1,), -1e-6)) if getattr(cfg, "gate_bias", None) else None

        # Activation, threshold, temperature
        gate_act_name = getattr(cfg, "gate_act_fn", "linear")
        self.gate_act_fn = ACT2FN[gate_act_name]
        self.gate_threshold = getattr(cfg, "gate_threshold", 0.5)
        self.gate_temperature = getattr(cfg, "gate_temperature", 1.0)

        # Whether to output gate_scores (for collecting router_logits by upper layer)
        self.output_gate_scores = getattr(cfg, "output_gate_scores", True)

    def _build_gate_norm(self, norm_type: str):
        if norm_type == "l1":
            return lambda x: torch.norm(x, p=1, dim=-1)
        if norm_type == "l2":
            return lambda x: torch.norm(x, p=2, dim=-1)
        if norm_type == "linf":
            return lambda x: torch.norm(x, p=float("inf"), dim=-1)
        return lambda x: torch.norm(x, p=2, dim=-1)  # default to l2

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None):
        """
        Args:
            x: [B, T, H] as Batch_size, seq_len, Hidden_size, or [N, H] as Number of valid tokens, Hidden_size
            mask: [B, T] as Batch_size, seq_len, Boolean, 1 for included, 0 for excluded or [N] as Number of valid tokens, Boolean, 1 for included, 0 for excluded
        Returns:
            gate_mask_full: Boolean mask of the same shape as the input, indicating which positions pass the threshold
            gate_score_full: If output_gate_scores is True, return the scores of the same shape as the input, filled with -inf for tokens that do not pass the threshold; otherwise None
        """
        orig_shape = x.shape
        x_flat = x.view(-1, self.hidden_size)

        if mask is not None:
            mask_flat = mask.view(-1).bool()
            idx = torch.nonzero(mask_flat, as_tuple=False).squeeze(-1)
            x_valid = x_flat[idx]
        else:
            idx = torch.arange(x_flat.shape[0], device=x.device)
            x_valid = x_flat

        # Calculate gate_score for valid tokens
        gate_hidden = self.gate_proj_A(x_valid)
        gate_score = self._gate_norm(gate_hidden)
        if self.gate_scale is not None:
            gate_score = gate_score * self.gate_scale
        if self.gate_bias is not None:
            gate_score = gate_score + self.gate_bias
        gate_score = self.gate_act_fn(gate_score)

        # Threshold/temperature
        gate_mask = gate_score >= (self.gate_threshold / self.gate_temperature)

        # Fill back to full shape
        gate_mask_full = torch.zeros(x_flat.shape[0], device=x.device, dtype=torch.bool)
        gate_mask_full[idx[gate_mask]] = True
        gate_mask_full = gate_mask_full.view(orig_shape[:-1])

        gate_score_full = None
        if self.output_gate_scores:
            gate_score_full = x_flat.new_ones(x_flat.shape[0]) * -float("inf")
            if gate_mask.any():
                gate_score_full[idx[gate_mask]] = gate_score[gate_mask]
            gate_score_full = gate_score_full.view(orig_shape[:-1])

        return gate_mask_full, gate_score_full


class RoutingFreeFFNWrapper(nn.Module):
    """
    Routing-free FFN:
    FFN(x) = [σ(x A_gate B_gate) ⊙ (x W_up)] W_down
    """

    def __init__(self, base_mlp: nn.Module, cfg):
        """
        base_mlp: Must contain up_proj, down_proj, act_fn attributes.
        cfg: Provide gating-related parameters (gate_proj_rank, gate_threshold, gate_temperature, etc.).
        """
        super().__init__()
        self.base_mlp = base_mlp
        self.hidden_size = getattr(cfg, "hidden_size")
        self.intermediate_size = getattr(
            cfg, "moe_intermediate_size", getattr(cfg, "intermediate_size", None)
        )
        if self.hidden_size is None or self.intermediate_size is None:
            raise ValueError("hidden_size / intermediate_size must be set in cfg.")

        self.gate_proj_rank = getattr(cfg, "gate_proj_rank", self.hidden_size // 4)

        # A_gate, B_gate Low-rank gating
        self.gate_proj_A = nn.Linear(self.hidden_size, self.gate_proj_rank, bias=False)
        self.gate_proj_B = nn.Linear(self.gate_proj_rank, self.intermediate_size, bias=False)

        # Reuse gating module to calculate score/mask, ensuring shared A parameters
        self.gate = RoutingFreeGate(self.hidden_size, cfg)
        self.gate.gate_proj_A = self.gate_proj_A

        self.output_gate_scores = getattr(cfg, "output_gate_scores", True)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None):
        """
        Returns:
            out: Same shape as x
            gate_score_full (or None): Same shape as x[...,0]
        """
        gate_mask_full, gate_score_full = self.gate(x, mask)

        if not gate_mask_full.any():
            out = torch.zeros_like(x)
            return out, gate_score_full if self.output_gate_scores else None

        orig_shape = x.shape
        x_flat = x.view(-1, orig_shape[-1])
        gate_mask_flat = gate_mask_full.view(-1)
        idx = torch.nonzero(gate_mask_flat, as_tuple=False).squeeze(-1)
        x_valid = x_flat[idx]

        # σ(x A_gate B_gate)
        gate_hidden = self.gate_proj_A(x_valid)
        gate_hidden = self.gate_proj_B(gate_hidden)
        gate_act = self.base_mlp.act_fn(gate_hidden)

        # (x W_up)
        up_out = self.base_mlp.up_proj(x_valid)

        # Down projection
        ffn_valid = self.base_mlp.down_proj(gate_act * up_out).to(dtype=x.dtype)

        out_flat = torch.zeros_like(x_flat)
        out_flat[idx] = ffn_valid
        out = out_flat.view(orig_shape)

        return out, gate_score_full if self.output_gate_scores else None

def wrap_mlp_with_routing_free(mlp, cfg):
    """
    Helper: wrap an existing MLP (with up_proj/down_proj/act_fn) by routing-free gating FFN.
    """
    '''usage:
    layer.mlp = wrap_mlp_with_routing_free(layer.mlp, config)
    # or MoE:
    for i, e in enumerate(layer.mlp.experts):
        layer.mlp.experts[i] = wrap_mlp_with_routing_free(e, config)
    '''
    return RoutingFreeFFNWrapper(mlp, cfg)

def balancing_loss_func(
    probs: torch.Tensor,
    mask: torch.Tensor,
    dim: int | None = None,
) -> torch.Tensor | int:
    """
    Switch-Transformer-style balancing loss.
    dim: None -> over all dimensions; 0 -> over expert dimension; 1 -> over token dimension.
    probs/mask must have the same shape.
    """
    if probs is None or mask is None:
        return 0
    assert dim in [0, 1, None], "dim must be 0, 1 or None"
    assert probs.shape == mask.shape, "probs and mask must have the same shape"

    density = torch.mean(mask.float(), dim=dim)
    density_proxy = torch.mean(probs, dim=dim)
    assert len(density.shape) == 1 and len(density_proxy.shape) == 1
    assert density.shape[0] == density_proxy.shape[0]
    loss = torch.mean(density * density_proxy)
    return loss

class RoutingFreeAuxLossMixin:
    def compute_routing_free_aux(self, gate_scores, cfg, training):
        """
        Args:
            gate_scores: List/tuple of [layer][B, T, n_experts] or None
            cfg: Provide gating-related parameters (gate_proj_rank, gate_threshold, gate_temperature, etc.).
            training: bool, whether in training mode (determines whether to update lambda_coef)
        Returns:
            aux_loss: Tensor or None (can be added to main loss)
            aux_dict: Dictionary for monitoring or None
            new_lambda: Updated lambda_coef (can be written back to self.lambda_coef)
        """
        if gate_scores is None:
            return None, None, getattr(self, "lambda_coef", None)

        gate_act_fn = ACT2FN[getattr(cfg, "gate_act_fn", "linear")]
        gate_temperature = getattr(cfg, "gate_temperature", 1.0)
        gate_threshold = getattr(cfg, "gate_threshold", 0.5)
        density_target = getattr(cfg, "density_target", 0.1)
        per_expert_coef = getattr(cfg, "per_expert_aux_loss_coef", 0.5)
        per_token_coef = getattr(cfg, "per_token_aux_loss_coef", 0.5)
        eta_coef = getattr(cfg, "eta_coef", 1.2)
        n_experts = getattr(cfg, "n_experts", None)

        # get or initialize lambda_coef
        lambda_init = getattr(cfg, "lambda_coef", 1e-6)
        lambda_coef = getattr(self, "lambda_coef", lambda_init)
        if not isinstance(lambda_coef, torch.Tensor):
            lambda_coef = torch.tensor(lambda_coef)

        gate_probs_list = []
        total_tokens = 0

        for layer_gate_scores in gate_scores:
            if layer_gate_scores is None:
                continue
            B, T, n = layer_gate_scores.shape
            total_tokens = B * T
            if total_tokens > 0:
                gate_probs_list.append(layer_gate_scores.view(-1, n))

        if not gate_probs_list:
            return None, None, lambda_coef

        gate_probs = torch.cat(gate_probs_list, dim=-1)  # [B*T, L*n_experts]
        # Ensure lambda and gate_probs are on the same device/type
        lambda_coef = lambda_coef.to(device=gate_probs.device, dtype=gate_probs.dtype)
        gate_probs = gate_act_fn(gate_probs)
        gate_probs = gate_probs / gate_temperature
        gate_mask = (gate_probs > gate_threshold).float()

        density = torch.mean(gate_mask)
        density_ratio = density / density_target
        density_per_expert = torch.mean(gate_mask, dim=0)
        density_proxy = torch.mean(gate_probs)
        density_proxy_per_expert = torch.mean(gate_probs, dim=0)

        per_expert_aux = balancing_loss_func(gate_probs, gate_mask, dim=0) if per_expert_coef > 0 else 0
        per_token_aux = balancing_loss_func(gate_probs, gate_mask, dim=1) if per_token_coef > 0 else 0
        reg_loss = per_expert_aux * per_expert_coef + per_token_aux * per_token_coef
        aux_loss = lambda_coef * reg_loss

        new_lambda = lambda_coef
        if training and aux_loss is not None:
            new_lambda = lambda_coef * (1 + eta_coef * (density - density_target) ** 2) ** torch.sign(density - density_target)
            new_lambda = torch.clamp(new_lambda, max=1.0)
            if hasattr(self, "lambda_coef"):
                self.lambda_coef = new_lambda
            if hasattr(self, "sync_lambda_coef"):
                self.sync_lambda_coef()

        aux_dict = {
            "aux_loss": aux_loss.item() if aux_loss is not None else None,
            "reg_loss": reg_loss.item() if reg_loss is not None else None,
            "lambda_coef": float(new_lambda.detach().cpu()) if isinstance(new_lambda, torch.Tensor) else new_lambda,
            "density": density.item(),
            "density_ratio": density_ratio.item(),
            "density_proxy": density_proxy.item(),
            "per_expert_aux_loss": per_expert_aux.item() if isinstance(per_expert_aux, torch.Tensor) else per_expert_aux,
            "per_token_aux_loss": per_token_aux.item() if isinstance(per_token_aux, torch.Tensor) else per_token_aux,
        }

        # Layer-wise expert density metrics, skip if n_experts is unknown
        if n_experts and density_per_expert is not None:
            n_layers = density_per_expert.numel() // n_experts
            layer_digits = len(str(n_layers - 1)) if n_layers > 0 else 1
            expert_digits = len(str(n_experts - 1)) if n_experts > 0 else 1
            for l in range(n_layers):
                for e in range(n_experts):
                    idx = l * n_experts + e
                    l_str = str(l).zfill(layer_digits)
                    e_str = str(e).zfill(expert_digits)
                    aux_dict[f"expert_density_L{l_str}E{e_str}"] = float(density_per_expert[idx].detach().cpu())
                    # Optional: record proxy: aux_dict[f"expert_density_proxy_L{l_str}E{e_str}"] = float(density_proxy_per_expert[idx].detach().cpu())

        return aux_loss, aux_dict, new_lambda

'''
# routing_free/deepseek_v3_rf.py
class RoutingFreeDeepseekV3Model(DeepseekV3Model):
    def __init__(self, config):
        super().__init__(config)
        for layer in self.layers:
            if hasattr(layer, "mlp"):
                if isinstance(layer.mlp, DeepseekV3MoE):
                    for i, e in enumerate(layer.mlp.experts):
                        layer.mlp.experts[i] = wrap_mlp_with_routing_free(e, config)
                else:
                    layer.mlp = wrap_mlp_with_routing_free(layer.mlp, config)

class RoutingFreeDeepseekV3ForCausalLM(RoutingFreeAuxLossMixin, DeepseekV3ForCausalLM):
    def forward(...):
        outputs = super().forward(..., output_gate_scores=True, ...)
        aux_loss, aux_dict, new_lambda = self.compute_routing_free_aux(outputs.router_logits, self.config, self.training)
        loss = outputs.loss
        if self.training and aux_loss is not None:
            loss = loss + aux_loss
        outputs.aux_loss = aux_dict
        self.lambda_coef = new_lambda
        return outputs
'''
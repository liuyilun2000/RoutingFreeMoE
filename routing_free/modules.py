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

def wrap_mlp_with_routing_free(mlp, cfg):
    """
    Helper: wrap an existing MLP (with up_proj/down_proj/act_fn) by routing-free gating FFN.
    当前返回占位实现，请在添加 RoutingFreeFFNWrapper 后替换。
    """
    raise NotImplementedError("wrap_mlp_with_routing_free 尚未实现")

class RoutingFreeAuxLossMixin:
    def compute_routing_free_aux(self, gate_scores, cfg, training):
        """
        占位：基于 gate_scores 计算辅助损失与 lambda 更新。
        """
        raise NotImplementedError("compute_routing_free_aux 尚未实现")

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
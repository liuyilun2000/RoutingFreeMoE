# coding=utf-8
"""Routing-free DeepSeekV3 config: inherit transformers' DeepseekV3Config and add routing-free fields."""

from transformers.modeling_rope_utils import rope_config_validation
from transformers.models.deepseek_v3.configuration_deepseek_v3 import DeepseekV3Config
from routing_free.modules import RoutingFreeConfigMixin

class RoutingFreeDeepseekV3Config(DeepseekV3Config, RoutingFreeConfigMixin):
    """Add routing-free hyperparameters on top of DeepseekV3Config."""

    model_type = "routing_free_deepseek_v3"

    def __init__(
        self,
        gate_proj_rank=None,
        gate_norm="l2",
        gate_bias=True,
        gate_scale=True,
        gate_act_fn=None,
        gate_threshold=0.05,
        gate_temperature=1.0,
        output_gate_scores=False,
        n_experts=16,
        density_target=0.1,
        lambda_coef=1e-5,
        eta_coef=0.2,
        per_expert_aux_loss_coef=0.5,
        per_token_aux_loss_coef=0.5,
        **kwargs,
    ):
        # Let base config handle official fields (num_attention_heads, num_hidden_layers, ...)
        super().__init__(**kwargs)

        # Backward-compat: map legacy n_* to num_* if present
        #import pdb; pdb.set_trace()
        #self.num_attention_heads = self.num_attention_heads
        #self.num_key_value_heads = self.num_key_value_heads
        #self.num_hidden_layers = self.num_hidden_layers

        # Keep RoPE validation aligned with upstream
        if getattr(self, "rope_parameters", None) is not None:
            rope_config_validation(self)

        # Initialize routing-free fields
        self._init_routing_free(
            gate_proj_rank=gate_proj_rank,
            gate_norm=gate_norm,
            gate_bias=gate_bias,
            gate_scale=gate_scale,
            gate_act_fn=gate_act_fn,
            gate_threshold=gate_threshold,
            gate_temperature=gate_temperature,
            output_gate_scores=output_gate_scores,
            n_experts=n_experts,
            density_target=density_target,
            lambda_coef=lambda_coef,
            eta_coef=eta_coef,
            per_expert_aux_loss_coef=per_expert_aux_loss_coef,
            per_token_aux_loss_coef=per_token_aux_loss_coef,
        )
        self.n_hidden_layers = self.num_hidden_layers
        self.n_attention_heads = self.num_attention_heads
        self.n_key_value_heads = self.num_key_value_heads


__all__ = ["RoutingFreeDeepseekV3Config", "RoutingFreeConfigMixin"]


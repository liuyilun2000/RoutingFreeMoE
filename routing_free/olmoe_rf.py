
import torch
from torch import nn
from transformers.modeling_outputs import MoeModelOutputWithPast, CausalLMOutputWithPast
from transformers.cache_utils import Cache, DynamicCache
from transformers.masking_utils import create_causal_mask

from transformers.models.olmoe.modeling_olmoe import (
    OlmoeModel,
    OlmoeForCausalLM,
    OlmoeDecoderLayer,
)
from transformers import OlmoeConfig

from typing import Optional
from dataclasses import dataclass

from .modules import RoutingFreeConfigMixin, wrap_mlp_with_routing_free, RoutingFreeAuxLossMixin, RoutingFreeMaskedMoE


class RoutingFreeOlmoeConfig(OlmoeConfig, RoutingFreeConfigMixin):
    """
    OlmoeConfig extended with routing-free gating hyperparameters.

    OLMoE-1B-7B defaults:
        hidden_size=2048, intermediate_size=1024, num_experts=64,
        num_hidden_layers=16, num_attention_heads=16, num_key_value_heads=8
    """

    model_type = "routing_free_olmoe"

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
        n_experts=64,           # OLMoE-1B-7B has 64 experts per layer
        density_target=0.25,
        lambda_coef=1e-5,
        eta_coef=0.2,
        per_expert_aux_loss_coef=0.5,
        per_token_aux_loss_coef=0.5,
        **kwargs,
    ):
        super().__init__(**kwargs)
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


def build_masked_moe_from_olmoe(config, olmoe_moe_block: nn.Module, wrap_fn):
    """
    Wrap every expert in an OlmoeMoE block with RoutingFreeFFNWrapper.

    OLMoE experts are always a plain ModuleList and already use
    gate_proj / up_proj / down_proj naming — no key renaming or fused-tensor
    handling is needed (unlike Mixtral).
    """
    rf_experts = [wrap_fn(expert, config) for expert in olmoe_moe_block.experts]
    return RoutingFreeMaskedMoE(config, rf_experts, shared_expert=None)


class RoutingFreeOlmoeDecoderLayer(OlmoeDecoderLayer):
    """
    OlmoeDecoderLayer with the MoE block replaced by RoutingFreeMaskedMoE.
    Returns (hidden_states, gate_score) so the model can collect gate scores
    for the auxiliary loss.
    """

    config_class = RoutingFreeOlmoeConfig

    def __init__(self, config: OlmoeConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        # Replace OlmoeMoE with the routing-free variant
        self.mlp = build_masked_moe_from_olmoe(config, self.mlp, wrap_mlp_with_routing_free)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        output_gate_scores: bool | None = None,
        use_cache: bool | None = False,
        cache_position: torch.LongTensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs,
    ):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # OlmoeSdpaAttention.forward() does not accept return_dict; strip it so
        # generation/caller can pass return_dict=True without breaking the layer.
        attn_kwargs = {k: v for k, v in kwargs.items() if k != "return_dict"}

        # Use [0] to safely extract hidden_states regardless of how many values
        # OlmoeAttention returns (varies by transformers version / Flash-Attn).
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **attn_kwargs,
        )[0]
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)

        mlp_out = self.mlp(hidden_states, output_gate_scores=output_gate_scores)
        gate_score = None
        if isinstance(mlp_out, tuple) and len(mlp_out) == 2:
            hidden_states, gate_score = mlp_out
        else:
            hidden_states = mlp_out

        hidden_states = residual + hidden_states
        return hidden_states, gate_score


class RoutingFreeOlmoeModel(OlmoeModel):
    """
    OlmoeModel with all decoder layers replaced by RoutingFreeOlmoeDecoderLayer.
    Collects per-layer gate_scores and attaches them to the output so the
    CausalLM head can compute the routing-free auxiliary loss.
    """

    config_class = RoutingFreeOlmoeConfig

    def __init__(self, config: OlmoeConfig):
        super().__init__(config)
        self.layers = nn.ModuleList(
            [RoutingFreeOlmoeDecoderLayer(config, i) for i in range(config.num_hidden_layers)]
        )

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        output_router_logits: bool | None = None,   # kept for API compat
        cache_position: torch.LongTensor | None = None,
        output_gate_scores: bool = False,             # RF-specific flag
        **kwargs,
    ) -> MoeModelOutputWithPast:

        if output_router_logits is not None:
            output_gate_scores = output_router_logits

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("Specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if cache_position is None:
            past_seen = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen, past_seen + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        # OLMoE uses plain causal masking (no sliding window)
        causal_mask = create_causal_mask(
            input_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
            position_ids=position_ids,
            config=self.config,
        )

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids=position_ids)

        gate_scores_all = [] if output_gate_scores else None
        for layer in self.layers:
            hidden_states, gate_score = layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                output_gate_scores=output_gate_scores,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )
            if gate_scores_all is not None:
                gate_scores_all.append(gate_score)

        hidden_states = self.norm(hidden_states)

        outputs = MoeModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )
        if gate_scores_all is not None:
            outputs.router_logits = gate_scores_all

        return outputs


@dataclass
class RoutingFreeOlmoeMoEOutput(CausalLMOutputWithPast):
    aux_dict: dict = None
    router_logits: list[torch.FloatTensor] = None
    lm_loss: float = None


class RoutingFreeOlmoeForCausalLM(RoutingFreeAuxLossMixin, OlmoeForCausalLM):
    """
    OlmoeForCausalLM with routing-free MoE and density-based auxiliary loss.
    Mirrors RoutingFreeMixtralForCausalLM but built on OLMoE base classes.
    """

    config_class = RoutingFreeOlmoeConfig

    def __init__(self, config):
        super().__init__(config)
        self.model = RoutingFreeOlmoeModel(config)
        lambda_init = getattr(config, "lambda_coef", 1e-6)
        self.lambda_coef = torch.tensor(lambda_init)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_router_logits=None,
        cache_position=None,
        logits_to_keep: int | torch.Tensor = 0,
        output_gate_scores: bool | None = None,
        **kwargs,
    ) -> RoutingFreeOlmoeMoEOutput:

        if output_router_logits is None:
            output_router_logits = self.config.output_router_logits

        if output_gate_scores is None:
            output_gate_scores = output_router_logits or self.training

        outputs: MoeModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_router_logits=output_gate_scores,
            cache_position=cache_position,
            output_gate_scores=output_gate_scores,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        lm_loss = None
        if labels is not None:
            loss = self.loss_function(logits, labels, self.vocab_size, **kwargs)
            lm_loss = loss.item()

        aux_loss = None
        aux_dict = None
        gate_scores = getattr(outputs, "router_logits", None) if output_gate_scores else None

        if self.training and gate_scores is not None:
            aux_loss, aux_dict, new_lambda = self.compute_routing_free_aux(
                gate_scores, self.config, self.training
            )
            if aux_loss is not None and loss is not None:
                loss = loss + aux_loss
            self.lambda_coef = new_lambda

        return RoutingFreeOlmoeMoEOutput(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=getattr(outputs, "hidden_states", None),
            attentions=getattr(outputs, "attentions", None),
            aux_dict=aux_dict,
            router_logits=gate_scores,
            lm_loss=lm_loss,
        )


__all__ = [
    "RoutingFreeOlmoeConfig",
    "RoutingFreeOlmoeModel",
    "RoutingFreeOlmoeForCausalLM",
]

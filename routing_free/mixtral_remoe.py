
import torch
from torch import nn
from transformers.modeling_outputs import MoeModelOutputWithPast, CausalLMOutputWithPast
from transformers.cache_utils import Cache, DynamicCache
from transformers.masking_utils import create_causal_mask

from transformers.models.mixtral.modeling_mixtral import (
    MixtralModel,
    MixtralForCausalLM,
    MixtralDecoderLayer,
    MixtralConfig,
)

from dataclasses import dataclass

from .mixtral_rf import MixtralRFExpert, RoutingFreeMoEOutput
from .modules import ReMoEMoE, ReMoEAuxLossMixin, ReMoEConfigMixin


class ReMoEMixtralConfig(MixtralConfig, ReMoEConfigMixin):
    """ReMoE config — standard Mixtral + l1_coef for ReLU router regularization."""

    model_type = "remoe_mixtral"

    def __init__(self, l1_coef=0.01, density_target=0.25, eta_coef=0.02, **kwargs):
        super().__init__(**kwargs)
        self._init_remoe(l1_coef=l1_coef, density_target=density_target, eta_coef=eta_coef)


def build_remoe_from_mixtral(config, mixtral_moe_block: nn.Module):
    """
    Decompose MixtralSparseMoeBlock into individual MixtralRFExpert instances,
    then wrap with ReMoEMoE (ReLU-weighted aggregation).
    """
    experts = []
    mixtral_experts = mixtral_moe_block.experts

    if isinstance(mixtral_experts, nn.ModuleList):
        num_experts = len(mixtral_experts)
        first_expert = mixtral_experts[0]
        hidden_dim = first_expert.w1.in_features
        intermediate_dim = first_expert.w1.out_features
        act_fn = first_expert.act_fn

        for i in range(num_experts):
            expert = MixtralRFExpert(hidden_dim, intermediate_dim, act_fn)
            source_expert = mixtral_experts[i]
            expert.gate_proj.weight.data.copy_(source_expert.w1.weight.data)
            expert.up_proj.weight.data.copy_(source_expert.w3.weight.data)
            expert.down_proj.weight.data.copy_(source_expert.w2.weight.data)
            experts.append(expert)
    else:
        num_experts = mixtral_experts.num_experts
        hidden_dim = mixtral_experts.hidden_dim
        intermediate_dim = mixtral_experts.intermediate_dim
        act_fn = mixtral_experts.act_fn

        for i in range(num_experts):
            expert = MixtralRFExpert(hidden_dim, intermediate_dim, act_fn)
            w_gate_up = mixtral_experts.gate_up_proj[i]
            w_gate = w_gate_up[:intermediate_dim, :]
            w_up = w_gate_up[intermediate_dim:, :]
            expert.gate_proj.weight.data.copy_(w_gate)
            expert.up_proj.weight.data.copy_(w_up)
            expert.down_proj.weight.data.copy_(mixtral_experts.down_proj[i])
            experts.append(expert)

    return ReMoEMoE(config.hidden_size, experts)


class ReMoEMixtralDecoderLayer(MixtralDecoderLayer):
    """Decoder layer with ReMoE (ReLU router, no softmax/top-k)."""

    config_class = ReMoEMixtralConfig

    def __init__(self, config: MixtralConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.block_sparse_moe = build_remoe_from_mixtral(config, self.block_sparse_moe)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        output_router_logits: bool | None = False,
        use_cache: bool | None = False,
        cache_position: torch.LongTensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs,
    ):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            position_embeddings=position_embeddings,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)

        mlp_out = self.block_sparse_moe(hidden_states)
        router_weights = None
        if isinstance(mlp_out, tuple) and len(mlp_out) == 2:
            hidden_states = mlp_out[0]
            router_weights = mlp_out[1]
        else:
            hidden_states = mlp_out

        hidden_states = residual + hidden_states
        return hidden_states, router_weights


class ReMoEMixtralModel(MixtralModel):
    """MixtralModel with ReMoE layers."""

    config_class = ReMoEMixtralConfig

    def __init__(self, config: MixtralConfig):
        super().__init__(config)
        self.layers = nn.ModuleList(
            [ReMoEMixtralDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        output_router_logits: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        **kwargs,
    ) -> MoeModelOutputWithPast:

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        if self.config.sliding_window is None:
            mask_function = create_causal_mask
        else:
            try:
                from transformers.masking_utils import create_sliding_window_causal_mask
                mask_function = create_sliding_window_causal_mask
            except ImportError:
                mask_function = create_causal_mask

        causal_mask = mask_function(
            input_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
            position_ids=position_ids,
            config=self.config,
        )

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids=position_ids)

        router_weights_all = []
        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            hidden_states, router_weights = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )
            if self.training:
                router_weights_all.append(router_weights)

        hidden_states = self.norm(hidden_states)

        outputs = MoeModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )
        if self.training:
            outputs.router_logits = router_weights_all
        return outputs


class ReMoEMixtralForCausalLM(ReMoEAuxLossMixin, MixtralForCausalLM):
    """CausalLM with ReMoE — L1 auxiliary loss on router weights."""

    config_class = ReMoEMixtralConfig

    def __init__(self, config):
        super().__init__(config)
        self.model = ReMoEMixtralModel(config)

        # Adaptive L1 coefficient (updated during training)
        l1_init = getattr(config, "l1_coef", 0.01)
        self.l1_coef_adaptive = torch.tensor(l1_init)

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
        **kwargs,
    ) -> RoutingFreeMoEOutput:

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
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
        router_weights_list = getattr(outputs, "router_logits", None)

        if self.training and router_weights_list:
            aux_loss, aux_dict, new_l1_coef = self.compute_remoe_aux(
                router_weights_list, self.config, self.training
            )
            if aux_loss is not None and loss is not None:
                loss = loss + aux_loss
            self.l1_coef_adaptive = new_l1_coef

        return RoutingFreeMoEOutput(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=getattr(outputs, "hidden_states", None),
            attentions=getattr(outputs, "attentions", None),
            aux_dict=aux_dict,
            router_logits=router_weights_list,
            lm_loss=lm_loss,
        )


__all__ = [
    "ReMoEMixtralConfig",
    "ReMoEMixtralModel",
    "ReMoEMixtralForCausalLM",
]

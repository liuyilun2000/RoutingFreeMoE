import torch
from torch import nn
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.utils.generic import check_model_inputs

from transformers.models.deepseek_v3.modeling_deepseek_v3 import (
    DeepseekV3Model,
    DeepseekV3ForCausalLM,
    DeepseekV3DecoderLayer,
    DeepseekV3RMSNorm,
    DeepseekV3Attention,
    create_causal_mask,
    DynamicCache,
    GenerationMixin,
)

from .modules import wrap_mlp_with_routing_free, RoutingFreeAuxLossMixin


class RoutingFreeDeepseekV3DecoderLayer(DeepseekV3DecoderLayer):
    """
    Decoder layer that swaps the FFN with routing-free gating wrapper and
    returns optional gate_scores for aux loss.
    """

    def __init__(self, config, layer_idx: int):
        super().__init__(config, layer_idx)

        # Replace MLP with routing-free wrapper when compatible
        if hasattr(self.mlp, "up_proj") and hasattr(self.mlp, "down_proj") and hasattr(self.mlp, "act_fn"):
            self.mlp = wrap_mlp_with_routing_free(self.mlp, config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values=None,
        use_cache: bool | None = False,
        cache_position: torch.LongTensor | None = None,
        position_embeddings=None,
        **kwargs,
    ):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        mlp_out = self.mlp(hidden_states)

        gate_score = None
        if isinstance(mlp_out, tuple) and len(mlp_out) == 2:
            hidden_states = mlp_out[0]
            gate_score = mlp_out[1]
        else:
            hidden_states = mlp_out

        hidden_states = residual + hidden_states
        return hidden_states, gate_score


class RoutingFreeDeepseekV3Model(DeepseekV3Model):
    """
    DeepseekV3Model with routing-free FFN per layer and gate_scores collection.
    """

    def __init__(self, config):
        super().__init__(config)
        # Replace decoder layers with routing-free versions
        self.layers = nn.ModuleList(
            [RoutingFreeDeepseekV3DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )

    @check_model_inputs
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values=None,
        inputs_embeds: torch.FloatTensor | None = None,
        cache_position: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        output_gate_scores: bool = True,
        **kwargs,
    ) -> BaseModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = create_causal_mask(
            config=self.config,
            input_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids=position_ids)

        gate_scores_all = []
        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            hidden_states, gate_score = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_embeddings=position_embeddings,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                **kwargs,
            )
            if output_gate_scores:
                gate_scores_all.append(gate_score)

        hidden_states = self.norm(hidden_states)

        outputs = BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )
        if output_gate_scores:
            outputs.router_logits = gate_scores_all  # attach for aux loss
        return outputs


class RoutingFreeDeepseekV3ForCausalLM(RoutingFreeAuxLossMixin, DeepseekV3ForCausalLM):
    """
    CausalLM head that uses routing-free decoder and computes aux loss from gate_scores.
    """

    def __init__(self, config):
        super().__init__(config)
        # swap model to routing-free version
        self.model = RoutingFreeDeepseekV3Model(config)

        # init lambda_coef if provided
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
        cache_position=None,
        logits_to_keep: int | torch.Tensor = 0,
        output_gate_scores: bool = True,
        **kwargs,
    ) -> CausalLMOutputWithPast:

        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            output_gate_scores=output_gate_scores,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

        aux_loss = None
        aux_dict = None
        gate_scores = getattr(outputs, "router_logits", None) if output_gate_scores else None
        aux_loss, aux_dict, new_lambda = self.compute_routing_free_aux(gate_scores, self.config, self.training)
        if self.training and aux_loss is not None and loss is not None:
            loss = loss + aux_loss
        self.lambda_coef = new_lambda

        out = CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=getattr(outputs, "hidden_states", None),
            attentions=getattr(outputs, "attentions", None),
        )
        out.aux_loss = aux_dict
        out.router_logits = gate_scores
        return out


__all__ = [
    "RoutingFreeDeepseekV3Model",
    "RoutingFreeDeepseekV3ForCausalLM",
]


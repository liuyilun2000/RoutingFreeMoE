
import torch
from torch import nn
from transformers.modeling_outputs import MoeModelOutputWithPast, MoeCausalLMOutputWithPast, CausalLMOutputWithPast
from transformers.utils.generic import check_model_inputs
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_rope_utils import rope_config_validation

from transformers.models.mixtral.modeling_mixtral import (
    MixtralModel,
    MixtralForCausalLM,
    MixtralDecoderLayer,
    MixtralConfig,
)
# Use create_causal_mask from modeling_utils or masking_utils depending on transformers version imports in original file
# Looking at modeling_mixtral.py imports:
from transformers.masking_utils import create_causal_mask

from typing import Optional#, Unpack
from dataclasses import dataclass

from routing_free.deepseek_v3.configuration_deepseek_v3_rf import RoutingFreeConfigMixin
from .modules import wrap_mlp_with_routing_free, RoutingFreeAuxLossMixin, RoutingFreeMaskedMoE

class RoutingFreeMixtralConfig(MixtralConfig, RoutingFreeConfigMixin):
    """Add routing-free hyperparameters on top of MixtralConfig."""

    model_type = "routing_free_mixtral"

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
        n_experts=8, # Mixtral usually has 8 experts
        density_target=0.1,
        lambda_coef=1e-5,
        eta_coef=0.2,
        per_expert_aux_loss_coef=0.5,
        per_token_aux_loss_coef=0.5,
        **kwargs,
    ):
        super().__init__(**kwargs)

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

# Helper class to represent a single decomposed expert from MixtralExperts
class MixtralRFExpert(nn.Module):
    def __init__(self, hidden_size, intermediate_size, hidden_act):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act_fn = hidden_act

def build_masked_moe_from_mixtral(config, mixtral_moe_block: nn.Module, wrap_fn):
    """
    Decompose MixtralSparseMoeBlock into individual routing-free wrapped experts.
    """
    rf_experts = []
    
    #import pdb; pdb.set_trace() 
    mixtral_experts = mixtral_moe_block.experts
    # Check if experts is a ModuleList (transformers implementation) or fused tensor (local implementation)
    if isinstance(mixtral_experts, nn.ModuleList):
        num_experts = len(mixtral_experts)
        # We need to get hidden/intermediate dims from the first expert
        # Transformers MixtralBlockSparseTop2MLP has w1, w2, w3
        # w1: in -> inter (gate)
        # w3: in -> inter (up)
        # w2: inter -> in (down)
        first_expert = mixtral_experts[0]
        hidden_dim = first_expert.w1.in_features
        intermediate_dim = first_expert.w1.out_features
        act_fn = first_expert.act_fn

        for i in range(num_experts):
            expert = MixtralRFExpert(hidden_dim, intermediate_dim, act_fn)
            source_expert = mixtral_experts[i]
            
            # Copy weights
            expert.gate_proj.weight.data.copy_(source_expert.w1.weight.data)
            expert.up_proj.weight.data.copy_(source_expert.w3.weight.data)
            expert.down_proj.weight.data.copy_(source_expert.w2.weight.data)
            
            wrapped_expert = wrap_fn(expert, config)
            rf_experts.append(wrapped_expert)

    else:
        # Fused implementation (local modeling_mixtral.py)
        num_experts = mixtral_experts.num_experts
        hidden_dim = mixtral_experts.hidden_dim
        intermediate_dim = mixtral_experts.intermediate_dim
        act_fn = mixtral_experts.act_fn
    
        for i in range(num_experts):
            # Create a single expert module
            expert = MixtralRFExpert(hidden_dim, intermediate_dim, act_fn)
            
            w_gate_up = mixtral_experts.gate_up_proj[i] # [2*int, hidden]
            w_gate = w_gate_up[:intermediate_dim, :]
            w_up = w_gate_up[intermediate_dim:, :]
            
            expert.gate_proj.weight.data.copy_(w_gate)
            expert.up_proj.weight.data.copy_(w_up)
            
            w_down = mixtral_experts.down_proj[i]
            expert.down_proj.weight.data.copy_(w_down)
            
            # Wrap with RoutingFreeFFNWrapper
            wrapped_expert = wrap_fn(expert, config)
            rf_experts.append(wrapped_expert)
        
    return RoutingFreeMaskedMoE(config, rf_experts, shared_expert=None)


class RoutingFreeMixtralDecoderLayer(MixtralDecoderLayer):
    """
    Decoder layer that swaps the FFN with routing-free gating wrapper and
    returns optional gate_scores for aux loss.
    """
    
    config_class = RoutingFreeMixtralConfig

    def __init__(self, config: MixtralConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        
        # Mixtral doesn't have first_k_dense_replace logic usually, but we implement similar check if present
        # If not present in config, we assume all layers are RF (if it was MoE)
        # Mixtral IS MoE for all layers. 
        # We might want to support "dense layers" if we reuse this for dense models, but Mixtral is MoE.
        
        # Assuming we replace the original MixtralSparseMoeBlock
        self.block_sparse_moe = build_masked_moe_from_mixtral(config, self.block_sparse_moe, wrap_mlp_with_routing_free)

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
        
        gate_score = None
        if isinstance(mlp_out, tuple) and len(mlp_out) == 2:
            hidden_states = mlp_out[0]
            gate_score = mlp_out[1]
        else:
            hidden_states = mlp_out

        hidden_states = residual + hidden_states
        return hidden_states, gate_score


class RoutingFreeMixtralModel(MixtralModel):
    """
    MixtralModel with routing-free FFN per layer and gate_scores collection.
    """
    config_class = RoutingFreeMixtralConfig

    def __init__(self, config: MixtralConfig):
        super().__init__(config)
        # Replace decoder layers with routing-free versions
        self.layers = nn.ModuleList(
            [RoutingFreeMixtralDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        output_router_logits: bool | None = None, # kept for signature compat, but we use output_gate_scores in RF logic usually
        cache_position: torch.LongTensor | None = None,
        output_gate_scores: bool = True, # Explicit RF arg
        **kwargs,
    ) -> MoeModelOutputWithPast:
        
        if output_router_logits is not None:
             output_gate_scores = output_router_logits

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

        # Mixtral uses create_causal_mask or create_sliding_window_causal_mask (not imported here to avoid complexities, leveraging standard create_causal_mask or config)
        # We'll use the logic from MixtralModel.forward
        # Copied from modeling_mixtral.py to ensure identical behavior
        if self.config.sliding_window is None:
            mask_function = create_causal_mask
        else:
             try:
                 from transformers.masking_utils import create_sliding_window_causal_mask
                 mask_function = create_sliding_window_causal_mask
             except ImportError:
                 # Fallback/Warning if transformers version is old, though unlikely given modeling_mixtral.py presence
                 print("Warning: create_sliding_window_causal_mask not found, falling back to create_causal_mask")
                 mask_function = create_causal_mask

        causal_mask = mask_function(
            input_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
            position_ids=position_ids,
            config=self.config, # create_sliding_window_causal_mask requires config
        )

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids=position_ids)

        gate_scores_all = []
        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            hidden_states, gate_score = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )
            
            if output_gate_scores:
                gate_scores_all.append(gate_score)

        hidden_states = self.norm(hidden_states)

        outputs = MoeModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )
        if output_gate_scores:
            outputs.router_logits = gate_scores_all  # attach for aux loss
            
        return outputs

@dataclass
class RoutingFreeMoEOutput(CausalLMOutputWithPast):
    aux_dict: dict = None
    router_logits: list[torch.FloatTensor] = None
    lm_loss: float = None
    orthogonality_loss: float = None


class RoutingFreeMixtralForCausalLM(RoutingFreeAuxLossMixin, MixtralForCausalLM):
    """
    CausalLM head that uses routing-free decoder and computes aux loss from gate_scores.
    """
    config_class = RoutingFreeMixtralConfig

    def __init__(self, config):
        super().__init__(config)
        self.model = RoutingFreeMixtralModel(config)
        
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
        output_router_logits=None,
        cache_position=None,
        logits_to_keep: int | torch.Tensor = 0,
        output_gate_scores: bool | None = None,
        **kwargs,
    ) -> RoutingFreeMoEOutput:
        
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
            aux_loss, aux_dict, new_lambda = self.compute_routing_free_aux(gate_scores, self.config, self.training)
            if aux_loss is not None and loss is not None:
                loss = loss + aux_loss
            self.lambda_coef = new_lambda
            
        # compute orthogonality loss
        orthogonality_loss = torch.tensor(0.0, device=self.device)
        orth_coef = getattr(self.config, "orthogonality_loss_coef", 0.0) # Check if config has this
        
        if self.training and orth_coef > 0:
             # Iterate over layers and sum orthogonality loss
             for layer in self.model.layers:
                 # Check if layer has mlp.experts (it should for RF)
                 if hasattr(layer.mlp, "experts"):
                     layer_orth_loss = self.compute_orthogonality_loss(layer.mlp.experts) 
                     if layer_orth_loss is not None:
                         orthogonality_loss += layer_orth_loss
             
             orthogonality_loss = orthogonality_loss * orth_coef
             aux_dict = aux_dict or {}
             aux_dict["orthogonality_loss"] = orthogonality_loss.item()
             if loss is not None:
                 loss = loss + orthogonality_loss

        out = RoutingFreeMoEOutput(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=getattr(outputs, "hidden_states", None),
            attentions=getattr(outputs, "attentions", None),
            aux_dict=aux_dict,
            router_logits=gate_scores,
            lm_loss=lm_loss,
            orthogonality_loss=orthogonality_loss
        )
        return out

__all__ = [
    "RoutingFreeMixtralModel",
    "RoutingFreeMixtralForCausalLM",
    "RoutingFreeMixtralConfig",
]

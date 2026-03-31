import torch
from torch import nn
import torch.nn.functional as F
from transformers.activations import ACT2FN


class RoutingFreeGate(nn.Module):
    """
    Basic gating module:
    Calculate gate_score and generate gate_mask.
    Only responsible for scoring and masking, not for specific MLP calculation, which is reusable for different MoE/MLP models.
    """

    def __init__(self, hidden_size: int, cfg, shared_gate_proj: nn.Linear | None = None):
        super().__init__()
        self.hidden_size = hidden_size
        self.gate_proj_rank = getattr(cfg, "gate_proj_rank", hidden_size // 4)

        # Low-rank projection A: hidden -> gate_proj_rank
        if shared_gate_proj is None:
            self.gate_proj_A = nn.Linear(self.hidden_size, self.gate_proj_rank, bias=False)
            self._shared_proj_fn = None
        else:
            # don't register shared module; use functional linear to avoid duplicate params
            self._shared_proj_fn = lambda x: F.linear(x, shared_gate_proj.weight, shared_gate_proj.bias)

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

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        output_gate_scores: bool | None = None,
    ):
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
        if self._shared_proj_fn is not None:
            gate_hidden = self._shared_proj_fn(x_valid)
        else:
            gate_hidden = self.gate_proj_A(x_valid)
        gate_score = self._gate_norm(gate_hidden)
        if self.gate_scale is not None:
            gate_score = gate_score * self.gate_scale
        if self.gate_bias is not None:
            gate_score = gate_score - self.gate_bias
        #gate_score = self.gate_act_fn(self._gate_norm(gate_hidden))

        # Threshold/temperature
        gate_mask = gate_score >= (self.gate_threshold / self.gate_temperature)
        '''
        num_active = gate_mask.sum().item()
        if num_active == 0:
            import pdb; pdb.set_trace()
            print(f"all experts are masked out")
        '''

        # Fill back to full shape
        gate_mask_full = torch.zeros(x_flat.shape[0], device=x.device, dtype=torch.bool)
        gate_mask_full[idx[gate_mask]] = True
        gate_mask_full = gate_mask_full.view(orig_shape[:-1])

        emit_gate_scores = self.output_gate_scores if output_gate_scores is None else output_gate_scores
        gate_score_full = None
        if emit_gate_scores:
            gate_score_full = x_flat.new_ones(x_flat.shape[0]) * -float("inf")
            if gate_mask.any():
                # ensure dtype aligns with destination
                gate_score_full[idx[gate_mask]] = gate_score[gate_mask].to(gate_score_full.dtype)
            gate_score_full = gate_score_full.view(orig_shape[:-1])

        return gate_mask_full, gate_score_full, gate_hidden


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
        self.gate_proj = base_mlp.gate_proj
        self.up_proj = base_mlp.up_proj
        self.down_proj = base_mlp.down_proj
        self.act_fn = base_mlp.act_fn

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
        self.gate = RoutingFreeGate(self.hidden_size, cfg, shared_gate_proj=self.gate_proj_A)

        self.output_gate_scores = getattr(cfg, "output_gate_scores", True)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        output_gate_scores: bool | None = None,
    ):
        """
        Returns:
            out: Same shape as x
            gate_score_full (or None): Same shape as x[...,0]
        """
        emit_gate_scores = self.output_gate_scores if output_gate_scores is None else output_gate_scores
        gate_mask_full, gate_score_full, gate_hidden = self.gate(
            x, mask, output_gate_scores=emit_gate_scores
        )
        gate_mask_flat = gate_mask_full.view(-1)

        orig_shape = x.shape
        x_flat = x.view(-1, orig_shape[-1])
        mlp_out_flat = torch.zeros_like(x_flat)

        if gate_mask_flat.any():
            x_valid = x_flat[gate_mask_flat]
            gate_hidden_valid = gate_hidden[gate_mask_flat]
            # gate_score_full already has -inf for inactive tokens (built in RoutingFreeGate);
            # just extract scores at active positions for weighting the FFN output.
            gate_score_valid = gate_score_full.view(-1)[gate_mask_flat]
            mlp_out_valid = self.down_proj(self.act_fn(self.gate_proj_B(gate_hidden_valid)) * self.up_proj(x_valid))
            mlp_out_flat[gate_mask_flat] = (mlp_out_valid * gate_score_valid.unsqueeze(1)).to(dtype=x.dtype)

        mlp_out = mlp_out_flat.view(orig_shape)
        # gate_score_full is already correctly shaped and filled by RoutingFreeGate — no rebuild needed.
        return mlp_out, gate_score_full if emit_gate_scores else None
        
    def forward_ffn(
        self,
        x_flat: torch.Tensor,
        gate_hidden: torch.Tensor,
        gate_score_flat: torch.Tensor,
        gate_mask_flat: torch.Tensor,
    ) -> torch.Tensor:
        """
        FFN-only path used by RoutingFreeMaskedMoE when gate scores are
        pre-computed externally via a batched gate_proj_A matmul.

        Args:
            x_flat:          [N, H]  — all tokens, flattened
            gate_hidden:     [N, R]  — raw gate_proj_A output for this expert
            gate_score_flat: [N]     — gate score after scale/bias (weights FFN output)
            gate_mask_flat:  [N]     — boolean, which tokens are active
        Returns:
            mlp_out_flat: [N, H]
        """
        mlp_out_flat = torch.zeros_like(x_flat)
        if gate_mask_flat.any():
            x_valid = x_flat[gate_mask_flat]
            gate_hidden_valid = gate_hidden[gate_mask_flat]
            gate_score_valid = gate_score_flat[gate_mask_flat]
            mlp_out_valid = self.down_proj(
                self.act_fn(self.gate_proj_B(gate_hidden_valid)) * self.up_proj(x_valid)
            )
            mlp_out_flat[gate_mask_flat] = (
                mlp_out_valid * gate_score_valid.unsqueeze(1)
            ).to(dtype=x_flat.dtype)
        return mlp_out_flat


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
    def sync_lambda_coef(self):
        import torch.distributed as dist
        if dist.is_initialized():
            # All-reduce with averaging across all ranks
            dist.all_reduce(self.lambda_coef, op=dist.ReduceOp.AVG)
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

        gate_act_fn = ACT2FN[getattr(cfg, "gate_act_fn", "relu")]
        gate_temperature = getattr(cfg, "gate_temperature", 1.0)
        gate_threshold = getattr(cfg, "gate_threshold", 0.5)
        density_target = getattr(cfg, "density_target", 0.25)
        per_expert_coef = getattr(cfg, "per_expert_aux_loss_coef", 0.5)
        per_token_coef = getattr(cfg, "per_token_aux_loss_coef", 0.5)
        eta_coef = getattr(cfg, "eta_coef", 1.2)
        n_experts = getattr(cfg, "n_experts", None)

        # get or initialize lambda_coef
        lambda_init = getattr(cfg, "lambda_coef", 1e-6)
        lambda_coef = getattr(self, "lambda_coef", lambda_init) # ?
        if not isinstance(lambda_coef, torch.Tensor):
            lambda_coef = torch.tensor(lambda_coef)

        gate_probs_list = []
        total_tokens = 0

        for layer_gate_scores in gate_scores:
            if layer_gate_scores is None:
                continue
            # skip empty tensors
            if layer_gate_scores.numel() == 0:
                continue
            # allow 2D gate scores (e.g., routing-free dense FFN) by adding a singleton expert dim
            if layer_gate_scores.dim() == 2:
                layer_gate_scores = layer_gate_scores.unsqueeze(-1)
            if layer_gate_scores.dim() != 3:
                # unsupported shape, skip
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
        gate_mask = (gate_probs >= (gate_threshold/gate_temperature)).float()

        density = torch.mean(gate_mask)
        density_ratio = density / density_target
        density_per_expert = torch.mean(gate_mask, dim=0)
        #density_per_token = torch.mean(gate_mask, dim=1)
        density_proxy = torch.mean(gate_probs)
        density_proxy_per_expert = torch.mean(gate_probs, dim=0)
        #density_proxy_per_token = torch.mean(gate_probs, dim=1)

        per_expert_aux = balancing_loss_func(gate_probs, gate_mask, dim=0) if per_expert_coef > 0 else 0
        per_token_aux = balancing_loss_func(gate_probs, gate_mask, dim=1) if per_token_coef > 0 else 0
        reg_loss = per_expert_aux * per_expert_coef + per_token_aux * per_token_coef
        aux_loss = lambda_coef * reg_loss

        new_lambda = lambda_coef
        if training and aux_loss is not None:
            #new_lambda = lambda_coef * (1 + eta_coef * (density - density_target) ** 2) ** torch.sign(density - density_target)
            #update_factor = (1 + eta_coef * (density - density_target) ** 2) ** torch.sign(density - density_target)
            
            #max_step = getattr(cfg, "lambda_max_step", 0.02) 
            #if update_factor > 1.01 and lambda_coef > 0.1:
                #import pdb; pdb.set_trace()
            #update_factor = torch.clamp(update_factor, 1.0 - max_step, 1.0 + max_step)
            
            #new_lambda = lambda_coef * update_factor
            
            update_factor = (1+eta_coef) ** torch.sign(density - density_target)
            new_lambda = lambda_coef * update_factor
            new_lambda = torch.clamp(new_lambda, max=1.0)
            if hasattr(self, "lambda_coef"):
                # limit relative update scale
                self.lambda_coef = new_lambda
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
        '''
        if density_proxy_per_expert is not None and density_per_expert is not None:
            n_layers = density_per_expert.numel() // n_experts
            layer_digits = len(str(n_layers - 1)) if n_layers > 0 else 1
            expert_digits = len(str(n_experts - 1)) if n_experts > 0 else 1
            for l in range(n_layers):
                for e in range(n_experts):
                    idx = l * n_experts + e
                    l_str = str(l).zfill(layer_digits)
                    e_str = str(e).zfill(expert_digits)
                    aux_dict[f"expert_density_L{l_str}E{e_str}"] = float(density_per_expert[idx].detach().cpu())
                    aux_dict[f"expert_density_proxy_L{l_str}E{e_str}"] = float(density_proxy_per_expert[idx].detach().cpu())
        '''

        return aux_loss, aux_dict, new_lambda

class RoutingFreeMaskedMoE(nn.Module):
    """
    通用的 routing-free MoE 聚合器：
    - 假设传入的 experts 都返回 (out, gate_score)，gate_score 形状为 [B, T]
    - 不使用原始 top-k router，而是按 gate_score 与阈值决定激活
    - shared_expert 保持原样（用于 dense residual FFN）
    """

    def __init__(self, cfg, experts: list[nn.Module], shared_expert: nn.Module | None = None):
        super().__init__()
        self.config = cfg
        self.experts = nn.ModuleList(experts)
        self.shared_expert = shared_expert
        self.output_gate_scores = getattr(cfg, "output_gate_scores", True)

    def forward(
        self,
        hidden_states: torch.Tensor,
        mask: torch.Tensor | None = None,
        output_gate_scores: bool | None = None,
    ):
        orig_shape = hidden_states.shape
        x_flat = hidden_states.view(-1, hidden_states.shape[-1])  # [N, H]
        N = x_flat.shape[0]
        E = len(self.experts)
        R = self.experts[0].gate_proj_rank
        threshold = self.config.gate_threshold / self.config.gate_temperature

        # Batch all gate_proj_A projections into one matmul instead of E separate ones.
        # W_A: [E*R, H]  →  gate_hidden_all: [N, E*R] → [N, E, R]
        W_A = torch.cat([e.gate_proj_A.weight for e in self.experts], dim=0)
        gate_hidden_all = F.linear(x_flat, W_A).view(N, E, R)

        # Compute all gate scores (norms) at once: [N, E]
        # _gate_norm applies norm over the last dim, so shape [N, E, R] → [N, E]
        gate_scores_all = self.experts[0].gate._gate_norm(gate_hidden_all)

        out_flat = torch.zeros_like(x_flat)
        emit_gate_scores = self.output_gate_scores if output_gate_scores is None else output_gate_scores
        expert_gate_scores_list = [] if emit_gate_scores else None

        for e, expert in enumerate(self.experts):
            gate_hidden_e = gate_hidden_all[:, e, :].contiguous()  # [N, R]
            gate_score_e  = gate_scores_all[:, e]                  # [N]

            # Apply per-expert learnable scale / bias (same as RoutingFreeGate.forward)
            g = expert.gate
            if g.gate_scale is not None:
                gate_score_e = gate_score_e * g.gate_scale
            if g.gate_bias is not None:
                gate_score_e = gate_score_e - g.gate_bias

            gate_mask_e = gate_score_e >= threshold  # [N]

            out_flat = out_flat + expert.forward_ffn(
                x_flat, gate_hidden_e, gate_score_e, gate_mask_e
            )

            if emit_gate_scores:
                score_full = gate_score_e.new_full((N,), -float('inf'))
                if gate_mask_e.any():
                    score_full[gate_mask_e] = gate_score_e[gate_mask_e]
                expert_gate_scores_list.append(score_full.view(orig_shape[:-1]))

        out = out_flat.view(orig_shape)
        return (
            out,
            torch.stack(expert_gate_scores_list, dim=-1) if emit_gate_scores else None,
        )

class AoEMoE(nn.Module):
    """
    Autonomy-of-Experts MoE: each expert self-evaluates via L2 norm of its
    low-rank projection (gate_proj_A), then top-K experts are selected and
    their outputs are aggregated with softmax-normalized weights.

    Experts are RoutingFreeFFNWrapper instances (with gate_proj_A, gate_proj_B,
    up_proj, down_proj, act_fn). No separate router network.

    Returns (output, None) — no aux loss needed.
    """

    def __init__(self, cfg, experts: list[nn.Module]):
        super().__init__()
        self.config = cfg
        self.experts = nn.ModuleList(experts)
        self.num_experts_per_tok = getattr(cfg, "num_experts_per_tok", 2)

    def forward(self, hidden_states: torch.Tensor, mask: torch.Tensor | None = None):
        orig_shape = hidden_states.shape  # [B, T, H]
        x_flat = hidden_states.view(-1, hidden_states.shape[-1])  # [N, H]
        N = x_flat.shape[0]
        E = len(self.experts)
        R = self.experts[0].gate_proj_rank
        K = self.num_experts_per_tok

        # Step 1: Batched low-rank self-evaluation — z_i = x @ W_down_r^i
        W_A = torch.cat([e.gate_proj_A.weight for e in self.experts], dim=0)  # [E*R, H]
        gate_hidden_all = F.linear(x_flat, W_A).view(N, E, R)  # [N, E, R]

        # Step 1b: L2 norm confidence scores — s_i = ||z_i||_2
        scores = torch.norm(gate_hidden_all, p=2, dim=-1)  # [N, E]

        # Step 2: Top-K expert selection
        topk_scores, topk_indices = torch.topk(scores, K, dim=-1)  # [N, K]

        # Step 3: Softmax normalization over selected experts
        topk_weights = F.softmax(topk_scores, dim=-1)  # [N, K]

        # Step 4: Weighted expert execution using cached z_j
        out_flat = torch.zeros_like(x_flat)
        for e, expert in enumerate(self.experts):
            # Find tokens that selected this expert and their weights
            expert_mask = (topk_indices == e)  # [N, K] boolean
            token_mask = expert_mask.any(dim=-1)  # [N] boolean

            if not token_mask.any():
                continue

            # Weight for this expert per token (sum across K slots, though typically appears once)
            weight_for_expert = (topk_weights * expert_mask.float()).sum(dim=-1)  # [N]

            x_selected = x_flat[token_mask]
            gate_hidden_selected = gate_hidden_all[token_mask, e, :]  # cached z_j
            w_selected = weight_for_expert[token_mask]

            # Full SwiGLU forward using cached z_j:
            # g_j = SiLU(z_j @ W_up_r^j), E_j(x) = (g_j ⊙ (x @ W_up_std^j)) @ W_down_std^j
            expert_out = expert.down_proj(
                expert.act_fn(expert.gate_proj_B(gate_hidden_selected)) * expert.up_proj(x_selected)
            )
            out_flat[token_mask] = out_flat[token_mask] + (expert_out * w_selected.unsqueeze(1)).to(dtype=x_flat.dtype)

        out = out_flat.view(orig_shape)
        return out, None


class ReMoERouter(nn.Module):
    """
    ReMoE router: shared linear projection → ReLU.
    Returns non-negative routing weights per expert.
    """

    def __init__(self, hidden_size: int, num_experts: int):
        super().__init__()
        self.linear = nn.Linear(hidden_size, num_experts, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, H] or [N, H]
        Returns:
            weights: same leading dims + [num_experts], non-negative (ReLU)
        """
        return F.relu(self.linear(x))


class ReMoEMoE(nn.Module):
    """
    ReMoE MoE (w/o softmax): ReLU-weighted expert aggregation.
    Skips zero-weight experts per token for efficiency.
    Returns (output, router_weights) where router_weights: [B, T, E].
    """

    def __init__(self, hidden_size: int, experts: list[nn.Module]):
        super().__init__()
        self.experts = nn.ModuleList(experts)
        self.router = ReMoERouter(hidden_size, len(experts))

    def forward(self, hidden_states: torch.Tensor, mask: torch.Tensor | None = None):
        orig_shape = hidden_states.shape  # [B, T, H]
        x_flat = hidden_states.view(-1, orig_shape[-1])  # [N, H]

        weights = self.router(x_flat)  # [N, E]
        out_flat = torch.zeros_like(x_flat)

        for e, expert in enumerate(self.experts):
            w_e = weights[:, e]  # [N]
            active = w_e > 0
            if active.any():
                x_active = x_flat[active]
                expert_out = expert.down_proj(expert.act_fn(expert.gate_proj(x_active)) * expert.up_proj(x_active))
                out_flat[active] = out_flat[active] + expert_out * w_e[active].unsqueeze(1)

        out = out_flat.view(orig_shape)
        router_weights = weights.view(orig_shape[:-1] + (len(self.experts),))
        return out, router_weights


class ReMoEAuxLossMixin:
    """L1 regularization on ReLU router outputs with adaptive λ coefficient."""

    def sync_remoe_lambda(self):
        import torch.distributed as dist
        if dist.is_initialized():
            dist.all_reduce(self.l1_coef_adaptive, op=dist.ReduceOp.AVG)

    def compute_remoe_aux(self, router_weights_list, cfg, training):
        """
        Args:
            router_weights_list: list of [B, T, E] tensors (one per layer)
            cfg: config with l1_coef, density_target, eta_coef
            training: bool, whether to update adaptive lambda
        Returns:
            aux_loss: scalar tensor
            aux_dict: dict for logging
        """
        l1_coef = getattr(self, "l1_coef_adaptive", None)
        if l1_coef is None:
            l1_coef = torch.tensor(getattr(cfg, "l1_coef", 0.01))

        if not router_weights_list:
            return None, None, l1_coef

        all_weights = []
        for w in router_weights_list:
            if w is not None:
                all_weights.append(w)

        if not all_weights:
            return None, None, l1_coef

        cat_weights = torch.cat([w.view(-1) for w in all_weights])
        l1_loss = cat_weights.sum()

        # Ensure lambda and weights are on the same device/dtype
        l1_coef = l1_coef.to(device=cat_weights.device, dtype=cat_weights.dtype)
        aux_loss = l1_coef * l1_loss

        # density = fraction of active experts (non-zero weights)
        density = (cat_weights > 0).float().mean()
        density_target = getattr(cfg, "density_target", 0.25)
        density_ratio = density / density_target

        # Adaptive lambda update: same pattern as RF
        new_l1_coef = l1_coef
        if training:
            eta_coef = getattr(cfg, "eta_coef", 0.02)
            update_factor = (1 + eta_coef) ** torch.sign(density - density_target)
            new_l1_coef = l1_coef * update_factor
            new_l1_coef = torch.clamp(new_l1_coef, max=1.0)
            if hasattr(self, "l1_coef_adaptive"):
                self.l1_coef_adaptive = new_l1_coef
                self.sync_remoe_lambda()

        aux_dict = {
            "aux_loss": aux_loss.item(),
            "l1_loss": l1_loss.item(),
            "l1_coef": float(new_l1_coef.detach().cpu()) if isinstance(new_l1_coef, torch.Tensor) else new_l1_coef,
            "density": density.item(),
            "density_ratio": density_ratio.item(),
        }
        return aux_loss, aux_dict, new_l1_coef


class ReMoEConfigMixin:
    """ReMoE config fields mixin."""

    remoe_fields = ("l1_coef", "density_target", "eta_coef")

    def _init_remoe(self, l1_coef=0.01, density_target=0.25, eta_coef=0.02):
        self.l1_coef = l1_coef
        self.density_target = density_target
        self.eta_coef = eta_coef


class RoutingFreeConfigMixin:
    """Routing-free fields mixin, reusable for other MoE configs."""

    routing_free_fields = (
        "gate_proj_rank",
        "gate_norm",
        "gate_bias",
        "gate_scale",
        "gate_act_fn",
        "gate_threshold",
        "gate_temperature",
        "output_gate_scores",
        "n_experts",
        "density_target",
        "lambda_coef",
        "eta_coef",
        "per_expert_aux_loss_coef",
        "per_token_aux_loss_coef",
    )

    def _init_routing_free(
        self,
        gate_proj_rank=None,
        gate_norm="l2",
        gate_bias=True,
        gate_scale=True,
        gate_act_fn=None,
        gate_threshold=0.05,
        gate_temperature=1.0,
        output_gate_scores=False,
        n_experts=12,
        density_target=0.1,
        lambda_coef=1e-5,
        eta_coef=0.2,
        per_expert_aux_loss_coef=0.5,
        per_token_aux_loss_coef=0.5,
    ):
        self.gate_proj_rank = gate_proj_rank
        self.gate_norm = gate_norm
        self.gate_bias = gate_bias
        self.gate_scale = gate_scale
        self.gate_act_fn = gate_act_fn
        self.gate_threshold = gate_threshold
        self.gate_temperature = gate_temperature
        self.output_gate_scores = output_gate_scores
        self.n_experts = n_experts
        self.density_target = density_target
        self.lambda_coef = lambda_coef
        self.eta_coef = eta_coef
        self.per_expert_aux_loss_coef = per_expert_aux_loss_coef
        self.per_token_aux_loss_coef = per_token_aux_loss_coef
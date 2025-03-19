"""
SwissAI style Language Model that is
compilable (avoids torch complex)
"""

import math

import tiktoken
import torch
import torch.nn as nn
from torch.nn import functional as F
from models.base import GPTBase

from .quantization import QuantizedLinear, QUANTIZER_CLASSES


def precompute_inv_freq(dim: int, base: float, device: torch.device) -> torch.Tensor:
    return 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64, device=device).float() / dim))


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class XIELU(nn.Module):
    def __init__(self, alpha_p_init=0.8, alpha_n_init=0.8, beta=0.5, eps=-1e-6):
        super(XIELU, self).__init__()
        self.beta = beta
        self.alpha_p = nn.Parameter(torch.log(torch.exp(torch.tensor(alpha_p_init)) - 1).unsqueeze(0))
        self.alpha_n = nn.Parameter(torch.log(torch.exp(torch.tensor(alpha_n_init - self.beta)) - 1).unsqueeze(0))
        self.eps = torch.tensor(eps)

    def forward(self, x):
        alpha_p = F.softplus(self.alpha_p)
        alpha_n = self.beta + F.softplus(self.alpha_n)
        return torch.where(x > 0,
                           alpha_p * x * x + self.beta * x,
                           alpha_n * torch.expm1(torch.min(x, self.eps)) - alpha_n * x + self.beta * x)


class SwissAIRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        SwissAIRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


class SwissAIAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.n_kv_head = config.n_kv_head
        self.num_key_value_groups = self.n_head // self.n_kv_head
        self.head_dim = config.n_embd // config.n_head
        self.dropout = config.dropout
        self.flash = hasattr(F, 'scaled_dot_product_attention')
        
        self.bias = config.bias
        self.qk_norm = config.qk_norm
        
        # Key, query, value projections
        self.q_proj = QuantizedLinear(
            config.n_embd, 
            self.n_head * self.head_dim, 
            bias=self.bias,
            weight_quantizer=QUANTIZER_CLASSES[config.w_quant](**config.w_quant_kwargs),
            activation_quantizer=QUANTIZER_CLASSES[config.a_quant](**config.a_quant_kwargs)
        )
        
        self.k_proj = QuantizedLinear(
            config.n_embd, 
            self.n_kv_head * self.head_dim, 
            bias=self.bias,
            weight_quantizer=QUANTIZER_CLASSES[config.w_quant](**config.w_quant_kwargs),
            activation_quantizer=QUANTIZER_CLASSES[config.a_quant](**config.a_quant_kwargs)
        )
        
        self.v_proj = QuantizedLinear(
            config.n_embd, 
            self.n_kv_head * self.head_dim, 
            bias=self.bias,
            weight_quantizer=QUANTIZER_CLASSES[config.w_quant](**config.w_quant_kwargs),
            activation_quantizer=QUANTIZER_CLASSES[config.a_quant](**config.a_quant_kwargs)
        )
        
        self.o_proj = QuantizedLinear(
            self.n_head * self.head_dim, 
            config.n_embd, 
            bias=self.bias,
            weight_quantizer=QUANTIZER_CLASSES[config.w_quant](**config.w_quant_kwargs),
            activation_quantizer=QUANTIZER_CLASSES[config.a_quant](**config.a_quant_kwargs)
        )
        
        if self.qk_norm:
            self.q_norm = SwissAIRMSNorm(self.head_dim, eps=config.rmsnorm_eps)
            self.k_norm = SwissAIRMSNorm(self.head_dim, eps=config.rmsnorm_eps)
        else:
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()
        
        # Causal mask to ensure that attention is only applied to the left in the input sequence
        if not self.flash:
            self.register_buffer(
                "bias",
                torch.tril(torch.ones(config.sequence_length, config.sequence_length))
                .view(1, 1, config.sequence_length, config.sequence_length)
            )
        
        self.scaling = self.head_dim ** -0.5

    def repeat_kv(self, hidden_states, n_rep):
        """
        Repeat key and values for multi-query attention
        """
        batch, n_kv_head, slen, head_dim = hidden_states.shape
        if n_rep == 1:
            return hidden_states
        hidden_states = hidden_states[:, :, None, :, :].expand(batch, n_kv_head, n_rep, slen, head_dim)
        return hidden_states.reshape(batch, n_kv_head * n_rep, slen, head_dim)

    def forward(self, x, freqs_cis, attention_mask=None):
        # batch size, sequence length, embedding dimensionality
        B, T, C = x.size()

        # Calculate query, key, values
        q = self.q_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
        
        # Apply RMSNorm to query and key if qk_norm is enabled
        q = self.q_norm(q)
        k = self.k_norm(k)
        
        # Apply rotary embeddings
        q, k = apply_rotary_pos_emb(q, k, freqs_cis[0], freqs_cis[1])
        
        # Repeat k and v for multi-query attention
        if self.num_key_value_groups > 1:
            k = self.repeat_kv(k, self.num_key_value_groups)
            v = self.repeat_kv(v, self.num_key_value_groups)

        # Compute attention with either flash attention or the standard implementation
        if self.flash:
            # Efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attention_mask, dropout_p=0.0 if not self.training else self.dropout,
                is_causal=True, scale=self.scaling,
            )
        else:
            raise NotImplementedError("Standard attention is not implemented")
            
        # Reshape and apply output projection
        y = y.transpose(1, 2).contiguous().view(B, T, -1)
        y = self.o_proj(y)
        
        return y


class SwissAIMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_act = getattr(config, "hidden_act", "xielu")
        self.intermediate_size = config.intermediate_size
        
        # If config doesn't specify a multiple_of value, use 256 as default
        self.multiple_of = getattr(config, "multiple_of", 256)
        
        # If needed, adjust intermediate size to be divisible by multiple_of
        if self.multiple_of > 0:
            self.intermediate_size = self.multiple_of * ((self.intermediate_size + self.multiple_of - 1) // self.multiple_of)
        
        # For SwissAI, up_proj replaces w1 from the original implementation
        self.up_proj = QuantizedLinear(
            config.n_embd,
            self.intermediate_size,
            bias=False,
            weight_quantizer=QUANTIZER_CLASSES[config.w_quant](**config.w_quant_kwargs),
            activation_quantizer=QUANTIZER_CLASSES[config.a_quant](**config.a_quant_kwargs),
        )
        
        # For non-XIELU activations, we need gate_proj (replacing w2 from original implementation)
        if self.hidden_act != "xielu":
            self.gate_proj = QuantizedLinear(
                config.n_embd,
                self.intermediate_size,
                bias=False,
                weight_quantizer=QUANTIZER_CLASSES[config.w_quant](**config.w_quant_kwargs),
                activation_quantizer=QUANTIZER_CLASSES[config.a_quant](**config.a_quant_kwargs),
            )
        
        # Down projection (c_proj in original implementation)
        self.down_proj = QuantizedLinear(
            self.intermediate_size,
            config.n_embd,
            bias=False,
            weight_quantizer=QUANTIZER_CLASSES[config.w_quant](**config.w_quant_kwargs),
            activation_quantizer=QUANTIZER_CLASSES[config.a_quant](**config.a_quant_kwargs),
        )
        
        # For XIELU activation function
        if self.hidden_act == "xielu":
            self.act_fn = XIELU()
        elif self.hidden_act == "silu":
            self.act_fn = F.silu
        else:
            self.act_fn = F.gelu

    def forward(self, x):
        if self.hidden_act == "xielu":
            # In case of xielu, no gated MLP
            return self.down_proj(self.act_fn(self.up_proj(x)))
        else:
            # Standard gated MLP architecture
            return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class SwissAIDecoderLayer(nn.Module):
    def __init__(self, config, layer_idx=None):
        super().__init__()
        self.hidden_size = config.n_embd
        self.self_attn = SwissAIAttention(config)
        self.mlp = SwissAIMLP(config)
        
        self.attention_layernorm = SwissAIRMSNorm(config.n_embd, eps=config.rmsnorm_eps)
        self.feedforward_layernorm = SwissAIRMSNorm(config.n_embd, eps=config.rmsnorm_eps)
        
        self.post_norm = getattr(config, "post_norm", False)

    def forward(self, hidden_states, freqs_cis, attention_mask=None):
        residual = hidden_states

        if not self.post_norm:
            hidden_states = self.attention_layernorm(hidden_states)

        # Self Attention
        hidden_states = self.self_attn(
            hidden_states,
            freqs_cis,
            attention_mask
        )
        
        # Apply normalization after attention if using post-norm
        if self.post_norm:
            hidden_states = self.attention_layernorm(hidden_states)
            
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        if not self.post_norm:
            hidden_states = self.feedforward_layernorm(hidden_states)
            
        hidden_states = self.mlp(hidden_states)
        
        if self.post_norm:
            hidden_states = self.feedforward_layernorm(hidden_states)
            
        hidden_states = residual + hidden_states

        return hidden_states


class SwissAI(GPTBase):
    def __init__(self, config):
        super().__init__(config)
        
        assert config.vocab_size is not None
        assert config.sequence_length is not None
        self.config = config
        self.tokenizer = None  # tiktoken.get_encoding("gpt2")

        # Model parameters
        self.vocab_size = config.vocab_size
        self.n_embd = config.n_embd
        self.n_head = config.n_head
        self.n_layer = config.n_layer
        self.sequence_length = config.sequence_length
        self.head_dim = config.n_embd // config.n_head
        
        # Rotary position embeddings
        self.inv_freq = precompute_inv_freq(self.head_dim, config.rope_theta, config.device)

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                drop=nn.Dropout(config.dropout),
                h=nn.ModuleList([SwissAIDecoderLayer(config, i) for i in range(config.n_layer)]),
                ln_f=SwissAIRMSNorm(config.n_embd, eps=config.rmsnorm_eps),
            )
        )
        
        # Language modeling head
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        # Weight tying (optional)
        if getattr(config, "tie_word_embeddings", False):
            self.lm_head.weight = self.wte.weight
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Apply special scaled init to residual projections
        for pn, p in self.named_parameters():
            if pn.endswith("down_proj.weight") or pn.endswith("o_proj.weight"):
                torch.nn.init.normal_(
                    p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer)
                )

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        """
        n_params = sum(p.numel() for p in self.parameters())
        return n_params

    def forward(self, idx, targets=None, target_logits=None, get_logits=False, all_logits=False):
        assert targets is None or target_logits is None, "Cannot provide both targets and target_logits"
        
        device = idx.device
        b, t = idx.size()
        assert (
            t <= self.config.sequence_length
        ), f"Cannot forward sequence of length {t}, block size is only {self.config.sequence_length}"
        
        # Token embeddings
        hidden_states = self.transformer.wte(idx)  # (b, t, n_embd)
        
        # Get the pre-computed position embeddings for the current sequence
        position_ids = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0)
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos().to(hidden_states.dtype)
        sin = emb.sin().to(hidden_states.dtype)
        
        # Forward through transformer layers
        for layer in self.transformer.h:
            hidden_states = layer(hidden_states, (cos, sin))
        
        # Apply final layer norm
        hidden_states = self.transformer.ln_f(hidden_states)
        
        # Handle different forwarding modes
        if targets is not None:
            # Training mode with explicit targets
            logits = self.lm_head(hidden_states)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
            )
        elif target_logits is not None:
            # KL-divergence training mode
            logits = self.lm_head(hidden_states)
            loss = F.kl_div(
                F.log_softmax(logits, dim=-1),
                F.log_softmax(target_logits, dim=-1),
                reduction="batchmean",
                log_target=True,
            )
        else:
            # Inference mode
            if all_logits:
                logits = self.lm_head(hidden_states)
            else:
                # Only compute logits for the last token (optimization)
                logits = self.lm_head(hidden_states[:, [-1], :])
            loss = None
        
        # Return logits only if requested
        logits = logits if get_logits else None
        
        return {
            "logits": logits,
            "loss": loss,
        }

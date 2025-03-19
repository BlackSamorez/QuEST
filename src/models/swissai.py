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


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> torch.Tensor:
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    cos_freqs = torch.cos(freqs)
    sin_freqs = torch.sin(freqs)
    # Stack the cos and sin parts in the last dimension to simulate complex numbers
    return torch.stack((cos_freqs, sin_freqs), dim=-1)


def _reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    freqs_cis: complex - (seq_len, head_dim / 2)
    x: complex - (bsz, seq_len, head_dim / 2)
    """
    ndim = x.ndim
    assert 1 < ndim
    assert freqs_cis.shape[:-1] == (x.shape[1], x.shape[-2])
    # New shape for broadcasting
    shape = [
        1 if i != 1 and i != ndim - 2 else d for i, d in enumerate(x.shape[:-1])
    ] + [2]
    return freqs_cis.view(*shape)


def apply_rotary_emb(q, k, freqs_cis):
    # q, k: (B, T, nh, hs)
    # freq_cis: (T, hs)
    # return: (B, T, nh, hs), (B, T, nh, hs)
    q = q.float().reshape(*q.shape[:-1], -1, 2)
    k = k.float().reshape(*k.shape[:-1], -1, 2)

    freqs_cis = _reshape_for_broadcast(freqs_cis, q)

    # Perform manual "complex" multiplication
    q_cos = q[..., 0] * freqs_cis[..., 0] - q[..., 1] * freqs_cis[..., 1]
    q_sin = q[..., 0] * freqs_cis[..., 1] + q[..., 1] * freqs_cis[..., 0]
    k_cos = k[..., 0] * freqs_cis[..., 0] - k[..., 1] * freqs_cis[..., 1]
    k_sin = k[..., 0] * freqs_cis[..., 1] + k[..., 1] * freqs_cis[..., 0]

    # Combine the results back into the interleaved format expected by q and k
    q_out = torch.stack((q_cos, q_sin), dim=-1).reshape(q.shape).flatten(3)
    k_out = torch.stack((k_cos, k_sin), dim=-1).reshape(k.shape).flatten(3)

    return q_out, k_out


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


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


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class SwissAIAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.n_head // self.num_key_value_heads
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
            self.num_key_value_heads * self.head_dim, 
            bias=self.bias,
            weight_quantizer=QUANTIZER_CLASSES[config.w_quant](**config.w_quant_kwargs),
            activation_quantizer=QUANTIZER_CLASSES[config.a_quant](**config.a_quant_kwargs)
        )
        
        self.v_proj = QuantizedLinear(
            config.n_embd, 
            self.num_key_value_heads * self.head_dim, 
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
            self.q_norm = RMSNorm(self.head_dim, eps=config.rmsnorm_eps)
            self.k_norm = RMSNorm(self.head_dim, eps=config.rmsnorm_eps)
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
        batch, num_key_value_heads, slen, head_dim = hidden_states.shape
        if n_rep == 1:
            return hidden_states
        hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

    def forward(self, x, freqs_cis, attention_mask=None):
        # batch size, sequence length, embedding dimensionality
        B, T, C = x.size()

        # Calculate query, key, values
        q = self.q_proj(x).view(B, T, self.n_head, self.head_dim)
        k = self.k_proj(x).view(B, T, self.num_key_value_heads, self.head_dim)
        v = self.v_proj(x).view(B, T, self.num_key_value_heads, self.head_dim)
        
        # Apply RMSNorm to query and key if qk_norm is enabled
        q = self.q_norm(q)
        k = self.k_norm(k)
        
        # Apply rotary embeddings
        q, k = apply_rotary_emb(q, k, freqs_cis)
        
        # Move head dimension to the front for attention calculation
        q = q.transpose(1, 2)  # (B, nh, T, hs)
        k = k.transpose(1, 2)  # (B, nkv, T, hs)
        v = v.transpose(1, 2)  # (B, nkv, T, hs)
        
        # Repeat k and v for multi-query attention
        if self.num_key_value_groups > 1:
            k = self.repeat_kv(k, self.num_key_value_groups)
            v = self.repeat_kv(v, self.num_key_value_groups)

        # Compute attention with either flash attention or the standard implementation
        if self.flash:
            # Efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=attention_mask, dropout_p=0.0 if not self.training else self.dropout, is_causal=True
            )
        else:
            # Manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * self.scaling
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
            att = F.dropout(att, p=0.0 if not self.training else self.dropout, training=self.training)
            y = att @ v  # (B, nh, T, hs) x (B, nh, T, hs) -> (B, nh, T, hs)
            
        # Reshape and apply output projection
        y = y.transpose(1, 2).contiguous().view(B, T, -1)
        y = self.o_proj(y)
        
        return y


class SwissAIMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_act = getattr(config, "hidden_act", "silu")
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
        
        self.attention_layernorm = RMSNorm(config.n_embd, eps=config.rmsnorm_eps)
        self.feedforward_layernorm = RMSNorm(config.n_embd, eps=config.rmsnorm_eps)
        
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
        self.freqs_cis = precompute_freqs_cis(self.head_dim, config.sequence_length)

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                drop=nn.Dropout(config.dropout),
                h=nn.ModuleList([SwissAIDecoderLayer(config, i) for i in range(config.n_layer)]),
                ln_f=RMSNorm(config.n_embd, eps=config.rmsnorm_eps),
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
        pos = torch.arange(0, t, dtype=torch.long, device=device)
        freqs_cis = self.freqs_cis.to(hidden_states.device)[pos]
        
        # Forward through transformer layers
        for layer in self.transformer.h:
            hidden_states = layer(hidden_states, freqs_cis)
        
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

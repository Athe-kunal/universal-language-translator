"""dInfer adapter for the dllm A2D Qwen3 MDLM checkpoints (dllm-hub/Qwen3-0.6B-diffusion-mdlm-*, and fine-tunes).

The A2D model is a plain Qwen3 decoder whose attention is bidirectional (padding-only mask), with no logit shift, so
dInfer's block-wise decoders can drive it directly once the model speaks dInfer's interface:

    out = model(input_ids, position_ids=..., past_key_values=KVCache | None, use_cache=bool, replace_position=(s, e))
    out.logits, out.past_key_values     # past_key_values: dinfer KVCache built from [k0, v0, k1, v1, ...]

With a KVCache, `input_ids` is only the current block; its keys/values are scattered into the cached full-length
keys/values at [replace_position[0]:replace_position[1]] and attention runs over the whole (cached) sequence, which is
dInfer's "dual cache". Plain torch only (no vLLM layers), so a 0.6B model needs no distributed/expert-parallel setup.

    model = Qwen3A2DModelLM.from_pretrained(ckpt_dir, dtype=torch.bfloat16, device="cuda")
Token ids for this family: mask 151669 (<|mask|>), eos 151645 (<|im_end|>); prompts use the tokenizer's chat template.
"""

import json
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file

MASK_ID, EOS_ID = 151669, 151645


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * x.to(dtype)


def _rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


class Attention(nn.Module):
    def __init__(self, cfg, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.n_heads, self.n_kv, self.head_dim = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
        h = cfg.hidden_size
        bias = cfg.attention_bias
        self.q_proj = nn.Linear(h, self.n_heads * self.head_dim, bias=bias)
        self.k_proj = nn.Linear(h, self.n_kv * self.head_dim, bias=bias)
        self.v_proj = nn.Linear(h, self.n_kv * self.head_dim, bias=bias)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, h, bias=bias)
        self.q_norm = RMSNorm(self.head_dim, cfg.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, cfg.rms_norm_eps)

    def forward(self, x, cos, sin, past_key_values, replace_position, use_cache):
        b, s, _ = x.shape
        q = self.q_norm(self.q_proj(x).view(b, s, self.n_heads, self.head_dim)).transpose(1, 2)
        k = self.k_norm(self.k_proj(x).view(b, s, self.n_kv, self.head_dim)).transpose(1, 2)
        v = self.v_proj(x).view(b, s, self.n_kv, self.head_dim).transpose(1, 2)
        q, k = q * cos + _rotate_half(q) * sin, k * cos + _rotate_half(k) * sin
        if past_key_values is not None:
            k, v = past_key_values.update(k, v, self.layer_idx, replace_position)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=False, enable_gqa=self.n_heads != self.n_kv)
        out = self.o_proj(out.transpose(1, 2).reshape(b, s, -1))
        return out, ((k, v) if use_cache else None)


class MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.gate_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.up_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.down_proj = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Layer(nn.Module):
    def __init__(self, cfg, layer_idx: int):
        super().__init__()
        self.self_attn = Attention(cfg, layer_idx)
        self.mlp = MLP(cfg)
        self.input_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)

    def forward(self, x, cos, sin, past_key_values, replace_position, use_cache):
        a, kv = self.self_attn(self.input_layernorm(x), cos, sin, past_key_values, replace_position, use_cache)
        x = x + a
        return x + self.mlp(self.post_attention_layernorm(x)), kv


class Backbone(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList(Layer(cfg, i) for i in range(cfg.num_hidden_layers))
        self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)


class H2Embed:
    """Iteration smoothing input: masked positions get a blend of the mask embedding and the expected embedding under the
    previous step's predicted distribution (same math as dinfer's H2Embed, without tensor-parallel/vLLM dependencies)."""

    def __init__(self, embedding: nn.Embedding, tau: float = 1.0):
        self.embedding, self.tau = embedding, tau

    def __call__(self, x, mask_index=None, logits=None, iter_cont_weight: float = 0.0):
        out = self.embedding(x)
        if mask_index is None or logits is None:
            return out
        expected = torch.softmax(logits / self.tau, dim=-1) @ self.embedding.weight
        return torch.where(mask_index.unsqueeze(-1), iter_cont_weight * expected + out, out)


class Qwen3A2DModelLM(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.config = cfg = SimpleNamespace(**config)
        self.model = Backbone(cfg)
        inv_freq = 1.0 / (cfg.rope_theta ** (torch.arange(0, cfg.head_dim, 2, dtype=torch.float32) / cfg.head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.h2e = H2Embed(self.model.embed_tokens)

    @property
    def device(self):
        return self.inv_freq.device

    @classmethod
    def from_pretrained(cls, path: str, dtype=torch.bfloat16, device="cuda") -> "Qwen3A2DModelLM":
        config = json.load(open(f"{path}/config.json"))
        assert config.get("tie_word_embeddings", True), "adapter assumes tied embeddings"
        model = cls(config).to(dtype).eval()
        missing, unexpected = model.load_state_dict(load_file(f"{path}/model.safetensors"), strict=False)
        assert not missing and not [k for k in unexpected if k != "lm_head.weight"], (missing, unexpected)
        return model.to(device)

    def forward(self, input_ids=None, attention_mask=None, position_ids=None, past_key_values=None, use_cache=None,
                replace_position=None, inputs_embeds=None, **kwargs):
        # attention_mask is ignored: the model is bidirectional and dInfer feeds unpadded / mask-padded blocks.
        h = inputs_embeds if inputs_embeds is not None else self.model.embed_tokens(input_ids)
        if position_ids is None:
            start = replace_position[0] if replace_position is not None else 0
            position_ids = torch.arange(start, start + h.shape[1], device=h.device).unsqueeze(0)
        freqs = position_ids[:, None, :, None].float() * self.inv_freq  # [b, 1, s, d/2]
        emb = torch.cat((freqs, freqs), dim=-1)
        cos, sin = emb.cos().to(h.dtype), emb.sin().to(h.dtype)
        cache = []
        for layer in self.model.layers:
            h, kv = layer(h, cos, sin, past_key_values, replace_position, use_cache)
            if use_cache:
                cache.extend(kv)
        logits = F.linear(self.model.norm(h), self.model.embed_tokens.weight)
        if use_cache:
            from dinfer.decoding.utils import KVCache  # lazy: the plain forward needs no dinfer install

            return SimpleNamespace(logits=logits, past_key_values=KVCache(cache))
        return SimpleNamespace(logits=logits, past_key_values=None)

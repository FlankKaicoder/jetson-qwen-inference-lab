from __future__ import annotations

import torch
from torch import nn


class RMSNorm(nn.Module):
    def __init__(self, size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        y = x.float()
        y = y * torch.rsqrt(y.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * y.to(dtype)


class PortableQwen3Layer(nn.Module):
    """Source-faithful Qwen3 layer with optional audit tensors."""

    def __init__(self, hidden=1024, q_heads=16, kv_heads=8, head_dim=128,
                 intermediate=3072, theta=1_000_000.0, eps=1e-6):
        super().__init__()
        self.q_heads = q_heads
        self.kv_heads = kv_heads
        self.head_dim = head_dim
        self.groups = q_heads // kv_heads
        self.theta = theta
        self.input_layernorm = RMSNorm(hidden, eps)
        self.self_attn = nn.Module()
        self.self_attn.q_proj = nn.Linear(hidden, q_heads * head_dim, bias=False)
        self.self_attn.k_proj = nn.Linear(hidden, kv_heads * head_dim, bias=False)
        self.self_attn.v_proj = nn.Linear(hidden, kv_heads * head_dim, bias=False)
        self.self_attn.o_proj = nn.Linear(q_heads * head_dim, hidden, bias=False)
        self.self_attn.q_norm = RMSNorm(head_dim, eps)
        self.self_attn.k_norm = RMSNorm(head_dim, eps)
        self.post_attention_layernorm = RMSNorm(hidden, eps)
        self.mlp = nn.Module()
        self.mlp.gate_proj = nn.Linear(hidden, intermediate, bias=False)
        self.mlp.up_proj = nn.Linear(hidden, intermediate, bias=False)
        self.mlp.down_proj = nn.Linear(intermediate, hidden, bias=False)

    def _rope(self, x, pos):
        inv = torch.pow(
            torch.tensor(self.theta, device=x.device, dtype=torch.float32),
            -torch.arange(self.head_dim // 2, device=x.device, dtype=torch.float32)
            / (self.head_dim // 2),
        )
        freq = pos.float().unsqueeze(-1) * inv
        emb = torch.cat((freq, freq), dim=-1)
        cos = emb.cos().unsqueeze(1).to(x.dtype)
        sin = emb.sin().unsqueeze(1).to(x.dtype)
        half = x.shape[-1] // 2
        rot = torch.cat((-x[..., half:], x[..., :half]), dim=-1)
        return x * cos + rot * sin

    def _run(self, hidden, pos, past_k=None, past_v=None):
        residual = hidden
        x = self.input_layernorm(hidden)
        b, s, _ = x.shape
        q = self.self_attn.q_norm(
            self.self_attn.q_proj(x).view(b, s, self.q_heads, self.head_dim)
        ).transpose(1, 2)
        k = self.self_attn.k_norm(
            self.self_attn.k_proj(x).view(b, s, self.kv_heads, self.head_dim)
        ).transpose(1, 2)
        v = self.self_attn.v_proj(x).view(b, s, self.kv_heads, self.head_dim).transpose(1, 2)
        q = self._rope(q, pos)
        k = self._rope(k, pos)
        if past_k is not None:
            k = torch.cat((past_k, k), dim=2)
            v = torch.cat((past_v, v), dim=2)
        ka = k[:, :, None].expand(b, self.kv_heads, self.groups, k.shape[2], self.head_dim)
        ka = ka.reshape(b, self.q_heads, k.shape[2], self.head_dim)
        va = v[:, :, None].expand(b, self.kv_heads, self.groups, v.shape[2], self.head_dim)
        va = va.reshape(b, self.q_heads, v.shape[2], self.head_dim)
        scores = torch.matmul(q, ka.transpose(-2, -1)) * (self.head_dim ** -0.5)
        if past_k is None:
            mask = torch.triu(torch.ones((s, s), device=x.device, dtype=torch.bool), diagonal=1)
            scores = scores.masked_fill(mask, torch.finfo(scores.dtype).min)
        attn = torch.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_out = torch.matmul(attn, va).transpose(1, 2).contiguous().reshape(b, s, -1)
        attn_out = self.self_attn.o_proj(attn_out)
        x = residual + attn_out
        residual = x
        x = self.post_attention_layernorm(x)
        x = self.mlp.down_proj(torch.nn.functional.silu(self.mlp.gate_proj(x)) * self.mlp.up_proj(x))
        return residual + x, k, v, attn_out

    def forward_prefill(self, hidden, pos):
        return self._run(hidden, pos)

    def forward_decode(self, hidden, pos, past_k, past_v):
        return self._run(hidden, pos, past_k, past_v)


class PortableFourLayerStack(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([PortableQwen3Layer() for _ in range(4)])

    def forward_prefill(self, hidden, pos):
        hidden_outs, ks, vs, attns = [], [], [], []
        for layer in self.layers:
            hidden, k, v, attn = layer.forward_prefill(hidden, pos)
            hidden_outs.append(hidden)
            ks.append(k)
            vs.append(v)
            attns.append(attn)
        return hidden_outs, ks, vs, attns

    def forward_decode(self, hidden, pos, past_ks, past_vs):
        hidden_outs, ks, vs, attns = [], [], [], []
        for i, layer in enumerate(self.layers):
            hidden, k, v, attn = layer.forward_decode(hidden, pos, past_ks[i], past_vs[i])
            hidden_outs.append(hidden)
            ks.append(k)
            vs.append(v)
            attns.append(attn)
        return hidden_outs, ks, vs, attns

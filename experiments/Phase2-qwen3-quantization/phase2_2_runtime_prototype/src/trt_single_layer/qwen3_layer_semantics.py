from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

CONFIG = {
    "hidden_size": 1024,
    "q_heads": 16,
    "kv_heads": 8,
    "head_dim": 128,
    "intermediate_size": 3072,
    "rope_theta": 1_000_000.0,
    "rms_norm_eps": 1e-6,
}


class Qwen3RMSNorm(nn.Module):
    def __init__(self, size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        y = x.float()
        y = y * torch.rsqrt(torch.mean(y.pow(2), dim=-1, keepdim=True) + self.eps)
        return self.weight * y.to(input_dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def rope(x: torch.Tensor, position_ids: torch.Tensor, theta: float) -> torch.Tensor:
    half = x.shape[-1] // 2
    idx = torch.arange(half, device=x.device, dtype=torch.float32)
    inv = torch.pow(torch.tensor(theta, device=x.device, dtype=torch.float32), -idx / half)
    angles = position_ids.float().unsqueeze(-1) * inv
    # Transformers applies cos/sin across the full head dimension; frequencies mirror across halves.
    cos = torch.cat((torch.cos(angles), torch.cos(angles)), dim=-1).unsqueeze(1).to(x.dtype)
    sin = torch.cat((torch.sin(angles), torch.sin(angles)), dim=-1).unsqueeze(1).to(x.dtype)
    return x * cos + rotate_half(x) * sin


class Qwen3SingleLayer(nn.Module):
    """Faithful single-layer semantic reference with explicit cache tensors."""

    def __init__(self, config=CONFIG):
        super().__init__()
        h, qh, kh, d, inter = (config["hidden_size"], config["q_heads"], config["kv_heads"], config["head_dim"], config["intermediate_size"])
        self.q_heads, self.kv_heads, self.head_dim, self.theta = qh, kh, d, config["rope_theta"]
        self.norm1 = Qwen3RMSNorm(h, config["rms_norm_eps"])
        self.q_proj = nn.Linear(h, qh * d, bias=False)
        self.k_proj = nn.Linear(h, kh * d, bias=False)
        self.v_proj = nn.Linear(h, kh * d, bias=False)
        self.q_norm = Qwen3RMSNorm(d, config["rms_norm_eps"])
        self.k_norm = Qwen3RMSNorm(d, config["rms_norm_eps"])
        self.o_proj = nn.Linear(qh * d, h, bias=False)
        self.norm2 = Qwen3RMSNorm(h, config["rms_norm_eps"])
        self.gate_proj = nn.Linear(h, inter, bias=False)
        self.up_proj = nn.Linear(h, inter, bias=False)
        self.down_proj = nn.Linear(inter, h, bias=False)

    def project_kv(self, x: torch.Tensor, position_ids: torch.Tensor):
        b, s, _ = x.shape
        shape_k = (b, s, self.kv_heads, self.head_dim)
        k = self.k_norm(self.k_proj(x).view(shape_k)).transpose(1, 2)
        v = self.v_proj(x).view(shape_k).transpose(1, 2)
        return rope(k, position_ids, self.theta), v

    def forward_prefill(self, hidden_states: torch.Tensor, position_ids: torch.Tensor):
        residual = hidden_states
        x = self.norm1(hidden_states)
        b, s, _ = x.shape
        q = self.q_norm(self.q_proj(x).view(b, s, self.q_heads, self.head_dim)).transpose(1, 2)
        k, v = self.project_kv(x, position_ids)
        q = rope(q, position_ids, self.theta)
        rep = self.q_heads // self.kv_heads
        k_attn = k[:, :, None].expand(b, self.kv_heads, rep, s, self.head_dim).reshape(b, self.q_heads, s, self.head_dim)
        v_attn = v[:, :, None].expand(b, self.kv_heads, rep, s, self.head_dim).reshape(b, self.q_heads, s, self.head_dim)
        scores = torch.matmul(q, k_attn.transpose(-2, -1)) * (self.head_dim ** -0.5)
        mask = torch.triu(torch.ones((s, s), device=x.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(mask, torch.finfo(scores.dtype).min)
        out = torch.matmul(torch.softmax(scores, dim=-1), v_attn).transpose(1, 2).reshape(b, s, -1)
        x = residual + self.o_proj(out)
        residual = x
        x = self.norm2(x)
        x = self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
        return residual + x, k, v

    def forward_decode(self, hidden_states: torch.Tensor, position_ids: torch.Tensor, past_k: torch.Tensor, past_v: torch.Tensor):
        residual = hidden_states
        x = self.norm1(hidden_states)
        b = x.shape[0]
        q = self.q_norm(self.q_proj(x).view(b, 1, self.q_heads, self.head_dim)).transpose(1, 2)
        k_new, v_new = self.project_kv(x, position_ids)
        q = rope(q, position_ids, self.theta)
        present_k = torch.cat((past_k, k_new), dim=2)
        present_v = torch.cat((past_v, v_new), dim=2)
        rep = self.q_heads // self.kv_heads
        k_attn = present_k[:, :, None].expand(b, self.kv_heads, rep, present_k.shape[2], self.head_dim).reshape(b, self.q_heads, present_k.shape[2], self.head_dim)
        v_attn = present_v[:, :, None].expand(b, self.kv_heads, rep, present_v.shape[2], self.head_dim).reshape(b, self.q_heads, present_v.shape[2], self.head_dim)
        out = torch.matmul(torch.softmax(torch.matmul(q, k_attn.transpose(-2, -1)) * (self.head_dim ** -0.5), dim=-1), v_attn).transpose(1, 2).reshape(b, 1, -1)
        x = residual + self.o_proj(out)
        residual = x
        x = self.norm2(x)
        x = self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
        return residual + x, present_k, present_v


def make_layer(seed: int = 20260902, device: str = "cpu", dtype: torch.dtype = torch.float16):
    torch.manual_seed(seed)
    return Qwen3SingleLayer().to(device=device, dtype=dtype).eval()

import torch
from torch import nn


CONFIG = {
    "hidden_size": 1024,
    "num_hidden_layers": 28,
    "num_attention_heads": 16,
    "num_key_value_heads": 8,
    "head_dim": 128,
    "intermediate_size": 3072,
    "rope_theta": 1000000.0,
    "rms_norm_eps": 1.0e-6,
    "sequence_length": 8,
}


class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x):
        return x * torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps) * self.weight


class Qwen3LikeDecoderBlock(nn.Module):
    """Single synthetic Qwen3-like decoder block; no checkpoint weights are loaded."""

    def __init__(self, config=CONFIG):
        super().__init__()
        h = config["hidden_size"]
        q = config["num_attention_heads"] * config["head_dim"]
        kv = config["num_key_value_heads"] * config["head_dim"]
        inter = config["intermediate_size"]
        self.num_heads = config["num_attention_heads"]
        self.num_kv_heads = config["num_key_value_heads"]
        self.head_dim = config["head_dim"]
        self.rope_theta = config["rope_theta"]
        self.norm1 = RMSNorm(h, config["rms_norm_eps"])
        self.q_proj = nn.Linear(h, q, bias=False)
        self.k_proj = nn.Linear(h, kv, bias=False)
        self.v_proj = nn.Linear(h, kv, bias=False)
        self.o_proj = nn.Linear(q, h, bias=False)
        self.norm2 = RMSNorm(h, config["rms_norm_eps"])
        self.gate_proj = nn.Linear(h, inter, bias=False)
        self.up_proj = nn.Linear(h, inter, bias=False)
        self.down_proj = nn.Linear(inter, h, bias=False)

    def _rope(self, x):
        # x: [B, heads, S, D], with fixed S=8 for this feasibility graph.
        half = self.head_dim // 2
        idx = torch.arange(half, device=x.device, dtype=torch.float32)
        inv_freq = torch.pow(torch.tensor(self.rope_theta, device=x.device, dtype=torch.float32), -idx / half)
        pos = torch.arange(x.shape[2], device=x.device, dtype=torch.float32)
        angles = pos[:, None] * inv_freq[None, :]
        cos = torch.cos(angles).to(dtype=x.dtype)[None, None, :, :]
        sin = torch.sin(angles).to(dtype=x.dtype)[None, None, :, :]
        first, second = x[..., :half], x[..., half:]
        return torch.cat((first * cos - second * sin, first * sin + second * cos), dim=-1)

    def forward(self, hidden_states):
        residual = hidden_states
        x = self.norm1(hidden_states)
        b, s, _ = x.shape
        q = self.q_proj(x).reshape(b, s, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).reshape(b, s, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).reshape(b, s, self.num_kv_heads, self.head_dim).transpose(1, 2)
        q, k = self._rope(q), self._rope(k)
        repeat = self.num_heads // self.num_kv_heads
        k = k.repeat_interleave(repeat, dim=1)
        v = v.repeat_interleave(repeat, dim=1)
        scores = torch.matmul(q, k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        causal = torch.triu(torch.ones((s, s), device=x.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(causal, torch.finfo(scores.dtype).min)
        attn = torch.softmax(scores, dim=-1)
        attn_out = torch.matmul(attn, v).transpose(1, 2).reshape(b, s, -1)
        x = residual + self.o_proj(attn_out)
        residual = x
        x = self.norm2(x)
        x = self.down_proj(torch.nn.functional.silu(self.gate_proj(x)) * self.up_proj(x))
        return residual + x


def make_block(seed=0, device="cuda", dtype=torch.float16):
    torch.manual_seed(seed)
    block = Qwen3LikeDecoderBlock().to(device=device, dtype=dtype).eval()
    return block

Source: transformers.models.qwen3.modeling_qwen3.py, Transformers 4.57.3.
Qwen3RMSNorm casts to float32 for pow/mean/rsqrt then casts back.
Qwen3Attention projects Q/K/V; Q/K RMSNorm runs on head-shaped tensors; RoPE is applied before cache.update().
DynamicCache.update() stores post-RoPE K and projected V in [B,KV_heads,L,head_dim]. repeat_kv occurs for attention consumption, not cache storage.
Decoder ordering: input RMSNorm -> attention -> residual -> post-attention RMSNorm -> SwiGLU MLP -> residual.

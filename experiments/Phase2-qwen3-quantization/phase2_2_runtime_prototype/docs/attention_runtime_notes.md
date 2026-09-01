# Attention Runtime Notes

## Grouped-query attention

Qwen3-0.6B has 16 query heads and 8 K/V heads. Two query heads conceptually share each K/V head. A runtime should preserve the compact 8-head K/V cache rather than materializing duplicated K/V values merely to match the number of query heads; the exact engine implementation remains to be designed.

## KV cache reuse

During prompt prefill, every layer projects K and V for every prompt token. In subsequent decode, prior keys and values are reused while only one new K/V token is produced and appended. Recomputing the whole prompt's K/V projections at each decode step defeats the cache's purpose.

## Prefill versus decode

Prefill is shaped `[B,S]`, has many queries and creates an initial cache span. Decode is shaped `[B,1]`, has a single new query but reads a cache whose length grows with context. These distinct shapes, cache lifetimes and tactic needs motivate separate optimization profiles or engines.

## Why decode is memory-bandwidth sensitive

For each generated token, every layer must read cached K/V across the existing sequence even though it produces only one new token. Arithmetic per cache byte therefore falls as the sequence grows, and cache layout, coalescing, allocation locality and bandwidth become important. This is a mechanism-oriented expectation, not a benchmark conclusion for this device.

No FlashAttention implementation or performance experiment is included in this preparation.

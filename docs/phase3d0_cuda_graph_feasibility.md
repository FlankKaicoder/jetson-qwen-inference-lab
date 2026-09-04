# Phase 3-D0 CUDA Graph Feasibility

- Date: 2026-09-04
- Branch: `phase/03d0-cuda-graph-feasibility`
- Starting checkpoint: `93e799fb2002f7b9884a9e3867ce93faf23cc173`
- Code commits: `2eadfd4869ed3bcfd1b491659716322af5d4c1b6`,
  `4bf561c9aae8c388eea11584f24706c88bfc1c45`,
  `6ba4e1ef9cde65eaa503aed895caae8e7c661135`
- Final Gate: `BLOCKED / BOUNDED`
- Classification: `DISPROVEN` for the current full-window and per-engine
  prototype paths; `UNKNOWN` for a redesigned runtime or native TensorRT graph
  integration.

## 1. Question and Scope

The question was narrowly scoped: can the current persistent Qwen3 TensorRT
runtime capture and replay the steady-state decode window with valid outputs and
reduce host-side launch or synchronization overhead?

Only CUDA Graph feasibility was authorized. No CUDA operator, TensorRT plugin,
attention kernel, RMSNorm kernel, quantization redesign, or runtime rewrite was
performed. The persistent stream path remained the reference.

## 2. Environment and Evidence State

- Device: Jetson Orin Nano Super, SM 8.7, batch 1.
- Software: CUDA 12.6, TensorRT 10.3.0, NVIDIA PyTorch
  `2.5.0a0+872d972e41.nv24.08`, Nsight Systems 2024.5.4.
- Power mode: `NV Power Mode: 25W`; clocks were not modified.
- Runtime: FP16 embedding, 28-layer decode, final RMSNorm and LM Head engines.
- Workload: deterministic first evaluation prompt, S=8 prefill, forced
  deterministic continuation tokens, CPU sampling outside graph.
- Primary evidence root:
  `results/phase3d0_cuda_graph/20260904T094952Z/`.
- Raw `.nsys-rep` files remain Jetson-local under `/tmp`; compact CSV/JSON
  summaries and hashes are committed.

## 3. D0-A Compatibility Audit

The audit found a bounded-capture shape: fixed decode steps can have fixed token,
position, hidden, logits and KV tensor addresses. CPU sampling must stay outside
the graph. Per-step output tensors can become the next step's KV inputs without
pointer overlap.

The static compatibility matrix therefore did not rule out a bounded decode
graph. The decisive failure occurred at execution: TensorRT work enqueued during
`torch.cuda.CUDAGraph` capture was not represented in replay. This is recorded as
an interop/runtime failure rather than a topology failure.

D0-A Gate: `PASS`. Capture boundaries were identified before implementation.

## 4. D0-B Capture Prototypes and Validation

Two graph modes were tested:

1. `full_window`: one `torch.cuda.CUDAGraph` around all four engine enqueues for
   a bounded decode window.
2. `per_engine`: one graph per TensorRT engine enqueue, replayed in sequence.

All runs used the same prompt, forced tokens, engines and persistent contexts.
The graph outputs were compared with the same-session persistent-stream outputs.
The validation gate requires finite outputs, exact KV equality, exact KV shapes
and a topology where each step's output addresses become the next step's inputs.

| Prototype | Steps | Captured graphs | Validation | Stream, ms | Graph, ms | Hidden rel-L2 | Logits rel-L2 | KV exact |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- |
| full-window | 1 | 1 | `BLOCKED` | 58.374672 | 22.299647 | 2.948851 | 1.000000 | false |
| full-window | 2 | 1 | `BLOCKED` | 95.679203 | 59.816322 | 0.676115 / 0.656258 | 1.000000 / 1.000000 | false |
| per-engine | 1 | 4 | `BLOCKED` | 61.595694 | 43.655578 | 2.948851 | 1.000000 | false |

Topology and address checks passed: fixed addresses were maintained after replay
and the cache chain was pointer-isolated. Functional validation failed. The
faster wall time is therefore invalid and is not a performance result.

D0-B Gate: `BLOCKED`.

## 5. D0-C Formal Benchmark Decision

The formal Phase 3-D0-C benchmark is
`BLOCKED_NO_VALID_GRAPH_PATH`.

No formal `PASS` benchmark table is presented because neither graph path produced
semantically valid outputs. Warmup 5 / repeats 10 measurements would only
characterize an invalid replay path. The persistent-stream baseline remains
available in the Phase 3-C report; it was not changed.

D0-C Gate: `BLOCKED`.

## 6. D0-D Nsight Systems A/B

Separate 8-step FP16 NSYS traces were collected for persistent stream and
per-engine graph replay with
`-t cuda,nvtx --sample=none --cpuctxsw=none`.

| Steady NVTX window metric | Persistent stream | CUDA graph replay |
| --- | ---: | ---: |
| Window wall, ms | 349.363232 | 193.934880 |
| Projected GPU operations | 3,392 | 64 |
| Kernel instances | 3,376 | 64 |
| Kernel time, ms | 217.708032 | 0.298048 |
| Memcpy operations | 16 | 0 |
| Memcpy time, ms | 0.019776 | 0 |

Persistent stream contained the expected TensorRT GEMM and Myelin kernels. The
graph-replay window contained only 64 PyTorch `FillFunctor<long>` kernels; it
contained no TensorRT kernels. This directly explains why graph replay was faster
but wrong: it was not replaying the decoder.

The same filtered windows had these CUDA API observations:

| Metric | Persistent stream | CUDA graph replay |
| --- | ---: | ---: |
| CUDA API calls | 3,484 | 151 |
| CUDA API time, ms | 197.225408 | 196.100256 |
| Kernel-launch API calls | 3,376 | 60 |
| Kernel-launch API time, ms | 44.973408 | 1.355904 |
| Synchronization calls | 47 | 1 |
| Synchronization time, ms | 144.971968 | 42.217056 |
| `cudaGraphLaunch` calls | 0 | 30 |
| `cudaGraphLaunch` time, ms | 0 | 152.393376 |

The API count and explicit kernel-launch overhead fell, but `cudaGraphLaunch`
dominated the graph-path API time. Since the GPU operations were absent and
validation failed, this is not evidence of a useful CUDA Graph optimization.

D0-D Gate: `BLOCKED`; profiler evidence is diagnostic only.

## 7. Classification and Recommendation

For the current implementation, CUDA Graph is classified
`DISPROVEN`: it neither preserved TensorRT execution nor passed correctness. The
more limited statement is also true: this does not disprove all possible CUDA
Graph designs.

Do not start Phase 3-D1 CUDA Graph runtime optimization. A future attempt would
require explicit authorization and a different boundary, likely native TensorRT
graph support or a redesigned fixed-shape runtime. It must not reuse the current
PyTorch capture result as evidence of speedup.

## 8. Limitations

- Only FP16, batch 1, one deterministic prompt and S=8 prefill were tested.
- Mixed precision was not tested because there was no valid graph path.
- The formal benchmark was intentionally not run after both capture paths failed.
- The 8-step profile is a short diagnostic window, not a long-tail profile.
- NVTX projection may fail to correlate graph-replayed TensorRT kernels to NVTX
  ranges; the kernel-count and kernel-time difference is the stronger evidence.
- The test used PyTorch CUDA Graph capture around `execute_async_v3`. Native
  TensorRT graph capture and engine/runtime redesign remain `UNKNOWN`.
- TensorRT default-stream warnings and `cudaGraphLaunch` overhead were preserved
  in evidence rather than tuned away.

## 9. Reproduction Commands

From the Jetson repository root:

```bash
python3 experiments/Phase2-qwen3-quantization/src/phase3d0/phase3d0_cuda_graph.py \
  --mode audit \
  --runtime fp16 \
  --out results/phase3d0_cuda_graph/<timestamp>/audit

python3 experiments/Phase2-qwen3-quantization/src/phase3d0/phase3d0_cuda_graph.py \
  --mode bench --runtime fp16 --decode-steps 1 --warmup 0 --repeats 1 \
  --graph-mode per_engine \
  --out results/phase3d0_cuda_graph/<timestamp>/per_engine_smoke

nsys profile -t cuda,nvtx --sample=none --cpuctxsw=none \
  -o /tmp/phase3d0_stream_<unique> \
  python3 experiments/Phase2-qwen3-quantization/src/phase3d0/phase3d0_cuda_graph.py \
  --mode profile --runtime fp16 --decode-steps 8 --warmup 5 --repeats 10 \
  --out results/phase3d0_cuda_graph/<timestamp>/stream_profile

nsys profile -t cuda,nvtx --sample=none --cpuctxsw=none \
  -o /tmp/phase3d0_per_engine_<unique> \
  python3 experiments/Phase2-qwen3-quantization/src/phase3d0/phase3d0_cuda_graph.py \
  --mode profile --runtime fp16 --decode-steps 8 --warmup 5 --repeats 10 \
  --graph-mode per_engine --graph \
  --out results/phase3d0_cuda_graph/<timestamp>/graph_profile
```

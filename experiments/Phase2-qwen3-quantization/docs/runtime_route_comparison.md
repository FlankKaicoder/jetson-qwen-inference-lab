# Runtime Route Comparison

This comparison is an architecture decision record for the current Jetson Orin SM87, CUDA 12.6, TensorRT 10.3 and NVIDIA PyTorch 2.5.0a0 environment. It does not install or build any runtime.

| Route | Qwen3 evidence | Current feasibility | Main work | Risk | Decision |
| --- | --- | --- | --- | --- | --- |
| Native TensorRT + custom runtime | Synthetic decoder block parses/builds/executes; full model not tested | `PARTIAL` | Full export, cache manager, prefill/decode scheduling, sampling and memory ownership | High | Best next design target, not production-ready |
| TensorRT-LLM | Latest source supports Qwen3 but official tested hardware/software intersection excludes current Orin stack; Jetson v0.12 route lacks Qwen3 model registration | `BLOCKED_NEEDS_RUNTIME_WORK` | Hardware/version reconciliation or a maintained Qwen3 backport | Very high | Do not install or modify stack |
| HF TensorRT backend | No repository evidence of a compatible backend on this Jetson stack | `UNKNOWN` | Backend-specific compatibility audit and cache integration | High | Not selected without evidence |
| HF PyTorch BF16 reference | Phase 1.1/1.2 closed and fully functional | `SUPPORTED_REFERENCE` | None for reference use | Medium memory footprint | Preserve as correctness oracle |

Native TensorRT offers the clearest learning path and can reuse the proven parser/build plumbing, but it requires a real autoregressive runtime. TensorRT-LLM offers mature cache and scheduling abstractions but has a documented current-platform intersection problem. The HF BF16 path remains the reference oracle for any later engine comparison.

# Phase 4-G Optimization Hypothesis Report

## Gate

**Phase 4-G: `PASS / BOUNDED`**

There is a bounded candidate direction: reduce or reshape memory/L2 traffic in
the shared h16816 GEMM path. However, there is **no proven optimization
hypothesis** and no closed `problem -> cause -> optimization method -> expected
benefit` chain. Therefore CUDA feasibility is `NOT READY`.

## Evidence inventory

| Evidence | Supports | Does not support |
| --- | --- | --- |
| Phase 3-E NCU rank-1 h16816 | Memory/L2 97.06%, HMMA 39.673148%, SM 39.57%, achieved occupancy 24.78% | Pure up_proj attribution, custom-kernel superiority, or direct DRAM throughput |
| Phase 4-E IProfiler | 28/28 up_proj layer timing; per-layer median 153.28-159.76 us | Kernel identity or production benchmark conclusion |
| Phase 4-F runtime correlation | 28/28 up_proj ranges, 28/28 layers, 196 launch rows, 7 launches/range | Future invocation stability or one-range-to-one-invocation proof |
| EngineInspector + ONNX | up_proj shape `C[1,3072] = A[1,1024] * B[1024,3072]` and 28 HIGH-confidence layers | Runtime kernel identity or exact GEMM semantics |
| Phase 3-E aggregate NSYS | h16816 was 39.590976 ms in Mixed runtime | Splitting shared `gate_proj;up_proj` time into a pure up_proj share |

## Kernel breakdown

All 196 observed up_proj-linked launches are GEMM candidates:

| Kernel | Count | Ranges | Operator | Confidence |
| --- | ---: | ---: | --- | --- |
| `trt_ampere_h16816gemm_128x64_ldg8_nn_v1` | 78 | 28 | `up_proj` | `HIGH` |
| `trt_ampere_h16816gemm_256x64_ldg8_tn_v1` | 6 | 28 | `up_proj` | `HIGH` |
| `sm80_xmma_gemm_f16f16_f16f16_f16_...execute_kernel_trt` | 100 | 28 | `up_proj` | `HIGH` |
| `sm80_xmma_gemm_f16f16_f16f32_f32_...execute_kernel_trt` | 12 | 28 | `up_proj` | `HIGH` |

The two h16816 variants total `84/196` launches. The two `sm80_xmma_gemm_*`
variants total `112/196`. Every observed range contains more than one kernel
name, but the per-logical-invocation distribution remains `UNKNOWN`.

Observed average kernel duration by name, derived only from the Phase 4-F
aggregate durations and launch counts, is approximately:

| Kernel | Mean duration/launch |
| --- | ---: |
| `trt_ampere_h16816gemm_128x64_ldg8_nn_v1` | `219.030 us` |
| `trt_ampere_h16816gemm_256x64_ldg8_tn_v1` | `194.720 us` |
| `sm80_xmma_gemm_f16f16_f16f16_f16_...` | `172.294 us` |
| `sm80_xmma_gemm_f16f16_f16f32_f32_...` | `172.669 us` |

## Bottleneck analysis

**Q1: What is the current bottleneck?**  
`EVIDENCE_INSUFFICIENT` for a pure up_proj conclusion.

The strongest hardware-counter signal is memory/L2 pressure in the shared
h16816 GEMM path: Phase 3-E reports `97.06%` memory/L2, `39.673148%` HMMA
active, and `24.78%` achieved occupancy against a `25%` theoretical limit. But
that NCU profile maps to `gate_proj;up_proj`, not pure `up_proj`, and direct
DRAM throughput is `N/A`.

**Q2: What would a CUDA kernel optimize?**  
The only evidence-supported candidate direction is to reduce or reshape
memory/L2 traffic. This is not yet a proven implementation direction because
the exact traffic source, pure up_proj share, layout semantics, accumulator
semantics, and same-shape alternative are unknown.

**Q3: Is Phase 5 CUDA feasibility authorized?**  
`NOT READY`.

## Candidate hypotheses

| ID | Hypothesis | Confidence | Expected benefit | Missing evidence |
| --- | --- | --- | --- | --- |
| H1 | Reduce or reshape memory/L2 traffic in h16816 | `MEDIUM_FOR_BOTTLENECK_DIRECTION` | `UNKNOWN` | Pure up_proj NCU, direct DRAM, traffic source, exact semantics, alternative baseline |
| H2 | Fuse GEMM with an adjacent operation | `LOW` | `UNKNOWN` | Adjacent operator identity, dependency graph, intermediate traffic, measured fusion opportunity |
| H3 | Reduce launch overhead | `LOW` | `UNKNOWN` | CPU launch API time, GPU gaps, per-invocation launch count |
| H4 | TensorRT is already optimal | `UNKNOWN` | `UNKNOWN` | Same-shape alternative, controlled benchmark, numerical-equivalence evidence |

No new optimization mechanism was invented. Each hypothesis is bounded by the
existing evidence.

## Confidence assessment

- `H1` is the strongest bottleneck-direction signal but only reaches
  `MEDIUM_FOR_BOTTLENECK_DIRECTION`; it is not a proven CUDA target.
- `H2` and `H3` are `LOW` because the current evidence does not establish the
  required adjacent operation or launch-bound behavior.
- `H4` cannot be claimed because there is no alternative comparison.
- The Phase 4-F correlation is `HIGH` for observed-launch operator attribution,
  but it does not prove future stability or per-invocation kernel identity.

## CUDA readiness decision

**`NOT_READY`**

The evidence does not prove that TensorRT's up_proj kernel has a correctable
performance defect, that a CUDA/CUTLASS/Plugin replacement would win, or that
h16816 has exploitable optimization space. The memory/L2-heavy signal is a
candidate direction, not a proven optimization hypothesis.

## Remaining unknowns

- Pure `up_proj` NCU behavior, separated from `gate_proj`.
- Direct DRAM traffic and the split between activation, weight, output,
  workspace and other memory movement.
- Why each observed range contains both h16816 and `sm80_xmma_gemm_*` kernels.
- Whether one NVTX range corresponds exactly to one logical up_proj invocation.
- Whether the observed kernel distribution is stable in real-token decode.
- Exact accumulator dtype, operand/output layout, `tn_n` semantics and
  alpha/beta semantics.
- Whether fusion, split reduction, or another method would preserve numerics
  and improve performance.

## Recommendation

Do not enter Phase 5 CUDA feasibility yet.

If a further evidence phase is authorized, the narrow next step should resolve
the remaining attribution boundary first: pure up_proj NCU and a stable
per-logical-invocation kernel distribution under a controlled real-token decode
workload. Only then can a concrete optimization hypothesis be evaluated.

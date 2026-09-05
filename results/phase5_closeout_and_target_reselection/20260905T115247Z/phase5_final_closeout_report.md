# Phase 5 Final Closeout: GEMM Investigation Freeze

## Final Gate

```text
PASS / BOUNDED / NO_PROVEN_OPTIMIZATION_TARGET
NO_PROVEN_CUDA_GEMM_OPTIMIZATION_TARGET
NO_PROVEN_TACTIC_DEFECT
EVENT_TIME_GAP_INCONCLUSIVE
TENSORRT_BACKEND_IDENTITY_UNKNOWN
NO_CUDA_IMPLEMENTATION_AUTHORIZED
```

The PASS component means the authorized feasibility question was closed with a
complete bounded evidence chain. The BOUNDED component reflects historical
timing boundaries, one-launch NCU samples, no numeric TensorRT tactic ID, and
unknown backend identity. It is not a claim that no possible optimization
exists anywhere.

## Question

Does the TensorRT decode-only `up_proj` FP16 GEMM,
`C[1,3072] = A[1,1024] * B[1024,3072]`, contain a proven performance space
worth pursuing through custom CUDA GEMM implementation?

## Evidence Chain

| Stage | Result | Primary artifact |
| --- | --- | --- |
| Phase 4 bottleneck attribution | `up_proj` was the first research target; shape frozen for 28 layers | `results/phase4b_target_selection/20260904T140726Z/` |
| TensorRT layer timing | Per-layer `IProfiler` median `153.280005-159.759998 us` | `results/phase4e_up_proj_baseline/20260904T161320Z/` |
| Kernel attribution | 196 correlated instances; `h16816=84`, `sm80_xmma=112` | `results/phase4f_kernel_attribution/20260904T164404Z/` |
| cuBLASLt baseline | Algorithm 21, FP32 accumulate, workspace 0, median `0.080077961 ms`, correctness PASS | `results/phase5a_cuda_feasibility_baseline/20260905T063059Z/` |
| CUTLASS feasibility | Best `0.083619133 ms`, about 4.42% slower than cuBLASLt | same directory |
| Boundary reconciliation | Historical steady-state correlated kernel median `147.424 us`; no attributed non-GEMM kernel | `results/phase5a_cuda_feasibility_baseline/20260905T072100Z/` |
| Inspector tactic attribution | 28/28 tactic strings recovered; numeric tactic ID `NOT_AVAILABLE` | `results/phase5b_tensorrt_gemm_path_investigation/20260905T103352Z/` |
| Matched NCU comparison | cuBLASLt `242.912 us`; TensorRT xmma `244.160 us` under `--clock-control none` | `results/phase5b_tensorrt_gemm_path_investigation/20260905T112513Z/` |

## Final Findings

1. **A. cuBLASLt is the fastest mature standalone baseline.** Direct
   `cublasLtMatmul` algorithm 21 reached median `0.080077961 ms` and passed
   the frozen correctness gate.
2. **B. CUTLASS v3.5.1 had no replacement advantage.** The best
   library-generated candidate was `0.083619133 ms`, about `4.42%` slower.
3. **C. TensorRT uses a Tensor Core xmma GEMM path.** Inspector recovered
   `sm80_xmma_gemm_*` tactic labels for all 28 `up_proj` layers and every
   label contains `tensor16x8x16`.
4. **D. Numeric tactic ID and backend identity remain unavailable.** The
   Inspector data model exposed tactic strings, but numeric tactic ID, runtime
   workspace, backend identity, accumulator semantics, and physical layout
   remain `NOT_AVAILABLE` or `UNKNOWN`.
5. **E. Matched NCU durations were nearly identical.** cuBLASLt and TensorRT
   xmma measured `242.912 us` and `244.160 us`; memory/L2 throughput was
   `76.06%` and `76.92%`. TensorRT used higher occupancy but lower
   tensor-cycle/HMMA activity.
6. **F. No evidence supports a custom CUDA GEMM implementation.** The frozen
   event-time gap was not reproduced in the NCU environment. The resource
   differences did not become a proven duration advantage or inefficiency.

## Historical Gap Boundary

The historical event-time comparison is:

| Path | Median / steady-state |
| --- | ---: |
| Direct cuBLASLt | `80.077961 us` |
| TensorRT steady-state correlated kernel, excluding first invocation | `147.424 us` |

This is not restated as a production `1.84x` speedup opportunity. The
measurement boundaries differ, clocks and invocation context were not
controlled, and matched NCU profiling did not reproduce the gap. Its cause is
`EVENT_TIME_GAP_INCONCLUSIVE`.

## up_proj Status

`up_proj` is `CLOSED_FOR_NOW`.

Reason:

```text
NO_PROVEN_OPTIMIZATION_TARGET
NO_PROVEN_CUDA_GEMM_OPTIMIZATION_TARGET
NO_PROVEN_TACTIC_DEFECT
NO_CUDA_IMPLEMENTATION_AUTHORIZED
```

It must not be returned to Rank 1 without a new independent evidence trigger,
for example a controlled paired event-time comparison, recovered numeric
tactic/backend identity, or a new engine/runtime boundary.

## Closure Limitations

- NCU captured one launch per backend with `--clock-control none`; it is not a
  production event-time reproduction.
- Direct DRAM throughput remains `N/A` and was not estimated.
- The observed `h16816` / `sm80_xmma_gemm` family switch remains `UNKNOWN`.
- Phase 4-E `IProfiler`, NSYS, CUDA event, and NCU timing boundaries are not
  interchangeable.
- This closeout freezes the authorized GEMM study; it does not prove that
  TensorRT is globally optimal.

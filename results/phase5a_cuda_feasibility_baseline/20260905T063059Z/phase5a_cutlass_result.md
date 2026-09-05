# Phase 5-A CUTLASS Library Candidate Result

## Gate

**Stage: `PASS / MEASURED`; NO CANDIDATE ADVANTAGE OVER DIRECT CUBLASLT.**

This candidate is a library-generated CUTLASS GEMM instantiation. No CUDA
kernel was hand-written and no TensorRT path was modified.

## Fixed Library And Configuration

| Field | Value |
| --- | --- |
| CUTLASS release/tag | `v3.5.1` |
| CUTLASS commit | `f7b19de32c5d1f3cedfc735c2849f12b537522ee` |
| API | `cutlass::gemm::device::Gemm` |
| Operands/output | FP16 row-major |
| Accumulator | FP32 |
| Instruction shape | `16x8x16` |
| Architecture tag / execution | Sm80 kernel on SM87 |
| Epilogue | `LinearCombination<cutlass::half_t,1,float,float>` |

The bounded candidate set used threadblock tiles `32x32x64`, `64x64x64` and
`128x64x64`, with split-K slices 1, 2, 4 or 8. Stages were 4 for the first two
tiles and 5 for `128x64x64`. All ten configurations completed.

## Best Timing

The best correctness-passing candidate was candidate 1:

| Field | Value |
| --- | ---: |
| Candidate | 1 |
| Configuration | `tb32x32x64`, warp `32x32x64`, stages 4, split-K 1 |
| Mean | `0.083616242 ms` |
| Median | `0.083619133 ms` |
| Sample std | `0.000008755 ms` |
| CV | `0.000104699` (0.0104699%) |
| Min | `0.083604395 ms` |
| Max | `0.083627254 ms` |
| P95 | `0.083627254 ms` |
| Iterations per trial | 5984 |
| TFLOPS | `0.075242033` |

Candidate 5 (`tb64x64x64`, split-K 1) was effectively tied at median
`0.083615869 ms`. Split-K did not improve this skinny shape; larger split-K
values were slower. The best CUTLASS median was `0.083619133 ms`, or
approximately 4.42% slower than the direct cuBLASLt FP32-accumulate median
`0.080077961 ms`.

## Limitations

- The configuration set was bounded, not exhaustive.
- Sm80 CUTLASS kernels ran on the integrated SM87 device; this is the recorded
  library candidate configuration, not a claim of native SM87 scheduling.
- Power and clock state remained uncontrolled and mostly unreadable.

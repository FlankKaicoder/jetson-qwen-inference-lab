# Phase 5-A Correctness Report

## Frozen Gate

The comparison used FP64 CPU reference over the same operands:

```text
PASS when max_abs_error <= 1e-3 + 1e-4 * max_abs_reference
```

All outputs were finite and copied only after measurement completed. The common
gate limit was `0.001066391`.

## Results

| Backend / variant | Best max abs error | Max abs error | Relative error | Gate |
| --- | ---: | ---: | ---: | --- |
| cuBLASLt FP32 accumulate | 8/8 PASS | `0.000218480-0.000269801` | `0.000481628-0.134635268` | PASS |
| cuBLASLt FP16 accumulate | 0/8 PASS | `0.001501896-0.002110384` | `0.978537944-1.787175012` | FAIL |
| CUTLASS library candidates | 10/10 PASS | `0.000218480-0.000446802` | `0.000651256-0.325293800` | PASS |

Relative error is informational. Some reference outputs are near zero, so the
frozen gate uses absolute error. The exact per-candidate values are in
`cublaslt_results.json` and `cutlass_results.json`.

Only correctness-passing variants participate in performance ranking.

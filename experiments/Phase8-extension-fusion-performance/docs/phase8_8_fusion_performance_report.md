# Phase 8.8 Transformer Operator Fusion Performance Evaluation

Date: 2026-09-20

Branch: `phase/08-extension-fusion-analysis`

Overall conclusion: `NO_PROVEN_FUSION_BENEFIT`

## 1. Motivation

Phase 8.8 asks whether operator fusion reduces kernel launches, memory traffic,
operator latency, and ultimately TensorRT/Qwen3 runtime latency. The staged
design requires CUDA operator evidence before any plugin or model integration.
This report stops at Phase 8.8-A because its dependency gate failed.

## 2. TensorRT fusion-boundary basis

Phase 8.7 found no proven missing FP16 TensorRT fusion target. FP16 Inspector
metadata is too compact to name exact boundaries, while detailed Mixed/INT8
evidence proves broad native fusion only for that different precision path.
The Phase 8.4-A standalone RMSNorm plugin was slower, but plugin-specific graph
and kernel attribution is unavailable. Phase 8.8 therefore began with an
independent CUDA microbenchmark and made no claim that TensorRT lacks fusion.

## 3. CUDA fusion result

The bounded FP16 `M=1/K=1024/N=3072` experiment compared RMSNorm plus cuBLAS
against one simple fused kernel. Correctness passed. Launch count fell from two
to one and CUDA Event mean fell by `6.3146%`, but median fell only `1.6886%`.
NCU L1 global-load sectors increased `1.4409%`; global-store sectors increased
from `256` to `3072` because the prototype's output stores are poorly
coalesced. The full analysis is in `phase8_8a_cuda_fusion_report.md`.

## 4. TensorRT plugin result

`NOT RUN`. Phase 8.8-B required `FUSION_BENEFIT_PROVEN` from Phase 8.8-A. The
actual gate was `NO_PROVEN_FUSION_BENEFIT`, so no `RMSNormLinearPlugin` was
implemented, built, serialized, deserialized, enqueued, or inspected.

## 5. Qwen3 runtime result

`NOT RUN`. No Qwen3 layer, TensorRT engine, full model, prefill, decode,
tokens/s, or end-to-end path was modified or measured. Runtime and end-to-end
fusion benefit remain `UNKNOWN`.

## 6. Performance table

| Metric | Baseline | Fusion | Interpretation |
| --- | ---: | ---: | --- |
| Kernel count | `2` | `1` | Launch reduction proven for the microbenchmark |
| CUDA Event mean | `0.1058619 ms` | `0.0991772 ms` | `6.3146%` lower |
| CUDA Event median | `0.1092214 ms` | `0.1073771 ms` | `1.6886%` lower; gate failed |
| L1 global-load sectors | `199872` | `202752` | `1.4409%` higher |
| L1 global-store sectors | `256` | `3072` | `12x` higher |
| NSYS GPU kernel time | `0.256064 ms` | `0.241824 ms` | `5.5611%` lower, bounded single captures |
| NCU duration sum | `0.174688 ms` | `0.253504 ms` | Conflicts with NSYS; diagnostic only |
| TensorRT runtime | `NOT RUN` | `NOT RUN` | Dependency gate failed |
| Qwen3 end-to-end | `NOT RUN` | `NOT RUN` | Benefit `UNKNOWN` |

## 7. Conclusion

```text
NO_PROVEN_FUSION_BENEFIT
```

Fusion reduced the static launch boundary, but the simple kernel did not
reduce measured global traffic and did not pass the median-latency gate. There
is no evidence basis to proceed to a TensorRT plugin or Qwen3 layer validation
under the authorized dependency chain. Stop after Phase 8.8-A.

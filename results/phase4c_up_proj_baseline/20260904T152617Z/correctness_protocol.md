# Phase 4-C up_proj Correctness Protocol

## Scope

This protocol freezes the correctness oracle for a future, separately
authorized `up_proj` decode GEMM implementation. Phase 4-C did not implement
or run a custom kernel.

## Exact problem

The logical target is:

```text
C[1,3072] = A[1,1024] * B[1024,3072]
```

where `A`, `B`, and `C` correspond to the Mixed Decode EngineInspector
activation, expanded constant, and output tensors. The logical GEMM contract is
row-major:

```text
C[m,n] = sum(k=0..1023) A[m,k] * B[k,n]
```

The physical TensorRT layout corresponding to the recorded `tn_n` tactic label
is `UNKNOWN`. The future probe must preserve the existing TensorRT operand
layout and record any layout transformation explicitly.

## Reference

Use the Exp04 Track A analogy:

1. Generate deterministic `A` and `B` operands as Half.
2. Convert them to FP32 on the host or reference device.
3. Compute the FP32 reference with the row-major formula or a trusted FP32
   library implementation.
4. Keep the reference in FP32 for error scaling.

This is inherited for standalone GEMM implementation correctness. Its use for
the real `up_proj` output is `PROPOSED` until measured on the actual activation
and weight tensors.

## Required checks

The future implementation must report all of the following:

- CUDA initialization, allocation, launch, last-error, and synchronization status.
- Allocation guard / canary status.
- Shape equality between `C_custom` and `C_reference`.
- Finite status for every `C_custom` element: no NaN and no Inf.
- `max_abs_error = max(abs(C_reference - C_custom))`.
- `max_rel_error` using `abs(reference) / max(abs(reference), 1e-12)`.
- Optional `mean_abs_error` and RMSE for diagnosis.

## Error gate

Inherit the Exp04 GEMM implementation gate:

```text
max_abs_error <= atol + rtol * max_abs_reference
atol = 1e-3
rtol = 1e-4
```

All required checks must pass. FP32 bitwise equality is not required. This gate
does not authorize an end-to-end model accuracy claim.

## Evidence required

Save deterministic input description, operand shapes and dtypes, reference
implementation, error metrics, guard status, CUDA status, software versions,
and PASS/FAIL for each check as new Phase 4-C-implementation artifacts. Do not
overwrite Phase 4-C.

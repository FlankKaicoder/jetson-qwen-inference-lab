# Exp04 Initial Correctness Analysis

V1 is compared with the V0 CPU FP32 reference using `atol=1e-3` and `rtol=1e-4` scaled by the maximum absolute reference value. The harness records max absolute/relative error and RMSE; FP32 bitwise equality is intentionally not required. A 16-element canary surrounds the output allocation and is checked after synchronization. CUDA allocation, copy, launch, last-error and synchronization failures are failures. Compute Sanitizer status is recorded separately by the run script.

The matrix includes tiny, 15/16/17 and 31/32/33 boundary-like shapes, rectangular/non-power-of-two shapes and `512x512x512`. Gate A1 requires every listed case to pass with intact guards and no CUDA failures.

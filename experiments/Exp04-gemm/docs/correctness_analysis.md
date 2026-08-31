# Exp04 Initial Correctness Analysis

V1 is compared with the V0 CPU FP32 reference using `atol=1e-3` and `rtol=1e-4` scaled by the maximum absolute reference value. The harness records max absolute/relative error and RMSE; FP32 bitwise equality is intentionally not required. A 16-element canary surrounds the output allocation and is checked after synchronization. CUDA allocation, copy, launch, last-error and synchronization failures are failures. Compute Sanitizer status is recorded separately by the run script.

The matrix includes tiny, 15/16/17 and 31/32/33 boundary-like shapes, rectangular/non-power-of-two shapes and `512x512x512`. Gate A1 requires every listed case to pass with intact guards and no CUDA failures.

## WMMA Dual-Reference Correctness and Precision Impact

The closure harness writes `benchmark/raw/wmma_correctness_dual_reference_20260831T165518Z.csv` for eight aligned shapes: `16^3`, `32^3`, `64x48x80`, `128^3`, `256^3`, `512x384x640`, `512^3`, and `1024^3`.

Original FP32 inputs feed two host references and the GPU path: CPU FP32 GEMM (Track B), CPU GEMM after FP16 conversion with FP32 accumulation (Track A), and V3 WMMA FP16xFP16 to FP32 output. All 8/8 rows report finite errors. Track A implementation correctness uses `max_abs_error <= 1e-3 + 1e-4 * max_abs_fp16_reference`; all canaries and CUDA statuses pass. Track B characterizes numerical precision impact, not FP32-equivalent correctness. The comparison against the original FP32 reference characterizes the end-to-end numerical impact of the mixed-precision WMMA path, including FP16 input quantization.

Measured Track B maximum absolute errors range from `0.001072` (`16^3`) to `0.013742` (`1024^3`). These are retained as evidence and do not redefine the implementation tolerance. This artifact closes Gate A3 without changing performance or profiler evidence.

# Environment

## Jetson Platform

| Field | Value |
| --- | --- |
| Host | `nvidia-desktop` |
| Platform | Linux-5.15.148-tegra-aarch64 |
| Jetson release | R36 (release), REVISION 4.3 |
| Device | Orin |
| Compute capability | 8.7 |
| NV power mode | 25W |
| GPU frequency during NCU | about 306 MHz |
| TensorRT | 10.3.0 |
| CUDA driver/runtime | 12.6 (12060) |
| cuBLASLt runtime | 120601 |
| PyTorch | 2.5.0a0+872d972e41.nv24.08 |
| NCU | 2024.3.1.0 |

## Profiling Boundaries

- NCU used `--clock-control none`; clocks were not modified.
- Both profiles used `--target-processes all` and captured one launch.
- cuBLASLt used `--launch-skip 100 --launch-count 1` after `100` warmup calls.
- TensorRT used an exact kernel-name filter and `--launch-count 1`.
- The Mixed Decode engine was read-only and had SHA-256
  `445fc7d295c5bbb91e5392182347aa0e59612a031b5556a3461e09f30a59005c`.
- The Jetson checkout was on `phase/03e-tensorrt-kernel-attribution` at
  `bf7abc67eb58662a68316045e166aa9f611330d7` and was not modified.
- Raw `.ncu-rep` files remain Jetson-local under
  `/tmp/phase5b_step2_20260905T112513Z/ncu/`.

## Target

| Field | Value |
| --- | --- |
| Operator | decode-only `up_proj` |
| Shape | `M=1, K=1024, N=3072` |
| Precision | FP16 input, FP16 output |
| TensorRT kernel | `sm80_xmma_gemm_f16f16_f16f16_f16_tn_n_tilesize32x32x64_stage6_warpsize2x2x1_tensor16x8x16_execute_kernel_trt` |
| cuBLASLt API | `cublasLtMatmul` |
| cuBLASLt compute type | FP16 input/output, FP32 accumulate |
| cuBLASLt heuristic / algorithm / workspace | index 4 / ID 21 / 0 bytes |

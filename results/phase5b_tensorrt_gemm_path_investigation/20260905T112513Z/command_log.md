# Command Log

All profiling and temporary build/run paths were Jetson-local under
`/tmp/phase5b_step2_20260905T112513Z/`. The Jetson repository was read-only.

## cuBLASLt Harness Build And Smoke Run

```bash
/usr/local/cuda-12.6/bin/nvcc -O3 -arch=sm_87 \
  -I/usr/local/cuda-12.6/include \
  -L/usr/local/cuda-12.6/lib64 -lcublasLt \
  -o /tmp/phase5b_step2_20260905T112513Z/bin/phase5b_step2_cublaslt_algo21_profile \
  /tmp/phase5b_step2_20260905T112513Z/src/phase5b_step2_cublaslt_algo21_profile.cu

/tmp/phase5b_step2_20260905T112513Z/bin/phase5b_step2_cublaslt_algo21_profile
```

## cuBLASLt NCU Profile

```bash
sudo -n /usr/local/cuda-12.6/bin/ncu \
  --clock-control none --target-processes all \
  --launch-skip 100 --launch-count 1 \
  --section SpeedOfLight --section ComputeWorkloadAnalysis \
  --section MemoryWorkloadAnalysis --section Occupancy \
  --section WarpStateStats --section SchedulerStats \
  --section LaunchStats --section SourceCounters \
  --export /tmp/phase5b_step2_20260905T112513Z/ncu/cublaslt_algo21_postwarmup \
  /tmp/phase5b_step2_20260905T112513Z/bin/phase5b_step2_cublaslt_algo21_profile
```

## TensorRT NCU Profile

```bash
cd /home/nvidia/projects/jetson-qwen-inference-lab
sudo -n /usr/local/cuda-12.6/bin/ncu \
  --clock-control none --target-processes all \
  --kernel-name regex:sm80_xmma_gemm_f16f16_f16f16_f16_tn_n_tilesize32x32x64_stage6_warpsize2x2x1_tensor16x8x16_execute_kernel_trt \
  --launch-count 1 \
  --section SpeedOfLight --section ComputeWorkloadAnalysis \
  --section MemoryWorkloadAnalysis --section Occupancy \
  --section WarpStateStats --section SchedulerStats \
  --section LaunchStats --section SourceCounters \
  --export /tmp/phase5b_step2_20260905T112513Z/ncu/trt_f16f16_execute_kernel_trt \
  python3.10 experiments/Phase2-qwen3-quantization/src/phase3b/phase3b_runtime_context.py \
    --mode profile \
    --out /tmp/phase5b_step2_20260905T112513Z/trt_f16f16_profile \
    --profile-runtime mixed --profile-seq-len 8 \
    --profile-decode-steps 1 \
    --lifetime persistent_context_lifetime
```

## CSV Export

```bash
/usr/local/cuda-12.6/bin/ncu --import <report>.ncu-rep --csv --page details > <report>_details.csv
/usr/local/cuda-12.6/bin/ncu --import <report>.ncu-rep --csv --page raw > <report>_raw.csv
```

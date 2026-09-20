#!/usr/bin/env bash
set -euo pipefail

SOURCE_DIR=${1:?source directory is required}
WORK_DIR=${2:?isolated work directory is required}
ARTIFACT_DIR=${3:?artifact directory is required}
CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-12.6}

mkdir -p "${WORK_DIR}/build" "${ARTIFACT_DIR}"

{
  date --iso-8601=seconds
  hostname
  uname -a
  "${CUDA_HOME}/bin/nvcc" --version
  nsys --version
  "${CUDA_HOME}/bin/ncu" --version
  nvidia-smi || true
} > "${ARTIFACT_DIR}/environment.txt" 2>&1

cmake -S "${SOURCE_DIR}/phase8_8_a_cuda_fusion_microbenchmark" \
  -B "${WORK_DIR}/build" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER="${CUDA_HOME}/bin/nvcc" \
  > "${ARTIFACT_DIR}/configure.log" 2>&1
cmake --build "${WORK_DIR}/build" --parallel 2 \
  > "${ARTIFACT_DIR}/build.log" 2>&1

BINARY="${WORK_DIR}/build/fusion_microbenchmark"
"${BINARY}" --output-dir "${ARTIFACT_DIR}" --warmup 50 \
  --repetitions 500 --trials 10 \
  > "${ARTIFACT_DIR}/benchmark.log" 2>&1

for mode in baseline fusion; do
  nsys profile --trace cuda --sample=none --cpuctxsw=none \
    --capture-range=cudaProfilerApi --capture-range-end=stop \
    --force-overwrite=true --output "${WORK_DIR}/nsys_${mode}" \
    "${BINARY}" --profile-mode "${mode}" --profile-warmup 20 \
    > "${ARTIFACT_DIR}/nsys_${mode}.log" 2>&1
  nsys stats --report cuda_gpu_kern_sum --format csv \
    "${WORK_DIR}/nsys_${mode}.nsys-rep" \
    > "${ARTIFACT_DIR}/nsys_${mode}_cuda_gpu_kern_sum.csv" 2>&1

  sudo -n "${CUDA_HOME}/bin/ncu" --clock-control none --csv --page raw \
    --metrics gpu__time_duration.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum,sm__warps_active.avg.pct_of_peak_sustained_active \
    "${BINARY}" --profile-mode "${mode}" --profile-warmup 0 \
    > "${ARTIFACT_DIR}/ncu_${mode}.csv" 2> "${ARTIFACT_DIR}/ncu_${mode}.log"
done

python3 "${SOURCE_DIR}/scripts/analyze_phase8_8a.py" \
  --artifact-dir "${ARTIFACT_DIR}"

sha256sum "${SOURCE_DIR}/phase8_8_a_cuda_fusion_microbenchmark/fusion_microbenchmark.cu" \
  "${SOURCE_DIR}/phase8_8_a_cuda_fusion_microbenchmark/CMakeLists.txt" \
  "${SOURCE_DIR}/scripts/analyze_phase8_8a.py" \
  "${SOURCE_DIR}/scripts/run_phase8_8a_jetson.sh" \
  "${ARTIFACT_DIR}"/*.csv "${ARTIFACT_DIR}"/run_metadata.txt \
  > "${ARTIFACT_DIR}/sha256sums.txt"

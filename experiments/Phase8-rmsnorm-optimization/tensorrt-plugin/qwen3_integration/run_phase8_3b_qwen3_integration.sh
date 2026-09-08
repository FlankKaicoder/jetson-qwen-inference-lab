#!/usr/bin/env bash
set -u -o pipefail

if [[ $# -ne 3 ]]; then
    echo "usage: $0 <output-dir> <x-fp16-bin> <gamma-fp16-bin>" >&2
    exit 2
fi

output_dir="$1"
x_path="$2"
gamma_path="$3"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
build_dir="${output_dir}/build"
artifact_dir="${output_dir}/artifacts"

mkdir -p "${artifact_dir}"
{
    cat /etc/nv_tegra_release 2>/dev/null || true
    /usr/local/cuda-12.6/bin/nvcc --version | tail -1
    dpkg-query -W -f='TensorRT=${Version}\n' tensorrt 2>/dev/null || true
    /home/nvidia/.venvs/jetson-qwen-phase1-hf/bin/python -c \
        'import torch; print("PyTorch=" + torch.__version__)'
} > "${artifact_dir}/environment.txt" 2>&1

cmake -S "${script_dir}" -B "${build_dir}" \
    -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.6/bin/nvcc \
    > "${artifact_dir}/configure.log" 2>&1
configure_status=$?
printf '%s\n' "${configure_status}" > "${artifact_dir}/configure_exit_code.txt"

if [[ ${configure_status} -eq 0 ]]; then
    cmake --build "${build_dir}" -j2 > "${artifact_dir}/build.log" 2>&1
    build_status=$?
else
    build_status=125
fi
printf '%s\n' "${build_status}" > "${artifact_dir}/build_exit_code.txt"

if [[ ${build_status} -eq 0 ]]; then
    "${build_dir}/qwen3_rmsnorm_integration" \
        "${build_dir}/librmsnorm_qwen3_plugin.so" \
        "${x_path}" "${gamma_path}" "${output_dir}/phase8_3b" \
        > "${artifact_dir}/integration.json" \
        2> "${artifact_dir}/integration.stderr"
    integration_status=$?
else
    integration_status=125
fi
printf '%s\n' "${integration_status}" > "${artifact_dir}/integration_exit_code.txt"

if [[ ${integration_status} -eq 0 ]]; then
    sha256sum "${x_path}" "${gamma_path}" \
        "${output_dir}/phase8_3b_original.engine" \
        "${output_dir}/phase8_3b_plugin.engine" \
        "${build_dir}/librmsnorm_qwen3_plugin.so" \
        > "${artifact_dir}/sha256sums.txt"
fi

printf 'configure=%s\nbuild=%s\nintegration=%s\n' \
    "${configure_status}" "${build_status}" "${integration_status}" \
    > "${artifact_dir}/run_status.txt"
exit "${integration_status}"

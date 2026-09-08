# Phase 8.3-A TensorRT RMSNorm Plugin

This directory contains a bounded TensorRT 10.3 `IPluginV3` prototype for the
Phase 8 RMSNorm study. It builds only a synthetic FP16 network with two inputs:
`X` shaped `[1,8,1024]` and `gamma` shaped `[1024]`. Its output is
`Y = X * rsqrt(mean(X^2) + epsilon) * gamma`.

`RMSNormPlugin` exposes the Core, Build, and Runtime V3 capability interfaces.
`RMSNormPluginCreator` registers the shared library through
`REGISTER_TENSORRT_PLUGIN`, accepts the serialized `epsilon` field, and creates
the plugin for both build and runtime phases. `enqueue` calls
`launch_rmsnorm_v2<__half>` from the Phase 8.1 CUDA source; it does not contain
a copied RMSNorm kernel.

The implementation deliberately accepts only TensorRT linear FP16 tensors and
the static test shapes. It is a minimal API and correctness exercise, not a
Qwen3 integration, an ONNX rewrite, a full-model engine rebuild, or a claim of
end-to-end acceleration.

On Jetson with CUDA 12.6 and TensorRT 10.3:

```bash
cmake -S . -B build -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.6/bin/nvcc
cmake --build build -j2
./build/rmsnorm_plugin_demo ./build/librmsnorm_trt_plugin.so
python3 pytorch_rmsnorm_baseline.py
```

The demo loads the library with `dlopen`, verifies registry lookup, builds and
deserializes an in-memory engine, runs inference, compares the FP16 result to
an explicit FP32 reference, and records five CUDA Event trials after 50 warmup
calls. The PyTorch script is a separate bounded control with the same shape,
seed, warmup, repetition, and trial counts.

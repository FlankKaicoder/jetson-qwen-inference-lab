# TensorRT Precision Probe

This probe never loads Qwen3. It creates synthetic `1024 -> {1024, 2048, 3072}` linear layers for `M=1` and `M=32`.

`build_linear_onnx.py` is an ONNX capability guard. Phase 2.1 does not install `onnx`, so that script records whether the requested ONNX route is available. `build_engine.py` uses the already installed TensorRT Python API to construct an equivalent direct-network FP16 baseline and an explicit Q/DQ INT8 candidate without changing the environment. Engines are written only to the caller-selected Jetson-local temporary directory.

`run_engine.py` binds CUDA tensors allocated by the existing PyTorch build to the TensorRT execution context. It records shape, device, finite output, and numerical error against the same seeded FP16 reference. It is a capability probe, not a performance benchmark.

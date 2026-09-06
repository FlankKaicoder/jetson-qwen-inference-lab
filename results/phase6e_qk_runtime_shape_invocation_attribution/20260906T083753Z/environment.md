# Environment

The Jetson host is `nvidia-desktop`, path
`/home/nvidia/projects/jetson-qwen-inference-lab`. The checkout remains
read-only on branch `phase/03e-tensorrt-kernel-attribution`, HEAD
`bf7abc67eb58662a68316045e166aa9f611330d7`. No repository switch, clean, or
historical file modification is performed.

| Field | Value | Evidence level |
| --- | --- | --- |
| GPU | NVIDIA Orin, SM `8.7` | HISTORICAL_EVIDENCE |
| TensorRT | `10.3.0` | HISTORICAL_EVIDENCE |
| CUDA | `12.6` | HISTORICAL_EVIDENCE |
| PyTorch | `2.5.0a0+872d972e41.nv24.08` | HISTORICAL_EVIDENCE |
| Nsys | `2024.5.4.34-245434855735v0` | HISTORICAL_EVIDENCE |
| Power mode | `NV Power Mode: 25W` | HISTORICAL_EVIDENCE |

Runtime API availability is verified on-device before execution:
`set_input_shape`, `set_tensor_address`, `get_tensor_shape`,
`get_tensor_strides`, `get_tensor_address`, `infer_shapes`, and
`execute_async_v3` on `trt.IExecutionContext`. Clock and power state remain
unchanged.

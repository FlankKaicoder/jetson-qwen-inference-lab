# Phase 5-A Step 3 Environment

No new profiling or benchmark was run. This step re-read the existing Phase 4-F
NSYS SQLite trace on Jetson in read-only mode.

| Field | Value |
| --- | --- |
| Artifact timestamp UTC | `20260905T072100Z` |
| Host | `nvidia-desktop`, Jetson Orin Nano Engineering Reference Developer Kit Super |
| GPU / capability | `Orin (nvgpu)`, SM `8.7` |
| JetPack / Tegra release | `R36 (release), REVISION: 4.3` |
| Kernel | `5.15.148-tegra` |
| Driver | `540.4.0` |
| CUDA | `12.6.68` |
| TensorRT | `10.3.0` |
| PyTorch | `2.5.0a0+872d972e41.nv24.08` |
| Python | `3.10.12` |
| Nsight Systems | `2024.5.4.34-245434855735v0` |
| Power/clock state | `N/A` or `UNKNOWN_NO_READABLE_NVIDIA_SMI_FIELDS`; not controlled |
| Source trace | `/tmp/phase3c_nsys_20260904T093500Z/mixed_persistent.sqlite` |
| Source trace SHA-256 | `ea9ea0bc4a369647b837def7f98d2bfec2765f1f6f9c9619b4388ab2ab4345a8` |
| Source trace bytes | `3,477,504` |
| Mixed Decode engine SHA-256 | `445fc7d295c5bbb91e5392182347aa0e59612a031b5556a3461e09f30a59005c` |
| Engine modified/rebuilt | `false` |
| TensorRT backend identity | `UNKNOWN` |

The historical SQLite trace was opened with SQLite read-only URI mode. The
engine was not deserialized or executed in this step.

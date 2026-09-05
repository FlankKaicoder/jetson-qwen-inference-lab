# Phase 5-B Step 1 Environment

No new profiling, engine deserialization, engine execution, benchmark, or
build was run. This step only read existing repository artifacts on Windows.

| Field | Value |
| --- | --- |
| Artifact timestamp UTC | `20260905T103352Z` |
| Analysis host | Windows repository checkout, `E:\nvidia-qwen` |
| Evidence platform | `nvidia-desktop`, Jetson Orin Nano Engineering Reference Developer Kit Super |
| GPU / capability | `Orin (nvgpu)`, SM `8.7` |
| JetPack / Tegra release | `R36 (release), REVISION: 4.3` |
| Kernel | `5.15.148-tegra` |
| Driver | `540.4.0` |
| CUDA | `12.6.68` |
| TensorRT | `10.3.0` |
| Mixed Decode engine SHA-256 | `445fc7d295c5bbb91e5392182347aa0e59612a031b5556a3461e09f30a59005c` |
| Engine modified/rebuilt/executed | `false` |
| Historical source inspector SHA-256 | `b9305150544221c601861aa7b7a86232bbc62d851ac45619fe6166758cc5fe71` |
| Historical NSYS source SHA-256 | `ea9ea0bc4a369647b837def7f98d2bfec2765f1f6f9c9619b4388ab2ab4345a8` |
| Power/clock state | `UNKNOWN`; historical and not controlled |

The analysis script read only the frozen Phase 4-A Inspector JSON, Phase 4-A.1
mapping CSV, and Phase 5-A Step 3 kernel breakdown CSV. No historical artifact
was modified. No engine, SQLite database, `.ncu-rep`, or raw profiler dump was
copied into this result directory.

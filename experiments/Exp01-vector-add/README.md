# Exp01 - CUDA Vector Add

## 1. 实验目标

用最基础的一元素一线程 FP32 Vector Add 建立 CUDA 执行模型基线，并通过 Jetson 实测理解：

`Kernel → Grid → Block → Warp → Thread → SM → Global Memory`。

本实验验证线程到数据的映射、launch 配置、block size 对延迟的影响、occupancy 与性能的关系、连续 global-memory 访问以及 CUDA Event 的 kernel-only 计时。代码跑通不是完成标准；correctness、benchmark、分析与 Gate 必须同时留档。

## 2. 为什么做这个实验

Vector Add 的计算很简单，可以把注意力集中在 CUDA 的执行与调度模型，而不被复杂算法掩盖。它也是一个低算术强度 kernel，适合建立 memory-bound 分析方法，并为后续 Reduction、GEMM 和 Transformer 算子实验统一计时与归档口径。

## 3. 理论背景

### 执行层级

- 一次 kernel launch 创建一个 grid。
- grid 由全部 logical blocks 组成；这些 block 不保证同时驻留。
- 每个 block 包含 `blockDim.x` 个 threads，并被划分为 warp；本机 `warpSize=32`。
- 硬件调度器把 block 分派到 SM。一个 block 在其生命周期内完整驻留在一个 SM 上，不会拆到多个 SM。
- 一个 SM 可同时驻留多个 block，但受到 threads、warps、registers、shared memory 和 architectural block limit 约束。

### 数据映射

基线 kernel 使用：

```cpp
idx = blockIdx.x * blockDim.x + threadIdx.x;
if (idx < N) {
    C[idx] = A[idx] + B[idx];
}
```

grid 使用向上取整：

`grid_size = (N + block_size - 1) / block_size`。

边界判断使不能整除 block size 的 `N` 仍然安全；多余 logical threads 不读写数组。

### 算术强度与实验前假设

每个元素执行一次加法，最基本数据流量为读取 A 4 B、读取 B 4 B、写入 C 4 B：

`arithmetic intensity ≈ 1 FLOP / 12 Bytes = 0.0833 FLOP/Byte`。

实验前假设：

- H1：block size 会影响性能，但 block 越大不一定越快。
- H2：block size 会影响 active warps / blocks / occupancy，但 occupancy 越高不保证 latency 越低。
- H3：Vector Add 算术强度很低，预计主要受 memory subsystem 限制。
- H4：一线程一连续元素的访问方式应具有良好的 coalescing 特征。

这些是实验前假设，不是预设结论。

## 4. 实验环境

采集时间：`2026-08-27T06:24:26Z`。完整只读记录见 `benchmark/environment.txt`。

| 项目 | 实测值 |
| --- | --- |
| Hostname / OS | `nvidia-desktop` / Ubuntu 22.04.5 LTS |
| Kernel | Linux 5.15.148-tegra aarch64 |
| GPU（CUDA API） | Orin |
| Compute capability | 8.7 |
| CUDA / nvcc | 12.6 / V12.6.68 |
| GCC / G++ | 11.4.0 / 11.4.0 |
| SM count | 8 |
| Warp size | 32 |
| Max threads/block | 1024 |
| Max threads/SM | 1536 |
| Max warps/SM | 48 |
| Max blocks/SM（device attribute） | 16 |
| Max thread dimensions | 1024, 1024, 64 |
| Max grid dimensions | 2147483647, 65535, 65535 |
| Shared memory | 49,152 B/block；167,936 B/SM |
| Registers | 65,536/block；65,536/SM |
| Kernel resources | 12 registers/thread；0 B static shared memory |
| Global memory / L2 | 7,989,903,360 B / 2,097,152 B |
| Supported sweep blocks | 16, 32, 64, 128, 256, 512, 1024 |

`nvidia-smi` 与 `tegrastats --help` 均可执行。未安装、升级或修改任何系统组件。

## 5. Baseline

Baseline 是单个基础 kernel：

- FP32：`C[i] = A[i] + B[i]`。
- 一线程处理一个元素。
- 无 FP16、vectorized load、shared memory、multiple elements/thread、CUDA Graph、Tensor Core 或 Thrust。
- CPU 使用同一输入生成 reference。
- GPU 输出逐元素比较，Gate 阈值为 `max_abs_error <= 1e-6`。

这不是“优化版本与旧版本”的速度比较；block-size sweep 是对同一 baseline launch geometry 的研究。

## 6. 实现方案

源码：`src/vector_add.cu`。

- 所有 `cudaMalloc`、`cudaMemcpy`、event、同步和正常路径 `cudaFree` 都检查返回值。
- kernel launch 后调用 `cudaGetLastError`，warmup 后执行 `cudaDeviceSynchronize`。
- CUDA Event 只包围重复 kernel launches；allocation、host initialization、H2D 和 D2H 均不计入 latency。
- 输出 `grid_size`、`total_threads`、`warps_per_block` 与多余 logical threads。
- 使用 `cudaOccupancyMaxActiveBlocksPerMultiprocessor` 计算本 kernel 在各 block size 下的理论 active blocks/SM，并结合实际 device properties 计算理论 occupancy。

可复现入口：

```bash
experiments/Exp01-vector-add/scripts/build.sh
experiments/Exp01-vector-add/scripts/run_experiment.sh
```

binary 生成在仓库外 `/tmp/jetson-qwen-exp01-build/vector_add`。

## 7. 正确性验证

测试了 11 个 N：

`1, 31, 32, 33, 255, 256, 257, 1000, 1024, 1025, 1048576`。

每个 N 覆盖设备支持的 7 个 block size：

`16, 32, 64, 128, 256, 512, 1024`。

结果：

- 总配置：77。
- CUDA runtime success：77/77。
- PASS：77/77。
- 全局最大 `max_abs_error`：`0`。
- 非整除边界（如 N=33、257、1025）全部通过，验证了 `if (idx < N)`。

完整数据：`benchmark/correctness_results.csv`；时间戳 raw 结果保存在 `benchmark/raw/`，未覆盖历史结果。

## 8. Benchmark

### 测量口径

- Performance N：`2^20`、`2^22`、`2^24`。
- Block sweep：设备支持的全部 7 个候选值。
- Warmup：20。
- Repetitions：200。
- Timing：CUDA Event，kernel-only。
- Effective bandwidth 使用 decimal GB/s：`1 GB = 1e9 bytes`。
- 公式：`(3 × N × sizeof(float)) / avg_kernel_time`。
- 该值是本 kernel 的 effective bandwidth，不是 GPU theoretical bandwidth。

### 完整 block-size sweep 摘要

| Block | Occupancy API | Active blocks/SM | 2^20 ms / GB/s | 2^22 ms / GB/s | 2^24 ms / GB/s |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | 33.33% | 16 | 0.974403 / 12.913 | 3.647107 / 13.800 | 14.144903 / 14.233 |
| 32 | 33.33% | 16 | 0.439934 / 28.602 | 1.743924 / 28.861 | 6.800838 / 29.603 |
| 64 | 66.67% | 16 | 0.227637 / 55.276 | 0.688694 / 73.083 | 2.702901 / 74.485 |
| **128** | **100.00%** | **12** | **0.202515 / 62.133** | **0.551186 / 91.315** | **2.171583 / 92.710** |
| 256 | 100.00% | 6 | 0.203268 / 61.903 | 0.570819 / 88.174 | 2.242899 / 89.762 |
| 512 | 100.00% | 3 | 0.206207 / 61.021 | 0.577960 / 87.085 | 2.297645 / 87.623 |
| 1024 | 66.67% | 1 | 0.262756 / 47.888 | 0.831504 / 60.531 | 3.298589 / 61.034 |

三个 N 的最快配置均为 block 128。代表性大规模 `N=16,777,216` 的结果是 `2.171583 ms`、`92.710 GB/s`。

完整 21 行结果：`benchmark/vector_add_benchmark.csv`；raw 结果和 console log 均保留。

## 9. Nsight分析

Nsight Compute 工具存在：

- Path：`/usr/local/cuda-12.6/bin/ncu`。
- Version：`2024.3.1.0`。

实际执行：

```bash
which ncu
ncu --version
ncu --list-sections
```

`ncu --list-sections` 返回：

`Insufficient privileges to launch app for profiling. Launch app with root privileges`

因此当前用户无法列出本版本真实 sections，不能安全选择 Launch Statistics、Occupancy、Speed Of Light 或 Memory Workload sections，也没有继续猜测 metric 名称。代表性 `N=16,777,216` 的 block 32、256、1024 profile 均未执行；没有生成任何 `.ncu-rep`。

Profiler Gate：`BLOCKED`。证据见 `benchmark/ncu_sections.txt` 和 `notes/ncu_profile_status.md`。

当前可用的 occupancy 数据来自 CUDA Occupancy API，是资源限制下的理论 active-block/warp 上限，不冒充 Nsight 实测 active occupancy。

## 10. 结果

### 主要结论

1. Block 16/32 的理论 occupancy 只有 33.33%，且大规模 effective bandwidth 仅 14.233/29.603 GB/s；可用于隐藏内存延迟的 warps 不足。
2. Block 64 提升到 66.67% occupancy 后，大规模达到 74.485 GB/s。
3. Block 128、256、512 的理论 occupancy 都是 100%，但 block 128 仍是最快，证明 occupancy 相同也不代表性能相同。
4. Block 1024 只能有 1 block/SM、32 active warps/SM，理论 occupancy 回落到 66.67%，大规模延迟升至 3.298589 ms。
5. 对 block 128，N 从 2^22 到 2^24 时 effective bandwidth 从 91.315 到 92.710 GB/s，表现出带宽平台；结合 0.0833 FLOP/Byte，支持 memory-bound 判断。

### 实验假设判断

| 假设 | 判断 | 依据 |
| --- | --- | --- |
| H1 | SUPPORTED | 三个规模均由 block 128 最快；更大的 256/512/1024 都没有更快。 |
| H2 | SUPPORTED | block 128/256/512 同为 100% 理论 occupancy，但延迟不同；128 最快。 |
| H3 | SUPPORTED | 算术强度仅 0.0833 FLOP/Byte，且大规模 effective bandwidth 在约 91–93 GB/s 形成平台。Nsight 吞吐量佐证因权限缺失。 |
| H4 | INCONCLUSIVE | 源码确认 warp 内相邻线程访问连续地址，结果与预期一致；但 Nsight memory transaction/coalescing 指标因权限阻塞，不能完整验证实际合并效率。 |

## 11. Gate判断

| Gate | 状态 | 依据 |
| --- | --- | --- |
| Gate A — Correctness | **PASS** | 77/77 受支持配置 runtime success，最大误差 0。 |
| Gate B — Benchmark | **PASS** | 3 个 N × 7 个 block，warmup=20、repetitions=200，CSV 已保存。 |
| Gate C — Nsight | **BLOCKED** | ncu 存在，但当前用户权限不足，连 section enumeration 都被拒绝；非代码失败。 |
| Overall Gate | **PARTIAL** | Correctness 与 Benchmark 完整通过；Profiler 证据受外部权限限制。 |

## 12. 为什么得到这个结果

### Q1. Launch 的两个参数是什么？

`vector_add<<<grid_size, block_size>>>` 中，第一个参数是 grid 内 block 数量，第二个参数是每个 block 的 thread 数量。这里都是一维配置。

### Q2. N=1,000,000、blockDim.x=256 时 gridDim.x 是多少？

`ceil(1,000,000 / 256) = 3907`。3906 个 block 只有 999,936 个 threads，不足以覆盖全部元素；3907 个 block 产生 1,000,192 个 logical threads，多出的 192 个由边界判断挡住。

### Q3. “一个 kernel 的所有 blocks”是什么意思？

一次 launch 创建一个 grid，grid 的逻辑成员就是这次 kernel 的全部 blocks。它不表示这些 blocks 同时驻留；本实验最大 grid 有 1,048,576 blocks，而设备只有 8 个 SM，硬件必须分批调度。

### Q4. Block 是 grid 管理还是硬件调度？

Grid 是程序定义的逻辑组织；CUDA 硬件调度器负责把可运行 block 分派到 SM。逻辑包含关系与物理调度是两件事。

### Q5. Warp 与 block 的关系？

本机 warp=32 threads。32、256、1024 threads/block 分别是 1、8、32 warps/block。16 threads 仍占用一个不满的 warp。

### Q6. “Block 驻留在 SM 上”是什么意思？

Block 被分配后，其 threads/warps、registers 和 shared-memory 配额属于同一个 SM，直到 block 完成并释放资源。当前设备有 8 个 SM，但同一 block 不会跨 SM 执行。

### Q7. 为什么一个 SM 能同时驻留多个 block？

只要总 threads、warps、registers、shared memory 和 architectural block limit 都未超过上限，SM 就能保留多个未完成 block。本机查询到 1536 threads/SM、48 warps/SM、65,536 registers/SM、167,936 B shared memory/SM、最多 16 blocks/SM。本 kernel 只用 12 registers/thread、0 B static shared memory；Occupancy API 对 block 128/256/512 分别给出 12/6/3 active blocks/SM。

### Q8. Occupancy 是什么？为什么越高不一定越快？

Occupancy 通常表示 active warps 相对 SM 最大 warps 的比例，它帮助隐藏延迟，但不会增加 DRAM 带宽，也不消除指令/调度开销。实验中 block 128、256、512 都是 100% 理论 occupancy，大规模延迟却分别是 2.171583、2.242899、2.297645 ms。

### Q9. 为什么 Vector Add 倾向 memory-bound？

每元素只有 1 FLOP，却至少搬运 12 B，算术强度约 0.0833 FLOP/Byte。大规模 block 128 的 effective bandwidth 在 91–93 GB/s 附近形成平台，而增加 block/occupancy 不再带来等比例收益，符合 memory subsystem 限制特征。

### Q10. 为什么当前访问有利于 coalescing？

Thread i 访问 A[i]、B[i]、C[i]；一个 warp 内相邻 threads 访问连续 FP32 地址，这符合合并为较少 memory transactions 的基本地址模式。实际 transaction efficiency 尚需解除 Nsight 权限后验证。

### Q11. 实测哪个 block size 最快？

Block 128 在 2^20、2^22、2^24 三个规模都最快。代表性 2^24 为 2.171583 ms、92.710 GB/s。没有预设 block 256 最快。

### Q12. 最佳 block 为什么可能不是 occupancy 最高的唯一答案？

Occupancy 到达足够隐藏延迟的水平后，性能还受 block 调度粒度、每个 SM 的 resident block 数、memory subsystem 和 launch geometry 影响。Block 128/256/512 都达到 100% 理论 occupancy，但 block 128 保留 12 blocks/SM 并实测最快；occupancy 只能作为解释变量，不能单独决定最佳 block。

## 13. 已知限制

- Nsight Compute 因当前用户 profiling 权限不足而阻塞，没有实测 SM throughput、memory throughput、memory transactions 或 achieved occupancy。
- 每个配置保存一次 200 repetitions 的平均值，未做多轮独立进程统计，也未报告 min/median。
- 未固定 Jetson DVFS/功耗模式，也未在每组 benchmark 同步采样 tegrastats；这是后续严格性能复验需要控制的变量。
- Effective bandwidth 使用最低 12 B/element 流量模型，不计 cache write policy 或额外 transaction。
- 本实验刻意不比较高级优化版本。

## 14. 学习总结

- Grid 定义工作总量，硬件按资源与可用 SM 分批调度 blocks。
- Warp 是实际 SIMD/SIMT 调度粒度；block size 会同时改变 warps/block 与 resident blocks/SM。
- 边界保护使向上取整 grid 对任意 N 都正确。
- CUDA Event 可以隔离 kernel-only latency，但必须明确 warmup、repetitions 和同步位置。
- Occupancy 的价值是提供延迟隐藏机会，不是“越高越快”的性能分数。
- 科学结论必须保留失败和阻塞：本次 Nsight 权限问题使 Overall Gate 只能是 PARTIAL。

## 15. 面试表达

我在 Jetson Orin 上手写过基础 FP32 CUDA Vector Add kernel，使用 `idx = blockIdx.x * blockDim.x + threadIdx.x` 做一线程一元素映射，并用边界判断支持任意 N。我实际 sweep 了 16 到 1024 threads/block；当前设备 warp size 是 32，所以 256 threads 对应 8 warps/block。Block 是 grid 的逻辑成员，但由硬件分派到 SM，并受 threads、warps、registers、shared memory 和 block limit 共同限制。

性能用 CUDA Event 只测 kernel，排除了 allocation、H2D 和 D2H，固定 warmup 20、repetitions 200。77 个 correctness 配置全部通过。对 N=16,777,216，block 128 最快，为 2.171583 ms、92.710 GB/s；block 128/256/512 的理论 occupancy 都是 100%，但延迟不同，因此不能简单说 block 越大或 occupancy 越高越好。Vector Add 每元素约 1 FLOP/12 B，算术强度很低，实测大规模带宽出现平台，符合 memory-bound 特征。Nsight Compute 当前受权限限制，我将 profiler Gate 如实记录为 BLOCKED，没有虚构指标。

## 16. Git信息

- Base：`main@d42ab4aeabc751723a4a2c1036b93a5ed16d3d01`。
- Branch：`exp/01-vector-add`。
- Commit：提交后以 `git rev-parse HEAD` 为准。
- Build artifact：仓库外 `/tmp/jetson-qwen-exp01-build`。
- 本实验不 merge `main`，不 force push，不开始 Exp02。

## 17. Exp01.1 Stability & Nsight Audit

本节是对 Original Exp01 的稳定性与 profiler 证据审计，不是新优化实验。原始 correctness、benchmark 数据和上述历史判断均保留；本节使用更严格口径给出审计后的结论。

### 17.1 当前设备状态

采集时间：`2026-08-27T07:54:33Z`。完整记录见 `benchmark/raw/environment_exp01_1_20260827T075423Z.txt`。

| 项目 | 只读实测 |
| --- | --- |
| nvpmodel | `NV Power Mode: 25W`，mode ID `1` |
| GPU devfreq（实验前） | current `306 MHz`，min `306 MHz`，max `918 MHz`；可用频点列表另含 `1020 MHz` |
| Memory clock | CUDA device property 报告 `918000 kHz`；非 root 用户无法读取 debugfs EMC rate，未把该值换算为模式理论带宽利用率 |
| jetson_clocks | `--show` 需要 root，未使用 sudo；GPU min/max 不相等，证明 GPU 频率未静态锁定，整体 jetson_clocks 状态不能由非 root 输出完整确认 |
| tegrastats | 前测 GPU 约 `49.8–50.0°C`，后测最高样本 `54.718°C`；`GR3D_FREQ` 是利用率/时钟状态信息，不是 occupancy |

本轮未执行 `nvpmodel -m`、未运行无参数 `jetson_clocks`，没有修改设备性能状态。

### 17.2 稳定性测量口径

- `N=16,777,216`，FP32，CUDA Event kernel-only timing。
- 每个配置每轮 `warmup=20`、`repetitions=200`；共 5 个独立 round，而不是把 repetitions 改成 1000。
- Block：`32, 64, 128, 256, 512, 1024`。
- 固定轮换顺序：
  - R1：`32 64 128 256 512 1024`
  - R2：`128 256 512 1024 32 64`
  - R3：`512 1024 32 64 128 256`
  - R4：`256 128 64 32 1024 512`
  - R5：`1024 512 256 128 64 32`
- 标准差为 5 个独立 round 的 sample standard deviation（分母 `n-1`）。
- Raw：`benchmark/stability_raw_20260827T075423Z.csv`；summary：`benchmark/stability_summary.csv`。

30/30 个独立配置运行成功且 correctness PASS，全部 `max_abs_error=0`。

### 17.3 五轮统计结果

| Block | Count | Mean ms | Median ms | Sample std ms | Min–max ms | CV | Mean / median GB/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 32 | 5 | 6.698908 | 6.682917 | 0.043458 | 6.675447–6.776365 | 0.649% | 30.055 / 30.126 |
| 64 | 5 | 2.726844 | 2.725990 | 0.005138 | 2.723132–2.735674 | 0.188% | 73.832 / 73.854 |
| **128** | **5** | **2.207088** | **2.206342** | **0.002887** | **2.204237–2.210511** | **0.131%** | **91.218 / 91.249** |
| 256 | 5 | 2.257612 | 2.259078 | 0.003938 | 2.253057–2.261157 | 0.174% | 89.177 / 89.119 |
| 512 | 5 | 2.337085 | 2.335803 | 0.002165 | 2.335330–2.339506 | 0.093% | 86.144 / 86.192 |
| 1024 | 5 | 3.331965 | 3.329194 | 0.004461 | 3.328536–3.338169 | 0.134% | 60.423 / 60.473 |

五轮最快配置分布为：`block 128 = 5/5`。每轮 `256 - 128` 延迟差分别为 `0.054815、0.056388、0.054841、0.043340、0.043235 ms`；均值差 `0.050524 ms`，约为 block 256 均值的 `2.238%`。该差异在五轮方向一致，并明显大于 block 128/256 各自 `0.002887/0.003938 ms` 的 run-to-run 标准差。因此：

> block 128 在当前记录的设备状态和本测试口径下表现为稳定 observed fastest configuration。

该结论不外推为所有设备、功耗模式或全部 CUDA Vector Add 的通用最优 block。

### 17.4 Nsight 权限审计

按授权首先执行 `sudo -n true`，返回 exit 1：`sudo: a password is required`。因此严格按任务要求停止 Nsight 部分，没有提示或猜测密码、没有修改 sudoers、没有运行四个 profile，也没有生成 `.ncu-rep`。

`Nsight Gate BLOCKED: interactive sudo authentication required`。证据见 `benchmark/ncu_sudo_status.txt`。

因此 block 32/128/256/1024 都没有 achieved occupancy、DRAM throughput、SM throughput、warp stall 或 transaction/sector 硬件计数器。Q1–Q3 的微架构解释仍只能由 benchmark 与 Occupancy API 约束，不能冒充 profiler 结论。

### 17.5 Theoretical 与 achieved occupancy

| Block | Active blocks/SM（API） | Active warps/SM（API 推导） | Theoretical occupancy | Achieved occupancy |
| ---: | ---: | ---: | ---: | --- |
| 32 | 16 | 16 | 33.33% | 未采集 |
| 128 | 12 | 48 | 100.00% | 未采集 |
| 256 | 6 | 48 | 100.00% | 未采集 |
| 1024 | 1 | 32 | 66.67% | 未采集 |

Theoretical occupancy 来自 CUDA Occupancy API 的资源驻留上限；achieved occupancy 必须来自 profiler 实际执行。本轮没有 achieved occupancy。tegrastats 的 `GR3D_FREQ` 也不是 occupancy。

### 17.6 H1–H4 审计

| 假设 | 最终判断 | 代码证据 | Benchmark 证据 | Occupancy API 证据 | Nsight 证据 |
| --- | --- | --- | --- | --- | --- |
| H1：block size 影响性能，更大不一定更快 | **SUPPORTED** | 同一 kernel 仅改变 launch block | 六个 block 的均值差异明显；128 为 5/5 最快，256/512/1024 更大但更慢 | 不同 block 形成 33.33%–100% 理论 occupancy | 无 |
| H2：higher occupancy 不自动等于 higher performance | **SUPPORTED** | kernel 资源用量相同 | 128/256/512 均值分别 2.207088/2.257612/2.337085 ms | 三者 theoretical occupancy 均 100% | achieved occupancy 未采集 |
| H3：Vector Add 是 memory-bound | **PARTIALLY SUPPORTED** | 每元素约 1 FLOP、12 B，算术强度约 0.0833 FLOP/B | 大规模有效带宽在 block 128 达 91.218 GB/s，计算量极低；但 effective bandwidth 不是 DRAM counter | Occupancy 提升后性能出现带宽型平台，但不是直接证明 | 无 DRAM/SM throughput counter |
| H4：连续地址具有良好 coalescing | **PARTIALLY SUPPORTED** | warp 内 thread i 连续访问 A[i]/B[i]/C[i] | 结果与连续访问预期相容，但 latency/带宽不能直接证明 transaction efficiency | 不适用 | 无 request/sector/transaction counter |

没有在未证明当前模式理论带宽的前提下计算 `92.7/102` 一类正式利用率。

### 17.7 Exp01.1 Gate

| Gate | 状态 | 依据 |
| --- | --- | --- |
| Gate A — Correctness | **PASS / FROZEN** | Original Exp01 77/77、最大误差 0；本轮未重跑完整 correctness |
| Gate B — Stability | **PASS** | 6 blocks × 5 independent rounds 完成；raw、summary、设备状态已保存 |
| Gate C — Nsight | **BLOCKED** | 非交互 sudo 认证失败，未获得任何 profiler counter；非代码错误 |
| Exp01 Overall | **PARTIAL** | Gate A 与 B 通过，Gate C 受外部权限阻塞 |

实现同时修复 `csvEscape()` 对原始双引号多写一个 quote 的问题；最小测试验证 `abc → abc` 与 `a"b → "a""b"` 均 PASS。旧 CSV 字段未触发该 bug，历史数据未删除、未覆盖、未重跑。

## 18. Exp01.2 Nsight Compute Gate Closure

Gate C 已在 `2026-08-30` 恢复并关闭。完整方法、原始指标解释和 128/256 分析见 `notes/exp01_2_nsight_compute.md`；统一数值摘要见 `benchmark/ncu_profile_summary_20260830T144903Z.csv`，Git 可审查 raw/details 位于 `benchmark/profiler/20260830T144618Z/` 与 `benchmark/profiler/20260830T144903Z/`。四个 `.ncu-rep` 仅保留在 Jetson `/tmp/jetson-qwen-exp01-ncu/`，未提交 Git。

### 18.1 Environment and methodology

- CUDA 12.6 / Nsight Compute 2024.3.1.0。
- `sudo -n ncu --version` 与 `sudo -n ncu --list-sections` PASS；未修改 sudoers。
- 原 workload：`N=16,777,216`、FP32、warmup 20、repetitions 200、原 `vectorAddKernel`。
- Blocks：32、128、256、1024。
- Sections：`LaunchStats`、`Occupancy`、`SpeedOfLight`、`MemoryWorkloadAnalysis_Tables`、`SchedulerStats`、`WarpStateStats`、`SourceCounters`。
- kernel filter 为 `regex:vectorAddKernel`；跳过 20 个 warmup launch，只 profile 第一个 measured launch。NCU replay latency 不作为 benchmark latency。
- 四个 profile 均成功，target correctness 均 PASS、最大误差 0。

### 18.2 Key profiler comparison

| Metric | 32 | 128 | 256 | 1024 |
| --- | ---: | ---: | ---: | ---: |
| Existing benchmark mean latency (ms) | 6.698908 | 2.207088 | 2.257612 | 3.331965 |
| Theoretical occupancy (%) | 33.33 | 100.00 | 100.00 | 66.67 |
| Achieved occupancy (%) | 24.69 | 85.85 | 83.37 | 57.25 |
| Theoretical active blocks/SM | 16 | 12 | 6 | 1 |
| Theoretical / achieved active warps/SM | 16 / 12.59 | 48 / 40.36 | 48 / 39.36 | 32 / 28.04 |
| Registers/thread | 16 | 16 | 16 | 16 |
| Static / dynamic shared memory | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| SM throughput (%) | 6.65 | 21.21 | 21.14 | 15.07 |
| Memory / L2 throughput (%) | 32.75 | 82.45 | 87.38 | 48.88 |
| DRAM throughput | N/A | N/A | N/A | N/A |
| Main warp stall | long scoreboard | long scoreboard | long scoreboard | long scoreboard |
| Load/store bytes per sector | 32 / 32 | 32 / 32 | 32 / 32 | 32 / 32 |

NCU 在该集成平台将直接 `DRAM Throughput` 报为 N/A，因此没有估算该值。对 128/256，SM throughput 近似相同，memory throughput 不支持“256 memory 利用更差”，stall 与 coalescing 也近似；profile SM frequency 还分别为 509.98/407.99 MHz。故不能从当前 metrics 建立可靠的 128 更快因果链。

**128 vs 256 final Case C：`microarchitectural cause remains inconclusive`。**

### 18.3 Final hypothesis and Gate status

| Item | Final status |
| --- | --- |
| H1 | `SUPPORTED` |
| H2 | `SUPPORTED` |
| H3 | `SUPPORTED` — memory throughput 明显高于 SM throughput，且 long scoreboard 主导 |
| H4 | `SUPPORTED` — 32 B/sector，L2 theoretical sectors 等于 ideal，excessive sectors 为 0 |
| Gate A — Correctness | `PASS / FROZEN` |
| Gate B — Stability | `PASS` |
| Gate C — Nsight | `PASS` |
| Exp01 Overall | `PASS` |

Exp01 已关闭并标记 `READY_FOR_EXP02`。本轮未开始 Exp02。

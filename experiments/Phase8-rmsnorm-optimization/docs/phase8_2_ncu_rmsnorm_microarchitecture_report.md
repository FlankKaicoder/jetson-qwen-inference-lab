# Phase 8.2-B RMSNorm Nsight Compute Microarchitecture Analysis

## Scope and Gate

This was the owner-authorized Phase 8.2-B follow-up to the Phase 8.2-A
permission audit. It profiled the standalone synthetic RMSNorm V0/V1/V2
kernels with Nsight Compute. No system permission, device node, clock, power
mode, TensorRT engine, ONNX graph, Qwen3 runtime, or kernel source was changed.

The result is **PASS / BOUNDED**: root NCU profiling succeeded and the
microarchitectural evidence explains the relative behavior of this synthetic
kernel. It does not demonstrate a full-model Qwen3 speedup or authorize a
TensorRT Plugin replacement.

## Profiling Environment

| Field | Value |
| --- | --- |
| Host/device | `nvidia-desktop`, Jetson Orin Nano Super |
| Compute capability | SM 8.7 |
| CUDA | 12.6.68 |
| Nsight Compute | 2024.3.1.0 (`/usr/local/cuda-12.6/bin/ncu`) |
| Remote checkout | `phase/03e-tensorrt-kernel-attribution@bf7abc67eb58662a68316045e166aa9f611330d7` (unchanged) |
| Target | `rmsnorm_ncu_target`, hidden size 1024, FP16 |
| Shapes | prefill `[1,8,1024]`; decode `[1,1,1024]` |
| NCU controls | `sudo -n ncu --clock-control none --set full --launch-count 1` |

The owner authorized a one-time root launch because ordinary NCU could not
launch under the Phase 8.2-A permission state. `sudo -n id` and all six NCU
launches succeeded. No `chmod`, group, modprobe, clock, or power change was
performed.

## Kernel Versions

- **V0**: one block per token, 1024 threads, scalar loads/stores, shared-memory
  tree reduction, dynamic shared memory 4096 B.
- **V1**: one block per token, warp-shuffle reduction plus one float per warp in
  shared memory, dynamic shared memory 128 B.
- **V2**: 512 threads per token, `half2` vectorized loads/stores and
  warp-shuffle reduction, dynamic shared memory 64 B.

## NCU Results

The table uses the prefill profile as the primary comparison. Values are the
single profiled launch; profiler duration is not the Phase 8.1 five-trial CUDA
Event benchmark.

| Version | Kernel duration (us) | Block | Registers/thread | Achieved active warp (%) | SM memory throughput (%) | Global sectors / ideal | Excessive sectors | Instructions |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| V0 | 25.664 | 1024 | 16 | 65.43 | 18.51 | 2048 / 2048 | 0 | 44,520 |
| V1 | 22.656 | 1024 | 18 | 65.32 | 6.69 | 2048 / 2048 | 0 | 29,520 |
| V2 | 20.448 | 512 | 20 | 29.78 | 3.56 | 1536 / 1536 | 0 | 14,672 |

Decode durations were V0 `26.464 us`, V1 `23.072 us`, and V2 `19.872 us`.
The complete six-profile data, including both shapes and all requested raw
counter fields, is in `artifacts/phase8_2_20260908T/ncu_summary.csv` and the
six `ncu_*_raw.csv` files.

## Memory Analysis

Global load and store efficiency counters report `32` bytes per sector for all
versions and both shapes. L2 theoretical global sectors equal their ideal
values, with zero excessive sectors. V0 and V1 issue the scalar access pattern
and move 2048 sectors for eight tokens. V2 processes two FP16 values per
thread, reducing the eight-token traffic to 1536 sectors and halving the
global load/store instruction counts. This is direct evidence for a lower
memory-instruction and traffic cost in V2.

The direct DRAM read-throughput counter is `0` in these reports. On this
integrated platform that counter is not a usable DRAM bandwidth measurement;
DRAM throughput is therefore **N/A**, not zero-bandwidth evidence.

## Warp and Occupancy Analysis

V1 keeps essentially the same achieved active-warp level as V0 (65.32% vs
65.43% for prefill), so its gain is not an occupancy increase. V1 removes the
full tree's repeated shared-memory reduction traffic: shared load/store
instructions fall from `840/552` to `264/264` for eight tokens.

The reported average stall ratios are:

| Version | Long scoreboard | Barrier | Not selected |
| --- | ---: | ---: | ---: |
| V0 | 5.640 | 5.538 | 1.596 |
| V1 | 8.499 | 6.508 | 1.342 |
| V2 | 3.174 | 5.813 | 0.581 |

Long-scoreboard and barrier ratios are issue-active normalized samples, not
percentages. V0-to-V1 speedup is consistent with fewer reduction barriers and
shared-memory instructions even though the sampled long-scoreboard ratio is
higher. V2 has the lowest long-scoreboard and not-selected ratios, but its
block size also changes from 1024 to 512; the profile supports the combined
vectorization/thread-count change, not vectorization in isolation.

V2's achieved active warp is lower (29.78%) because it launches half as many
threads per block. This does not prevent lower latency because each thread
performs a vector operation and the kernel executes substantially fewer
instructions.

## Instruction Analysis

For prefill, total instructions are V0 `44,520`, V1 `29,520`, and V2
`14,672`. V1's reduction is principally shared-memory work: shared
load/store instructions are `840/552` for V0 versus `264/264` for V1. V2
further reduces global loads/stores from `768/256` to `256/128` for eight
tokens, consistent with vectorized memory operations. LSU pipe utilization is
`26.81%`, `11.09%`, and `6.01%` respectively.

## Why V1 and V2 Are Faster

**V1 vs V0:** both use the same scalar global-memory pattern and the same
1024-thread block. Warp-shuffle reduction replaces the full shared-memory tree,
cutting shared-memory instructions by roughly 50-69% in the profiled launch.
The lower synchronization and shared-memory work explains the measured
`22.656 us` versus `25.664 us` prefill duration.

**V2 vs V1:** `half2` loads/stores process two FP16 elements per thread. Global
instruction counts and theoretical sectors decrease, and total instructions
fall by about 50%; the measured prefill duration falls to `20.448 us`. V2 also
uses a 512-thread block and different register/shared-memory shape, so the
isolated causal contribution of vectorization is **INCONCLUSIVE**. The combined
implementation is nevertheless faster in both prefill and decode profiles.

## Limitations

1. This is a standalone hidden-size-1024 synthetic target; it is not wired into
   the Qwen3 TensorRT graph or runtime.
2. NCU runs use unmodified GPU clocks. Relative comparisons are bounded to this
   profiling session; profiler duration is not a production latency claim.
3. The integrated Jetson DRAM counter is unavailable/unusable here; no DRAM
   bandwidth number is inferred from effective or L2 throughput.
4. NCU samples one launch per profile. Phase 8.1 correctness evidence remains
   the formal 12-case FP16/BF16 correctness gate.

## Phase 8.2 Gate

```text
PASS / BOUNDED
```

Root profiling succeeded for V0/V1/V2 at both requested shapes. The evidence
supports the V1 reduction/synchronization explanation and supports a combined
V2 vectorized-access/thread-shape explanation. Exact V2 attribution and any
full-model benefit remain unproven.

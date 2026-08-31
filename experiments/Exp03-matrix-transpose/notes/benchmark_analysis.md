# Exp03.2 Formal Benchmark Analysis

## Method

Six dimensions (1024x1024, 2048x2048, 4096x4096, 4093x4093, 4096x2048, 2048x4096) were measured for V1-V4. Each trial used 20 warmups and 100 CUDA Event timed kernel launches; allocation and transfers were excluded. Five trials used deterministic version rotation. Useful effective bandwidth is `2*width*height*4/time`; it is not direct DRAM throughput.

## Controlled repeat: 4096x4096

| Version | Mean ms | Median ms | CV % | Effective GB/s |
| --- | ---: | ---: | ---: | ---: |
| V1 | 2.063998 | 1.965309 | 10.251 | 65.028 |
| V2 | 9.986927 | 9.986102 | 0.131 | 13.439 |
| V3 | 2.742423 | 2.761392 | 1.406 | 48.941 |
| V4 | 1.576546 | 1.581530 | 6.567 | 85.134 |

Observed mean ratios: V2/V1=4.838x slower, V3/V2=3.642x faster, V4/V3=1.739x faster, V4/V1=1.309x faster. These are benchmark observations, not microarchitectural conclusions.

## Gate B

B1 correctness sanity: PASS. B2 timing validity: PASS. B3 stability: INCONCLUSIVE. The controlled repeat retained CV >5% for V1/1024^2 (13.88%), V3/1024^2 (21.70%), V4/1024^2 (18.19%), V1/4096^2 (10.25%), and V4/4096^2 (6.57%). No trial was deleted and power/clocks were not changed. Gate C was not started because Gate B is unresolved.

## Exp03.2b Stability Diagnosis

The historical method used 20 warmups and 100 measured launches regardless of kernel duration. The replacement method independently calibrates every version/dimension with CUDA Events, performs a time-based warmup, then times a launch batch outside allocation, copies, telemetry and file I/O. Seven trials use deterministic rotation and retain every observation.

The valid Diagnostic A (`20260831T112500Z`) uses a 1000 ms warmup target, 500 ms measurement target, 20/20000 warmup bounds, 100/10000 measurement bounds, and a 2.0 iteration safety factor. The safety factor is necessary because short calibration batches were slower than sustained execution; all valid A windows were at least 788.957 ms. Earlier `20260831T103500Z` diagnostic and `/tmp/jetson-qwen-exp03-benchmark-v2-20260831T104800Z` attempts are retained but rejected from Gate statistics because some actual windows were below 500 ms.

| Configuration | Historical CV % | Diagnostic A CV % | Change pp |
| --- | ---: | ---: | ---: |
| V1 / 1024^2 | 13.880 | 0.535 | -13.345 |
| V2 / 1024^2 | 4.796 | 0.300 | -4.496 |
| V3 / 1024^2 | 21.697 | 0.336 | -21.361 |
| V4 / 1024^2 | 18.195 | 0.373 | -17.822 |
| V1 / 4096^2 | 10.251 | 0.487 | -9.764 |
| V2 / 4096^2 | 0.131 | 0.111 | -0.020 |
| V3 / 4096^2 | 1.406 | 0.426 | -0.981 |
| V4 / 4096^2 | 6.567 | 1.047 | -5.520 |

No Diagnostic B was run because all four 4096^2 configurations passed CV <= 5% in A. The short fixed measurement window is a `SUPPORTED CONTRIBUTOR`, not proven to be the sole cause. Diagnostic A tegrastats reported GPU temperature 50.406-59.062 C and VDD_IN 8830-16498 mW. GPU and EMC frequencies were unavailable in these post-workload point samples, so device-state/DVFS contribution is `INCONCLUSIVE`; telemetry correlations are not interpreted causally.

## Formal Benchmark V2

The valid Formal Benchmark V2 (`20260831T113200Z`) applies the same adaptive method to all six dimensions. It retained 168/168 trials, all 24 configuration CVs were <= 0.791%, the minimum actual timed window was 793.641 ms, and pre/post V1-V4 sanity passed.

| Version | Mean ms (4096^2) | Median ms | CV % | Useful effective GB/s |
| --- | ---: | ---: | ---: | ---: |
| V1 | 1.961324 | 1.957106 | 0.657 | 68.432 |
| V2 | 10.018314 | 9.989181 | 0.535 | 13.397 |
| V3 | 2.702256 | 2.696119 | 0.528 | 49.669 |
| V4 | 1.425565 | 1.428211 | 0.791 | 94.151 |

V2 is 5.108x slower than V1; V3 is 3.707x faster than V2; V4 is 1.896x faster than V3; V4 measured latency is 1.376x lower than V1. These remain timing observations. Useful effective bandwidth is not direct DRAM throughput, and the V4/V1 mechanism requires profiler evidence.

B1 correctness sanity: `PASS`. B2 timing validity: `PASS`. B3 stability: `PASS`. Gate B: `PASS`.

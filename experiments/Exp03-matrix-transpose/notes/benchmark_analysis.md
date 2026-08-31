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
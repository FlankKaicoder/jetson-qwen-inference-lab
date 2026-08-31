# Exp03 Matrix Transpose Experiment Design

## Scope and fixed semantics

Exp03 studies row-major FP32 matrices. Input has `height x width` elements and uses `input[y * width + x]`. The transpose output has `width x height` elements and uses `output[x * height + y]`. Width and height retain these meanings in every version; all index arithmetic uses `std::size_t`.

The mechanism chain is row-major global access -> warp access pattern -> global coalescing -> shared-memory tiling -> global address reordering -> shared bank mapping -> +1-column padding.

## Versions and mechanisms

| Version | Mechanism | Expected role |
| --- | --- | --- |
| V0 | CPU `out[x * height + y] = in[y * width + x]` | bitwise reference |
| V1 | Direct copy, contiguous load/store | coalesced global-memory control; not a transpose baseline |
| V2 | Direct transpose | coalesced input read, height-strided output store |
| V3 | `tile[32][32]`, barrier, swapped tile coordinates | reorder global stores through shared memory; unpadded shared transpose |
| V4 | `tile[32][33]`, otherwise identical V3 | alter bank mapping with one-column padding |

All kernels use `TILE_DIM=32`, `BLOCK_ROWS=8`, `block(32,8)`, and guard both global loads and stores. V3/V4 always execute an unconditional `__syncthreads()` between tile write and tile read, including partial tiles.

## Correctness matrix

The formal dimensions are: 1x1; 1x17; 17x1; 7x13; 13x7; 31x31; 32x32; 33x33; 31x32; 32x31; 32x33; 33x32; 63x65; 65x63; 127x129; 129x127; 511x513; 513x511; 997x1000; 1000x997; 2048x2048; 4093x4096; 4096x4093.

Patterns are deterministic coordinate-coded values, sequential values, and fixed-seed signed values (`seed=0x03C0FFEE`). No NaN or Inf is generated. Every version x dimension x pattern is run three independent times: 23 x 3 x 4 x 3 = 828 executions.

Correctness is bitwise exact. V1 compares output bytes with input bytes; V2-V4 compare output bytes with the CPU reference bytes. No floating-point tolerance is used.

The output allocation has 16 guard words before and after the logical output, initialized to fixed `0x7FC12345` bit patterns and checked after synchronization. This catches many OOB writes but is not memcheck equivalence. Runtime checks cover `cudaMalloc`, `cudaMemcpy`, kernel launch, `cudaGetLastError`, and `cudaDeviceSynchronize`.

## Frozen hypotheses

- H1: V2 has worse global transaction/coalescing efficiency than V1 because output stores are height-strided. This requires Gate C evidence.
- H2: V3 improves global-store coalescing over V2 and can reduce large-matrix transpose latency. This requires Gate B + C.
- H3: V3's 32-column shared tile produces measurable bank conflicts. If the installed NCU counter is unsupported, status is `INCONCLUSIVE`.
- H4: V4's 33-column tile significantly reduces bank conflicts and latency versus V3 only when both bank-conflict and benchmark evidence support it.
- H5: V4 may remain slower than V1 because transpose adds shared staging, synchronization and address reordering; V1 is a control, not a transpose winner.

No hypothesis is a conclusion before the relevant Gate.

## Gate A result

The final correctness run (`correctness_20260831T044304Z.csv`) passed 828/828 executions: each of V1-V4 passed 207/207 cases, with zero guard failures and zero CUDA failures. An initial V1/V2 non-tiled `grid_y` bug left rows uncovered for heights greater than `BLOCK_ROWS`; it was fixed by using `ceil(height/BLOCK_ROWS)` and the complete matrix was rerun. The failed first attempt remains retained as historical evidence on Jetson and is not used for the final PASS.

## Gate definitions and deferred work

Gate A requires all 828 bitwise passes, guard PASS, CUDA runtime PASS, rectangular/partial/tiny/large coverage and source mechanism audit. Gate B and C remain `NOT_STARTED` in this phase. Future benchmark candidates are 1024x1024, 2048x2048, 4096x4096, 4093x4096 and 4096x4093 using CUDA Events, warmup, repetitions and rotating order; no timing is run here. Future NCU work will use only metrics actually supported by Jetson's installed Nsight Compute and will separate effective matrix bandwidth from direct DRAM throughput.

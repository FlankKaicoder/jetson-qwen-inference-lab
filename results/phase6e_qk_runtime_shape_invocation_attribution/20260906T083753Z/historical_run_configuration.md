# Historical Run Configuration

## Reproduction Decision

The Phase 6-E run is a `NEW_CONTROLLED_RUN`. It uses the exact historical
runtime and workload sequence but is not claimed to be byte-for-byte
reproducible. Historical raw trace identity is preserved separately.

## Engine Chain

| Role | Historical path | Evidence level | Identity |
| --- | --- | --- | --- |
| Embedding | `/tmp/phase2_2c1_20260902T090000Z/embedding_fp16.engine` | HISTORICAL_EVIDENCE | SHA256 `f13851d5236767e62ef1f596fe95d4f1f721282863f49f1751682a7cd615a8c9` |
| Final RMSNorm | `/tmp/phase2_2c2_20260903T/final_rmsnorm_fp32_reduce.engine` | HISTORICAL_EVIDENCE | SHA256 `27724efd7cde97b3b6a71b5722cc2477091770a509c912e872523caf1f7def83` |
| LM head | `/tmp/phase2_2c3_20260903T024500Z/lm_head_fp16.engine` | HISTORICAL_EVIDENCE | SHA256 `ff9c0dd793345b922083be0a650dae7b5f415411f7ae365ef49ecef7d897a6bd` |
| Mixed prefill decoder | `/tmp/phase2_3e_20260904T020000Z/work/mixed_prefill_28layer.engine` | HISTORICAL_EVIDENCE | SHA256 `3258b0a82fd4884a291e2bf0e3dded179434a9d1a5d034c4fc4e9f772c8964ec` |
| Mixed decode decoder | `/tmp/phase2_3e_20260904T020000Z/work/mixed_decode_28layer.engine` | HISTORICAL_EVIDENCE | SHA256 `445fc7d295c5bbb91e5392182347aa0e59612a031b5556a3461e09f30a59005c` |

## Frozen Runtime Facts

| Field | Value | Evidence level |
| --- | --- | --- |
| Runtime lifetime | `persistent_context_lifetime` | HISTORICAL_EVIDENCE |
| TensorRT | `10.3.0` | HISTORICAL_EVIDENCE |
| CUDA | `12.6` | HISTORICAL_EVIDENCE |
| PyTorch | `2.5.0a0+872d972e41.nv24.08` | HISTORICAL_EVIDENCE |
| Nsys | `2024.5.4.34-245434855735v0` | HISTORICAL_EVIDENCE |
| Power mode | `NV Power Mode: 25W` | HISTORICAL_EVIDENCE |
| Profile workload | two prefill warmups, one steady prefill S=8, decode steps 0-3 | HISTORICAL_EVIDENCE |
| Profile mode | `mixed_persistent_context_lifetime` | HISTORICAL_EVIDENCE |

## Input Configuration

| Field | Value | Evidence level |
| --- | --- | --- |
| Manifest | `experiments/Phase2-qwen3-quantization/artifacts/phase2_3b_20260903T205610Z/evaluation_manifest.json` | HISTORICAL_EVIDENCE |
| Selection | first row with `split == "evaluation"` | HISTORICAL_EVIDENCE |
| Sample ID | `eva_025` | HISTORICAL_EVIDENCE |
| First 8 token IDs | `[840, 20772, 54809, 23045, 304, 825, 11652, 624]` | HISTORICAL_EVIDENCE |
| Forced continuation | `experiments/Phase2-qwen3-quantization/src/phase2_3f/force_cont.json` = `[21806, 0, 358, 2776, 11, 14582, 2585, 1184]` | HISTORICAL_EVIDENCE |
| Batch | 1 | HISTORICAL_EVIDENCE |
| Prefill length | 8 | HISTORICAL_EVIDENCE |
| Decode count | 4 | HISTORICAL_EVIDENCE |
| Decode cache progression | code-derived `old_len = 8 + step` | CODE_DERIVED |

## Historical Harness Facts

The historical profile used `phase3b_runtime_context.py` in profile mode and
the reusable `phase2_3f_compare.py` TensorRT pipeline. Each engine execution
created or reused one execution context, set input shapes and tensor addresses,
allocated output tensors from context-resolved shapes, called
`execute_async_v3`, and synchronized the current CUDA stream. The Mixed
persistent profile reported five context creations and reused contexts across
executions.

## Historical Launch Context

Phase 6-C recovered exactly eleven `/MatMul` launches: seven
`trt_ampere_h16816gemm_128x64_ldg8_nn_v1` and four
`sm80_xmma_gemm_f16f16_f16f32_f32_nn_n_tilesize32x32x64_stage6_warpsize2x2x1_tensor16x8x16_aligna2_alignc2_execute_kernel_trt`.
For decode steps 0-3, an xmma launch preceded an h16816 launch. Their immediate
NVTX parents differed. Therefore same graph region, engine invocation, and
mathematical workload remain `UNKNOWN` for the historical pair.

## Unrecovered Historical Fields

| Field | Value |
| --- | --- |
| Historical runtime Q shape | UNKNOWN |
| Historical runtime K shape | UNKNOWN |
| Historical runtime QK output shape | UNKNOWN |
| Historical effective GEMM dimensions | UNKNOWN |
| Historical kernel argument identity | UNKNOWN |
| Exact historical input byte identity | UNKNOWN |
| Exact historical allocator layout identity | UNKNOWN |

These fields remain `UNKNOWN` because the historical Nsys SQLite schema does not
record kernel arguments or runtime tensor shapes.

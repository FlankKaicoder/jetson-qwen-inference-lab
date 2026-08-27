# Exp01.1 Stability and Nsight Audit Notes

## Scope

Exp01.1 只审计 Original Exp01 的设备状态、独立轮次稳定性和 Nsight 证据。没有修改 kernel 算法，没有重跑或覆盖旧 correctness/benchmark，没有开始 Exp02。

## Read-only device state

- UTC pre-run：`2026-08-27T07:54:33Z`。
- `nvpmodel -q`：`25W`，mode ID `1`。
- GPU devfreq：pre-run current/min/max = `306/306/918 MHz`；available list 还包含 `1020 MHz`。
- CUDA device property memory clock：`918000 kHz`。debugfs EMC rate 对非 root 用户不可读。
- `jetson_clocks --show`：非 root exit 1；没有使用 sudo，没有改变 clocks。GPU min/max 不相等，GPU 未静态锁频。
- tegrastats：pre-run GPU 约 `49.8–50.0°C`；post-run 最高样本 `54.718°C`。
- `GR3D_FREQ` 只作为状态信息，不当作 occupancy。

完整原始输出：`../benchmark/raw/environment_exp01_1_20260827T075423Z.txt`。

## CSV escaping

原实现命中 quote 时先追加两个 quote，随后又无条件追加原 quote，导致三 quote。修复为 quote 分支追加两个 quote、非 quote 分支追加原字符。`--test-csv-escape` 验证：

- `abc` → `abc`
- `a"b` → `"a""b"`

结果：PASS。证据：`csv_escape_test_20260827T075423Z.txt`。

## Stability protocol and result

- `N=16777216`，FP32，warmup 20，repetitions 200，CUDA Event kernel-only。
- 5 个独立 round；顺序严格为：
  - R1 `32 64 128 256 512 1024`
  - R2 `128 256 512 1024 32 64`
  - R3 `512 1024 32 64 128 256`
  - R4 `256 128 64 32 1024 512`
  - R5 `1024 512 256 128 64 32`
- 30/30 correctness PASS，全部 max error 0。
- Block 128 是 5/5 round 最快。
- Block 128 mean/sample-std = `2.207088/0.002887 ms`。
- Block 256 mean/sample-std = `2.257612/0.003938 ms`。
- 均值差 `0.050524 ms`；逐轮差 `0.043235–0.056388 ms`，五轮方向一致，明显大于 run-to-run variation。

Raw：`../benchmark/stability_raw_20260827T075423Z.csv`。Summary：`../benchmark/stability_summary.csv`。

## Nsight audit

`sudo -n true` 返回：

`sudo: a password is required`

因此：

- `Nsight Gate BLOCKED: interactive sudo authentication required`。
- 没有交互式请求或猜测 sudo 密码。
- 没有修改 sudoers 或系统 profiler 配置。
- 没有对 block 32/128/256/1024 运行 profile。
- 没有 `.ncu-rep` 进入 Git。

证据：`../benchmark/ncu_sudo_status.txt`。

由于没有 profiler counter，block 32 的低 latency-hiding、block 128/256 的相近行为、block 1024 的 resident-block 限制都只能由 benchmark 与 theoretical occupancy 解释。Achieved occupancy、DRAM/SM throughput、warp stalls、memory sectors/transactions 均未知。

## Final evidence judgment

- H1：SUPPORTED。
- H2：SUPPORTED。
- H3：PARTIALLY SUPPORTED；低算术强度和 benchmark 支持，但无 DRAM/SM counter。
- H4：PARTIALLY SUPPORTED；源码地址模式连续，但无 transaction/sector counter。
- Gate A：PASS / FROZEN。
- Gate B：PASS。
- Gate C：BLOCKED。
- Overall：PARTIAL。

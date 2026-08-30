# Experiment Index

这是实验状态的快速索引；详细结论以 report 和 result artifacts 为准。不得虚构实验、提前填写未发生的结果，或用本索引替代原始证据。

| Exp ID | Title | Branch | Status | Gate | Important commit | Report | Result directory | Key conclusion | Next dependency |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Exp00 | Environment & Repository Bootstrap | `main` | `PASS` | `PASS` | `bd5aad316e372d0660b9f133b2a0c58fdb78ee03` | Historical state in commit `bd5aad3` | `UNKNOWN` — no dedicated result directory | Windows and Jetson repository bootstrap and initial environment check completed; GitHub synchronization followed in `d42ab4a`. | Exp01 baseline |
| Exp01 | CUDA Vector Add + stability and Nsight audits | `exp/01-vector-add` | `PASS` | Overall `PASS`; A `PASS / FROZEN`, B `PASS`, C `PASS` | `249ddfb0a6873765bc391922111acfdd489e6d5c`, `74087eddc3815c45bae655978b57e99279dd4bd8`, `HEAD` (Gate C closure) | `experiments/Exp01-vector-add/README.md` | `experiments/Exp01-vector-add/benchmark/` | Block 128 fastest in 5/5 stability rounds; memory-bound and coalescing hypotheses supported; precise 128/256 cause remains inconclusive after complete NCU profiling. | Owner review; Exp02 may be designed only after explicit authorization. |

No other experiment directory or completed experiment result was present when this index was verified on `2026-08-30`. Exp02 has not started.

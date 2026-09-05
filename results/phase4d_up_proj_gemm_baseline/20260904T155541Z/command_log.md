# Phase 4-D Command Log

## Local preflight

```powershell
git status --short
git branch --show-current
git rev-parse HEAD
git log -1 --oneline
git remote -v
git diff --check
```

Observed local state:

- Branch: `phase/04a-tensorrt-operator-attribution-recovery`
- HEAD: `bf7abc67eb58662a68316045e166aa9f611330d7`
- Tracked diff: none
- Deleted tracked files: none
- Existing untracked artifact directories preserved.

## Remote read-only preflight

```powershell
ssh -o BatchMode=yes -o ConnectTimeout=8 jetson "hostname && pwd && git -C /home/nvidia/projects/jetson-qwen-inference-lab branch --show-current && git -C /home/nvidia/projects/jetson-qwen-inference-lab rev-parse HEAD && git -C /home/nvidia/projects/jetson-qwen-inference-lab status --short && test -f /tmp/phase2_3e_20260904T020000Z/work/mixed_decode_28layer.engine && echo ENGINE_PRESENT"
```

Observed:

- Host: `nvidia-desktop`
- Branch: `phase/03e-tensorrt-kernel-attribution`
- HEAD: `bf7abc67eb58662a68316045e166aa9f611330d7`
- Existing engine present.
- No remote repository modification was made.

## Standalone probe

Two ephemeral Python probes were streamed over SSH to the frozen
`/home/nvidia/.venvs/jetson-qwen-phase2-quant/bin/python` environment. They did
not write remote files, modify TensorRT, PyTorch, CUDA, ONNX, engines, or the
runtime.

The first probe measured both `torch.matmul` and
`torch.nn.functional.linear`. The second probe reran only `torch.matmul` with a
larger 10,000-iteration calibration to reduce startup sensitivity. Both probes
used CUDA Events, 7 trials, deterministic Half operands, and the exact shape
`M=1,K=1024,N=3072`.

Primary rerun result:

```text
torch.matmul median = 0.086364701 ms
torch.matmul CV = 0.045405%
```

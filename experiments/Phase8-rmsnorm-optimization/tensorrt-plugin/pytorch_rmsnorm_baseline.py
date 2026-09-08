#!/usr/bin/env python3
"""Measure the bounded PyTorch FP16 RMSNorm control for Phase 8.3-A."""

import json
import math

import torch
import torch.nn.functional as functional


TOKENS = 8
HIDDEN = 1024
EPSILON = 1.0e-6
WARMUP = 50
REPETITIONS = 200
TRIALS = 5
SEED = 20260908


def main() -> None:
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    x = torch.randn((1, TOKENS, HIDDEN), device="cuda", dtype=torch.float16)
    gamma = torch.randn((HIDDEN,), device="cuda", dtype=torch.float16)
    reference = x.float() * torch.rsqrt(x.float().square().mean(dim=-1, keepdim=True) + EPSILON)
    reference = reference * gamma.float()
    output = functional.rms_norm(x, (HIDDEN,), gamma, EPSILON)
    relative_l2 = torch.linalg.vector_norm((output.float() - reference).reshape(-1))
    relative_l2 /= torch.linalg.vector_norm(reference.reshape(-1))
    max_abs = (output.float() - reference).abs().max()

    for _ in range(WARMUP):
        functional.rms_norm(x, (HIDDEN,), gamma, EPSILON)
    torch.cuda.synchronize()

    trial_latency_ms = []
    for _ in range(TRIALS):
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(REPETITIONS):
            functional.rms_norm(x, (HIDDEN,), gamma, EPSILON)
        stop.record()
        stop.synchronize()
        trial_latency_ms.append(start.elapsed_time(stop) / REPETITIONS)

    mean = sum(trial_latency_ms) / len(trial_latency_ms)
    variance = sum((value - mean) ** 2 for value in trial_latency_ms) / len(trial_latency_ms)
    print(json.dumps({
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "shape": [1, TOKENS, HIDDEN],
        "dtype": "float16",
        "epsilon": EPSILON,
        "seed": SEED,
        "relative_l2_error": relative_l2.item(),
        "max_abs_error": max_abs.item(),
        "warmup": WARMUP,
        "repetitions_per_trial": REPETITIONS,
        "trials": TRIALS,
        "trial_latency_ms": trial_latency_ms,
        "mean_latency_ms": mean,
        "stddev_latency_ms": math.sqrt(variance),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

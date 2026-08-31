# Exp04 Initial Benchmark Analysis

The V1 benchmark uses per-shape calibration, approximately 1000 ms time-based warmup, an adaptive approximately 500 ms CUDA Event window (2x safety factor), and seven trials. Events surround only repeated kernel launches. Allocation, input generation, H2D, D2H, initialization and CPU reference work are outside the timed region. Raw trials retain warmup/measurement iterations, calibrated latency, actual event window, latency and GFLOPS. Summary statistics are mean, median, sample standard deviation, CV, min/max, mean GFLOPS, mean window and launch count.

GFLOPS is `2*M*N*K / latency_seconds / 1e9`. This stage makes no compute-bound or memory-bound claim; that requires later Nsight evidence.

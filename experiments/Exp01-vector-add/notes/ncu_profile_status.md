# Nsight Compute Profile Status

## Tool

- Path: `/usr/local/cuda-12.6/bin/ncu`
- Version: `2024.3.1.0`

## Commands executed

```bash
which ncu
ncu --version
ncu --list-sections
```

## Result

`ncu --list-sections` returned:

`Insufficient privileges to launch app for profiling. Launch app with root privileges`

The current user therefore could not enumerate the sections actually supported by this Nsight Compute installation. No metric or section name was guessed, no privileged command was attempted, and no `.ncu-rep` file was created.

Representative profiles for `N=16777216` with block sizes 32, 256, and 1024 were not launched because section discovery is a prerequisite.

## Gate

`Profiler Gate = BLOCKED`

This is an environment-permission limitation, not a CUDA implementation failure.

## Suggested follow-up

After an administrator explicitly enables non-root GPU performance-counter access, or explicitly authorizes a privileged profiling session:

1. Re-run `ncu --list-sections`.
2. Select only sections reported by this installed version.
3. Profile `N=16777216` for block 32, 256, and 1024.
4. Store large reports outside Git and commit only small text/CSV summaries.

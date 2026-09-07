# Phase 8.2-A NCU Permission Audit Report

## Audit Purpose

Phase 8.2-A was a read-only permission audit for Nsight Compute on the Jetson
Orin Nano Super. Its only goal was to determine whether the current user can
profile a CUDA kernel without a system-permission change.

It did not perform NCU performance analysis, modify CUDA kernels or benchmark
logic, modify TensorRT or Qwen3, change clock or power mode, or alter device
nodes, user groups, or modprobe configuration.

## Jetson Environment

| Field | Value |
| --- | --- |
| Host | `nvidia-desktop` |
| Device | Jetson Orin Nano Super |
| Compute capability | `8.7` |
| Nsight Compute | `2024.3.1.0` |
| NCU path | `/usr/local/cuda-12.6/bin/ncu` |
| Audit date | `2026-09-07` |

## User Permissions

The current user is:

```text
uid=1000(nvidia) gid=1000(nvidia)
```

Group membership includes:

```text
nvidia adm cdrom sudo audio dip video plugdev render i2c lpadmin gdm
sambashare weston-launch gpio jtop
```

The required `video` and `render` groups are present. Membership in `sudo`
is also present, but no elevated command was executed during this audit.

## Device Node Permissions

| Node | Observed permissions |
| --- | --- |
| `/dev/nvidia0` | `crw-rw-rw- root root` |
| `/dev/nvidiactl` | `crw-rw-rw- root root` |
| `/dev/nvidia-modeset` | `crw-rw-rw- root root` |
| `/dev/nvidia-caps/` | `NOT_AVAILABLE` — directory does not exist |

The ordinary NVIDIA device nodes are broadly readable and writable, but this
does not by itself grant NCU profiling privileges.

## NVIDIA Profiling Capability

The requested path was checked read-only:

```text
/proc/driver/nvidia/capabilities/profiling
```

Result:

```text
NOT_AVAILABLE
```

The file does not exist on this system. It was not created or modified.

## NCU Test Result

NCU was available at:

```text
/usr/local/cuda-12.6/bin/ncu
```

Version check:

```text
Version 2024.3.1.0 (build 34702747) (public-release)
```

A minimal smoke test was run against the existing, already-built
`rmsnorm_ncu_target` binary. It was not recompiled. The command used
`--clock-control none` and did not change device clock or power state.

Command:

```bash
/usr/local/cuda-12.6/bin/ncu \
  --clock-control none \
  --set basic \
  --launch-count 1 \
  /tmp/phase8_2_rmsnorm_20260907T100117Z/build/rmsnorm_ncu_target \
  --version V0 \
  --dtype float16 \
  --tokens 8
```

Result:

```text
==WARNING== Insufficient privileges to launch app for profiling.
Launch app with root privileges
```

No NCU report or microarchitecture metrics were collected.

## Blocking Cause

Although the current user is in `video` and `render`, NCU still cannot launch
the target process for profiling on this Jetson configuration. The existing
profiling-capability interface is unavailable, and NCU explicitly requires
root privileges or an equivalent system-level profiling authorization.

## Recommended Solution

No system change should be made without explicit owner authorization. The
next permitted options are:

1. Run one authorized elevated NCU smoke test as root.
2. Grant an explicitly bounded profiling permission for the current user.
3. Leave Phase 8.2 blocked if system-permission changes are not acceptable.

No `sudo ncu`, `sudo chmod`, device-node modification, user-group change, or
modprobe configuration change was attempted in this audit.

## Gate

```text
BLOCKED
```

Phase 8.2 remains blocked until the owner explicitly authorizes one of the
documented permission solutions.

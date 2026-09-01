#!/usr/bin/env python3
import argparse
import os
import signal
import subprocess
import time


stopping = False


def request_stop(_signum, _frame):
    global stopping
    stopping = True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--interval-ms", type=int, default=200)
    args = parser.parse_args()

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    child = subprocess.Popen(
        ["tegrastats", "--interval", str(args.interval_ms)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    try:
        with open(args.output, "w", encoding="utf-8", buffering=1) as output:
            output.write(f"# sampler_pid={os.getpid()} tegrastats_pid={child.pid}\n")
            while not stopping:
                line = child.stdout.readline()
                if not line:
                    if child.poll() is not None:
                        break
                    continue
                output.write(f"{time.time_ns()}\t{line.rstrip()}\n")
    finally:
        if child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=3)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()


if __name__ == "__main__":
    main()

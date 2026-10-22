#!/usr/bin/env bash
# Block until no other WSL process holds the GPU (/dev/dxg) open, so benchmarks
# don't time-slice with someone else's kernels. Windows-side GPU users are invisible
# from WSL; keep the desktop idle while measuring.
# Usage: tools/wait_gpu_quiet.sh [timeout_seconds]
timeout=${1:-3600}
start=$(date +%s)
self=$$
while true; do
    busy=""
    for p in /proc/[0-9]*; do
        pid=${p#/proc/}
        [ "$pid" = "$self" ] && continue
        if ls -l "$p/fd" 2>/dev/null | grep -q "/dev/dxg"; then
            busy="$busy $pid"
        fi
    done
    if [ -z "$busy" ]; then
        echo "gpu quiet"
        exit 0
    fi
    if [ $(( $(date +%s) - start )) -ge "$timeout" ]; then
        echo "still busy after ${timeout}s:$busy" >&2
        exit 1
    fi
    sleep 5
done

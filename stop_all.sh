#!/usr/bin/env bash
#
# stop_all.sh — stop every VA process launched by run_all.sh.
# Reads the PIDs recorded in run_all.pids, sends SIGTERM, then SIGKILL to any
# straggler that ignores it.

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIDFILE="$ROOT/run_all.pids"

if [ ! -f "$PIDFILE" ]; then
    echo "[stop_all] no run_all.pids found — nothing to stop."
    exit 0
fi

# Phase 1: polite SIGTERM.
while read -r pid; do
    [ -n "$pid" ] || continue
    if kill -0 "$pid" 2>/dev/null; then
        echo "[stop_all] SIGTERM -> PID $pid"
        kill "$pid" 2>/dev/null || true
    fi
done < "$PIDFILE"

# Give them a couple of seconds to shut down cleanly.
sleep 2

# Phase 2: hard-kill anything still alive.
while read -r pid; do
    [ -n "$pid" ] || continue
    if kill -0 "$pid" 2>/dev/null; then
        echo "[stop_all] SIGKILL -> PID $pid (did not exit)"
        kill -9 "$pid" 2>/dev/null || true
    fi
done < "$PIDFILE"

rm -f "$PIDFILE"
echo "[stop_all] done."

#!/usr/bin/env bash
#
# run_all.sh - thin wrapper around the Python supervisor (Linux/container).
#
# The multi-process launch is now owned by supervisor.py (invoked as
# `python main.py --supervise`). The camera GROUPS table, per-group core slicing
# (sized from the cores this process is allowed - sched_getaffinity, so it
# respects docker --cpuset-cpus), PID file, log redirection and teardown all
# live there. This script only resolves the venv python and forwards flags, so
# there is ONE source of truth for the topology (edit GROUPS in supervisor.py).
#
# NOTE: the previous version hard-pinned each group with `taskset`. The
# supervisor partitions cores SOFTLY via the *_NUM_THREADS caps instead, so
# taskset/util-linux is no longer required. Constrain cores with the container
# runtime (docker --cpuset-cpus) if you need hard isolation.
#
# USAGE
#   ./run_all.sh                       # start + supervise all groups (Ctrl+C stops all)
#   ./run_all.sh --reset-plates        # wipe ALL plate identities once, then start
#   ./run_all.sh --foreground          # mirror group logs to stdout (container PID 1)
#   tail -f ./logs/va_b1-gate.out.log  # follow a group's log
#   ./stop_all.sh                      # stop groups by PID file (if run detached)
#
# NOTE: the external 8080 server must POST ANPR events to
#   http://<this-host-ip>:8000/api/anpr/event   (NOT :8080, NOT localhost from
# another box) and port 8000 must be open in the host firewall (ufw/firewalld).

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"   # so main.py / config.yaml resolve regardless of caller's cwd

# Forward recognised flags straight through to `main.py --supervise`. Reject
# anything else so a typo doesn't silently launch the wrong topology.
SUP_ARGS=()
for arg in "$@"; do
    case "$arg" in
        --reset-plates)   SUP_ARGS+=(--reset-plates) ;;
        --foreground|-f)  SUP_ARGS+=(--foreground) ;;
        *) echo "[run_all] unknown arg: '$arg' (usage: ./run_all.sh [--reset-plates] [--foreground])" >&2; exit 1 ;;
    esac
done

# Prefer the project venv python; fall back to system python3/python.
PYTHON="$ROOT/.venv/bin/python"
if [ ! -x "$PYTHON" ]; then
    PYTHON="$(command -v python3 || command -v python || true)"
fi
if [ -z "${PYTHON:-}" ]; then
    echo "[run_all] ERROR: no python found (.venv/bin/python missing, no python3 on PATH)" >&2
    exit 1
fi

# exec so the supervisor replaces this shell: it becomes PID 1 under a container
# entrypoint and receives docker stop's SIGTERM directly.
exec "$PYTHON" main.py --supervise "${SUP_ARGS[@]}"

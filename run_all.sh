#!/usr/bin/env bash
#
# run_all.sh — Linux launcher for the Video Analytics pipeline as parallel
# per-process camera groups. This is the LINUX equivalent of run_all.ps1.
#
# WHY PER-PROCESS GROUPS (read this before asking "I have 16 cores, why split?")
#   A single VA process runs inference SERIALLY — one camera at a time, ~0.5s
#   per inference (FP32@640) on this box — so ALL cameras in one process share
#   ~2 inferences/sec no matter how many cores exist. One process cannot spread
#   that serial loop across 16 cores (the in-process threading that would have
#   done so was tried and reverted). Multiple OS processes = multiple concurrent
#   inference workers = the only current way to actually use all 16 cores.
#
# WHY taskset (the part that makes it work on a 16-core box)
#   With no pinning, EACH process's OpenVINO/ultralytics/OpenCV grabs a large
#   share of the 16 cores. Five processes then oversubscribe (~5x) and thrash —
#   every inference gets SLOWER, not faster. `taskset -c <range>` pins each
#   group to a DISJOINT set of cores so the 16 cores are partitioned, not
#   fought over. The *_NUM_THREADS env vars below size the math/BLAS thread
#   pools to match each group's core count.
#
# TOPOLOGY RULES (same as run_all.ps1)
#   * EXACTLY ONE group runs --api (api=1). That process receives the ANPR /
#     ramp / Park_Entry webhooks POSTed by the external 8080 server and binds
#     plates in its OWN in-memory registry, so it MUST own the entrance /
#     Park_Entry camera (CAM-23).
#   * All groups share the DB + on-disk gallery; cross-process identity flows
#     only through those. Keep cameras that must hand identity off live
#     (same floor) in the SAME group.
#
# USAGE
#   ./run_all.sh                       # start all groups
#   tail -f ./logs/va_b1-gate.out.log  # follow a group's log
#   ./stop_all.sh                      # stop everything started here
#
# NOTE: the external 8080 server must POST ANPR events to
#   http://<this-host-ip>:8000/api/anpr/event   (NOT :8080, NOT localhost from
# another box) and port 8000 must be open in the host firewall (ufw/firewalld).

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"   # so main.py / config.yaml resolve regardless of caller's cwd

# Prefer the project venv python; fall back to system python3/python.
PYTHON="$ROOT/.venv/bin/python"
if [ ! -x "$PYTHON" ]; then
    PYTHON="$(command -v python3 || command -v python || true)"
fi
if [ -z "${PYTHON:-}" ]; then
    echo "[run_all] ERROR: no python found (.venv/bin/python missing, no python3 on PATH)" >&2
    exit 1
fi

if ! command -v taskset >/dev/null 2>&1; then
    echo "[run_all] ERROR: 'taskset' not found. Install util-linux (apt-get install util-linux)." >&2
    exit 1
fi

LOGDIR="$ROOT/logs"
mkdir -p "$LOGDIR"

API_PORT=8000

# ---------------------------------------------------------------------------
# EDIT THESE to match your DB camera roster.
#   Fields (pipe-separated):  name | cameras | api(0/1) | cpu-range
#     name     : short label for log filenames
#     cameras  : comma-separated camera IDs passed to --cameras
#     api      : 1 on EXACTLY ONE group — the entrance/Park_Entry (CAM-23) owner
#     cpu-range: taskset -c spec (e.g. "0-3" or "0,2,4"). Ranges MUST be
#                disjoint and together cover the cores you want to use. Below
#                partitions all 16 cores (4+3+3+3+3). Give the --api/gate group
#                the extra core — it also serves webhooks + gate matching.
# ---------------------------------------------------------------------------
GROUPS=(
    "b1-gate|CAM-23,CAM-03,CAM-04,CAM-05,CAM-06,CAM-07|1|0-3"
    "b1-areas|CAM-08,CAM-20,CAM-21,CAM-22,CAM-24|0|4-6"
    "b2-1|CAM-09,CAM-10,CAM-11,CAM-12,CAM-13,CAM-14|0|7-9"
    "b2-2|CAM-15,CAM-16,CAM-17,CAM-18,CAM-19,CAM-25|0|10-12"
    "ground|CAM-00,CAM-01,CAM-02|0|13-15"
)
# ---------------------------------------------------------------------------

# Count cores in a taskset -c spec like "0-3" or "0,2,5-7".
count_cores() {
    local spec="$1" total=0 part
    IFS=',' read -ra parts <<< "$spec"
    for part in "${parts[@]}"; do
        if [[ "$part" == *-* ]]; then
            total=$(( total + ${part#*-} - ${part%%-*} + 1 ))
        else
            total=$(( total + 1 ))
        fi
    done
    echo "$total"
}

# Sanity: exactly one API host.
api_count=0
for g in "${GROUPS[@]}"; do
    IFS='|' read -r _n _c api _r <<< "$g"
    [ "$api" = "1" ] && api_count=$(( api_count + 1 ))
done
if [ "$api_count" -ne 1 ]; then
    echo "[run_all] ERROR: exactly one group must have api=1 (found $api_count). Fix GROUPS." >&2
    exit 1
fi

PIDFILE="$ROOT/run_all.pids"
: > "$PIDFILE"

for g in "${GROUPS[@]}"; do
    IFS='|' read -r name cams api cores <<< "$g"
    n="$(count_cores "$cores")"

    args=(main.py --cameras "$cams")
    label="$name"
    if [ "$api" = "1" ]; then
        args+=(--api --port "$API_PORT")
        label="$name (+API :$API_PORT)"
    fi

    out="$LOGDIR/va_${name}.out.log"
    err="$LOGDIR/va_${name}.err.log"

    echo "[run_all] starting '$label' on cores [$cores] (${n} threads) -> $cams"

    OMP_NUM_THREADS="$n" \
    OPENBLAS_NUM_THREADS="$n" \
    MKL_NUM_THREADS="$n" \
    NUMEXPR_NUM_THREADS="$n" \
    VECLIB_MAXIMUM_THREADS="$n" \
        taskset -c "$cores" nohup "$PYTHON" "${args[@]}" \
            >"$out" 2>"$err" &

    pid=$!
    echo "$pid" >> "$PIDFILE"
    printf '           pid=%-7s log=%s\n' "$pid" "$out"
done

echo
echo "[run_all] PIDs written to $PIDFILE"
echo "[run_all] follow a log:  tail -f $LOGDIR/va_b1-gate.out.log"
echo "[run_all] health check:  curl -s http://localhost:$API_PORT/api/health"
echo "[run_all] stop all:      ./stop_all.sh"

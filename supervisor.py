"""
supervisor.py — single-entry launcher for the per-process camera groups.

This is the Python equivalent of run_all.ps1 / run_all.sh: it spawns ONE OS
process per camera group (each an independent serial inference worker) and
supervises them from a single parent. It does NOT change the scaling model —
the process boundary is still the parallelism. It only folds the N separate
launches into one entry point with unified startup, core-slicing, logging and
teardown.

TOPOLOGY RULES (identical to run_all.ps1 — read before editing GROUPS)
  * EXACTLY ONE group runs --api. That worker owns the entrance / Park_Entry
    camera and receives the ANPR webhooks, binding plates in its OWN in-memory
    registry. Set "api": True on that group only.
  * All groups share the DB + on-disk gallery, so slot status, sessions and
    per-plate gallery folders are visible across every worker.
  * Cameras in DIFFERENT groups do NOT share live in-memory track state;
    cross-process identity flows only through DB + gallery. Keep cameras that
    hand identity off live (typically the same floor) in the SAME group.

USAGE
    python supervisor.py                  # start all groups, supervise, block
    python supervisor.py --reset-plates   # wipe ALL plate identities once first
    python supervisor.py --foreground     # also mirror child logs to stdout
                                          #   (use this as the Docker PID 1)
    Ctrl+C / SIGTERM                       # graceful stop of every child
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LOG_DIR = ROOT / "logs"
PID_FILE = ROOT / "run_all.pids"
API_PORT = 8000

# ---------------------------------------------------------------------------
# EDIT THESE to match your DB camera roster — same table as run_all.ps1.
#   name : short label used for log filenames
#   cams : comma-separated camera IDs passed to --cameras
#   api  : True on EXACTLY ONE group — the one owning the entrance camera
# ---------------------------------------------------------------------------
GROUPS = [
    {"name": "b1-gate",  "cams": "CAM-23,CAM-03,CAM-04,CAM-05,CAM-06,CAM-07", "api": True},
    {"name": "b1-areas", "cams": "CAM-08,CAM-20,CAM-21,CAM-22,CAM-24",        "api": False},
    {"name": "b2-1",     "cams": "CAM-09,CAM-10,CAM-11,CAM-12,CAM-13,CAM-14", "api": False},
    {"name": "b2-2",     "cams": "CAM-15,CAM-16,CAM-17,CAM-18,CAM-19,CAM-25", "api": False},
    {"name": "ground",   "cams": "CAM-00,CAM-01,CAM-02",                      "api": False},
]
# ---------------------------------------------------------------------------


def _cam_count(group: dict) -> int:
    return len([c for c in group["cams"].split(",") if c.strip()])


def _available_cores() -> int:
    """Cores this process is actually allowed to run on — NOT the host total.
    Under Docker --cpuset-cpus (or taskset), os.cpu_count() still reports every
    host CPU, which would over-slice the per-group thread budget and
    oversubscribe. sched_getaffinity reflects the cgroup/affinity mask; fall
    back to os.cpu_count() where it's unavailable (e.g. Windows)."""
    if hasattr(os, "sched_getaffinity"):
        try:
            return max(1, len(os.sched_getaffinity(0)))
        except OSError:
            pass
    return os.cpu_count() or 1


def _thread_slice(cam_count: int, total_cams: int, total_cores: int) -> int:
    """Cores handed to a group, proportional to its camera share, at least 1.
    Partitions the CPU so parallel workers don't each spin up all-core pools
    and thrash (BLAS/OpenMP/OpenVINO all read these env vars)."""
    return max(1, (total_cores * cam_count) // total_cams)


def _child_env(threads: int) -> dict:
    """Copy of the parent env with the native-thread caps pinned for one group."""
    env = os.environ.copy()
    for var in (
        "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
    ):
        env[var] = str(threads)
    return env


def _reset_plates() -> None:
    """One-shot GLOBAL plate wipe, run ONCE before the groups start (never per
    group — that would race N wipes). Clears then exits."""
    print("[supervisor] --reset-plates: wiping ALL slot plate identities + "
          "per-car galleries (one-shot)...")
    rc = subprocess.run(
        [sys.executable, "main.py", "--reset-plates-only"], cwd=str(ROOT)
    ).returncode
    if rc != 0:
        raise SystemExit(f"[supervisor] reset-plates-only failed (exit {rc}).")
    print("[supervisor] reset complete.\n")


def _tail_to_stdout(path: Path, label: str, stop: threading.Event) -> None:
    """Mirror a child log file to this process's stdout, prefixed with the
    group label — so `--foreground` (Docker PID 1) surfaces every group's
    output in one stream. Best-effort; polls for the file to appear."""
    while not stop.is_set() and not path.exists():
        time.sleep(0.2)
    if not path.exists():
        return
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        while not stop.is_set():
            line = fh.readline()
            if line:
                sys.stdout.write(f"[{label}] {line}")
                sys.stdout.flush()
            else:
                time.sleep(0.25)


def run(reset_plates: bool = False, foreground: bool = False,
        port: int = API_PORT) -> int:
    """Spawn and supervise one worker process per camera group. Blocks until a
    signal (Ctrl+C / SIGTERM) or an unexpected child exit tears the fleet down.
    Callable directly (this is what ``main.py --supervise`` invokes) so it does
    not parse argv itself. Returns a process exit code."""
    # Sanity: exactly one API host owns the entrance camera + ANPR webhooks.
    api_groups = [g for g in GROUPS if g.get("api")]
    if len(api_groups) != 1:
        raise SystemExit(
            f"Exactly one group must have api=True (found {len(api_groups)}). Fix GROUPS."
        )

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    if reset_plates:
        _reset_plates()

    total_cores = _available_cores()
    total_cams = sum(_cam_count(g) for g in GROUPS)

    # On Windows: NEW_PROCESS_GROUP so a stray CTRL_C to us isn't broadcast to
    # the children (we tear them down explicitly), and NO_WINDOW so each worker
    # doesn't pop its own console window — the reason the old launcher had to use
    # pythonw.exe. stdout/stderr are redirected to files below regardless.
    # On POSIX: a new session isolates the children so our SIGTERM handler drives
    # the teardown instead of the shell's own signal fan-out.
    if os.name == "nt":
        popen_kwargs = {
            "creationflags": (
                subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW
            )
        }
    else:
        popen_kwargs = {"start_new_session": True}

    children: list[tuple[str, subprocess.Popen]] = []
    tail_threads: list[threading.Thread] = []
    stop_tail = threading.Event()

    PID_FILE.unlink(missing_ok=True)

    for g in GROUPS:
        cam_count = _cam_count(g)
        threads = _thread_slice(cam_count, total_cams, total_cores)
        cli = [sys.executable, "main.py", "--cameras", g["cams"]]
        if g.get("api"):
            cli += ["--api", "--port", str(port)]

        out_path = LOG_DIR / f"va_{g['name']}.out.log"
        err_path = LOG_DIR / f"va_{g['name']}.err.log"
        label = f"{g['name']} (+API :{port})" if g.get("api") else g["name"]
        print(f"[supervisor] starting group '{label}' "
              f"({threads} threads/{total_cores} cores) -> {g['cams']}")

        out_fh = out_path.open("w", encoding="utf-8")
        err_fh = err_path.open("w", encoding="utf-8")
        proc = subprocess.Popen(
            cli, cwd=str(ROOT), env=_child_env(threads),
            stdout=out_fh, stderr=err_fh, **popen_kwargs,
        )
        children.append((g["name"], proc))
        with PID_FILE.open("a") as pf:
            pf.write(f"{proc.pid}\n")

        if foreground:
            t = threading.Thread(
                target=_tail_to_stdout, args=(out_path, g["name"], stop_tail),
                daemon=True,
            )
            t.start()
            tail_threads.append(t)

    print(f"\n[supervisor] {len(children)} groups up. PIDs -> {PID_FILE}")
    print(f"[supervisor] follow a log:  Get-Content -Wait '{LOG_DIR / 'va_b1-gate.out.log'}'")
    print("[supervisor] stop: Ctrl+C here (or send SIGTERM to this process)\n")

    shutting_down = threading.Event()

    def _shutdown(*_a) -> None:
        if shutting_down.is_set():
            return
        shutting_down.set()
        print("\n[supervisor] stopping all groups...")
        stop_tail.set()
        for name, proc in children:
            if proc.poll() is None:
                proc.terminate()  # SIGTERM / CTRL-equivalent via Popen
        deadline = time.time() + 10
        for name, proc in children:
            remaining = max(0.0, deadline - time.time())
            try:
                proc.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                print(f"[supervisor] group '{name}' didn't exit in time — killing.")
                proc.kill()
        PID_FILE.unlink(missing_ok=True)

    signal.signal(signal.SIGINT, _shutdown)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _shutdown)

    # Supervise: if any worker dies unexpectedly, tear the rest down so the
    # whole system fails loud rather than silently running degraded.
    try:
        while not shutting_down.is_set():
            for name, proc in children:
                rc = proc.poll()
                if rc is not None and not shutting_down.is_set():
                    print(f"[supervisor] group '{name}' exited (code {rc}) — "
                          f"shutting down the rest.")
                    _shutdown()
                    break
            time.sleep(1.0)
    except KeyboardInterrupt:
        _shutdown()

    # Non-zero if a worker crashed us out; zero on a clean signal-driven stop.
    return 0 if not any(p.returncode not in (0, None) for _, p in children) else 1


def main(argv: list[str] | None = None) -> int:
    """Standalone CLI entry: parse argv, then delegate to run()."""
    parser = argparse.ArgumentParser(description="Supervise per-group VA workers.")
    parser.add_argument(
        "--reset-plates", action="store_true",
        help="Wipe ALL slot plate identities + per-car galleries ONCE, then start.",
    )
    parser.add_argument(
        "--foreground", action="store_true",
        help="Block and mirror all child logs to stdout (use as Docker PID 1).",
    )
    parser.add_argument(
        "--port", type=int, default=API_PORT,
        help=f"Port for the --api group (default {API_PORT}).",
    )
    args = parser.parse_args(argv)
    return run(
        reset_plates=args.reset_plates,
        foreground=args.foreground,
        port=args.port,
    )


if __name__ == "__main__":
    raise SystemExit(main())

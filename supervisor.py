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
from typing import Optional

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
# These 5 groups are the STATIC FALLBACK only. The live topology is now built
# per-AREA from the DB `cameras.area` column at startup (see resolve_groups /
# _groups_from_db). This list is used verbatim only if that DB read fails or the
# ANPR/entrance camera can't be located — so a bad DB never leaves the demo worse
# than today. Force this fallback with env VA_STATIC_GROUPS=1.
GROUPS = [
    {"name": "b1-gate",  "cams": "CAM-23,CAM-03,CAM-04,CAM-05,CAM-06,CAM-07", "api": True},
    {"name": "b1-areas", "cams": "CAM-08,CAM-20,CAM-21,CAM-22,CAM-24",        "api": False},
    {"name": "b2-1",     "cams": "CAM-09,CAM-10,CAM-11,CAM-12,CAM-13,CAM-14", "api": False},
    {"name": "b2-2",     "cams": "CAM-15,CAM-16,CAM-17,CAM-18,CAM-19,CAM-25", "api": False},
    {"name": "ground",   "cams": "CAM-00,CAM-01,CAM-02",                      "api": False},
]

# The camera that receives the ANPR webhooks / owns the entrance. Its group is the
# one (and only one) that runs --api. Override with env VA_API_CAMERA.
API_CAMERA = os.environ.get("VA_API_CAMERA", "CAM-23").strip().upper()

# Entry/gate cameras forced together into the single --api group, pulled OUT of
# their DB area buckets. WHY: the ANPR handler pulls LIVE frames from CAM-23 (ramp
# top) AND CAM-03 (B1 entrance line-crossing) through a PROCESS-LOCAL buffer
# (main.py get_camera_frame -> engine.cam_manager), so if those two aren't in the
# same process as the API server, entrance plate->identity seeding silently fails.
# ONLY those two need co-location — CAM-05/06/07 are the B1<->B2 RAMP cameras
# (boot log: CAM-05 RAMP-UP<->B1-C, CAM-07 B1-C<->RAMP-DN), so they stay in their
# own area groups. Override with env VA_GATE_CAMERAS (comma-separated); set it
# empty to disable the gate group and fall back to a pure per-area split.
GATE_CAMERAS = [
    c.strip().upper()
    for c in os.environ.get("VA_GATE_CAMERAS", "CAM-23,CAM-03").split(",")
    if c.strip()
]
# ---------------------------------------------------------------------------


def _grouping_mode() -> str:
    """``"area"`` (one group per aisle) or ``"floor"`` (one group per floor).

    Per-aisle was designed for the PINNED regime: many small groups = many
    concurrent serial-inference loops = idle cores filled. Under a CFS quota the
    same fan-out is what kills us — 8 processes x (OpenVINO TBB + OpenCV +
    PaddleOCR OMP + ReID) pools burst together, blow the period budget, and the
    WHOLE cgroup freezes (cpu.stat 2026-07-16: 57% of periods throttled at both
    14- and 20-core limits, ~16 cores burned for ~8 inferences/s — ~99% of the
    spend was pool spin + scheduling churn, not detection). Fewer, fatter
    processes is the right shape there.

    ``VA_GROUP_BY=area|floor`` forces either without a rebuild — the escape
    hatch if the isolated bench (tools/bench_yolo.py) shows inference is
    genuinely compute-bound, where fewer serial loops would halve throughput.
    """
    forced = os.environ.get("VA_GROUP_BY", "").strip().lower()
    if forced in ("area", "floor"):
        return forced
    return "floor" if _quota_cores() is not None else "area"


def _groups_from_db() -> list[dict]:
    """Build one supervisor group per parking AREA (pinned/cpuset regime) or
    per FLOOR (quota regime) from the DB `cameras` roster — the same roster
    main.py loads (ANPR cams already excluded). See _grouping_mode for why.

    WHY PER-AREA when pinned: per-camera fps is 1/(cams_in_group x ~50ms) and
    only one inference runs at a time per group, so a few big groups leave cores
    idle. One group per aisle (B1-A/B/C, B2-A/B/C, ...) gives ~3-4 cams/group =~
    2x the concurrent inferences while keeping each aisle's cameras in ONE
    process, so live identity handoff WITHIN an aisle is preserved. Per-floor
    grouping keeps that property too (a floor is a superset of its aisles).

    Raises on any problem (no cameras, entrance cam missing) so the caller can
    fall back to the static GROUPS. Never returns a topology with != 1 api group.
    """
    from collections import OrderedDict

    from src.config import load_config
    from src.database import init_db
    from src.services.config_service import (
        ensure_areas_initialized,
        load_cameras_from_db,
        sync_areas_from_db,
    )

    config = load_config(str(ROOT / "config.yaml"))
    db = init_db(config.database.url)
    db.create_tables()
    session = db.SessionLocal()
    try:
        # Areas first so a camera's `area` resolves against the synced table,
        # then the DB-first camera roster (mirrors main.py's startup order).
        ensure_areas_initialized(session, config)
        sync_areas_from_db(session, config)
        load_cameras_from_db(session, config)
    finally:
        session.close()

    if not config.cameras:
        raise RuntimeError("no cameras returned from DB")

    # Resolve the gate roster against the actual cameras, preserving GATE_CAMERAS
    # order (a gate camera that isn't enabled/loaded is simply skipped).
    by_upper = {cam.id.strip().upper(): cam.id for cam in config.cameras}
    gate_ids = [by_upper[g] for g in GATE_CAMERAS if g in by_upper]
    gate_set = set(gate_ids)

    # Bucket the REMAINING cameras by area or floor (mode-dependent); a camera
    # missing its primary key falls back to the other so it stays grouped
    # instead of scattering into one "unzoned" blob.
    mode = _grouping_mode()
    print(f"[supervisor] grouping cameras by {mode.upper()} "
          f"(override with VA_GROUP_BY=area|floor)")
    buckets: "OrderedDict[str, list[str]]" = OrderedDict()
    for cam in config.cameras:
        if cam.id in gate_set:
            continue
        if mode == "floor":
            key = (cam.floor or "").strip() or (cam.area or "").strip() or "unzoned"
        else:
            key = (cam.area or "").strip() or (cam.floor or "").strip() or "unzoned"
        buckets.setdefault(key, []).append(cam.id)

    groups = [
        {"name": key.lower().replace(" ", "-"), "cams": ",".join(cams), "api": False}
        for key, cams in sorted(buckets.items())
    ]

    if gate_ids:
        # The gate group is THE --api group by construction; put it first.
        if API_CAMERA not in {g.upper() for g in gate_ids}:
            raise RuntimeError(
                f"api camera {API_CAMERA} not in gate set {gate_ids}"
            )
        groups.insert(0, {"name": "gate", "cams": ",".join(gate_ids), "api": True})
    else:
        # No gate cameras present — fall back to tagging whichever area owns the
        # entrance/ANPR camera as --api (pure per-area split).
        for g in groups:
            if API_CAMERA in [c.strip().upper() for c in g["cams"].split(",")]:
                g["api"] = True
                break
        else:
            raise RuntimeError(
                f"entrance/ANPR camera {API_CAMERA} not in any area group"
            )

    return groups


def resolve_groups() -> list[dict]:
    """The live group topology: per-area from the DB, or the static GROUPS if the
    DB read fails (or VA_STATIC_GROUPS is set). Logs which one is used."""
    if os.environ.get("VA_STATIC_GROUPS", "").strip().lower() in ("1", "true", "yes", "on"):
        print("[supervisor] VA_STATIC_GROUPS set — using the static fallback GROUPS.")
        return GROUPS
    try:
        groups = _groups_from_db()
        print(f"[supervisor] topology from DB cameras.area: {len(groups)} area group(s).")
        return groups
    except Exception as exc:  # noqa: BLE001 — never let grouping block startup
        print(f"[supervisor] DB area-grouping failed ({exc!r}); "
              f"falling back to static GROUPS.")
        return GROUPS


def _merge_groups(groups: list[dict], target: int) -> list[dict]:
    """Merge the resolved groups down to ``target`` OS processes, balanced by
    camera count, keeping the single api group's cameras inside one bucket.

    MEASUREMENT KNOB (VA_MERGE_GROUPS): halving the process count halves the
    number of independent OpenVINO thread pools competing on the shared cores.
    If per-inference `ov` moves toward the isolated single-process floor, the
    dominant cost is PROCESS-LEVEL contention (pool interaction / cache /
    bandwidth / scheduler — this test does not distinguish the mechanism), which
    makes "fewer processes + async" the right direction. A no-op unless
    0 < target < len(groups). This re-buckets cameras across floors, so it is a
    THROUGHPUT-measurement topology, not a correctness one (cross-floor identity
    flows only through the DB); use it to measure, then revert.
    """
    if target <= 0 or target >= len(groups):
        return groups
    # api group first so it anchors bucket 0 (exactly one group has api=True).
    ordered = sorted(groups, key=lambda g: not g.get("api"))
    buckets: list[list[dict]] = [[] for _ in range(target)]
    sizes = [0] * target
    buckets[0].append(ordered[0])
    sizes[0] += _cam_count(ordered[0])
    for g in ordered[1:]:
        j = sizes.index(min(sizes))       # least-full bucket
        buckets[j].append(g)
        sizes[j] += _cam_count(g)
    merged = []
    for bucket in buckets:
        if not bucket:
            continue
        merged.append({
            "name": "+".join(g["name"] for g in bucket),
            "cams": ",".join(g["cams"] for g in bucket),
            "api": any(g.get("api") for g in bucket),
        })
    return merged
# ---------------------------------------------------------------------------


def _cam_count(group: dict) -> int:
    return len([c for c in group["cams"].split(",") if c.strip()])


def _affinity_cores() -> int:
    """Cores this process is allowed to run ON (the cpuset/affinity mask).
    Under Docker --cpuset-cpus (or taskset), os.cpu_count() still reports every
    host CPU, which would over-slice the per-group thread budget and
    oversubscribe. sched_getaffinity reflects the cpuset mask; fall back to
    os.cpu_count() where it's unavailable (e.g. Windows)."""
    if hasattr(os, "sched_getaffinity"):
        try:
            return max(1, len(os.sched_getaffinity(0)))
        except OSError:
            pass
    return os.cpu_count() or 1


def _quota_cores() -> Optional[float]:
    """CPU-seconds per second this cgroup may burn, or None if unlimited.

    THE MASK IS NOT THE BUDGET. A Kubernetes CPU *limit* is enforced with a CFS
    quota (cgroup v2 `cpu.max`, v1 `cpu.cfs_quota_us`), NOT a cpuset — so
    sched_getaffinity still returns every core on the NODE. Sizing the per-group
    slices from that mask hands out cores the pod does not own and pins each
    worker to host CPUs shared with every other pod on the node. Measured
    2026-07-15: the mask said 20, the pod's real budget was ~12, and the four
    groups pinned to contended host CPUs ran at 0.003-0.015 fps/camera (5-26
    MINUTES per slot flip) while the four on quiet cores held 1.3-2.0 fps. The
    kernel could not rebalance them, because affinity forbade it.
    """
    # cgroup v2: "<quota> <period>", or "max <period>" when unlimited.
    try:
        with open("/sys/fs/cgroup/cpu.max", encoding="ascii") as fh:
            quota_s, period_s = fh.read().split()[:2]
        if quota_s != "max":
            period = float(period_s)
            if period > 0:
                return float(quota_s) / period
        return None
    except (OSError, ValueError):
        pass
    # cgroup v1: quota is -1 when unlimited.
    try:
        with open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us", encoding="ascii") as fh:
            quota = float(fh.read().strip())
        with open("/sys/fs/cgroup/cpu/cpu.cfs_period_us", encoding="ascii") as fh:
            period = float(fh.read().strip())
        if quota > 0 and period > 0:
            return quota / period
    except (OSError, ValueError):
        pass
    return None


# Threads per CPU-second of quota, when the quota (not a cpuset) is the limit.
# A QUOTA IS NOT A THREAD COUNT — but bursting past it is NOT free either.
#
# History, both measured in production:
#   2026-07-15: apportioning exactly `quota` threads (14 across 8 groups) gave gate
#   and ground ONE thread each — 1.8/1.6 fps -> 0.40/0.20, single-threaded HEVC
#   decode was the floor. The fix that mattered was _MIN_THREADS_PER_GROUP, but it
#   shipped bundled with 1.5x oversubscription "to keep the quota busy".
#   2026-07-16: cpu.stat deltas showed 57% of CFS periods THROTTLED at BOTH the 14-
#   and 20-core limits, ~16 cores burned for ~8 inferences/s (~2 core-seconds per
#   11ms inference — ~99% of the spend was pool spin + churn). CFS does not
#   "smooth" a burst: threads that wake together burn the period budget early and
#   the WHOLE cgroup freezes for the rest of the 100ms — which is exactly how an
#   11ms inference stretches to ~1000ms wall.
#
# So: size to the quota, no oversubscription. The per-group floor below is what
# actually protects decode concurrency; the excess only bought throttling.
_QUOTA_THREAD_OVERSUBSCRIBE = 1.0

# Never choke a group below this, whatever the arithmetic says. One thread cannot
# decode a 1080p HEVC stream and run inference on it.
_MIN_THREADS_PER_GROUP = 2

# Cap each group's OpenVINO pool regardless of slice width. Past 8 threads a
# 320px int8 inference has too little parallel work left (src/ov_tuning.py:
# widening 8->15 measured 0-10%, sometimes negative) and the spare TBB workers
# spin-wait — under a CFS quota, spin IS spend. BELOW 8 is a different story on
# weak server cores: tools/bench_yolo.py on the production Xeon Silver 4410Y
# (2.0GHz) measured 5.4 inf/s at 2 threads vs 28.7 at 8 (185ms -> 35ms), so an
# aggressive cap starves the big floor groups. 8 = where scaling ends, both boxes.
_OV_THREADS_MAX = 8


def _cpu_budget(n_groups: int = 1) -> tuple[int, bool]:
    """(threads to apportion, whether pinning is safe).

    PIN ONLY WHAT WE EXCLUSIVELY OWN. A cpuset (docker --cpuset-cpus / taskset)
    grants us those cores outright, so slicing them per group is real isolation.
    A CFS quota grants us a SHARE OF TIME on cores everybody else is also using —
    the mask still lists the whole node. Pinning against a quota is strictly
    worse than not pinning: it cannot add throughput, and it forfeits the
    kernel's ability to route around a core a neighbouring pod is hogging.

    So the presence of a quota — at ANY value — is what decides PINNING, not
    whether the quota is smaller than the mask. Raising the pod's CPU limit
    above the mask does NOT make pinning safe again: the cores were never ours.
    That is exactly the trap this hit in production — the limit was raised, the
    pod stayed at ~11.9 cores, because affinity (not quota) was the real cap.

    THREAD COUNT IS A SEPARATE QUESTION. See _QUOTA_THREAD_OVERSUBSCRIBE: the
    quota caps the burn, so pools are sized for concurrency (to keep the quota
    busy across I/O waits), bounded by the mask — there is no point spinning more
    threads than the node has cores.
    """
    mask = _affinity_cores()
    quota = _quota_cores()
    # Explicit total OpenVINO thread budget across ALL groups, overriding the
    # mask/quota heuristic. Set VA_OV_TOTAL_THREADS to the pod's REAL usable core
    # count when the affinity mask overreports it (the 2026-07-15 trap: mask said
    # 20, real budget ~12), so the per-group pools don't oversubscribe the cores.
    # Apportioned by camera share downstream, so e.g. 15 -> gate2/ground2/b1 5/b2 6.
    override = os.environ.get("VA_OV_TOTAL_THREADS", "").strip()
    if override:
        try:
            n = int(override)
        except ValueError:
            print(f"[supervisor] VA_OV_TOTAL_THREADS={override!r} not an int — ignored.")
        else:
            if n > 0:
                # Unpinned (thread-cap only) whenever a quota is present, same as
                # the heuristic path; a cpuset (no quota) can still pin its slice.
                return max(1, n), (quota is None)
    if quota is None:
        # No quota: the mask IS our exclusive budget (cpuset, or a whole box).
        return mask, True
    want = int(_QUOTA_THREAD_OVERSUBSCRIBE * quota + 0.5)
    floor = _MIN_THREADS_PER_GROUP * max(1, n_groups)
    return max(1, min(mask, max(want, floor))), False


def _core_slices(
    groups: list[dict], total_cores: int, min_size: int = 1
) -> list[list[int]]:
    """Carve the CPU into one DISJOINT core range per group, sized by camera share.

    Every group gets at least ``min_size`` cores and the slices together cover
    all cores (largest-remainder apportionment, ties broken toward the busier
    group), so no core sits idle and no core is claimed twice. ``min_size`` is
    _MIN_THREADS_PER_GROUP in the unpinned regime — camera-share apportionment
    alone can hand a 2-camera group a single thread, which cannot decode HEVC
    and run inference at once (the 0.40/0.20 fps regression).

    Why ranges and not just a thread count: OMP_NUM_THREADS only reaches
    numpy/BLAS/torch. OpenVINO (TBB) and OpenCV ignore it and size their pools
    to the whole machine, so all 5 groups were each spinning an all-core pool.
    Both libraries DO respect the CPU affinity mask, so handing each group a
    disjoint slice is what actually caps them. See src/cpu_affinity.py.
    """
    # Fewer cores than groups: there is nothing to partition, so don't pretend to.
    # Every group sees the whole CPU (i.e. affinity becomes a no-op) rather than
    # us handing out overlapping "slices" that would just mislead the logs.
    if total_cores < len(groups):
        return [list(range(total_cores)) for _ in groups]
    if total_cores < min_size * len(groups):
        min_size = 1  # not enough to honour the floor — degrade, don't overflow

    total_cams = sum(_cam_count(g) for g in groups) or 1
    exact = [total_cores * _cam_count(g) / total_cams for g in groups]
    sizes = [max(min_size, int(e)) for e in exact]

    # Hand the remaining cores to whoever was shortchanged most by the floor
    # (most cameras wins a tie). Measure the shortfall against what each group
    # was ACTUALLY granted, not against floor(exact) — otherwise a tiny group
    # rounded up to its 1-core minimum still looks starved and outbids a group
    # three times its size.
    leftover = total_cores - sum(sizes)
    order = sorted(
        range(len(groups)),
        key=lambda i: (exact[i] - sizes[i], _cam_count(groups[i])),
        reverse=True,
    )
    for i in range(max(0, leftover)):
        sizes[order[i % len(order)]] += 1

    # The minimums can also push the floors PAST the total (e.g. many tiny
    # groups lifted to min_size next to one big one). Claw back from whoever is
    # most over-granted relative to their camera share, never below min_size —
    # otherwise the slice loop below would silently hand later groups EMPTY
    # ranges once next_core runs off the end.
    excess = sum(sizes) - total_cores
    while excess > 0:
        candidates = [i for i in range(len(groups)) if sizes[i] > min_size]
        if not candidates:
            break
        victim = max(
            candidates,
            key=lambda i: (sizes[i] - exact[i], -_cam_count(groups[i])),
        )
        sizes[victim] -= 1
        excess -= 1

    slices, next_core = [], 0
    for size in sizes:
        slices.append(list(range(next_core, min(next_core + size, total_cores))))
        next_core += size
    return slices


def _child_env(cores: list[int], pin: bool = True) -> dict:
    """Copy of the parent env carrying one group's CPU slice.

    VA_CPU_LIST is applied by main.py before it imports cv2/torch/openvino; the
    *_NUM_THREADS vars still cap the OpenMP/BLAS pools (NMS, ByteTrack, numpy)
    which read them at first use rather than from the affinity mask.

    ``pin=False`` (quota-limited: see _cpu_budget) keeps the THREAD CAP but drops
    the affinity pin, so a group can still only run len(cores) threads yet may
    run them on whichever cores are idle. The thread cap is what prevents the
    oversubscription the pinning was introduced to stop — OpenVINO (TBB) and
    OpenCV size their pools from the affinity mask and ignore OMP_NUM_THREADS,
    so they are capped explicitly by VA_OV_NUM_THREADS / VA_CV_NUM_THREADS
    rather than by starving them of cores.
    """
    env = os.environ.copy()
    if pin:
        env["VA_CPU_LIST"] = ",".join(str(c) for c in cores)
    else:
        env.pop("VA_CPU_LIST", None)
        env["VA_NO_AFFINITY"] = "1"
        # TBB/OpenCV would otherwise each spin a pool sized to the whole node.
        # OpenVINO is additionally capped at _OV_THREADS_MAX (=8, where 320px
        # int8 scaling ends — measured on both the dev box and the production
        # Xeon); the spare workers past that only spin-burn the quota. Decode
        # (OpenCV) keeps the full slice — that pool does real concurrent work,
        # one stream per camera.
        env["VA_OV_NUM_THREADS"] = str(min(len(cores), _OV_THREADS_MAX))
        env["VA_CV_NUM_THREADS"] = str(len(cores))
    # Pinned: OMP/BLAS pools bounded by the slice, as always. Unpinned (quota):
    # one thread each — PaddleOCR's own boot warning says OMP>1 hurts it, the
    # numpy/BLAS ops here are tiny, and under CFS an idle-spinning OMP worker
    # burns period budget the detector and decoder needed.
    omp = str(len(cores)) if pin else "1"
    for var in (
        "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
    ):
        env[var] = omp
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


def _tail(path: Path, label: str, stop: threading.Event, stream=None) -> None:
    """Mirror a child log file to this process's stdout (or stderr), prefixed
    with the group label — so `--foreground` (Docker PID 1) surfaces every
    group's output in one stream. Best-effort; polls for the file to appear.

    Each group is tailed twice: its .out.log to our stdout, its .err.log to our
    stderr. Tailing only stdout would silently swallow every worker traceback —
    a group that dies on import would just vanish from `docker logs`, leaving
    the fleet looking healthy minus one camera group.
    """
    out = stream if stream is not None else sys.stdout
    while not stop.is_set() and not path.exists():
        time.sleep(0.2)
    if not path.exists():
        return
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        while not stop.is_set():
            line = fh.readline()
            if line:
                out.write(f"[{label}] {line}")
                out.flush()
            else:
                time.sleep(0.25)


def run(reset_plates: bool = False, foreground: bool = False,
        port: int = API_PORT) -> int:
    """Spawn and supervise one worker process per camera group. Blocks until a
    signal (Ctrl+C / SIGTERM) or an unexpected child exit tears the fleet down.
    Callable directly (this is what ``main.py --supervise`` invokes) so it does
    not parse argv itself. Returns a process exit code."""
    # Per-area topology from the DB (or the static fallback). Resolve ONCE here so
    # the sanity check, core-slicing and the spawn loop all see the same groups.
    groups = resolve_groups()

    # MEASUREMENT: collapse to fewer OS processes (fewer OpenVINO pools) to test
    # whether the residual inference contention is process-level. Reverts by
    # unsetting VA_MERGE_GROUPS.
    _merge_env = os.environ.get("VA_MERGE_GROUPS", "").strip()
    if _merge_env:
        try:
            _target = int(_merge_env)
        except ValueError:
            print(f"[supervisor] VA_MERGE_GROUPS={_merge_env!r} not an int — ignored.")
        else:
            before = len(groups)
            groups = _merge_groups(groups, _target)
            if len(groups) != before:
                print(f"[supervisor] VA_MERGE_GROUPS={_target}: merged "
                      f"{before} groups -> {len(groups)} "
                      f"({', '.join(g['name'] for g in groups)})")

    # Sanity: exactly one API host owns the entrance camera + ANPR webhooks.
    api_groups = [g for g in groups if g.get("api")]
    if len(api_groups) != 1:
        raise SystemExit(
            f"Exactly one group must have api=True (found {len(api_groups)}). Fix GROUPS."
        )

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    if reset_plates:
        _reset_plates()

    total_cores, pin = _cpu_budget(len(groups))
    if not pin:
        print(
            f"[supervisor] cgroup CPU quota detected ({_quota_cores():.1f} cores) "
            f"alongside a {_affinity_cores()}-core affinity mask: this is a CPU "
            f"*limit*, not a cpuset, so the mask is the whole node and those cores "
            f"are shared. Running UNPINNED with {total_cores} threads apportioned "
            f"across {len(groups)} groups — pinning would nail each group to host "
            f"CPUs it does not own and make raising the CPU limit a no-op. Threads "
            f"are sized TO the quota (min {_MIN_THREADS_PER_GROUP}/group): bursting "
            f"past it doesn't add throughput, it freezes the whole cgroup for the "
            f"rest of each CFS period (measured 57% of periods throttled, 2026-07-16)."
        )
    total_cams = sum(_cam_count(g) for g in groups)

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

    slices = _core_slices(
        groups, total_cores, min_size=1 if pin else _MIN_THREADS_PER_GROUP
    )

    for g, cores in zip(groups, slices):
        cli = [sys.executable, "main.py", "--cameras", g["cams"]]
        if g.get("api"):
            cli += ["--api", "--port", str(port)]

        out_path = LOG_DIR / f"va_{g['name']}.out.log"
        err_path = LOG_DIR / f"va_{g['name']}.err.log"
        label = f"{g['name']} (+API :{port})" if g.get("api") else g["name"]
        if pin:
            span = f"{cores[0]}-{cores[-1]}" if len(cores) > 1 else str(cores[0])
            budget = f"CPUs {span} = {len(cores)}/{total_cores}"
        else:
            budget = f"{len(cores)}/{total_cores} threads, unpinned"
        print(f"[supervisor] starting group '{label}' "
              f"({budget}) -> {g['cams']}")

        out_fh = out_path.open("w", encoding="utf-8")
        err_fh = err_path.open("w", encoding="utf-8")
        child_env = _child_env(cores, pin=pin)
        child_env["VA_GROUP"] = g["name"]
        child_env["VA_PROCESS_COUNT"] = str(len(groups))
        proc = subprocess.Popen(
            cli, cwd=str(ROOT), env=child_env,
            stdout=out_fh, stderr=err_fh, **popen_kwargs,
        )
        children.append((g["name"], proc))
        with PID_FILE.open("a") as pf:
            pf.write(f"{proc.pid}\n")

        if foreground:
            # Two tails per group: stdout carries the [PERF]/effective-FPS lines,
            # stderr carries the tracebacks. Both must reach PID 1 or `docker
            # logs` shows a half-story.
            for tail_path, tail_label, tail_stream in (
                (out_path, g["name"], sys.stdout),
                (err_path, f"{g['name']}:err", sys.stderr),
            ):
                t = threading.Thread(
                    target=_tail,
                    args=(tail_path, tail_label, stop_tail, tail_stream),
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

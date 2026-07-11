"""
cpu_affinity.py — pin a worker process to the CPU slice the supervisor gave it.

WHY THIS EXISTS
---------------
The supervisor already sets OMP_NUM_THREADS & friends per group, but those only
reach numpy/BLAS/torch. OpenVINO's CPU plugin is TBB-based and ignores them, and
Ultralytics compiles the detector with PERFORMANCE_HINT=LATENCY and no thread cap
(autobackend.py), so *every* group process sizes its inference pool to the whole
machine. OpenCV's parallel_for does the same for CLAHE/resize/bitwise_and. With 5
groups that means ~5x oversubscription on both pools — threads thrashing caches
instead of doing work.

Both TBB and OpenCV size their pools from the process's CPU affinity mask, so
pinning each group to a disjoint slice caps them at the source, with no
per-library configuration and no patching of Ultralytics.

MUST RUN BEFORE cv2 / torch / openvino are imported — those libraries read the
mask when they build their pools, and never re-read it.

Set VA_NO_AFFINITY=1 to opt out (e.g. when profiling a single group alone).
"""

from __future__ import annotations

import os
import sys
from typing import List, Optional


def parse_cpu_list(spec: str) -> List[int]:
    """Parse a taskset-style CPU list: "0-3", "0,2,4", "0-3,8" -> [0,1,2,3,8]."""
    cpus: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            cpus.update(range(int(lo), int(hi) + 1))
        else:
            cpus.add(int(part))
    return sorted(cpus)


def _set_affinity_windows(cpus: List[int]) -> bool:
    """SetProcessAffinityMask via ctypes — no psutil dependency.

    Single processor group only (<=64 CPUs), which covers every box we run on.
    """
    import ctypes

    if any(c >= 64 for c in cpus):
        return False
    mask = 0
    for c in cpus:
        mask |= 1 << c

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    # argtypes/restype are NOT optional here: HANDLE is pointer-sized, and
    # ctypes defaults to c_int, which truncates the 64-bit pseudo-handle and
    # makes the call fail silently.
    kernel32.GetCurrentProcess.restype = ctypes.c_void_p
    kernel32.GetCurrentProcess.argtypes = []
    kernel32.SetProcessAffinityMask.restype = ctypes.c_int  # BOOL
    kernel32.SetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.c_size_t]

    return bool(kernel32.SetProcessAffinityMask(kernel32.GetCurrentProcess(), mask))


def apply_from_env(env_var: str = "VA_CPU_LIST") -> Optional[List[int]]:
    """Pin this process to the CPUs named in *env_var*. Returns them, or None.

    Best-effort: an unparseable spec or an OS that refuses the call is never
    fatal — the worker still runs, just unpinned (and oversubscribed).
    """
    if os.environ.get("VA_NO_AFFINITY", "").strip().lower() in ("1", "true", "yes", "on"):
        return None

    spec = os.environ.get(env_var, "").strip()
    if not spec:
        return None

    try:
        cpus = parse_cpu_list(spec)
    except ValueError:
        print(f"[WARN] {env_var}='{spec}' is not a valid CPU list — running unpinned.")
        return None
    if not cpus:
        return None

    try:
        if hasattr(os, "sched_setaffinity"):
            os.sched_setaffinity(0, cpus)          # Linux — the deployment target
        elif sys.platform == "win32":
            if not _set_affinity_windows(cpus):
                raise OSError("SetProcessAffinityMask failed")
        else:
            return None
    except OSError as exc:
        print(f"[WARN] could not pin to CPUs {cpus}: {exc} — running unpinned.")
        return None

    return cpus

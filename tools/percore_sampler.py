"""
tools/percore_sampler.py — per-core CPU sampler for the VA pods.

Answers "is the load spread across cores or concentrated on a few?" — the
question a single average (9 of 15 cores) cannot. Emits one timestamped line
per interval so it lines up with the engine's [PERF] infer-breakdown lines.

Two independent views, because they answer different questions:

  * NODE  (from /proc/stat): busy% of every host core the container can see.
    This is the WHOLE node — includes other tenants. Shows node saturation and
    noisy neighbours. No external tools needed (pure /proc); this is the
    portable equivalent of `mpstat -P ALL 1`.

  * POD   (from cgroup v1 cpuacct.usage_percpu): CPU-nanoseconds THIS cgroup
    burned per core, so you see *your* distribution isolated from neighbours.
    Only available on cgroup v1; silently skipped on v2 (use the NODE view there).

Each line reports, across cores: how many are >80% busy (NODE) or carrying load
(POD), the min/median/max, and the top few cores — so concentration is obvious
at a glance without dumping 15+ numbers every second.

Usage (run inside the pod, alongside the PERF_TRACE=1 engine):
    python tools/percore_sampler.py --interval 1 --duration 600
    python tools/percore_sampler.py --interval 1 --duration 600 --full   # every core
Linux only (reads /proc and /sys/fs/cgroup).
"""

from __future__ import annotations

import argparse
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple


def _read_proc_stat() -> Dict[int, Tuple[int, int]]:
    """Per-cpu (busy_jiffies, total_jiffies) from /proc/stat cpuN lines."""
    out: Dict[int, Tuple[int, int]] = {}
    try:
        with open("/proc/stat") as f:
            for line in f:
                if not line.startswith("cpu") or line.startswith("cpu "):
                    continue
                parts = line.split()
                try:
                    core = int(parts[0][3:])
                except ValueError:
                    continue
                vals = [int(x) for x in parts[1:]]
                idle = vals[3] + (vals[4] if len(vals) > 4 else 0)  # idle + iowait
                total = sum(vals)
                out[core] = (total - idle, total)
    except OSError:
        pass
    return out


def _read_cpuacct_percpu() -> Optional[List[int]]:
    """Per-cpu cumulative CPU-ns for THIS cgroup (v1). None if unavailable (v2)."""
    for p in (
        "/sys/fs/cgroup/cpuacct/cpuacct.usage_percpu",
        "/sys/fs/cgroup/cpu,cpuacct/cpuacct.usage_percpu",
    ):
        try:
            with open(p) as f:
                return [int(x) for x in f.read().split()]
        except OSError:
            continue
    return None


def _summary(vals: Dict[int, float], busy_thresh: float, top: int, full: bool) -> str:
    if not vals:
        return "n/a"
    items = sorted(vals.items(), key=lambda kv: kv[1], reverse=True)
    pcts = sorted(vals.values())
    n = len(pcts)
    median = pcts[n // 2]
    hot = sum(1 for v in vals.values() if v >= busy_thresh)
    head = (
        f"cores={n} hot(>={busy_thresh:.0f}%)={hot} "
        f"min={pcts[0]:.0f}% med={median:.0f}% max={pcts[-1]:.0f}%"
    )
    if full:
        allc = " ".join(f"c{c}={v:.0f}" for c, v in sorted(vals.items()))
        return f"{head} | {allc}"
    topc = " ".join(f"c{c}={v:.0f}%" for c, v in items[:top])
    return f"{head} | top: {topc}"


def main() -> None:
    ap = argparse.ArgumentParser(description="Per-core CPU sampler (NODE + POD).")
    ap.add_argument("--interval", type=float, default=1.0, help="seconds between samples")
    ap.add_argument("--duration", type=float, default=600.0, help="total seconds to run")
    ap.add_argument("--busy", type=float, default=80.0, help="hot-core threshold %%")
    ap.add_argument("--top", type=int, default=4, help="how many busiest cores to show")
    ap.add_argument("--full", action="store_true", help="print every core, not just top")
    args = ap.parse_args()

    prev_stat = _read_proc_stat()
    prev_pod = _read_cpuacct_percpu()
    prev_t = time.perf_counter()
    if prev_pod is None:
        print("[PERCORE] cpuacct.usage_percpu unavailable (cgroup v2?) — NODE view only.")

    deadline = time.perf_counter() + args.duration
    while time.perf_counter() < deadline:
        time.sleep(args.interval)
        now_t = time.perf_counter()
        dt = now_t - prev_t

        cur_stat = _read_proc_stat()
        node: Dict[int, float] = {}
        for core, (busy, total) in cur_stat.items():
            if core in prev_stat:
                db = busy - prev_stat[core][0]
                dtot = total - prev_stat[core][1]
                if dtot > 0:
                    node[core] = 100.0 * db / dtot
        prev_stat = cur_stat

        pod: Dict[int, float] = {}
        cur_pod = _read_cpuacct_percpu()
        if cur_pod is not None and prev_pod is not None and dt > 0:
            for core, ns in enumerate(cur_pod):
                if core < len(prev_pod):
                    # ns of CPU used / ns of wall elapsed = fraction of one core
                    pod[core] = 100.0 * (ns - prev_pod[core]) / (dt * 1e9)
        prev_pod = cur_pod

        ts = datetime.now().strftime("%H:%M:%S")
        node_busy_sum = sum(node.values()) / 100.0  # cores-worth busy, node-wide
        print(f"[PERCORE] {ts} NODE {_summary(node, args.busy, args.top, args.full)} "
              f"| node_busy={node_busy_sum:.1f} cores")
        if pod:
            pod_busy_sum = sum(pod.values()) / 100.0
            print(f"[PERCORE] {ts} POD  {_summary(pod, args.busy, args.top, args.full)} "
                  f"| pod_busy={pod_busy_sum:.1f} cores")
        prev_t = now_t


if __name__ == "__main__":
    main()

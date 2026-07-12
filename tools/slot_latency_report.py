"""
slot_latency_report.py — how long do slots ACTUALLY take to change status, and why?

Reads the [SLOTLAT] lines emitted by SlotStateMachine and answers the one question that
decides the fix:

  Is a slow slot flip the DEBOUNCE (nominal frames / achieved fps), or is it FLICKER
  (the counters demand CONSECUTIVE frames, so one contrary frame throws the whole run
  away and the flip starts over)?

If observed p50 ~= nominal, the debounce length is simply the cost -> lower
confirm_leave_frames / raise fps. If observed >> nominal and resets are common, the
debounce length is NOT the problem and lowering it would only paper over unstable
detection.

Usage:
    python tools/slot_latency_report.py                 # reads logs/va_*.out.log
    python tools/slot_latency_report.py logs/va_b2-2.out.log
"""

import glob
import math
import re
import statistics
import sys
from collections import defaultdict

# Slot ids may contain SPACES (e.g. "B11 CFO"), so the slot field cannot be \S+ —
# that silently dropped every space-containing slot from the stats. Anchor on the
# literal text that follows instead.
CONFIRM_RE = re.compile(
    r"\[SLOTLAT\] CONFIRM slot=(?P<slot>.+?) -> (?P<to>\w+) "
    r"after (?P<frames>\d+) frames in (?P<secs>[\d.]+|nan)s \(resets=(?P<resets>\d+)\)"
)
RESET_RE = re.compile(
    r"\[SLOTLAT\] RESET\s+slot=(?P<slot>.+?) (?P<transition>\w+->\w+) "
    r"at (?P<had>\d+)/(?P<need>\d+) frames"
)
FPS_RE = re.compile(r"effective FPS/camera: (?P<fps>[\d.]+) \(target (?P<target>\d+)\)")


def _pct(values, p):
    """Nearest-rank percentile on an ALREADY-SORTED list.

    Deliberately not statistics.quantiles: that interpolates and, on the small samples
    this report often has, extrapolates a p90 ABOVE the observed max — a nonsense
    number to hand someone who is trying to decide whether the tail is real.
    """
    if not values:
        return float("nan")
    k = max(0, min(len(values) - 1, math.ceil(p / 100.0 * len(values)) - 1))
    return values[k]


def main(paths):
    if not paths:
        # BOTH streams: the [SLOTLAT] lines come from `logger` (stderr -> .err.log),
        # while the "effective FPS/camera" line is a bare print() (stdout -> .out.log).
        # Globbing only .out.log finds the fps and none of the measurements.
        paths = sorted(glob.glob("logs/va_*.out.log") + glob.glob("logs/va_*.err.log"))
    if not paths:
        print("No logs found. Pass a path, or run from the repo root.")
        return 1

    confirms = defaultdict(list)          # to_state -> [seconds]
    resets_per_flip = defaultdict(list)   # to_state -> [reset count at confirm]
    reset_events = defaultdict(int)       # transition -> count
    reset_by_slot = defaultdict(int)
    fps_samples = []

    for path in paths:
        with open(path, "r", errors="replace") as fh:
            for line in fh:
                m = CONFIRM_RE.search(line)
                if m:
                    secs = m.group("secs")
                    if secs != "nan":
                        confirms[m.group("to")].append(float(secs))
                    resets_per_flip[m.group("to")].append(int(m.group("resets")))
                    continue
                m = RESET_RE.search(line)
                if m:
                    reset_events[m.group("transition")] += 1
                    reset_by_slot[m.group("slot")] += 1
                    continue
                m = FPS_RE.search(line)
                if m:
                    fps_samples.append(float(m.group("fps")))

    if not confirms and not reset_events:
        print(
            "No [SLOTLAT] lines found in:\n  "
            + "\n  ".join(paths)
            + "\n\nThe instrumentation ships in SlotStateMachine but only appears once the\n"
              "engine has been restarted with it. Restart the workers, let real traffic\n"
              "flow, then re-run this."
        )
        return 1

    fps = statistics.mean(fps_samples) if fps_samples else None
    print("=" * 72)
    print("SLOT STATUS LATENCY")
    print("=" * 72)
    if fps:
        print(f"effective fps/camera (mean over logs): {fps:.2f}")

    for to_state, label in (("OCCUPIED", "free -> occupied"), ("VACANT", "occupied -> free")):
        secs = sorted(confirms.get(to_state, []))
        rs = resets_per_flip.get(to_state, [])
        print()
        print(f"--- {label}  (n={len(secs)}) ---")
        if not secs:
            print("    no confirmed flips seen")
            continue
        print(
            f"    observed   p50 {_pct(secs,50):5.2f}s   p90 {_pct(secs,90):5.2f}s   "
            f"max {max(secs):5.2f}s"
        )
        clean = sum(1 for r in rs if r == 0)
        print(
            f"    flips with NO flicker: {clean}/{len(rs)} "
            f"({100.0*clean/len(rs):.0f}%)   mean resets/flip: {statistics.mean(rs):.2f}"
        )

    print()
    print("--- flicker (counter resets) ---")
    if not reset_events:
        print("    none — detection is stable; latency is purely the debounce")
    else:
        for transition, n in sorted(reset_events.items(), key=lambda kv: -kv[1]):
            print(f"    {transition:20s} {n}")
        print()
        print("    worst slots:")
        for slot, n in sorted(reset_by_slot.items(), key=lambda kv: -kv[1])[:8]:
            print(f"      {slot:16s} {n} resets")

    print()
    print("=" * 72)
    print("READ IT LIKE THIS")
    print("=" * 72)
    print(
        "  observed p50 ~= nominal (frames / fps), few resets\n"
        "      -> the debounce length IS the cost. Lower confirm_leave_frames in the\n"
        "         Config DB row, and/or raise effective fps.\n"
        "\n"
        "  observed p90/max >> nominal, resets common\n"
        "      -> FLICKER is the cost, not the debounce. The counters require\n"
        "         CONSECUTIVE frames, so one dropped detection discards the whole run.\n"
        "         Lowering the debounce would NOT fix this. Make the counters tolerate\n"
        "         isolated misses (decrement instead of hard-reset), and/or attack the\n"
        "         instability itself (assigner overlap_threshold is 0.3 in the DB vs\n"
        "         0.2 in YAML; detector confidence is 0.35 at imgsz 320).\n"
        "\n"
        "  NOTE: this measures ENGINE -> DB only. A normal park/leave is never PUSHED to\n"
        "  any client (src/api.py drops non-alert events from the SSE stream, and there\n"
        "  is no WebSocket), so the dashboard must poll /api/slots. Whatever the client's\n"
        "  poll interval is, it adds on top of every number above."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

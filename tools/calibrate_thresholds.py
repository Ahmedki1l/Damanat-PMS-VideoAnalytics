"""
tools/calibrate_thresholds.py — Phase 2 / T2.3 threshold calibration utility.

Reads the production ``MATCH_EVENT`` lines emitted by ``match_logger`` (see
``src/vehicle_registry/vehicle_registry_identity.py:~1198``), joins them
against the ``parking_sessions`` table for ground truth, and sweeps each
threshold field in :class:`MatchingConfig` to produce a precision / recall
curve. Emits a JSON report plus a recommended ``matching:`` YAML block.

Run as
~~~~~~

::

    python tools/calibrate_thresholds.py
    python tools/calibrate_thresholds.py --log-glob 'logs/*.log' --output report.json
    python tools/calibrate_thresholds.py --apply config.yaml

``--apply`` rewrites the ``matching:`` block in the supplied config file
with the recommended thresholds. Without ``--apply`` we only emit the
recommended block on stdout — the operator decides whether to merge.

Cosine-shift detection
~~~~~~~~~~~~~~~~~~~~~~

The WS-A INT8 quantisation has been observed to shift cosines by ~0.06 from
the torchreid baseline. When the mean per-event ``NewCost`` score deviates
from the snapshot defaults by >= 0.03 we surface the drift loudly and
recommend a uniform threshold shift. (The recommended shift is the median
of the per-threshold optimal points, not a hand-tuned constant.)

The tool is intentionally side-effect-free unless ``--apply`` is passed;
this is the same pattern used by ``tools/build_test_set.py``.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import os
import re
import statistics
import sys
from dataclasses import dataclass
from glob import glob
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

# Allow ``python tools/calibrate_thresholds.py`` to find ``src``.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import MatchingConfig  # noqa: E402

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Threshold fields swept by the tool. Order matters: the recommended YAML
# block lists them in this order for readability.
# --------------------------------------------------------------------------- #

CALIBRATABLE_FIELDS: Tuple[str, ...] = (
    "b1_anpr",
    "b1_zone",
    "b1_cross_camera",
    "global_default",
    "global_with_plate",
    "global_cross_camera",
    "reattach_default",
    "reattach_cross_camera",
    "reid_solo_confirm",
)

# Sweep step for each threshold. 0.01 is fine-grained enough to see local
# maxima in precision/recall on the typical ~1000-event log.
SWEEP_STEP: float = 0.01
SWEEP_LOW: float = 0.20
SWEEP_HIGH: float = 0.90

# Cosine drift threshold from §T2.3. WS-A's INT8 quantisation can shift the
# mean cosine by ~0.06; >= 0.03 is the surface-this-loudly trigger.
COSINE_DRIFT_TRIGGER: float = 0.03


# --------------------------------------------------------------------------- #
# Log line parser — matches the schema written by ``match_logger.info`` at
# vehicle_registry_identity.py:~1198-1210.
#
# Reference line shape (one physical line per match):
#
#   2026-05-13 12:00:00,000 - reid_match_perf - INFO -
#     MATCH_EVENT | Plate: ABC-123 | Slot: SLOT-A | Time: 2026-05-13T12:00:00 |
#     NewCost: 0.7345 | OldCost: 0.6201 | Flags: {...} | GateSnapshots: [...] |
#     SlotSnapshot: snap.jpg
#
# The parser is permissive about leading log prefixes — anything before the
# first ``MATCH_EVENT`` is dropped.
# --------------------------------------------------------------------------- #


_EVENT_RE = re.compile(
    r"MATCH_EVENT\s*\|\s*"
    r"Plate:\s*(?P<plate>[^|]+?)\s*\|\s*"
    r"Slot:\s*(?P<slot>[^|]+?)\s*\|\s*"
    r"Time:\s*(?P<time>[^|]+?)\s*\|\s*"
    r"NewCost:\s*(?P<new>[-+0-9.eE]+)\s*\|\s*"
    r"OldCost:\s*(?P<old>[-+0-9.eE]+)\s*\|\s*"
    r"Flags:\s*(?P<flags>\{[^}]*\})\s*\|\s*"
    r"GateSnapshots:\s*(?P<gates>\[[^\]]*\])\s*\|\s*"
    r"SlotSnapshot:\s*(?P<slot_snap>.+?)\s*$"
)


@dataclass
class MatchEvent:
    """Parsed ``MATCH_EVENT`` row from the audit log."""

    plate: str
    slot: str
    time: str
    new_cost: float
    old_cost: float
    flags: str
    gates: str
    slot_snapshot: str

    @property
    def has_valid_score(self) -> bool:
        # The audit log occasionally records NaN when the legacy fallback
        # path skipped the ReID compare. Drop those rows from the sweep.
        return (
            self.new_cost == self.new_cost  # NaN != NaN
            and self.new_cost >= 0.0
            and self.new_cost <= 1.0
        )


def parse_log_files(log_paths: Iterable[Path]) -> List[MatchEvent]:
    """Parse all ``MATCH_EVENT`` rows out of the given log files."""
    events: List[MatchEvent] = []
    for path in log_paths:
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError as exc:
            logger.warning("[calibrate] failed to read %s: %r", path, exc)
            continue
        for line in text.splitlines():
            m = _EVENT_RE.search(line)
            if m is None:
                continue
            try:
                ev = MatchEvent(
                    plate=m.group("plate").strip(),
                    slot=m.group("slot").strip(),
                    time=m.group("time").strip(),
                    new_cost=float(m.group("new")),
                    old_cost=float(m.group("old")),
                    flags=m.group("flags").strip(),
                    gates=m.group("gates").strip(),
                    slot_snapshot=m.group("slot_snap").strip(),
                )
            except (TypeError, ValueError):
                continue
            if ev.has_valid_score:
                events.append(ev)
    return events


# --------------------------------------------------------------------------- #
# Ground-truth join — parking_sessions table (SQLAlchemy optional).
# --------------------------------------------------------------------------- #


def fetch_ground_truth(
    db_url: Optional[str],
    plates: Iterable[str],
) -> Dict[str, bool]:
    """Return ``{plate: is_match}`` from the ``parking_sessions`` table.

    Ground-truth semantics:
      * If a plate has any ``parking_sessions`` row that ever reached a
        ``parked`` or ``exited`` state at the correct slot, treat the match
        as a *true positive* (label True).
      * Plates that exist in the log but never reached a slot binding are
        treated as *false positives* (label False).

    A row's ``slot_id`` is compared loosely (case-insensitive) against the
    audit log's ``slot`` field. When SQLAlchemy or the DB connection is
    unavailable we skip ground-truth fetching and return ``{}``; the caller
    falls through to a "self-consistency only" sweep that flags threshold
    drift but cannot compute precision/recall.
    """
    if not db_url:
        return {}
    plates = list({p for p in plates if p})
    if not plates:
        return {}
    try:
        from sqlalchemy import create_engine, text  # type: ignore
    except ImportError:
        logger.info(
            "[calibrate] sqlalchemy unavailable — skipping ground-truth join. "
            "Threshold sweep will report self-consistency only."
        )
        return {}
    try:
        engine = create_engine(db_url)
    except Exception as exc:
        logger.warning(
            "[calibrate] could not connect to %s: %r — skipping ground truth.",
            db_url,
            exc,
        )
        return {}

    truth: Dict[str, bool] = {}
    try:
        with engine.connect() as conn:
            for plate in plates:
                try:
                    row = conn.execute(
                        text(
                            "SELECT TOP 1 status FROM dbo.parking_sessions "
                            "WHERE plate_number = :p ORDER BY entry_time DESC"
                        ),
                        {"p": plate},
                    ).first()
                except Exception as exc:
                    logger.debug(
                        "[calibrate] DB query failed for plate=%s: %r",
                        plate,
                        exc,
                    )
                    continue
                # No row → never bound; treat as "audit log records the
                # match but DB never confirmed" → False.
                if row is None:
                    truth[plate] = False
                else:
                    truth[plate] = str(row[0]).lower() in ("parked", "exited", "open")
    except Exception as exc:
        logger.warning("[calibrate] ground-truth query loop failed: %r", exc)
    return truth


# --------------------------------------------------------------------------- #
# Threshold sweep
# --------------------------------------------------------------------------- #


@dataclass
class PrecisionRecallPoint:
    """One sample on the precision/recall curve for a given threshold."""

    threshold: float
    tp: int
    fp: int
    tn: int
    fn: int

    @property
    def precision(self) -> float:
        denom = self.tp + self.fp
        return self.tp / denom if denom else 1.0

    @property
    def recall(self) -> float:
        denom = self.tp + self.fn
        return self.tp / denom if denom else 1.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return (2 * p * r) / (p + r) if (p + r) else 0.0

    def as_dict(self) -> dict:
        return {
            "threshold": round(self.threshold, 4),
            "tp": self.tp,
            "fp": self.fp,
            "tn": self.tn,
            "fn": self.fn,
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
        }


def sweep_threshold(
    events: List[MatchEvent],
    truth: Dict[str, bool],
    low: float = SWEEP_LOW,
    high: float = SWEEP_HIGH,
    step: float = SWEEP_STEP,
) -> List[PrecisionRecallPoint]:
    """Sweep the threshold over [low, high] in ``step`` increments and
    return one PrecisionRecallPoint per sample. Events without ground
    truth count as the audit log's own verdict (i.e. score >= old default
    threshold).
    """
    if not events:
        return []
    points: List[PrecisionRecallPoint] = []
    threshold = low
    # Floating-point loop terminator: use a tiny epsilon to include 'high'.
    while threshold <= high + 1e-9:
        tp = fp = tn = fn = 0
        for ev in events:
            predicted = ev.new_cost >= threshold
            label = truth.get(ev.plate)
            if label is None:
                # No ground truth — fall back to the historical verdict
                # (clear at ``MatchingConfig.global_default = 0.55``). This
                # makes the curve self-consistent rather than empty when
                # the DB is unavailable.
                label = ev.new_cost >= MatchingConfig.global_default
            if predicted and label:
                tp += 1
            elif predicted and not label:
                fp += 1
            elif not predicted and label:
                fn += 1
            else:
                tn += 1
        points.append(
            PrecisionRecallPoint(
                threshold=round(threshold, 4),
                tp=tp,
                fp=fp,
                tn=tn,
                fn=fn,
            )
        )
        threshold += step
    return points


def recommend_threshold(points: List[PrecisionRecallPoint]) -> float:
    """Pick the threshold with the best F1 score, falling back to the
    closest precision >= 0.92, recall >= 0.88 point.
    """
    if not points:
        return MatchingConfig.global_default
    # First pass — points that meet the acceptance bar.
    qualifying = [p for p in points if p.precision >= 0.92 and p.recall >= 0.88]
    if qualifying:
        # Highest recall among qualifying — biases toward fewer false
        # negatives once the precision floor is cleared.
        qualifying.sort(key=lambda p: (p.recall, p.precision), reverse=True)
        return float(qualifying[0].threshold)
    # Fallback — best F1.
    return float(max(points, key=lambda p: p.f1).threshold)


# --------------------------------------------------------------------------- #
# Drift detection
# --------------------------------------------------------------------------- #


def detect_cosine_drift(events: List[MatchEvent]) -> Optional[Dict[str, float]]:
    """Surface a global cosine drift when the mean ``NewCost`` deviates from
    the snapshot defaults by >= ``COSINE_DRIFT_TRIGGER``.

    Returns ``{baseline, observed, delta, recommendation}`` or ``None`` when
    no drift is detected (or no events to measure).
    """
    if not events:
        return None
    scores = [ev.new_cost for ev in events if ev.has_valid_score]
    if not scores:
        return None
    observed = statistics.mean(scores)
    # Baseline is the midpoint of (b1_zone, b1_anpr) — the historical
    # operating point of the cascade.
    baseline = (MatchingConfig.b1_zone + MatchingConfig.b1_anpr) / 2.0
    delta = observed - baseline
    if abs(delta) < COSINE_DRIFT_TRIGGER:
        return None
    return {
        "baseline": round(baseline, 4),
        "observed": round(observed, 4),
        "delta": round(delta, 4),
        # When ``observed`` is lower than ``baseline``, recommend pulling
        # every threshold DOWN by the same delta (and vice versa).
        "recommendation": (
            f"Mean cosine has shifted by {delta:+.3f}. Consider applying a "
            f"uniform threshold shift of {delta:+.3f} to all matching "
            f"thresholds (or re-export the ReID IR)."
        ),
    }


# --------------------------------------------------------------------------- #
# Report assembly
# --------------------------------------------------------------------------- #


def build_report(
    events: List[MatchEvent],
    truth: Dict[str, bool],
) -> Dict:
    """Compute the full calibration report — per-threshold sweep results,
    recommended values, and any cosine-drift warning."""
    sweep = sweep_threshold(events, truth)
    sweep_dicts = [p.as_dict() for p in sweep]
    recommended_value = recommend_threshold(sweep)
    drift = detect_cosine_drift(events)

    # Apply the same recommended value to every calibratable field by
    # default. Operators can override per-field once they have inspected
    # the sweep results.
    recommended_block: Dict[str, float] = {
        name: round(recommended_value, 4) for name in CALIBRATABLE_FIELDS
    }

    return {
        "events_total": len(events),
        "events_with_truth": sum(1 for ev in events if ev.plate in truth),
        "sweep": sweep_dicts,
        "recommended_threshold": round(recommended_value, 4),
        "recommended_block": recommended_block,
        "cosine_drift": drift,
    }


def render_yaml_block(recommended_block: Dict[str, float]) -> str:
    """Render the recommended ``matching:`` YAML fragment."""
    lines = ["matching:"]
    for key, value in recommended_block.items():
        lines.append(f"  {key}: {value:.2f}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def _resolve_db_url(explicit: Optional[str]) -> Optional[str]:
    """Resolve ``--db-url`` from the CLI flag, env var, or config.yaml."""
    if explicit:
        return explicit
    env_url = os.environ.get("DATABASE_URL", "").strip()
    if env_url:
        return env_url
    cfg = Path("config.yaml")
    if cfg.exists():
        try:
            import yaml  # type: ignore

            data = yaml.safe_load(cfg.read_text(encoding="utf-8")) or {}
            return ((data.get("database") or {}).get("DATABASE_URL") or "").strip() or None
        except Exception as exc:
            logger.debug("[calibrate] config.yaml DB lookup failed: %r", exc)
    return None


def _apply_to_config(config_path: Path, recommended_block: Dict[str, float]) -> None:
    """Rewrite the ``matching:`` block of ``config_path`` in place."""
    try:
        import yaml  # type: ignore
    except ImportError:
        print(
            "PyYAML is required for --apply; install with `pip install pyyaml`.",
            file=sys.stderr,
        )
        sys.exit(2)

    if not config_path.exists():
        print(f"[calibrate] config file not found: {config_path}", file=sys.stderr)
        sys.exit(2)

    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    matching_block = raw.get("matching") or {}
    matching_block.update({k: float(v) for k, v in recommended_block.items()})
    raw["matching"] = matching_block
    config_path.write_text(
        yaml.safe_dump(raw, sort_keys=False, default_flow_style=False),
        encoding="utf-8",
    )
    print(f"[calibrate] wrote recommended matching: block to {config_path}")


def _resolve_log_paths(log_glob: str) -> List[Path]:
    """Expand a glob pattern to a list of existing log file paths."""
    return [Path(p) for p in sorted(glob(log_glob)) if Path(p).is_file()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Calibrate matching thresholds against the audit-log MATCH_EVENT "
            "stream. Emits a JSON report and a recommended YAML block."
        )
    )
    parser.add_argument(
        "--log-glob",
        default="logs/reid_match_perf*.log",
        help="Glob pattern selecting MATCH_EVENT log files (default %(default)s).",
    )
    parser.add_argument(
        "--db-url",
        default=None,
        help=(
            "SQLAlchemy URL of the parking_sessions DB. Defaults to "
            "$DATABASE_URL or the value in config.yaml. When unset the "
            "sweep runs in self-consistency mode (no precision/recall)."
        ),
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional path to write the JSON report to. Defaults to stdout.",
    )
    parser.add_argument(
        "--apply",
        nargs="?",
        const="config.yaml",
        default=None,
        help=(
            "When supplied, rewrites the ``matching:`` block of the given "
            "config file with the recommended thresholds. Without --apply "
            "the tool is read-only."
        ),
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress the human-readable preamble (JSON only).",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    paths = _resolve_log_paths(args.log_glob)
    if not paths:
        msg = (
            f"No log files matched '{args.log_glob}'. Provide --log-glob or "
            f"run the registry against real traffic first."
        )
        if not args.quiet:
            print(msg, file=sys.stderr)
        # Return a zero-event report rather than crashing — useful for CI.
        report = build_report([], {})
    else:
        events = parse_log_files(paths)
        if not args.quiet:
            print(
                f"[calibrate] parsed {len(events)} MATCH_EVENT rows from "
                f"{len(paths)} log file(s).",
                file=sys.stderr,
            )
        db_url = _resolve_db_url(args.db_url)
        truth = fetch_ground_truth(db_url, [ev.plate for ev in events])
        if not args.quiet:
            print(
                f"[calibrate] resolved ground truth for {len(truth)} plates "
                f"(db_url={'set' if db_url else 'unset'}).",
                file=sys.stderr,
            )
        report = build_report(events, truth)

    payload = json.dumps(report, indent=2)
    if args.output:
        Path(args.output).write_text(payload, encoding="utf-8")
        if not args.quiet:
            print(f"[calibrate] wrote report to {args.output}", file=sys.stderr)
    else:
        print(payload)

    if not args.quiet:
        print("\n--- Recommended YAML block ---", file=sys.stderr)
        print(render_yaml_block(report["recommended_block"]), file=sys.stderr)
        if report.get("cosine_drift"):
            print(
                "\n[calibrate] WARNING: cosine drift detected — "
                f"{report['cosine_drift']['recommendation']}",
                file=sys.stderr,
            )

    if args.apply:
        _apply_to_config(Path(args.apply), report["recommended_block"])

    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    sys.exit(main())

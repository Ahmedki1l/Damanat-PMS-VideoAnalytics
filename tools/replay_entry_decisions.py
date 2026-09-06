"""Replay every recorded Entry decision through the live engine.

WHY THIS IS POSSIBLE. The decision record was designed as a calibration corpus,
so it stores not just what happened but everything the policy needed to decide:
the full ranked candidate list after the colour veto, both runners-up, and the
three thresholds the decision was actually held to. That is a complete
description of one `evaluate_unique_match` call — everything except the raw
embeddings, which are the one thing the policy never looks at directly. It only
ever consumes cosine similarities.

So the embeddings can be RECONSTRUCTED rather than stored. For a query `e0` and
a target similarity `s`, the vector

    v = s * e0 + sqrt(1 - s^2) * e_k        (e_k orthogonal, unit)

is a unit vector whose cosine against the query is exactly `s`. Give every
candidate its own orthogonal axis and the whole similarity structure of a
recorded decision reproduces to machine precision. What comes out is not a
simulation of the engine — it IS the engine, run on evidence shaped to match a
real day at the gate.

WHAT IT IS FOR

    --verify    Every recorded decision must replay to the same verdict with
                the gallery disabled. This is the regression gate: it proves a
                change to the matcher did not silently move any decision that
                has already been reviewed. Exit code 1 if any diverge.

    --sweep     Re-run every scenario with synthetic gallery references at a
                range of uplift levels, and report what each level would change.
                A PROJECTION, clearly labelled: the real references live on the
                pod and their true uplift is what the shadow window measures.
                Its job is to bound the answer in advance — in particular to
                show whether an uplift large enough to help is also large
                enough to start confirming things that should stay refused.

Usage:
    python tools/replay_entry_decisions.py <dir-or-files...> [--verify] [--sweep]
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
from collections import Counter
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.entry.decision import EntryDecisionEngine  # noqa: E402
from src.entry.domain import (  # noqa: E402
    AttemptGroup,
    AttemptInput,
    AttemptRecord,
    CrossingInput,
    CrossingRecord,
    CrossingRole,
    FrameEvidence,
    PlateEvidence,
    PlateReadState,
    RecordStatus,
)
from src.entry.settings import EntrySettings  # noqa: E402

DIMENSIONS = 96
_BASE = datetime(2026, 1, 1, tzinfo=timezone.utc)


# ---------------------------------------------------------------- vectors --


def _axis(index: int) -> List[float]:
    vector = [0.0] * DIMENSIONS
    vector[index % DIMENSIONS] = 1.0
    return vector


def _at_similarity(similarity: float, axis: int) -> Tuple[float, ...]:
    """A unit vector whose cosine against axis 0 is exactly `similarity`."""
    similarity = max(-1.0, min(1.0, float(similarity)))
    residual = math.sqrt(max(0.0, 1.0 - similarity * similarity))
    vector = [0.0] * DIMENSIONS
    vector[0] = similarity
    vector[axis % DIMENSIONS] += residual
    return tuple(vector)


def _blend(base: Tuple[float, ...], target: float, axis: int) -> Tuple[float, ...]:
    """A unit vector at cosine `target` from `base`, in a fresh direction."""
    target = max(-1.0, min(1.0, float(target)))
    residual = math.sqrt(max(0.0, 1.0 - target * target))
    extra = _axis(axis)
    vector = [target * b + residual * e for b, e in zip(base, extra)]
    norm = math.sqrt(sum(value * value for value in vector)) or 1.0
    return tuple(value / norm for value in vector)


# --------------------------------------------------------------- scenario --


def _evidence(embedding: Tuple[float, ...], evidence_id: str) -> FrameEvidence:
    return FrameEvidence(
        evidence_id=evidence_id,
        embedding=embedding,
        plate=PlateEvidence(
            evidence_id=evidence_id,
            camera_id="CAM-23",
            source_role="primary",
            state=PlateReadState.NO_PLATE,
            text="",
            confidence=0.0,
        ),
        colour_hsv=None,
    )


def _group(
    group_id: str,
    embeddings: List[Tuple[float, ...]],
    captured_at: datetime,
    gallery: Tuple[Tuple[float, ...], ...] = (),
) -> AttemptGroup:
    request = AttemptInput(
        attempt_id="attempt-" + group_id,
        source_event_id="event-" + group_id,
        camera_id="CAM-ENTRY",
        captured_at=captured_at,
        reported_plate="AAA-1111",
        reported_confidence=0.96,
        metadata={},
    )
    record = AttemptRecord(
        request=request,
        evidence=tuple(
            _evidence(embedding, f"{group_id}-{index}")
            for index, embedding in enumerate(embeddings)
        ),
        group_id=group_id,
    )
    return AttemptGroup(
        group_id=group_id,
        attempts={request.attempt_id: record},
        status=RecordStatus.PENDING,
        identity_key="AAA1111",
        gallery_embeddings=gallery,
    )


def _crossing(
    crossing_id: str,
    embedding: Tuple[float, ...],
    captured_at: datetime,
    camera_id: str,
) -> CrossingRecord:
    request = CrossingInput(
        crossing_id=crossing_id,
        source_event_id="src-" + crossing_id,
        camera_id=camera_id,
        captured_at=captured_at,
        line_id="1",
        direction="ramp-entry",
        role=CrossingRole.PRIMARY,
        metadata={},
    )
    return CrossingRecord(
        request=request,
        evidence=(_evidence(embedding, "obs-" + crossing_id),),
        status=RecordStatus.PENDING,
    )


class Scenario:
    """One recorded `reid_evaluation`, rebuilt as live engine inputs."""

    def __init__(self, record: dict):
        reid = record["reid"]
        self.record = record
        self.day = str(record.get("ts", ""))[:10]
        self.camera = (record.get("observation") or {}).get("camera", "?")
        self.expected_score = float(reid["score"])
        self.expected_row_margin = float(reid["row_margin"])
        self.expected_column_margin = float(reid["column_margin"])
        self.expected_reason = str(record.get("reason", ""))
        self.expected_accepted = bool(reid.get("accepted"))
        self.winner = str(reid["argmax"])
        self.row_runner = float(reid.get("row_runner") or 0.0)
        self.column_runner = float(reid.get("column_runner") or 0.0)
        self.ranked = [(str(g), float(s)) for g, s in (record.get("ranked") or [])]
        self.settings = EntrySettings(
            reid_min_score=float(reid["min_score"]),
            reid_row_margin=float(reid["min_row_margin"]),
            reid_column_margin=float(reid["min_column_margin"]),
            # The corpus has already had the veto applied — `ranked` is the list
            # AFTER it. Re-applying it here on synthetic colour would remove
            # candidates the recorded decision actually considered.
            colour_veto_enabled=False,
        )

    def build(self, gallery_uplift: Optional[float] = None):
        """Materialise groups and crossings. `gallery_uplift` is a projection."""
        query = tuple(_axis(0))
        crossing = _crossing("obs", query, _BASE + timedelta(minutes=5), self.camera)
        attempt_at = _BASE + timedelta(minutes=1)

        groups: Dict[str, AttemptGroup] = {}
        for index, (group_id, score) in enumerate(self.ranked):
            gallery: Tuple[Tuple[float, ...], ...] = ()
            if gallery_uplift is not None:
                # One synthetic reference per candidate, at the candidate's own
                # score lifted by `gallery_uplift`. Every candidate gets exactly
                # one, so the comparison stays like-for-like and the sweep
                # measures the uplift rather than the reference count.
                gallery = (
                    _at_similarity(
                        min(1.0, score + gallery_uplift), 40 + index
                    ),
                )
            groups[group_id] = _group(
                group_id,
                [_at_similarity(score, 1 + index)],
                attempt_at,
                gallery=gallery,
            )

        crossings = [crossing]
        if self.column_runner:
            # A second observation on the SAME camera, scoring exactly the
            # recorded column runner against the winner. That is what the
            # column gate compares, and nothing else in the scenario reads it.
            winner_vector = groups[self.winner].attempts[
                "attempt-" + self.winner
            ].evidence[0].embedding
            crossings.append(
                _crossing(
                    "obs-column",
                    _blend(winner_vector, self.column_runner, 80),
                    _BASE + timedelta(minutes=6),
                    self.camera,
                )
            )

        # A row runner ABOVE every live candidate is a remembered contest: a
        # competitor that has since been consumed or aged out. The record keeps
        # it precisely so a margin cannot be won by the pool shrinking.
        live_runner = self.ranked[1][1] if len(self.ranked) > 1 else 0.0
        if self.row_runner > live_runner + 1e-9:
            crossing.contested_scores["remembered"] = self.row_runner

        return crossing, groups, crossings

    def replay(self, gallery_uplift: Optional[float] = None):
        crossing, groups, crossings = self.build(gallery_uplift)
        engine = EntryDecisionEngine(self.settings)
        return engine.evaluate_unique_match(crossing, groups, crossings)

    # `reid_block` rounds each scalar to 6 decimals INDEPENDENTLY, so a score
    # replayed from the record can be off by 5e-7 — and a margin, being the
    # difference of two such values, by twice that. Comparing margins at 1e-6
    # therefore flags 180 exact replays as divergent purely on the last digit.
    # These are the tightest tolerances the stored precision actually supports.
    SCORE_TOLERANCE = 1e-6
    MARGIN_TOLERANCE = 2e-6

    def matches(self, evaluation) -> List[str]:
        """Empty when the replay reproduced the record."""
        if evaluation is None:
            return ["engine returned no evaluation"]
        problems = []
        if evaluation.group_id != self.winner:
            problems.append(f"argmax {evaluation.group_id} != {self.winner}")
        for name, actual, expected, tolerance in (
            ("score", evaluation.score, self.expected_score, self.SCORE_TOLERANCE),
            (
                "row_margin",
                evaluation.row_margin,
                self.expected_row_margin,
                self.MARGIN_TOLERANCE,
            ),
            (
                "column_margin",
                evaluation.column_margin,
                self.expected_column_margin,
                self.MARGIN_TOLERANCE,
            ),
        ):
            if abs(float(actual) - expected) > tolerance:
                problems.append(f"{name} {actual:.6f} != {expected:.6f}")
        if bool(evaluation.accepted) != self.expected_accepted:
            problems.append(
                f"accepted {evaluation.accepted} != {self.expected_accepted}"
            )
        if evaluation.reason != self.expected_reason:
            problems.append(f"reason {evaluation.reason} != {self.expected_reason}")
        return problems


# ------------------------------------------------------------------ input --


def load_records(paths: List[str]) -> List[dict]:
    files: List[str] = []
    for path in paths:
        if os.path.isdir(path):
            files.extend(sorted(glob.glob(os.path.join(path, "*.jsonl"))))
        else:
            files.append(path)
    records = []
    for path in files:
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    # A torn final line is expected on a log copied off a live
                    # pod. Skipping it is correct; failing on it is not.
                    continue
    return records


def scenarios(records: List[dict]) -> List[Scenario]:
    out = []
    for record in records:
        if record.get("stage") != "reid_evaluation":
            continue
        if not record.get("reid") or not record.get("ranked"):
            continue
        out.append(Scenario(record))
    return out


# ----------------------------------------------------------------- verify --


def verify(cases: List[Scenario]) -> int:
    print("=" * 72)
    print(f"REPLAY  {len(cases)} recorded Re-ID decisions, gallery DISABLED")
    print("=" * 72)
    failures = []
    by_day = Counter()
    by_reason = Counter()
    for case in cases:
        problems = case.matches(case.replay())
        by_day[case.day] += 1
        by_reason[case.expected_reason] += 1
        if problems:
            failures.append((case, problems))

    for day, count in sorted(by_day.items()):
        print(f"  {day}  {count:4d} decisions")
    print()
    print("  outcomes replayed:")
    for reason, count in by_reason.most_common():
        print(f"    {reason:<34} {count}")
    print()
    if failures:
        print(f"  DIVERGED: {len(failures)}")
        for case, problems in failures[:20]:
            print(f"    {case.day} {case.camera} {case.winner[:14]}: "
                  f"{'; '.join(problems)}")
        if len(failures) > 20:
            print(f"    ... and {len(failures) - 20} more")
        return 1
    print("  every decision reproduced exactly. No behaviour change.")
    return 0


# ------------------------------------------------------------------ sweep --


def sweep(cases: List[Scenario], uplifts: List[float]) -> None:
    print()
    print("=" * 72)
    print("PROJECTION  what a gallery of the given uplift would change")
    print("=" * 72)
    print("  A PROJECTION, not a measurement. Real references live on the pod;")
    print("  the shadow window measures the true uplift. This bounds it.")
    print()
    baseline = {}
    for index, case in enumerate(cases):
        evaluation = case.replay()
        baseline[index] = (
            bool(evaluation and evaluation.accepted),
            case.expected_reason,
        )

    header = f"  {'uplift':>7} {'accept':>7} {'rescued':>8} {'newly-refused':>14}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for uplift in uplifts:
        accepted = rescued = refused = 0
        for index, case in enumerate(cases):
            evaluation = case.replay(uplift)
            now_ok = bool(evaluation and evaluation.accepted)
            was_ok = baseline[index][0]
            accepted += int(now_ok)
            rescued += int(now_ok and not was_ok)
            refused += int(was_ok and not now_ok)
        total = len(cases)
        print(
            f"  {uplift:>7.2f} {accepted:>4d}/{total:<3d} {rescued:>8d} "
            f"{refused:>14d}"
        )
    print()
    print("  'rescued'       = refused before, accepted with the gallery.")
    print("  'newly-refused' = accepted before, refused with it. Must stay 0:")
    print("                    a uniform uplift lifts every candidate, so any")
    print("                    non-zero value means the margin gates are")
    print("                    reacting to the reference count, not the car.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", help="decision-log files or a directory")
    parser.add_argument("--verify", action="store_true", help="regression gate")
    parser.add_argument("--sweep", action="store_true", help="gallery projection")
    args = parser.parse_args()

    records = load_records(args.paths)
    cases = scenarios(records)
    if not cases:
        print("no reid_evaluation records found", file=sys.stderr)
        return 2

    status = 0
    if args.verify or not args.sweep:
        status = verify(cases)
    if args.sweep:
        sweep(cases, [0.05, 0.10, 0.15, 0.20, 0.30, 0.40])
    return status


if __name__ == "__main__":
    raise SystemExit(main())

"""Read a shadow run and answer the go/no-go question.

The Entry V2 pipeline raises no alerts. Every outcome is a JSONL record, which
makes this file the review: if the run is not queried, nobody has looked at it.
So this is not a debug convenience, it is the instrument the flip to
authoritative is decided with, and it is deliberately blunt about it.

Two kinds of output:

  HARD STOPS   things that must be zero. Any one of them fails the gate, no
               judgement required and no discussion.
  REVIEW       distributions and samples a person has to actually read. These
               cannot be automated into a verdict, and pretending otherwise
               would be the whole problem with review gates.

Usage:
    # pull the day off the pod first
    POD=$(kubectl get pod -l app=pms-video-analytics -o name | head -1)
    kubectl cp "${POD#pod/}:/app/vehicle_images/entry_v2_shadow" ./shadow-$(date +%F)

    python tools/summarize_entry_shadow.py ./shadow-2026-08-29
    python tools/summarize_entry_shadow.py ./shadow-2026-08-29 --json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable


def load_records(paths: Iterable[Path]) -> list[dict]:
    """Every decision record in the given files or directories."""
    records: list[dict] = []
    for path in paths:
        files = (
            sorted(path.glob("entry_decisions_*.jsonl"))
            if path.is_dir()
            else [path]
        )
        for file in files:
            with open(file, encoding="utf-8") as handle:
                for line_no, line in enumerate(handle, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        # A torn last line is normal: the writer flushes but
                        # never fsyncs, so a copy taken mid-write can clip one.
                        print(
                            f"  ! skipped unparsable line {file.name}:{line_no}",
                            file=sys.stderr,
                        )
                        continue
                    if isinstance(record, dict):
                        records.append(record)
    return records


def _stage(records, stage):
    return [r for r in records if r.get("stage") == stage]


def hard_stops(records: list[dict]) -> list[tuple[str, int, list]]:
    """Checks that must all be zero. Returns (name, count, examples)."""
    checks: list[tuple[str, int, list]] = []

    # A single observation never proves a car entered. Two independent ones do.
    offenders = [
        r
        for r in _stage(records, "physical_confirm")
        if r.get("result") == "confirmed" and len(r.get("witnesses") or []) < 2
    ]
    checks.append(("confirmations with fewer than two witnesses", len(offenders),
                   offenders[:3]))

    # A wrong plate opens a session under someone else's name.
    offenders = [
        r
        for r in _stage(records, "plate_consensus")
        if r.get("result") == "confirmed"
        and (r.get("plate") or {}).get("outcome") != "consensus"
    ]
    checks.append(("confirmations with no plate consensus", len(offenders),
                   offenders[:3]))

    # The ramp cameras read no plates. If one ever appears as a source, the
    # separation this design rests on has been lost somewhere.
    offenders = []
    for record in _stage(records, "plate_consensus"):
        available = (record.get("plate") or {}).get("available") or []
        if any(src not in ("anpr", "hik_text", "our_ocr") for src in available):
            offenders.append(record)
    checks.append(("plate sources that are not one of the three", len(offenders),
                   offenders[:3]))

    # Degrading to DisabledEvidenceProcessor is SAFE but records nothing, so a
    # clean-looking run can mean the pipeline never ran at all.
    offenders = [
        r for r in records if "invalid_configuration" in str(r.get("reason", ""))
    ]
    checks.append(("entry_v2_invalid_configuration", len(offenders),
                   offenders[:3]))

    # HIK substitutes for a MISSING ANPR read; it never adds a second witness
    # alongside one, because it is the platform's log of the same gate event.
    offenders = [
        r
        for r in _stage(records, "physical_confirm")
        if r.get("result") == "confirmed"
        and {"anpr", "hik"} <= set(r.get("witnesses") or [])
    ]
    checks.append(("confirmations counting HIK alongside ANPR", len(offenders),
                   offenders[:3]))

    return checks


def review_sections(records: list[dict]) -> dict:
    """Distributions and samples a person has to read. No verdicts here."""
    out: dict = {}

    out["records"] = len(records)
    out["by_stage"] = Counter(r.get("stage", "?") for r in records)
    out["by_result"] = Counter(r.get("result", "?") for r in records)

    # Which pair actually fired. A pair that never fires is a path that is not
    # being exercised, which is as important as one that misbehaves.
    pairs = Counter()
    for record in _stage(records, "physical_confirm"):
        if record.get("result") == "confirmed":
            pairs["+".join(sorted(record.get("witnesses") or []))] += 1
    out["witness_pairs"] = pairs

    # FIFO is measured and never enforced. If it turns out to agree with Re-ID
    # almost always, that is an argument for using it as a tie-break - made
    # from evidence rather than assumption. If it disagrees often, that is the
    # justification for having refused to enforce it.
    fifo = [r.get("fifo") for r in records if r.get("fifo")]
    agreed = sum(1 for f in fifo if f.get("agreed"))
    out["fifo"] = {
        "compared": len(fifo),
        "agreed": agreed,
        "disagreed": len(fifo) - agreed,
    }

    # Every colour veto is a candidate REMOVED from contention. A veto that
    # removed the right car is invisible here and has to be read case by case.
    vetoes = [
        r for r in records if (r.get("colour") or {}).get("vetoed")
    ]
    out["colour_vetoes"] = len(vetoes)

    # Did we ask HikCentral, and did the answer help?
    hik = [r.get("hik") for r in records if r.get("hik")]
    out["hik"] = {
        "queries": len(hik),
        "not_configured": sum(1 for h in hik if h.get("configured") is False),
        "api_errors": sum(1 for h in hik if h.get("api_error")),
        "records_returned": sum(int(h.get("records") or 0) for h in hik),
        "images": sum(int(h.get("images") or 0) for h in hik),
    }

    # Re-ID distributions - the reason the corpus exists. The shipped
    # thresholds should sit on a plateau, not a cliff.
    accepted, rejected = [], []
    for record in _stage(records, "reid_evaluation"):
        reid = record.get("reid") or {}
        score = reid.get("score")
        if score is None:
            continue
        (accepted if reid.get("accepted") else rejected).append(reid)
    out["reid"] = {
        "accepted": _distribution([r["score"] for r in accepted]),
        "rejected": _distribution([r["score"] for r in rejected]),
        "row_margin_accepted": _distribution(
            [r.get("row_margin", 0.0) for r in accepted]
        ),
    }
    out["reid_reject_reasons"] = Counter(
        r.get("reason", "?")
        for r in _stage(records, "reid_evaluation")
        if not (r.get("reid") or {}).get("accepted")
    )

    # The cases a person must open individually.
    out["needs_eyes"] = {
        "ambiguous": [r for r in records if r.get("result") == "ambiguous"][:5],
        "unreadable": [r for r in records if r.get("result") == "unreadable"][:5],
        "expired": [r for r in records if r.get("result") == "expired"][:5],
    }
    out["needs_eyes_counts"] = {
        key: sum(1 for r in records if r.get("result") == key)
        for key in ("ambiguous", "unreadable", "expired", "hik_degraded")
    }

    # Identity hygiene: a rising same-key split rate means ANPR is misreading,
    # or the appearance guard is too tight for this facility's gate imagery.
    identities = _stage(records, "anpr_identity")
    out["identity"] = {
        "created": sum(
            1 for r in identities if (r.get("identity") or {}).get("created")
        ),
        "enriched": sum(
            1 for r in identities if not (r.get("identity") or {}).get("created")
        ),
        "same_key_splits": sum(
            1 for r in identities
            if (r.get("identity") or {}).get("same_key_split")
        ),
        "correction_candidates": sum(
            1 for r in identities
            if (r.get("identity") or {}).get("correction_candidate_of")
        ),
        "hik_sourced": sum(
            1 for r in identities if (r.get("identity") or {}).get("hik_sourced")
        ),
    }
    return out


def _distribution(values: list[float]) -> dict:
    if not values:
        return {"n": 0}
    ordered = sorted(values)
    def pick(q):
        return round(ordered[min(len(ordered) - 1, int(q * len(ordered)))], 4)
    return {
        "n": len(ordered),
        "min": round(ordered[0], 4),
        "p10": pick(0.10),
        "p50": pick(0.50),
        "p90": pick(0.90),
        "max": round(ordered[-1], 4),
    }


def render(records: list[dict]) -> int:
    """Print the report. Returns a shell exit code: 0 clean, 1 hard stop."""
    if not records:
        print(
            "NO RECORDS.\n"
            "  That is not a clean run, it is an absent one. Check that\n"
            "  ENTRY_V2_DECISION_LOG_DIR is set on the VA deployment, that the\n"
            "  directory is writable, and that build_entry_coordinator did not\n"
            "  degrade to DisabledEvidenceProcessor on a config error."
        )
        return 1

    checks = hard_stops(records)
    review = review_sections(records)

    print("=" * 72)
    print(f"ENTRY V2 SHADOW REVIEW   {review['records']} records")
    print("=" * 72)

    print("\nHARD STOPS  (all must be zero)")
    failed = 0
    for name, count, examples in checks:
        mark = "ok  " if count == 0 else "FAIL"
        print(f"  [{mark}] {name}: {count}")
        if count:
            failed += 1
            for example in examples:
                print(f"          e.g. {_brief(example)}")

    print("\nVOLUME")
    for stage, count in sorted(review["by_stage"].items()):
        print(f"  {stage:<20} {count}")
    print("  --")
    for result, count in sorted(review["by_result"].items()):
        print(f"  {result:<20} {count}")

    print("\nWITNESS PAIRS  (a pair that never fires is a path never exercised)")
    if review["witness_pairs"]:
        for pair, count in review["witness_pairs"].most_common():
            print(f"  {pair:<20} {count}")
    else:
        print("  none - nothing was confirmed")

    print("\nRe-ID  (thresholds should sit on a plateau, not a cliff)")
    print(f"  accepted  {review['reid']['accepted']}")
    print(f"  rejected  {review['reid']['rejected']}")
    print(f"  row margin, accepted: {review['reid']['row_margin_accepted']}")
    for reason, count in review["reid_reject_reasons"].most_common(6):
        print(f"    {reason:<34} {count}")

    fifo = review["fifo"]
    print("\nFIFO vs Re-ID  (measured, never enforced)")
    print(
        f"  compared {fifo['compared']}   agreed {fifo['agreed']}   "
        f"disagreed {fifo['disagreed']}"
    )
    if fifo["disagreed"]:
        print("  -> read each disagreement: Re-ID should have been right in all.")

    print(f"\nCOLOUR VETOES  {review['colour_vetoes']}")
    if review["colour_vetoes"]:
        print("  -> confirm none of them removed the CORRECT identity.")

    hik = review["hik"]
    print("\nHIKCENTRAL  (we query it; it never calls us)")
    print(
        f"  queries {hik['queries']}   records {hik['records_returned']}   "
        f"images {hik['images']}   api errors {hik['api_errors']}"
    )
    if hik["not_configured"]:
        print(
            f"  !! {hik['not_configured']} query/queries were never made: ramp\n"
            "     indexCodes unset. That is NOT 'no events' - run\n"
            "     scripts/setup/probe_hik_camera_events.py in the pod."
        )

    identity = review["identity"]
    print("\nIDENTITY")
    print(
        f"  created {identity['created']}   enriched {identity['enriched']}   "
        f"hik-sourced {identity['hik_sourced']}"
    )
    print(
        f"  same-key splits {identity['same_key_splits']}   "
        f"correction candidates {identity['correction_candidates']}"
    )

    print("\nNEEDS EYES  (no tool can turn these into a verdict)")
    for key, count in review["needs_eyes_counts"].items():
        print(f"  {key:<14} {count}")
    for key, samples in review["needs_eyes"].items():
        for sample in samples:
            print(f"    {key}: {_brief(sample)}")

    print("\n" + "=" * 72)
    if failed:
        print(f"VERDICT: NOT READY - {failed} hard stop(s) failed.")
        print("         Fix and re-run the window. Do not flip to authoritative.")
    else:
        print("VERDICT: hard stops clean.")
        print("         This is necessary, NOT sufficient. The sections above")
        print("         marked 'needs eyes' still have to be read by a person")
        print("         before the mode is changed.")
    print("=" * 72)
    return 1 if failed else 0


def _brief(record: dict) -> str:
    observation = record.get("observation") or {}
    identity = record.get("identity") or {}
    return (
        f"{record.get('stage')}/{record.get('result')} "
        f"reason={record.get('reason')} "
        f"obs={observation.get('id', '-')} "
        f"key={identity.get('identity_key', '-')}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths", nargs="+", type=Path,
        help="JSONL files, or directories holding entry_decisions_*.jsonl",
    )
    parser.add_argument("--json", action="store_true", help="machine-readable")
    args = parser.parse_args()

    missing = [p for p in args.paths if not p.exists()]
    if missing:
        for path in missing:
            print(f"no such path: {path}", file=sys.stderr)
        return 2

    records = load_records(args.paths)
    if args.json:
        payload = {
            "records": len(records),
            "hard_stops": {name: count for name, count, _ in hard_stops(records)},
            "review": {
                key: (dict(value) if isinstance(value, Counter) else value)
                for key, value in review_sections(records).items()
                if key != "needs_eyes"
            },
        }
        print(json.dumps(payload, indent=2, default=str))
        return 1 if any(payload["hard_stops"].values()) else 0
    return render(records)


if __name__ == "__main__":
    raise SystemExit(main())

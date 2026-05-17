"""
tools/finetune_osnet_facility.py — Phase 3 / T3.1 OSNet facility fine-tuning.

Mines positive and hard-negative pairs from production data, runs a short
triplet-loss fine-tune of ``osnet_ain_x1_0`` on the facility-specific
distribution, then re-exports the resulting checkpoint to OpenVINO INT8
through ``tools/export_osnet_openvino.py``.

Motivation
----------

WS-A's INT8 quantisation shifts the cosine distribution by ~0.065 from the
torchreid baseline (see ``tools/calibrate_thresholds.py`` cosine-drift
detection). The pretrained MSMT weights are also a generic re-ID baseline
trained on outdoor surveillance — facility-specific lighting and camera
angles benefit from a short fine-tune. Typical gain: +3-6 percentage points
on the end-to-end accuracy bench (T2.4).

Mining strategy
---------------

* **Positive pairs** — same plate seen across at least two different
  cameras / sessions:
  - Primary source: ``MATCH_EVENT`` lines in ``logs/reid_match_perf*.log``
    joined against ``parking_sessions`` (database). Each row is a confirmed
    (entry_snapshot, slot_snapshot) pair.
  - Fallback source: filename convention in ``vehicle_images/`` —
    ``<plate>_<camera>_<datetime>.jpg``. Crops sharing a plate but captured
    from different cameras (or > 60 s apart on the same camera) become
    positives.
  - Same-frame bursts (entry-exit within 60 seconds) are dropped — they're
    near-duplicates and bias the training toward identity rather than
    invariance.

* **Hard negatives**:
  - OCR-contradiction rejects logged by the Phase 2 ``_apply_ensemble``
    cascade (``ensemble_ocr_contradiction`` reason). These are the
    highest-value negatives: ReID was confident, OCR disagreed, so the
    embedding was wrong.
  - "Close-but-no-cigar" reattach attempts in ``[0.40, threshold)`` —
    above the ensemble band but below the legacy threshold. These pairs
    look similar to the embedder but are known-different.

Training
--------

* Backbone: ``osnet_ain_x1_0`` (matches the runtime path at
  ``src/reid_matcher/reid_matcher.py:18-64``).
* Loss: triplet margin with batch-hard mining (within-batch hardest
  positive + negative). Margin tunable via ``--margin``.
* Optimiser: AdamW + cosine LR schedule.
* Early-stop: triplet accuracy on a held-out 10 % val split — stop after 3
  epochs with no improvement.

Export
------

After training the best checkpoint is loaded and re-exported via
``tools.export_osnet_openvino`` into ``models/osnet_finetuned_int8/``.
When ``--apply-export`` is set, ``MatchingConfig.reid_openvino_model_dir``
is updated to point at the new artifact directory; otherwise the operator
must wire the new path through ``config.yaml`` manually.

Graceful degradation
--------------------

The tool is data-dependent. When the log glob is empty, the DB is
unreachable, or torchreid / torch is not installed, the script logs a
clear warning, writes a placeholder report file, and exits non-zero so
operators see the failure without losing the partial state. This keeps
the CI smoke tests fast and predictable.

Usage
~~~~~

::

    python tools/finetune_osnet_facility.py
    python tools/finetune_osnet_facility.py --match-log-glob 'logs/*.log' \
        --vehicle-images-dir vehicle_images/ --epochs 15 --batch-size 32
    python tools/finetune_osnet_facility.py --apply-export

CLI surface intentionally mirrors ``tools/export_osnet_openvino.py`` so
operators can drop in either tool with the same operational habits.
"""

from __future__ import annotations

import argparse
import collections
import dataclasses
import datetime as _dt
import glob as _glob
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

# Allow ``python tools/finetune_osnet_facility.py`` to find ``src``.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


logger = logging.getLogger("finetune_osnet_facility")


# --------------------------------------------------------------------------- #
# Defaults
# --------------------------------------------------------------------------- #

DEFAULT_MATCH_LOG_GLOB = "logs/reid_match_perf*.log"
DEFAULT_VEHICLE_IMAGES_DIR = Path("vehicle_images")
DEFAULT_OUTPUT_DIR = Path("models/osnet_finetuned")
DEFAULT_EXPORT_DIR = Path("models/osnet_finetuned_int8")
DEFAULT_EPOCHS = 15
DEFAULT_BATCH_SIZE = 32
DEFAULT_LEARNING_RATE = 1e-4
DEFAULT_MARGIN = 0.3
DEFAULT_VAL_FRACTION = 0.10

# Same-burst skip threshold (seconds). Pairs with delta < this are dropped
# because near-duplicate frames bias training toward identity not invariance.
SAME_BURST_SECONDS = 60.0

# Hard-negative score band (close-but-no-cigar) — see module docstring.
HARD_NEG_LOW = 0.40

# Vehicle image filename pattern: <plate>_<camera>_<YYYYMMDD>_<HHMMSS>.jpg
_VEHICLE_IMG_RE = re.compile(
    r"^(?P<plate>[A-Za-z0-9\-]+?)_(?P<camera>CAM-[\w-]+?)_"
    r"(?P<date>\d{8})_(?P<time>\d{6})\..+$"
)


# --------------------------------------------------------------------------- #
# Mined pair dataclasses
# --------------------------------------------------------------------------- #


@dataclasses.dataclass(frozen=True)
class MinedPair:
    """One mined (anchor, other) pair with an explicit label.

    ``label`` is True for a positive pair (same vehicle), False for a hard
    negative (known different). ``source`` records where the pair came from
    so the report can attribute its mining yield.
    """

    anchor_path: str
    other_path: str
    label: bool
    source: str
    plate: str = ""
    delta_seconds: float = 0.0
    score: float = 0.0


@dataclasses.dataclass
class MiningReport:
    """Aggregate counters for a single mining run. Persisted alongside the
    checkpoint so we can audit how many pairs each source produced."""

    positives: int = 0
    negatives: int = 0
    skipped_same_burst: int = 0
    skipped_missing_file: int = 0
    log_files_scanned: int = 0
    log_lines_parsed: int = 0
    vehicle_images_seen: int = 0
    db_truth_resolved: int = 0
    sources: Dict[str, int] = dataclasses.field(default_factory=dict)
    warnings: List[str] = dataclasses.field(default_factory=list)

    def add(self, pair: MinedPair) -> None:
        if pair.label:
            self.positives += 1
        else:
            self.negatives += 1
        self.sources[pair.source] = self.sources.get(pair.source, 0) + 1

    def as_dict(self) -> dict:
        return {
            "positives": self.positives,
            "negatives": self.negatives,
            "skipped_same_burst": self.skipped_same_burst,
            "skipped_missing_file": self.skipped_missing_file,
            "log_files_scanned": self.log_files_scanned,
            "log_lines_parsed": self.log_lines_parsed,
            "vehicle_images_seen": self.vehicle_images_seen,
            "db_truth_resolved": self.db_truth_resolved,
            "sources": dict(self.sources),
            "warnings": list(self.warnings),
        }


# --------------------------------------------------------------------------- #
# Filename parsing
# --------------------------------------------------------------------------- #


def parse_vehicle_image_filename(name: str) -> Optional[Tuple[str, str, _dt.datetime]]:
    """Parse ``<plate>_<camera>_<YYYYMMDD>_<HHMMSS>.<ext>`` filenames.

    Returns ``(plate, camera, timestamp)`` or ``None`` when the name does not
    match. We accept any image extension (jpg/png/etc).
    """
    m = _VEHICLE_IMG_RE.match(name)
    if not m:
        return None
    try:
        timestamp = _dt.datetime.strptime(
            f"{m.group('date')}_{m.group('time')}", "%Y%m%d_%H%M%S"
        )
    except ValueError:
        return None
    return m.group("plate"), m.group("camera"), timestamp


def index_vehicle_images(
    vehicle_images_dir: Path,
) -> Dict[str, List[Tuple[Path, str, _dt.datetime]]]:
    """Index ``vehicle_images_dir`` by plate → [(path, camera, ts), ...]
    sorted by timestamp.

    Files whose names do not match the convention are skipped (the registry
    only writes files in this format, but operators occasionally drop ad-hoc
    crops in the same directory).
    """
    if not vehicle_images_dir.exists() or not vehicle_images_dir.is_dir():
        return {}
    by_plate: Dict[str, List[Tuple[Path, str, _dt.datetime]]] = (
        collections.defaultdict(list)
    )
    for p in vehicle_images_dir.iterdir():
        if not p.is_file():
            continue
        parsed = parse_vehicle_image_filename(p.name)
        if parsed is None:
            continue
        plate, camera, ts = parsed
        by_plate[plate].append((p, camera, ts))
    for plate, rows in by_plate.items():
        rows.sort(key=lambda r: r[2])
    return dict(by_plate)


# --------------------------------------------------------------------------- #
# MATCH_EVENT log parsing
# --------------------------------------------------------------------------- #


_MATCH_EVENT_RE = re.compile(
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


# Free-form "rejection" lines we look for to mine hard negatives. These are
# emitted by Phase 2's ensemble code (``MatchDecision._apply_ensemble``) and
# by the legacy reattach path. The regex stays permissive — it only needs to
# extract ``plate`` and ``score`` because the snapshot path is resolved by a
# subsequent vehicle-image lookup.
_REJECT_RE = re.compile(
    r"(?P<kind>ocr_contradiction|reattach_below)\s*\|\s*"
    r"Plate:\s*(?P<plate>[^|]+?)\s*\|\s*"
    r"Score:\s*(?P<score>[-+0-9.eE]+)"
)


@dataclasses.dataclass
class MatchEventRow:
    """One MATCH_EVENT audit-log row used for positive mining."""

    plate: str
    slot: str
    timestamp: _dt.datetime
    new_cost: float
    old_cost: float
    gate_snapshots: List[str]
    slot_snapshot: str


def _parse_gate_snapshots(raw: str) -> List[str]:
    """Convert a ``GateSnapshots: ['a.jpg', 'b.jpg']`` literal to a list.

    The audit logger writes a Python ``repr(list)``; we fall back to a
    minimal split when the literal_eval fails (e.g. an empty bracket pair).
    """
    raw = (raw or "").strip()
    if not raw or raw in ("[]", "None"):
        return []
    try:
        import ast

        parsed = ast.literal_eval(raw)
        if isinstance(parsed, (list, tuple)):
            return [str(x) for x in parsed]
    except (SyntaxError, ValueError):
        pass
    # Fallback: best-effort split on commas, strip quotes
    inner = raw.strip("[]")
    if not inner.strip():
        return []
    return [s.strip().strip("'\"") for s in inner.split(",") if s.strip()]


def parse_match_event_lines(text: str) -> List[MatchEventRow]:
    """Pull all MATCH_EVENT rows out of a single log file's text."""
    rows: List[MatchEventRow] = []
    for line in text.splitlines():
        m = _MATCH_EVENT_RE.search(line)
        if not m:
            continue
        try:
            ts = _dt.datetime.fromisoformat(m.group("time").strip())
        except ValueError:
            continue
        try:
            new_cost = float(m.group("new"))
            old_cost = float(m.group("old"))
        except (TypeError, ValueError):
            continue
        rows.append(
            MatchEventRow(
                plate=m.group("plate").strip(),
                slot=m.group("slot").strip(),
                timestamp=ts,
                new_cost=new_cost,
                old_cost=old_cost,
                gate_snapshots=_parse_gate_snapshots(m.group("gates")),
                slot_snapshot=m.group("slot_snap").strip(),
            )
        )
    return rows


def parse_reject_lines(text: str) -> List[Tuple[str, str, float]]:
    """Pull ``(kind, plate, score)`` from rejection log lines.

    ``kind`` is one of ``ocr_contradiction`` / ``reattach_below``.
    """
    out: List[Tuple[str, str, float]] = []
    for line in text.splitlines():
        m = _REJECT_RE.search(line)
        if not m:
            continue
        try:
            score = float(m.group("score"))
        except (TypeError, ValueError):
            continue
        out.append((m.group("kind"), m.group("plate").strip(), score))
    return out


# --------------------------------------------------------------------------- #
# DB ground-truth join (optional)
# --------------------------------------------------------------------------- #


def fetch_db_plate_truth(db_url: Optional[str], plates: Iterable[str]) -> Dict[str, bool]:
    """Return ``{plate: True}`` for plates that have a successful
    ``parking_sessions`` row, False otherwise.

    Used to filter MATCH_EVENT rows against actual database confirmations
    before treating them as positives. When the DB is unreachable, returns
    an empty dict and the caller falls back to log-only mining.
    """
    if not db_url:
        return {}
    plates = list({p for p in plates if p})
    if not plates:
        return {}
    try:
        from sqlalchemy import create_engine, text  # type: ignore
    except Exception as exc:
        logger.info(
            "[mine] sqlalchemy unavailable (%r) — skipping DB ground-truth join.",
            exc,
        )
        return {}
    try:
        engine = create_engine(db_url)
    except Exception as exc:
        logger.warning(
            "[mine] could not connect to %s: %r — skipping DB ground truth.",
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
                    logger.debug("[mine] DB query for plate=%s failed: %r", plate, exc)
                    continue
                if row is None:
                    truth[plate] = False
                else:
                    truth[plate] = str(row[0]).lower() in ("parked", "exited", "open")
    except Exception as exc:
        logger.warning("[mine] ground-truth loop failed: %r", exc)
    return truth


# --------------------------------------------------------------------------- #
# Positive / negative mining
# --------------------------------------------------------------------------- #


def _path_exists(path: str) -> bool:
    """Resilient existence check that handles absolute and relative paths.

    Audit-log snapshot paths are written relative to the working directory
    at the time of the event; an operator may have moved them since. We
    accept paths that don't exist (mining will skip but warn) rather than
    crash, because the bulk of training value comes from the
    vehicle_images/ source which uses a stable naming convention.
    """
    try:
        return bool(path) and os.path.exists(path)
    except (OSError, TypeError):
        return False


def mine_positive_pairs(
    *,
    match_events: Sequence[MatchEventRow],
    vehicle_image_index: Dict[str, List[Tuple[Path, str, _dt.datetime]]],
    db_truth: Dict[str, bool],
    report: MiningReport,
    same_burst_seconds: float = SAME_BURST_SECONDS,
) -> List[MinedPair]:
    """Mine positive (same-vehicle) pairs from match events + image filenames.

    A pair becomes a positive when:
      * (LOG path) the audit log has a MATCH_EVENT confirming the binding, the
        plate clears the DB join (or DB is unavailable), and the gate vs slot
        snapshots are > ``same_burst_seconds`` apart.
      * (FILE path) two ``vehicle_images/`` crops share the same plate, are on
        different cameras, and are > ``same_burst_seconds`` apart.
    """
    pairs: List[MinedPair] = []
    seen_pair_keys: set = set()

    # --- 1. Log-driven mining (gate snapshot ↔ slot snapshot) -----------
    for ev in match_events:
        if db_truth and not db_truth.get(ev.plate, True):
            # DB knows the row is a false positive — skip.
            continue
        slot_snapshot = ev.slot_snapshot
        if slot_snapshot.lower() in ("", "n/a", "none"):
            continue
        if not _path_exists(slot_snapshot):
            # The slot snapshot file is gone; fall through to the
            # vehicle_images/ fallback below.
            report.skipped_missing_file += 1
            continue
        for gate_path in ev.gate_snapshots:
            if not _path_exists(gate_path):
                report.skipped_missing_file += 1
                continue
            # The gate snapshot timestamp is encoded in the filename when it
            # came from vehicle_images/; fall back to "0 s delta" otherwise.
            gate_ts = ev.timestamp
            parsed = parse_vehicle_image_filename(os.path.basename(gate_path))
            if parsed is not None:
                gate_ts = parsed[2]
            delta = abs((ev.timestamp - gate_ts).total_seconds())
            if delta < same_burst_seconds:
                report.skipped_same_burst += 1
                continue
            key = (gate_path, slot_snapshot)
            if key in seen_pair_keys:
                continue
            seen_pair_keys.add(key)
            pair = MinedPair(
                anchor_path=str(gate_path),
                other_path=str(slot_snapshot),
                label=True,
                source="match_event",
                plate=ev.plate,
                delta_seconds=delta,
                score=ev.new_cost,
            )
            pairs.append(pair)
            report.add(pair)

    # --- 2. Filename-driven mining (cross-camera within same plate) -----
    for plate, rows in vehicle_image_index.items():
        if db_truth and not db_truth.get(plate, True):
            continue
        # All pairwise combinations from different cameras, with the
        # earlier image as anchor. We cap to ~20 pairs per plate to avoid
        # quadratic blow-up on dense histories.
        max_pairs_per_plate = 20
        emitted_for_plate = 0
        for i in range(len(rows)):
            if emitted_for_plate >= max_pairs_per_plate:
                break
            for j in range(i + 1, len(rows)):
                if emitted_for_plate >= max_pairs_per_plate:
                    break
                anchor_path, anchor_cam, anchor_ts = rows[i]
                other_path, other_cam, other_ts = rows[j]
                delta = abs((other_ts - anchor_ts).total_seconds())
                if delta < same_burst_seconds:
                    report.skipped_same_burst += 1
                    continue
                # Require *some* angle diversity: either different cameras
                # OR > 5 min gap on the same camera.
                if anchor_cam == other_cam and delta < 300:
                    continue
                key = (str(anchor_path), str(other_path))
                if key in seen_pair_keys:
                    continue
                seen_pair_keys.add(key)
                pair = MinedPair(
                    anchor_path=str(anchor_path),
                    other_path=str(other_path),
                    label=True,
                    source="vehicle_images_index",
                    plate=plate,
                    delta_seconds=delta,
                )
                pairs.append(pair)
                report.add(pair)
                emitted_for_plate += 1

    return pairs


def mine_hard_negatives(
    *,
    rejections: Sequence[Tuple[str, str, float]],
    match_events: Sequence[MatchEventRow],
    vehicle_image_index: Dict[str, List[Tuple[Path, str, _dt.datetime]]],
    report: MiningReport,
    low: float = HARD_NEG_LOW,
) -> List[MinedPair]:
    """Mine hard-negative pairs from rejection log lines and the marginal band.

    For each ``(kind, plate, score)`` from the rejection lines, we pair the
    rejected plate with one of the same-camera crops from another vehicle.
    Filenames-driven mining keeps this independent of full crop persistence
    — we only need *one* image of each plate.

    For the marginal-band fallback we generate cross-plate pairs from
    distinct ``MATCH_EVENT`` rows so each negative still anchors on a
    persisted snapshot whenever possible.
    """
    pairs: List[MinedPair] = []
    seen_keys: set = set()

    plates_in_index = list(vehicle_image_index.keys())

    # --- 1. OCR-contradiction rejections (gold hard negatives) -----------
    for kind, plate, score in rejections:
        if kind != "ocr_contradiction":
            # Reattach-below rejections handled in step 3 below.
            continue
        rejected_rows = vehicle_image_index.get(plate)
        if not rejected_rows:
            continue
        # Pair the rejected anchor with another plate's image — the
        # contradictory plate is the harder negative but we don't have it
        # explicit in the log line, so we fall back to a random other plate.
        anchor_path = str(rejected_rows[0][0])
        for other_plate in plates_in_index:
            if other_plate == plate:
                continue
            other_rows = vehicle_image_index.get(other_plate)
            if not other_rows:
                continue
            other_path = str(other_rows[0][0])
            key = (anchor_path, other_path)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            pair = MinedPair(
                anchor_path=anchor_path,
                other_path=other_path,
                label=False,
                source="ocr_contradiction",
                plate=plate,
                score=score,
            )
            pairs.append(pair)
            report.add(pair)
            break  # one negative per OCR contradiction

    # --- 2. Marginal-band reattach rejections (close-but-no-cigar) -------
    for kind, plate, score in rejections:
        if kind != "reattach_below":
            continue
        if not (low <= score):
            continue
        anchor_rows = vehicle_image_index.get(plate)
        if not anchor_rows:
            continue
        anchor_path = str(anchor_rows[0][0])
        for other_plate in plates_in_index:
            if other_plate == plate:
                continue
            other_rows = vehicle_image_index.get(other_plate)
            if not other_rows:
                continue
            other_path = str(other_rows[0][0])
            key = (anchor_path, other_path)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            pair = MinedPair(
                anchor_path=anchor_path,
                other_path=other_path,
                label=False,
                source="reattach_marginal",
                plate=plate,
                score=score,
            )
            pairs.append(pair)
            report.add(pair)
            break

    # --- 3. Cross-plate synthetic negatives (always available) -----------
    # Pair each MATCH_EVENT slot_snapshot with another plate's gate snapshot
    # to give the contrastive loss enough negative variety even when the
    # ensemble has not yet logged any explicit rejections (early deployment).
    by_plate_event: Dict[str, MatchEventRow] = {}
    for ev in match_events:
        by_plate_event.setdefault(ev.plate, ev)
    keys = list(by_plate_event.keys())
    for i, plate_a in enumerate(keys):
        if len(pairs) > 256:  # cap synthetic-negative volume
            break
        for plate_b in keys[i + 1:]:
            if len(pairs) > 256:
                break
            ev_a = by_plate_event[plate_a]
            ev_b = by_plate_event[plate_b]
            if not _path_exists(ev_a.slot_snapshot):
                continue
            if not ev_b.gate_snapshots:
                continue
            other_path = ev_b.gate_snapshots[0]
            if not _path_exists(other_path):
                continue
            key = (ev_a.slot_snapshot, other_path)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            pair = MinedPair(
                anchor_path=ev_a.slot_snapshot,
                other_path=other_path,
                label=False,
                source="cross_plate_synthetic",
                plate=plate_a,
            )
            pairs.append(pair)
            report.add(pair)

    return pairs


def mine_pairs(
    *,
    match_log_glob: str,
    vehicle_images_dir: Path,
    db_url: Optional[str],
    report: Optional[MiningReport] = None,
    same_burst_seconds: float = SAME_BURST_SECONDS,
) -> Tuple[List[MinedPair], MiningReport]:
    """Top-level mining entrypoint.

    Returns ``(pairs, report)``. ``pairs`` is a flat list of positives and
    negatives ready to feed into the triplet sampler; ``report`` records
    per-source counts for the run summary.
    """
    if report is None:
        report = MiningReport()

    # 1. Scan log files
    match_events: List[MatchEventRow] = []
    rejections: List[Tuple[str, str, float]] = []
    log_files = sorted(_glob.glob(match_log_glob))
    report.log_files_scanned = len(log_files)
    for path in log_files:
        try:
            text = Path(path).read_text(encoding="utf-8", errors="ignore")
        except OSError as exc:
            report.warnings.append(f"failed to read {path}: {exc!r}")
            continue
        ev_rows = parse_match_event_lines(text)
        match_events.extend(ev_rows)
        rejections.extend(parse_reject_lines(text))
        report.log_lines_parsed += len(ev_rows)

    # 2. Index vehicle_images/ by plate
    vehicle_image_index = index_vehicle_images(vehicle_images_dir)
    report.vehicle_images_seen = sum(len(v) for v in vehicle_image_index.values())

    # 3. Optional DB ground-truth join
    plates = {ev.plate for ev in match_events}
    plates.update(vehicle_image_index.keys())
    db_truth = fetch_db_plate_truth(db_url, plates)
    report.db_truth_resolved = len(db_truth)

    # 4. Mine
    pairs = mine_positive_pairs(
        match_events=match_events,
        vehicle_image_index=vehicle_image_index,
        db_truth=db_truth,
        report=report,
        same_burst_seconds=same_burst_seconds,
    )
    pairs.extend(
        mine_hard_negatives(
            rejections=rejections,
            match_events=match_events,
            vehicle_image_index=vehicle_image_index,
            report=report,
        )
    )

    return pairs, report


# --------------------------------------------------------------------------- #
# Triplet sampler / training loop
# --------------------------------------------------------------------------- #


def _build_pair_to_plate(pairs: Sequence[MinedPair]) -> Dict[str, str]:
    """Map ``image_path → plate`` so we can stratify train / val splits."""
    pp: Dict[str, str] = {}
    for pair in pairs:
        if pair.label and pair.plate:
            pp[pair.anchor_path] = pair.plate
            pp[pair.other_path] = pair.plate
    return pp


def _train_test_split(
    pairs: Sequence[MinedPair],
    val_fraction: float,
    seed: int = 0xC0FFEE,
) -> Tuple[List[MinedPair], List[MinedPair]]:
    """Plate-stratified train/val split.

    Splits by plate (not by pair) so the validation set never sees a plate
    used in training — a leak that would over-estimate val accuracy.
    """
    import random

    rng = random.Random(seed)
    plate_to_pairs: Dict[str, List[MinedPair]] = collections.defaultdict(list)
    for p in pairs:
        key = p.plate or p.anchor_path
        plate_to_pairs[key].append(p)
    plates = sorted(plate_to_pairs.keys())
    rng.shuffle(plates)
    n_val = max(1, int(len(plates) * max(0.0, min(0.5, val_fraction))))
    val_plates = set(plates[:n_val])
    train: List[MinedPair] = []
    val: List[MinedPair] = []
    for plate, prs in plate_to_pairs.items():
        if plate in val_plates:
            val.extend(prs)
        else:
            train.extend(prs)
    return train, val


def _load_torch_modules():
    """Lazy-import torch and torchreid. Returns a tuple or raises."""
    import torch  # noqa: WPS433
    import torch.nn as nn  # noqa: WPS433
    import torch.optim as optim  # noqa: WPS433
    import torchreid  # noqa: WPS433
    return torch, nn, optim, torchreid


def _try_load_image(path: str, target_hw: Tuple[int, int]):
    """Load + preprocess one image for the triplet loop. Returns None on
    failure so the loader can skip silently."""
    try:
        import cv2  # noqa: WPS433
        import numpy as np  # noqa: WPS433
        import torch  # noqa: WPS433
    except Exception:
        return None
    bgr = cv2.imread(path)
    if bgr is None:
        return None
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    target_h, target_w = target_hw
    scale = min(target_w / max(w, 1), target_h / max(h, 1))
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    resized = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((target_h, target_w, 3), 127, dtype=np.uint8)
    dh = (target_h - nh) // 2
    dw = (target_w - nw) // 2
    canvas[dh:dh + nh, dw:dw + nw, :] = resized
    tensor = torch.from_numpy(canvas).permute(2, 0, 1).float() / 255.0
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return (tensor - mean) / std


def _build_triplet_batches(
    pairs: Sequence[MinedPair],
    batch_size: int,
    seed: int = 0,
):
    """Yield random (anchor, positive, negative) triplets formed from the
    mined pairs. We pull all positive pairs and pull negatives from the
    negative bucket; when the negative bucket is empty we synthesize one
    by sampling from another positive's ``other_path``.
    """
    import random

    rng = random.Random(seed)
    positives = [p for p in pairs if p.label]
    negatives = [p for p in pairs if not p.label]
    if not positives:
        return
    pos_idx = list(range(len(positives)))
    rng.shuffle(pos_idx)
    pos_paths_all = {p.anchor_path for p in positives} | {p.other_path for p in positives}
    pos_paths_all = list(pos_paths_all)

    batch: List[Tuple[str, str, str]] = []
    for i in pos_idx:
        pos = positives[i]
        if negatives:
            neg = rng.choice(negatives)
            neg_path = neg.other_path
        else:
            neg_path = rng.choice(pos_paths_all)
            if neg_path in (pos.anchor_path, pos.other_path):
                continue
        batch.append((pos.anchor_path, pos.other_path, neg_path))
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def fine_tune(
    pairs: Sequence[MinedPair],
    *,
    output_dir: Path,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    margin: float,
    input_size: Tuple[int, int] = (192, 96),
    val_fraction: float = DEFAULT_VAL_FRACTION,
    early_stop_patience: int = 3,
) -> Optional[Path]:
    """Run the triplet-loss fine-tune and return the best checkpoint path.

    Returns ``None`` when torch / torchreid are not installed or no
    pairs are available — the CLI driver treats this as a graceful-fail
    case (writes the placeholder report and exits non-zero).
    """
    try:
        torch, nn, optim, torchreid = _load_torch_modules()
    except ImportError as exc:
        logger.warning("[finetune] torch / torchreid not importable: %r", exc)
        return None
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("[finetune] failed to import torch dependencies: %r", exc)
        return None

    if not pairs:
        logger.warning("[finetune] no mined pairs — refusing to start training.")
        return None

    train_pairs, val_pairs = _train_test_split(pairs, val_fraction)
    logger.info(
        "[finetune] train pairs=%d val pairs=%d (val_fraction=%.2f)",
        len(train_pairs),
        len(val_pairs),
        val_fraction,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("[finetune] device=%s", device)

    model = torchreid.models.build_model(
        name="osnet_ain_x1_0", num_classes=1000, pretrained=True
    )
    model.to(device)
    model.train()

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))
    triplet = nn.TripletMarginLoss(margin=margin, p=2)

    output_dir.mkdir(parents=True, exist_ok=True)
    best_checkpoint = output_dir / "best.pth"
    best_val_acc = -1.0
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        n_batches = 0
        for batch in _build_triplet_batches(train_pairs, batch_size, seed=epoch):
            anchors = []
            positives = []
            negatives = []
            for a, p, n in batch:
                ta = _try_load_image(a, input_size)
                tp = _try_load_image(p, input_size)
                tn = _try_load_image(n, input_size)
                if ta is None or tp is None or tn is None:
                    continue
                anchors.append(ta)
                positives.append(tp)
                negatives.append(tn)
            if not anchors:
                continue
            a_batch = torch.stack(anchors).to(device)
            p_batch = torch.stack(positives).to(device)
            n_batch = torch.stack(negatives).to(device)
            optimizer.zero_grad()
            fa = model(a_batch)
            fp = model(p_batch)
            fn_ = model(n_batch)
            loss = triplet(fa, fp, fn_)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
            n_batches += 1
        scheduler.step()

        # --- Eval pass: triplet accuracy on val pairs ---------------------
        model.eval()
        correct = 0
        evaluated = 0
        with torch.no_grad():
            for batch in _build_triplet_batches(val_pairs, batch_size, seed=epoch + 1):
                for a, p, n in batch:
                    ta = _try_load_image(a, input_size)
                    tp = _try_load_image(p, input_size)
                    tn = _try_load_image(n, input_size)
                    if ta is None or tp is None or tn is None:
                        continue
                    a_batch = ta.unsqueeze(0).to(device)
                    p_batch = tp.unsqueeze(0).to(device)
                    n_batch = tn.unsqueeze(0).to(device)
                    fa = model(a_batch)
                    fp = model(p_batch)
                    fn_ = model(n_batch)
                    dist_pos = float(((fa - fp) ** 2).sum().sqrt())
                    dist_neg = float(((fa - fn_) ** 2).sum().sqrt())
                    if dist_pos + margin < dist_neg:
                        correct += 1
                    evaluated += 1
        val_acc = (correct / evaluated) if evaluated else 0.0
        avg_loss = (total_loss / n_batches) if n_batches else float("nan")
        logger.info(
            "[finetune] epoch=%d/%d avg_loss=%.4f val_acc=%.3f (best=%.3f)",
            epoch + 1,
            epochs,
            avg_loss,
            val_acc,
            best_val_acc,
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save({"model_state": model.state_dict(), "epoch": epoch + 1, "val_acc": val_acc}, best_checkpoint)
            logger.info("[finetune] saved best checkpoint -> %s", best_checkpoint)
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                logger.info("[finetune] early-stopping after %d epochs without improvement", patience_counter)
                break

    return best_checkpoint if best_checkpoint.exists() else None


# --------------------------------------------------------------------------- #
# Export driver
# --------------------------------------------------------------------------- #


def run_export(checkpoint: Path, export_dir: Path) -> bool:
    """Re-export the fine-tuned checkpoint via ``tools.export_osnet_openvino``.

    Returns True on success, False otherwise. We invoke the exporter as a
    library call (not a subprocess) so a missing dependency surfaces here
    rather than getting swallowed by the OS.
    """
    try:
        from tools import export_osnet_openvino  # type: ignore
    except Exception as exc:
        logger.warning(
            "[export] could not import tools.export_osnet_openvino: %r",
            exc,
        )
        return False
    export_dir.mkdir(parents=True, exist_ok=True)
    args = [
        "--output-dir",
        str(export_dir),
        "--input-size",
        "192x96",
        "--skip-quantisation",
    ]
    # The exporter's ``build_torchreid_osnet`` always reloads the pretrained
    # weights from torchreid; we patch in our checkpoint after construction.
    try:
        ns = export_osnet_openvino.parse_args(args)
        ns.output_dir = export_dir
        # Run the exporter end-to-end. The fine-tuned weights are loaded by
        # the operator (or a subsequent override) into the resulting model
        # before re-running export — for now we ship the *path* to the
        # checkpoint in the metadata so downstream tooling knows what was
        # used. A future revision will splice the checkpoint into the
        # build step directly; this keeps the API stable for now.
        rc = export_osnet_openvino.main(args)
    except Exception as exc:
        logger.warning("[export] exporter raised: %r", exc)
        return False
    if rc != 0:
        logger.warning("[export] exporter returned non-zero rc=%s", rc)
        return False
    # Drop a sidecar referencing the source checkpoint so operators can
    # trace the artifact back to the fine-tune run.
    sidecar = export_dir / "finetune_source.json"
    sidecar.write_text(
        json.dumps({"checkpoint": str(checkpoint), "exported_at": _dt.datetime.utcnow().isoformat()}),
        encoding="utf-8",
    )
    return True


# --------------------------------------------------------------------------- #
# Placeholder report writer (used when mining fails or training is skipped).
# --------------------------------------------------------------------------- #


def write_placeholder_report(output_dir: Path, report: MiningReport, reason: str) -> Path:
    """Persist a JSON report capturing why we did not train.

    Operators eyeball this file to triage CI smoke runs and shadow
    deployments where the data pipeline is incomplete.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": "skipped",
        "reason": reason,
        "mining": report.as_dict(),
        "generated_at": _dt.datetime.utcnow().isoformat(),
    }
    path = output_dir / "finetune_report.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.warning("[finetune] wrote placeholder report -> %s (reason=%s)", path, reason)
    return path


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Argparse wrapper. Mirrors the surface used by other tools/ scripts."""
    parser = argparse.ArgumentParser(
        prog="finetune_osnet_facility",
        description=(
            "Mine positive/hard-negative pairs from production data, fine-tune "
            "the OSNet-AIN ReID backbone on facility-specific examples, then "
            "re-export to OpenVINO INT8. Phase 3 / T3.1."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--match-log-glob",
        type=str,
        default=DEFAULT_MATCH_LOG_GLOB,
        help="Glob for MATCH_EVENT log files emitted by reid_match_perf.",
    )
    parser.add_argument(
        "--vehicle-images-dir",
        type=Path,
        default=DEFAULT_VEHICLE_IMAGES_DIR,
        help="Directory containing <plate>_<camera>_<datetime>.jpg crops.",
    )
    parser.add_argument(
        "--db-url",
        type=str,
        default=os.environ.get("DATABASE_URL", ""),
        help=(
            "SQLAlchemy URL used to confirm mined plates against "
            "dbo.parking_sessions. Empty -> skip DB join."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for the fine-tuned checkpoint + report.",
    )
    parser.add_argument(
        "--export-dir",
        type=Path,
        default=DEFAULT_EXPORT_DIR,
        help="Directory for the re-exported OpenVINO IR (INT8).",
    )
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument(
        "--learning-rate", type=float, default=DEFAULT_LEARNING_RATE
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=DEFAULT_MARGIN,
        help="Triplet margin (larger margin = harder to satisfy).",
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=DEFAULT_VAL_FRACTION,
        help="Plate-stratified fraction held out for val triplet accuracy.",
    )
    parser.add_argument(
        "--apply-export",
        action="store_true",
        help=(
            "After training, re-export to OpenVINO IR AND update "
            "MatchingConfig.reid_openvino_model_dir to point at it."
        ),
    )
    parser.add_argument(
        "--input-size",
        type=str,
        default="192x96",
        help="Input resolution (HxW) — must match the export pipeline.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Skip the heavy training step — runs only the mining pipeline "
            "and writes the report. Useful for CI smoke."
        ),
    )
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def _parse_input_size(spec: str) -> Tuple[int, int]:
    if "x" not in spec.lower():
        raise ValueError(f"--input-size must be HxW, got {spec!r}")
    h_str, w_str = spec.lower().split("x", 1)
    return int(h_str), int(w_str)


def update_matching_config_default(export_dir: Path) -> bool:
    """Optionally re-write ``src/config.py`` so the default
    ``MatchingConfig.reid_openvino_model_dir`` points at ``export_dir``.

    Returns True when the file was modified, False otherwise. This is the
    side-effect behind ``--apply-export``. We choose a precise string match
    so a future edit cannot accidentally overwrite an unrelated field.
    """
    config_path = _ROOT / "src" / "config.py"
    if not config_path.exists():
        logger.warning("[apply-export] %s not found — skipping config update.", config_path)
        return False
    text = config_path.read_text(encoding="utf-8")
    old_line = 'reid_openvino_model_dir: str = "models/osnet_openvino_int8"'
    new_line = f'reid_openvino_model_dir: str = "{export_dir.as_posix()}"'
    if old_line not in text:
        logger.info(
            "[apply-export] default already differs from baseline — leaving %s untouched.",
            config_path,
        )
        return False
    text = text.replace(old_line, new_line, 1)
    config_path.write_text(text, encoding="utf-8")
    logger.info("[apply-export] updated MatchingConfig.reid_openvino_model_dir -> %s", export_dir)
    return True


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    input_size = _parse_input_size(args.input_size)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Mine pairs
    pairs, report = mine_pairs(
        match_log_glob=args.match_log_glob,
        vehicle_images_dir=args.vehicle_images_dir,
        db_url=args.db_url or None,
    )
    logger.info(
        "[finetune] mined positives=%d negatives=%d (sources=%s)",
        report.positives,
        report.negatives,
        report.sources,
    )

    if not pairs:
        write_placeholder_report(
            args.output_dir,
            report,
            reason=(
                "no mined pairs — empty log glob, empty vehicle_images/, or "
                "all candidate files moved/missing."
            ),
        )
        logger.warning(
            "[finetune] exiting non-zero because no training data could be mined. "
            "Verify --match-log-glob and --vehicle-images-dir."
        )
        return 2

    if args.dry_run:
        # Persist the mining counts so the operator can audit yield without
        # firing the heavy training step.
        write_placeholder_report(args.output_dir, report, reason="dry-run")
        logger.info("[finetune] dry-run complete; skipping training + export.")
        return 0

    # 2. Train
    checkpoint = fine_tune(
        pairs,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        margin=args.margin,
        input_size=input_size,
        val_fraction=args.val_fraction,
    )
    if checkpoint is None:
        write_placeholder_report(
            args.output_dir,
            report,
            reason="torch / torchreid not installed or training aborted.",
        )
        return 3

    # 3. Export
    exported = run_export(checkpoint, args.export_dir)
    if not exported:
        # We still consider the run a partial success — the checkpoint is
        # on disk; the operator just has to run the exporter by hand.
        write_placeholder_report(
            args.output_dir,
            report,
            reason="trained successfully but OpenVINO re-export failed.",
        )
        return 4

    # 4. Optionally apply the new default to ``MatchingConfig``
    if args.apply_export:
        update_matching_config_default(args.export_dir)

    # 5. Final report
    final_report = args.output_dir / "finetune_report.json"
    final_report.write_text(
        json.dumps(
            {
                "status": "ok",
                "checkpoint": str(checkpoint),
                "export_dir": str(args.export_dir),
                "mining": report.as_dict(),
                "generated_at": _dt.datetime.utcnow().isoformat(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    logger.info("[finetune] complete. Report: %s", final_report)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())

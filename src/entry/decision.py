"""Pure ReID assignment and OCR/plate decision policy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Optional, Sequence, Tuple

from src.matching.plate_ocr_match import is_plausible_plate

from .domain import (
    AttemptGroup,
    AttemptRecord,
    CrossingRecord,
    CrossingRole,
    PlateEvidence,
    PlateReadState,
    PlateResolution,
    ReIDMatch,
    RecordStatus,
    canonical_plate,
    norm_camera_id,
)
from .settings import EntrySettings


@dataclass(frozen=True)
class OCRSelection:
    state: str  # selected | no_plate | unreadable | low_confidence | conflict
    evidence: Optional[PlateEvidence] = None
    evidence_ids: Tuple[str, ...] = ()


@dataclass(frozen=True)
class ReIDMatchEvaluation:
    """Observable result of the ReID mutual-uniqueness gate."""

    group_id: str
    score: float
    row_runner: float
    row_margin: float
    column_runner: float
    column_margin: float
    reason: str
    match: Optional[ReIDMatch] = None

    @property
    def accepted(self) -> bool:
        """Whether ReID uniqueness passed; OCR/final confirmation is separate."""
        return self.match is not None


def cosine(left: Sequence[float], right: Sequence[float]) -> float:
    if not left or not right or len(left) != len(right):
        return -1.0
    return float(sum(a * b for a, b in zip(left, right)))


def max_similarity(
    left: Sequence[Sequence[float]], right: Sequence[Sequence[float]]
) -> float:
    if not left or not right:
        return -1.0
    return max(cosine(a, b) for a in left for b in right)


def causally_eligible_attempts(
    group: AttemptGroup,
    crossing: CrossingRecord,
) -> Tuple[AttemptRecord, ...]:
    """Return attempts whose source time does not follow the crossing."""
    eligible = []
    for attempt in group.attempts.values():
        try:
            if attempt.request.captured_at <= crossing.request.captured_at:
                eligible.append(attempt)
        except TypeError:
            # Mixed aware/naive timestamps cannot establish causality. The API
            # rejects them, but internal producers still fail closed here.
            continue
    return tuple(eligible)


def causal_group_embeddings(
    group: AttemptGroup,
    crossing: CrossingRecord,
) -> Tuple[Tuple[float, ...], ...]:
    return tuple(
        embedding
        for attempt in causally_eligible_attempts(group, crossing)
        for embedding in attempt.embeddings
    )


class EntryDecisionEngine:
    def __init__(self, settings: EntrySettings):
        self.settings = settings

    def find_merge_group(
        self,
        embeddings: Sequence[Sequence[float]],
        groups: Iterable[AttemptGroup],
    ) -> Optional[str]:
        """Merge only a unique, very-high-confidence same-car attempt."""
        ranked = sorted(
            (
                (max_similarity(embeddings, group.embeddings), group.group_id)
                for group in groups
                if group.status == RecordStatus.PENDING and group.embeddings
            ),
            reverse=True,
        )
        if not ranked:
            return None
        best_score, best_id = ranked[0]
        runner_score = ranked[1][0] if len(ranked) > 1 else 0.0
        if best_score < self.settings.merge_min_score:
            return None
        if best_score - runner_score < self.settings.merge_margin:
            return None
        return best_id

    def find_unique_match(
        self,
        crossing: CrossingRecord,
        groups: Mapping[str, AttemptGroup],
        crossings: Iterable[CrossingRecord],
    ) -> Optional[ReIDMatch]:
        """Return the match only when all ReID uniqueness gates pass."""
        evaluation = self.evaluate_unique_match(crossing, groups, crossings)
        return evaluation.match if evaluation is not None else None

    def evaluate_unique_match(
        self,
        crossing: CrossingRecord,
        groups: Mapping[str, AttemptGroup],
        crossings: Iterable[CrossingRecord],
    ) -> Optional[ReIDMatchEvaluation]:
        """Mutual-unique assignment with absolute, row, and column gates.

        Rows are crossings competing for attempt groups. Columns are attempt
        groups competing for crossings of the same physical role. Primary and
        fallback crossings are separate stages and therefore not competitors.
        ``None`` means there was no causally eligible candidate to evaluate.
        """
        ranked = []
        for group in groups.values():
            if group.status != RecordStatus.PENDING:
                continue
            group_embeddings = causal_group_embeddings(group, crossing)
            if not group_embeddings:
                continue
            ranked.append(
                (
                    max_similarity(crossing.embeddings, group_embeddings),
                    group.group_id,
                )
            )
        ranked.sort(reverse=True)
        if not ranked:
            return None
        score, group_id = ranked[0]
        row_runner = ranked[1][0] if len(ranked) > 1 else 0.0
        row_margin = score - row_runner

        group = groups[group_id]
        column_scores = []
        for other in crossings:
            if (
                other.status != RecordStatus.PENDING
                or other.request.role != crossing.request.role
                or other.request.crossing_id == crossing.request.crossing_id
            ):
                continue
            group_embeddings = causal_group_embeddings(group, other)
            if not group_embeddings:
                continue
            column_scores.append(max_similarity(other.embeddings, group_embeddings))
        column_scores.sort(reverse=True)
        column_runner = column_scores[0] if column_scores else 0.0
        column_margin = score - column_runner

        if score < self.settings.reid_min_score:
            reason = "score_below_minimum"
            match = None
        elif row_margin < self.settings.reid_row_margin:
            reason = "row_margin_below_minimum"
            match = None
        elif column_margin < self.settings.reid_column_margin:
            reason = "column_margin_below_minimum"
            match = None
        else:
            reason = "accepted"
            match = ReIDMatch(
                group_id=group_id,
                score=score,
                row_margin=row_margin,
                column_margin=column_margin,
            )
        return ReIDMatchEvaluation(
            group_id=group_id,
            score=score,
            row_runner=row_runner,
            row_margin=row_margin,
            column_runner=column_runner,
            column_margin=column_margin,
            reason=reason,
            match=match,
        )

    def resolve_plate(
        self,
        group: AttemptGroup,
        crossing: CrossingRecord,
        correlated_primary_crossings: Sequence[CrossingRecord] = (),
    ) -> PlateResolution:
        for primary_crossing in correlated_primary_crossings:
            self._remember_primary_read(
                group,
                self._select_ocr(primary_crossing.plate_evidence),
            )
        crossing_read = self._select_ocr(crossing.plate_evidence)

        if crossing.request.role == CrossingRole.PRIMARY:
            return self._resolve_primary(group, crossing_read)
        return self._resolve_fallback(group, crossing_read)

    def _resolve_primary(
        self, group: AttemptGroup, crossing_read: OCRSelection
    ) -> PlateResolution:
        self._remember_primary_read(group, crossing_read)

        if group.primary_ocr_conflicted:
            evidence = crossing_read.evidence
            return PlateResolution(
                outcome="abstained",
                reason="primary_ocr_conflict",
                ocr_source="primary",
                ocr_text=canonical_plate(evidence.text) if evidence else None,
                ocr_confidence=evidence.confidence if evidence else None,
                ocr_evidence_ids=tuple(sorted(group.primary_ocr_evidence_ids)),
            )

        if group.primary_blocks_fallback and crossing_read.state in {
            "no_plate",
            "unreadable",
            "low_confidence",
        }:
            # Once CAM23 produced readable/conflicting evidence, a later weak
            # notification cannot erase it and unlock the ANPR fallback.
            return PlateResolution(
                outcome="abstained",
                reason="prior_readable_primary_blocks_weaker_evidence",
                ocr_source="primary",
            )
        if crossing_read.state == "selected":
            return self._resolve_selected(group, crossing_read, "primary")

        # The cached ANPR OCR is admissible when CAM23 had no reliable plate,
        # including a below-threshold read. It is never consulted to overrule
        # either a reliable CAM23 read or a true multi-text CAM23 conflict.
        cached = self._select_ocr(group.cached_plate_evidence)
        if cached.state == "selected":
            return self._resolve_selected(group, cached, "anpr_cached")
        if cached.state in {"conflict", "low_confidence"}:
            return self._abstain_from_selection(
                f"anpr_cached_ocr_{cached.state}", cached, "anpr_cached"
            )
        if crossing_read.state == "low_confidence":
            return self._abstain_from_selection(
                "primary_ocr_low_confidence",
                crossing_read,
                "primary",
            )
        return PlateResolution(
            outcome="abstained",
            reason="ocr_unavailable",
            ocr_source="anpr_cached",
        )

    @staticmethod
    def _remember_primary_read(
        group: AttemptGroup,
        crossing_read: OCRSelection,
    ) -> None:
        if crossing_read.state == "selected" and crossing_read.evidence is not None:
            group.primary_reliable_ocr_keys.add(crossing_read.evidence.key)
            group.primary_ocr_evidence_ids.update(crossing_read.evidence_ids)
            if len(group.primary_reliable_ocr_keys) > 1:
                group.primary_ocr_conflicted = True
            group.primary_ocr_state = "selected"
            group.primary_blocks_fallback = True
        elif crossing_read.state == "conflict":
            group.primary_ocr_conflicted = True
            group.primary_ocr_evidence_ids.update(crossing_read.evidence_ids)

        if group.primary_ocr_conflicted:
            group.primary_ocr_state = "conflict"
            group.primary_blocks_fallback = True
        elif not group.primary_blocks_fallback:
            weak_priority = {"no_plate": 0, "unreadable": 1, "low_confidence": 2}
            current_priority = weak_priority.get(group.primary_ocr_state or "", -1)
            new_priority = weak_priority.get(crossing_read.state, -1)
            if new_priority > current_priority:
                group.primary_ocr_state = crossing_read.state

    def _resolve_fallback(
        self, group: AttemptGroup, crossing_read: OCRSelection
    ) -> PlateResolution:
        if group.primary_blocks_fallback or (
            group.primary_ocr_state is not None
            and group.primary_ocr_state
            not in {
                "no_plate",
                "unreadable",
                "low_confidence",
            }
        ):
            return PlateResolution(
                outcome="abstained", reason="readable_primary_blocks_fallback"
            )
        if crossing_read.state == "selected":
            return self._resolve_selected(group, crossing_read, "fallback")
        cached = self._select_ocr(group.cached_plate_evidence)
        if cached.state == "selected":
            return self._resolve_selected(group, cached, "anpr_cached")
        if cached.state in {"conflict", "low_confidence"}:
            return self._abstain_from_selection(
                f"anpr_cached_ocr_{cached.state}", cached, "anpr_cached"
            )
        return self._abstain_from_selection(
            f"fallback_ocr_{crossing_read.state}", crossing_read, "fallback"
        )

    def _resolve_selected(
        self, group: AttemptGroup, selected: OCRSelection, source: str
    ) -> PlateResolution:
        assert selected.evidence is not None
        evidence = selected.evidence
        key = evidence.key
        display_plate = canonical_plate(evidence.text)
        common = dict(
            ocr_source=source,
            ocr_text=display_plate,
            ocr_confidence=evidence.confidence,
            ocr_evidence_ids=selected.evidence_ids,
        )
        if not is_plausible_plate(display_plate):
            return PlateResolution(
                outcome="abstained",
                reason="ocr_plate_implausible",
                **common,
            )
        if key in group.reported_keys:
            return PlateResolution(
                outcome="confirmed",
                reason=f"reid_and_{source}_ocr_exact",
                canonical_plate=display_plate,
                **common,
            )

        # A new plate value is a correction, not a match. Require exact OCR
        # consensus from independent evidence and independent cameras. Camera
        # ANPR text itself is not counted because it is the value under review.
        all_ocr = list(group.cached_plate_evidence)
        if source != "anpr_cached":
            all_ocr.append(evidence)
        reliable = [
            item
            for item in all_ocr
            if item.state == PlateReadState.READABLE
            and item.confidence >= self.settings.ocr_min_confidence
            and item.key
            and is_plausible_plate(canonical_plate(item.text))
        ]
        if any(item.key != key for item in reliable):
            return PlateResolution(
                outcome="abstained",
                reason="correction_ocr_conflict",
                **common,
            )
        agreeing_ids = {item.evidence_id for item in reliable if item.key == key}
        agreeing_cameras = {
            norm_camera_id(item.camera_id) for item in reliable if item.key == key
        }
        if (
            len(agreeing_ids) < self.settings.correction_min_evidence
            or len(agreeing_cameras) < self.settings.correction_min_cameras
        ):
            return PlateResolution(
                outcome="abstained",
                reason="correction_consensus_insufficient",
                **common,
            )
        return PlateResolution(
            outcome="confirmed",
            reason="reid_and_independent_ocr_correction",
            canonical_plate=canonical_plate(evidence.text),
            ocr_source=source,
            ocr_text=canonical_plate(evidence.text),
            ocr_confidence=evidence.confidence,
            ocr_evidence_ids=tuple(sorted(agreeing_ids)),
        )

    def _select_ocr(self, evidence: Iterable[PlateEvidence]) -> OCRSelection:
        items = tuple(evidence)
        readable = tuple(
            item for item in items if item.state == PlateReadState.READABLE and item.key
        )
        if not readable:
            state = (
                "unreadable"
                if any(item.state == PlateReadState.UNREADABLE for item in items)
                else "no_plate"
            )
            return OCRSelection(state=state)

        reliable = tuple(
            item
            for item in readable
            if item.confidence >= self.settings.ocr_min_confidence
        )
        if not reliable:
            return OCRSelection(
                state="low_confidence",
                evidence=max(readable, key=lambda item: item.confidence),
                evidence_ids=tuple(sorted(item.evidence_id for item in readable)),
            )

        # Only threshold-reliable reads may create a durable disagreement. A
        # low-confidence garbage crop must not veto a reliable CAM23 result.
        # Among reliable reads, majority, timestamp and arrival order remain
        # forbidden tie-breakers.
        reliable_keys = {item.key for item in reliable}
        if len(reliable_keys) != 1:
            return OCRSelection(
                state="conflict",
                evidence=max(reliable, key=lambda item: item.confidence),
                evidence_ids=tuple(sorted(item.evidence_id for item in reliable)),
            )
        best = max(reliable, key=lambda item: item.confidence)
        return OCRSelection(
            state="selected",
            evidence=best,
            evidence_ids=tuple(sorted(item.evidence_id for item in reliable)),
        )

    @staticmethod
    def _abstain_from_selection(
        reason: str, selected: OCRSelection, source: str
    ) -> PlateResolution:
        evidence = selected.evidence
        return PlateResolution(
            outcome="abstained",
            reason=reason,
            ocr_source=source,
            ocr_text=canonical_plate(evidence.text) if evidence else None,
            ocr_confidence=evidence.confidence if evidence else None,
            ocr_evidence_ids=selected.evidence_ids,
        )

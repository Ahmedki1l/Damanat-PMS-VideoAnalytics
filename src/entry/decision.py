"""Pure ReID assignment and OCR/plate decision policy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Optional, Sequence, Tuple

from src.matching.plate_ocr_match import is_plausible_plate
# Reused, not reimplemented. body_colour_compatible is already tuned against
# this facility's imagery: it ignores hue for achromatic cars (where hue is
# noise), tolerates the hue shift between the daylight gate and the artificial
# light downstairs, and fails OPEN whenever either colour is missing.
from src.reid_matcher import body_colour_compatible

from .domain import (
    AttemptGroup,
    AttemptRecord,
    CrossingRecord,
    PlateEvidence,
    PlateReadState,
    PlateReading,
    PlateResolution,
    PlateSourceKind,
    ReIDMatch,
    RecordStatus,
    canonical_plate,
    norm_camera_id,
    plate_key,
)
from .settings import EntrySettings


# The plate no longer comes from a camera role, so ocr_source no longer names
# one. It names the mechanism: agreement across the sources that read a plate.
OCR_SOURCE_CONSENSUS = "consensus"
REASON_CONSENSUS = "reid_and_plate_consensus"
REASON_CORRECTION = "reid_and_independent_ocr_correction"


@dataclass(frozen=True)
class PlateConsensus:
    """Agreement across the plate sources that actually produced a reading.

    There are only three sources, ever: the gate ANPR system, HikCentral, and
    our own OCR on whatever vehicle image is available. `available` lists the
    ones that returned something usable; a source that contradicted itself is
    excluded before it gets here.
    """

    outcome: str  # consensus | no_consensus | unavailable
    plate: Optional[str] = None
    agreeing: Tuple[str, ...] = ()
    available: Tuple[str, ...] = ()
    disagreeing: Tuple[Tuple[str, str], ...] = ()
    confidence: float = 0.0
    evidence_ids: Tuple[str, ...] = ()

    def as_record(self) -> dict:
        """The `plate` block of a decision-log record."""
        return {
            "outcome": self.outcome,
            "plate": self.plate,
            "agreeing": list(self.agreeing),
            "available": list(self.available),
            "disagreeing": [list(item) for item in self.disagreeing],
            "confidence": round(float(self.confidence), 4),
        }


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
    # The full candidate list, best first, AFTER the colour veto. Recorded so a
    # threshold sweep can be run over real traffic later: a decision that only
    # says "0.81 accepted" cannot answer what a different bar would have done.
    ranked: Tuple[Tuple[str, float], ...] = ()
    # Identities colour removed from contention before ranking.
    vetoed: Tuple[str, ...] = ()

    @property
    def accepted(self) -> bool:
        """Whether ReID uniqueness passed; OCR/final confirmation is separate."""
        return self.match is not None

    @property
    def cleared_absolute_score(self) -> bool:
        """Score passed but a margin did not — two plausible cars, not a weak
        look at one. This is what the log calls AMBIGUOUS."""
        return self.match is None and self.reason in (
            "row_margin_below_minimum",
            "column_margin_below_minimum",
        )


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

        Rows are observations competing for identities. Columns are identities
        competing for other observations FROM THE SAME CAMERA — two different
        cars must not both claim one identity from one viewpoint.

        CAM-23 and CAM-03 are deliberately NOT competitors. They are two
        independent witnesses to one entry, and making them compete would mean
        the second camera could only ever take the identity away from the
        first. (This is the rule the previous design got backwards, treating
        CAM-03 as a separate fallback STAGE rather than a peer.)

        ``None`` means there was no causally eligible candidate to evaluate.
        """
        ranked = []
        vetoed = []
        query_colour = crossing.colour_hsv
        for group in groups.values():
            if group.status != RecordStatus.PENDING:
                continue
            group_embeddings = causal_group_embeddings(group, crossing)
            if not group_embeddings:
                continue
            # COLOUR VETO. Colour is subtractive only: it removes a candidate
            # that cannot be this car, and the margin is then recomputed over
            # the survivors. It never adds score to anything, because colour is
            # low-entropy — most cars are black, grey, silver or white, and two
            # white sedans agreeing on colour is not evidence they are one car.
            #
            # Removing a candidate is also how an ambiguity gets broken: with
            # the impostor gone the true match's row margin can clear on its
            # own merit, rather than being credited for the colour agreeing.
            if self.settings.colour_veto_enabled and not body_colour_compatible(
                query_colour, group.colour_hsv
            ):
                vetoed.append(group.group_id)
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
            # Column competition is scoped PER CAMERA, not per role.
            #
            # It exists to stop two different cars claiming one identity from
            # the same viewpoint. CAM-23 and CAM-03 seeing the SAME car are not
            # competitors — they are two independent witnesses to one entry, and
            # making them compete would mean the second camera could only ever
            # take the identity away from the first.
            #
            # Behaviourally identical to the previous role comparison, because
            # role and camera are one-to-one today. Written as camera because
            # that is the actual reason, and because CAM-03 is no longer a
            # "fallback stage" whose role is what matters about it.
            if (
                other.status != RecordStatus.PENDING
                or norm_camera_id(other.request.camera_id)
                != norm_camera_id(crossing.request.camera_id)
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
            ranked=tuple((gid, value) for value, gid in ranked),
            vetoed=tuple(vetoed),
        )

    # ------------------------------------------------------------------ #
    # Plate consensus
    # ------------------------------------------------------------------ #
    def our_ocr_reading(self, group: AttemptGroup) -> Optional[PlateReading]:
        """Fold every image OUR reader looked at into ONE plate source.

        Our OCR is a single opinion however many images produced it. Counting a
        read of the ANPR image and a read of the HikCentral image as two
        sources would let our own model reach consensus with itself and
        manufacture a two-source agreement out of one reader.

        Disagreeing with itself marks the source CONFLICTED, which removes it
        from the available set entirely. A reader that contradicts itself is
        evidence of unreliability, not a tie to be broken.

        Note what is NOT in this pool: CAM-23 and CAM-03 evidence. The ramp
        cameras are visual observation sources and read no plates.
        """
        reliable = [
            item
            for item in group.cached_plate_evidence
            if item.state == PlateReadState.READABLE
            and item.confidence >= self.settings.ocr_min_confidence
            and item.key
            and is_plausible_plate(canonical_plate(item.text))
        ]
        if not reliable:
            return None
        keys = {item.key for item in reliable}
        best = max(reliable, key=lambda item: item.confidence)
        return PlateReading(
            source=PlateSourceKind.OUR_OCR,
            text=best.text,
            confidence=best.confidence,
            origin=",".join(sorted(item.evidence_id for item in reliable)),
            conflicted=len(keys) > 1,
        )

    def anpr_reading(self, group: AttemptGroup) -> Optional[PlateReading]:
        """What the gate ANPR system reported, folded across its attempts.

        Derived from the group's attempts rather than stored, for the same
        reason our OCR is: the causal projection restricts an identity to the
        attempts that PRECEDE a crossing, and a stored copy would happily hand
        the decision a plate read after the car had already gone past.
        """
        readings = [
            attempt.request
            for attempt in group.attempts.values()
            if plate_key(attempt.request.reported_plate)
        ]
        if not readings:
            return None
        keys = {plate_key(item.reported_plate) for item in readings}
        best = max(readings, key=lambda item: item.reported_confidence or 0.0)
        return PlateReading(
            source=PlateSourceKind.ANPR,
            text=best.reported_plate,
            confidence=float(best.reported_confidence or 0.0),
            origin=",".join(sorted(item.attempt_id for item in readings)),
            conflicted=len(keys) > 1,
        )

    def available_plate_sources(self, group: AttemptGroup):
        """The plate sources that actually produced a usable reading.

        ANPR and our own OCR are DERIVED from the identity's attempts, so they
        automatically respect the causal projection and can never fall out of
        step with the evidence they are folded from. Only what we FETCHED from
        HikCentral is stored — and stage 5 must attach it with its own
        causality check, since it does not come from an attempt.
        """
        sources = {
            kind: reading
            for kind, reading in group.plate_sources.items()
            if kind is PlateSourceKind.HIK_TEXT
        }
        for derived in (self.anpr_reading(group), self.our_ocr_reading(group)):
            if derived is not None:
                sources[derived.source] = derived
        return {
            kind: reading
            for kind, reading in sources.items()
            if reading.key and not reading.conflicted
        }

    def plate_consensus(self, group: AttemptGroup) -> PlateConsensus:
        """Agreement across the sources that actually read a plate.

        The rule is "at least two agreeing", not "two out of exactly three":
        with n sources available, the winning group must have at least
        min(2, n) members and must not be tied. So

            three sources, two agree      -> those two win
            three sources, all disagree   -> no consensus
            two sources, both agree       -> consensus
            two sources, they disagree    -> no consensus
            one source                    -> it stands

        The single-source case is deliberate. HikCentral being unreachable or
        an image being unusable must not stop ordinary traffic, and by the time
        this runs two independent observations have already agreed on the
        physical vehicle. What is refused is a CONTRADICTION, not a thin
        record.
        """
        available = self.available_plate_sources(group)
        if not available:
            return PlateConsensus(outcome="unavailable", plate=None)

        by_key = {}
        for kind, reading in available.items():
            by_key.setdefault(reading.key, []).append((kind, reading))

        ranked = sorted(by_key.items(), key=lambda item: -len(item[1]))
        top_key, top = ranked[0]
        tied = len(ranked) > 1 and len(ranked[1][1]) == len(top)
        needed = min(2, len(available))

        disagreeing = tuple(
            (kind.value, reading.text)
            for key, entries in ranked[1:]
            for kind, reading in entries
        )
        if len(top) < needed or tied:
            return PlateConsensus(
                outcome="no_consensus",
                plate=None,
                available=tuple(sorted(k.value for k in available)),
                disagreeing=tuple(
                    (kind.value, reading.text)
                    for kind, reading in available.items()
                ),
            )

        best = max(top, key=lambda item: item[1].confidence)
        return PlateConsensus(
            outcome="consensus",
            plate=canonical_plate(best[1].text),
            agreeing=tuple(sorted(kind.value for kind, _ in top)),
            available=tuple(sorted(k.value for k in available)),
            disagreeing=disagreeing,
            confidence=best[1].confidence,
            evidence_ids=tuple(
                sorted(
                    part
                    for _, reading in top
                    for part in (reading.origin.split(",") if reading.origin else ())
                    if part
                )
            ),
        )

    def observation_contradicts(
        self,
        crossing: CrossingRecord,
        consensus: PlateConsensus,
    ) -> bool:
        """Does the observation itself read a DIFFERENT plate? Then withhold.

        A ramp camera is not a plate source and can never name a car. But when
        it reads a plate reliably and that plate is not the one the sources
        agreed on, it is evidence that Re-ID has matched the wrong identity —
        and refusing on that is not the same as naming a plate with it.

        This closes a real hole. Two similar cars at Re-ID 0.92 with only one
        ANPR read between them will otherwise confirm the second car's crossing
        under the first car's plate: the witnesses agree, the consensus has
        nothing to contradict it, and nothing else is left to object. The old
        design caught this with its CAM-23 plate policy; with that gone, the
        veto has to.

        Subtractive, like the colour check and the producer-family gate: it can
        withhold an entry, never create one. Erring toward an entry we do not
        open rather than a session opened under someone else's name.

        Off via ENTRY_V2_OBSERVATION_PLATE_VETO_ENABLED if the shadow window
        shows ramp reads are too unreliable to withhold on.
        """
        if not self.settings.observation_plate_veto_enabled:
            return False
        if consensus.outcome != "consensus" or not consensus.plate:
            return False
        key = plate_key(consensus.plate)
        for item in crossing.plate_evidence:
            if (
                item.state == PlateReadState.READABLE
                and item.confidence >= self.settings.ocr_min_confidence
                and item.key
                and is_plausible_plate(canonical_plate(item.text))
                and item.key != key
            ):
                return True
        return False

    def resolve_plate(
        self,
        group: AttemptGroup,
        crossing: CrossingRecord,
        correlated_primary_crossings: Sequence[CrossingRecord] = (),
    ) -> PlateResolution:
        """Name the vehicle a confirmed entry belongs to.

        Re-ID chose WHICH identity; this chooses what its plate is, from the
        plate sources alone. The crossing is passed only so callers keep one
        signature — nothing about it is read, because CAM-23 and CAM-03 do not
        read plates. `correlated_primary_crossings` is likewise ignored and
        kept only so existing call sites do not have to change in this stage.
        """
        del correlated_primary_crossings

        consensus = self.plate_consensus(group)
        contradiction = self.observation_contradicts(crossing, consensus)
        if contradiction:
            return PlateResolution(
                outcome="abstained",
                reason="observation_plate_contradiction",
                ocr_source=OCR_SOURCE_CONSENSUS,
                ocr_text=consensus.plate,
                ocr_confidence=consensus.confidence,
            )
        common = dict(
            ocr_source=OCR_SOURCE_CONSENSUS,
            ocr_text=consensus.plate,
            ocr_confidence=consensus.confidence,
            ocr_evidence_ids=consensus.evidence_ids,
        )
        if consensus.outcome == "unavailable":
            return PlateResolution(
                outcome="abstained",
                reason="plate_sources_unavailable",
                ocr_source=OCR_SOURCE_CONSENSUS,
            )
        if consensus.outcome == "no_consensus":
            return PlateResolution(
                outcome="abstained",
                reason="plate_no_consensus",
                ocr_source=OCR_SOURCE_CONSENSUS,
            )
        if not is_plausible_plate(consensus.plate or ""):
            return PlateResolution(
                outcome="abstained",
                reason="plate_implausible",
                **common,
            )

        # A consensus that disagrees with every plate ANPR reported is a
        # CORRECTION, and is held to a higher bar for teaching the durable
        # gallery — see identity.build_gallery_authorization_proof.
        corrected = plate_key(consensus.plate or "") not in group.reported_keys
        return PlateResolution(
            outcome="confirmed",
            reason=(
                REASON_CORRECTION if corrected else REASON_CONSENSUS
            ),
            canonical_plate=consensus.plate,
            **common,
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

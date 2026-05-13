"""
src.matching.match_voter — Temporal voting wrapper around :class:`MatchDecision`.

Phase 2 / T2.2. ``MatchVoter`` maintains a thread-safe per-``(camera_id,
track_id)`` rolling buffer of recent :class:`Decision` objects. A decision is
committed only when the same plate wins ``voting_min_agree`` of the last
``voting_window_frames`` decisions; otherwise the voter "defers" by returning
``None``.

This addresses the false-confirm failure mode where a single noisy frame
yields a wrong plate that subsequent frames would have voted down. Default
config (3-of-5 at the documented frame rate) trades ~200-500 ms commit
latency for materially fewer wrong matches.

Public surface
~~~~~~~~~~~~~~

    MatchVoter(config)
        .submit(camera_id, track_id, decision) -> Optional[Decision]
        .cleanup_stale(max_age_seconds=None)
        .clear(camera_id=None, track_id=None)
        .snapshot()

The voter is feature-flagged via ``MatchingConfig.voting_enabled``: when off,
``submit`` is a pass-through that commits the incoming decision immediately
(no buffering, no deferral). Callers that care can check
``config.voting_enabled`` before calling ``submit`` to skip the deque
bookkeeping entirely.

Audit logging
~~~~~~~~~~~~~

Voter-tick events log to a dedicated ``match_voter_perf`` logger so they can
be routed to a separate file (or muted) without affecting the existing
``reid_match_perf`` commit audit trail at
``vehicle_registry_identity.py:~1116``. The commit-time log line in
``try_link_to_slot`` is untouched.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import Counter, deque
from typing import Deque, Dict, Optional, Tuple

from src.config import MatchingConfig
from src.matching.match_decision import Decision

# Dedicated voter logger — separate from ``reid_match_perf`` (the commit audit
# trail) so tick-level chatter can be routed independently.
voter_logger = logging.getLogger("match_voter_perf")
logger = logging.getLogger(__name__)


# Key for the per-track buffer map: (camera_id, track_id). track_id may be
# ``None`` for callers without a track id (e.g. shadow-mode probes); keep the
# type permissive so the voter is reusable from non-identity callsites too.
_BufferKey = Tuple[str, Optional[int]]


class _BufferEntry:
    """Internal per-key wrapper carrying the decision deque and a last-touch
    timestamp for ``cleanup_stale``.

    A plain dataclass would do but we want O(1) appends + bounded growth via
    ``deque(maxlen=...)`` plus a separate timestamp for staleness — those two
    concerns are easier to read as a small class.
    """

    __slots__ = ("decisions", "last_seen")

    def __init__(self, window: int):
        self.decisions: Deque[Decision] = deque(maxlen=max(1, int(window)))
        self.last_seen: float = time.time()

    def touch(self) -> None:
        self.last_seen = time.time()


class MatchVoter:
    """Temporal vote aggregator around :class:`MatchDecision` outputs.

    Args:
        config: The active :class:`MatchingConfig`. Reads ``voting_enabled``,
            ``voting_window_frames`` and ``voting_min_agree`` at every
            ``submit`` so config swaps are picked up without rebuilding the
            voter. Defaults are ``False / 5 / 3`` — feature is off by default.
    """

    # Cleanup defaults — generous so a slow camera does not have its vote
    # window dropped between two snapshots. ``cleanup_stale_default_seconds``
    # is what the registry GC thread will use when calling without an arg.
    DEFAULT_STALE_SECONDS: float = 60.0

    def __init__(self, config: MatchingConfig):
        self._config = config
        self._lock = threading.RLock()
        self._buffers: Dict[_BufferKey, _BufferEntry] = {}

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    @property
    def config(self) -> MatchingConfig:
        return self._config

    def submit(
        self,
        camera_id: str,
        track_id: Optional[int],
        decision: Decision,
    ) -> Optional[Decision]:
        """Append ``decision`` to the per-track buffer and return a committed
        verdict only when the same plate (or session id) wins the vote.

        Args:
            camera_id: source camera. Buffers are partitioned per camera so a
                cross-camera handoff does not bleed into another camera's vote.
            track_id: integer track id (or ``None`` for non-tracked submissions).
            decision: the :class:`Decision` returned by
                :class:`MatchDecision`. The voter inspects only
                ``decision.verdict`` and ``decision.scores.get('plate')`` /
                ``scores.get('session_id')`` — the same shape used by the
                identity mixin's commit log.

        Returns:
            * ``decision`` itself (pass-through) when ``voting_enabled`` is
              False.
            * ``decision`` when its plate wins the K-of-N vote.
            * ``None`` otherwise — the caller should defer the commit.
        """
        cfg = self._config

        if not cfg.voting_enabled:
            # Feature flag off — no buffering, immediate commit.
            return decision

        window = max(1, int(cfg.voting_window_frames))
        threshold = max(1, int(cfg.voting_min_agree))
        # A min-agree larger than the window is unsatisfiable; clamp.
        if threshold > window:
            threshold = window

        key: _BufferKey = (str(camera_id), track_id)

        with self._lock:
            entry = self._buffers.get(key)
            if entry is None:
                entry = _BufferEntry(window=window)
                self._buffers[key] = entry
            # If the config window shrank, ``entry.decisions.maxlen`` may
            # exceed the new ``window``. ``deque`` doesn't support live
            # maxlen mutation, so we rebuild on a mismatch.
            if entry.decisions.maxlen != window:
                entry.decisions = deque(entry.decisions, maxlen=window)

            entry.decisions.append(decision)
            entry.touch()

            voter_logger.debug(
                "[VOTER] cam=%s track=%s window=%d min_agree=%d size=%d "
                "verdict=%s reason=%s plate=%s",
                camera_id,
                track_id,
                window,
                threshold,
                len(entry.decisions),
                decision.verdict,
                decision.reason,
                self._decision_key(decision),
            )

            # Defer until the buffer is at least min_agree deep.
            if len(entry.decisions) < threshold:
                return None

            # Tally only "confirm" verdicts — a "reject"/"tentative" does
            # not count toward the K-of-N agreement on a specific plate.
            confirm_keys = [
                self._decision_key(d)
                for d in entry.decisions
                if d.verdict == "confirm"
            ]
            if not confirm_keys:
                return None

            counts = Counter(k for k in confirm_keys if k is not None)
            if not counts:
                # All keys were None — fall back to "any confirm" majority.
                if len(confirm_keys) >= threshold:
                    voter_logger.info(
                        "[VOTER] commit (anonymous) cam=%s track=%s "
                        "confirms=%d/%d",
                        camera_id,
                        track_id,
                        len(confirm_keys),
                        threshold,
                    )
                    return decision
                return None

            top_key, top_count = counts.most_common(1)[0]
            if top_count < threshold:
                return None

            # Commit the most-recent decision matching the winning key, so
            # the caller sees the freshest scores / snapshot path.
            for d in reversed(entry.decisions):
                if self._decision_key(d) == top_key and d.verdict == "confirm":
                    voter_logger.info(
                        "[VOTER] commit cam=%s track=%s plate=%s confirms=%d/%d",
                        camera_id,
                        track_id,
                        top_key,
                        top_count,
                        threshold,
                    )
                    return d
            return decision

    def cleanup_stale(self, max_age_seconds: Optional[float] = None) -> int:
        """Drop per-track buffers not updated within ``max_age_seconds``.

        Called from the registry's GC thread (see
        ``vehicle_registry_core._gc_loop``). Returns the number of dropped
        entries — useful for sanity tests.
        """
        max_age = (
            float(max_age_seconds)
            if max_age_seconds is not None
            else self.DEFAULT_STALE_SECONDS
        )
        cutoff = time.time() - max_age
        dropped = 0
        with self._lock:
            stale = [k for k, v in self._buffers.items() if v.last_seen < cutoff]
            for k in stale:
                self._buffers.pop(k, None)
                dropped += 1
        if dropped:
            voter_logger.debug(
                "[VOTER] cleanup_stale dropped %d buffers (max_age=%.1fs)",
                dropped,
                max_age,
            )
        return dropped

    def clear(
        self,
        camera_id: Optional[str] = None,
        track_id: Optional[int] = None,
    ) -> int:
        """Drop a specific track's buffer (or everything).

        With no args, wipes the entire buffer map. With ``(camera_id,
        track_id)`` drops just that key. Returns the number of buffers
        dropped.
        """
        with self._lock:
            if camera_id is None and track_id is None:
                count = len(self._buffers)
                self._buffers.clear()
                return count
            removed = 0
            for k in list(self._buffers):
                if camera_id is not None and k[0] != camera_id:
                    continue
                if track_id is not None and k[1] != track_id:
                    continue
                self._buffers.pop(k, None)
                removed += 1
            return removed

    def snapshot(self) -> Dict[_BufferKey, Tuple[int, float]]:
        """Return a read-only snapshot ``{key: (len(buffer), last_seen)}``
        for tests and the registry's debug endpoints. Buffers themselves are
        not exposed — callers should not be inspecting Decision tuples
        directly outside of unit tests."""
        with self._lock:
            return {
                k: (len(v.decisions), float(v.last_seen))
                for k, v in self._buffers.items()
            }

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    @staticmethod
    def _decision_key(decision: Decision) -> Optional[str]:
        """Derive the vote key from a :class:`Decision`.

        Order of preference:
          1. ``scores['plate']`` — the canonical key when an ANPR plate is
             known (B1 / global cases).
          2. ``scores['session_id']`` — for plate-less / appearance-only
             sessions (the reattach path).
          3. ``decision.reason`` — last resort so anonymous reject votes are
             still distinct from anonymous confirm votes; the caller will
             usually filter on ``verdict == 'confirm'`` before voting.

        Returns ``None`` when no key can be derived — the voter then falls
        back to "any confirm" majority counting.
        """
        scores = decision.scores or {}
        for field in ("plate", "session_id"):
            value = scores.get(field)
            if value:
                return str(value)
        return None


__all__ = ["MatchVoter"]

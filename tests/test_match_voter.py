"""
tests.test_match_voter — Unit tests for the Phase 2 / T2.2 ``MatchVoter``.

Covers:

  * pass-through when ``voting_enabled`` is False
  * deferral while the buffer is still under ``voting_min_agree``
  * commit when the same plate wins ``voting_min_agree`` of the window
  * minority no-commit (different plates) returns None
  * plate-flip mid-window resets the vote
  * stale buffer cleanup drops idle tracks
  * partition by ``(camera_id, track_id)`` — votes don't bleed across keys

These tests do not import :class:`VehicleRegistry`; they exercise the voter
in isolation so failures point at the voter itself.
"""

from __future__ import annotations

import time
import unittest
from typing import Optional

from src.config import MatchingConfig
from src.matching.match_decision import Decision
from src.matching.match_voter import MatchVoter


def _decision(plate: Optional[str], verdict: str = "confirm") -> Decision:
    """Build a ``Decision`` whose scores carry ``plate`` so the voter has a
    vote key to count on."""
    return Decision(
        verdict=verdict,
        reason="test",
        scores={"plate": plate} if plate is not None else {},
    )


class TestMatchVoterDisabled(unittest.TestCase):
    """When ``voting_enabled`` is False the voter is a no-op pass-through."""

    def test_disabled_voter_returns_input_unchanged(self) -> None:
        cfg = MatchingConfig(voting_enabled=False)
        voter = MatchVoter(cfg)
        decision = _decision("PASS-001")

        result = voter.submit("CAM-01", 1, decision)

        self.assertIs(result, decision)
        # No buffers were created — pass-through doesn't bookkeep.
        self.assertEqual(voter.snapshot(), {})


class TestMatchVoterCommit(unittest.TestCase):
    """3-of-5 happy path: same plate wins the K-of-N vote."""

    def setUp(self) -> None:
        self.cfg = MatchingConfig(
            voting_enabled=True,
            voting_window_frames=5,
            voting_min_agree=3,
        )
        self.voter = MatchVoter(self.cfg)

    def test_buffer_filling_returns_none_until_min_agree(self) -> None:
        # First decision — buffer size 1 < 3 → None.
        self.assertIsNone(self.voter.submit("CAM-01", 7, _decision("ABC-001")))
        self.assertIsNone(self.voter.submit("CAM-01", 7, _decision("ABC-001")))
        # Third decision — size 3 == 3 → commit.
        committed = self.voter.submit("CAM-01", 7, _decision("ABC-001"))
        self.assertIsNotNone(committed)
        self.assertEqual(committed.scores["plate"], "ABC-001")

    def test_majority_commit_when_minority_disagrees(self) -> None:
        for plate in ["ABC-001", "ABC-001", "WRONG-009", "ABC-001"]:
            committed = self.voter.submit("CAM-01", 7, _decision(plate))
        # 3 of 4 votes are ABC-001 — committed.
        self.assertIsNotNone(committed)
        self.assertEqual(committed.scores["plate"], "ABC-001")

    def test_window_overflow_drops_oldest_decision(self) -> None:
        # Submit 6 decisions for a window of 5 — oldest is dropped.
        for plate in ["AAA-001", "BBB-002", "AAA-001", "AAA-001", "BBB-002", "BBB-002"]:
            self.voter.submit("CAM-01", 7, _decision(plate))
        # Last 5 votes: [BBB-002, AAA-001, AAA-001, BBB-002, BBB-002] — 3 BBB.
        # Add one more BBB-002 to ensure commit.
        committed = self.voter.submit("CAM-01", 7, _decision("BBB-002"))
        self.assertIsNotNone(committed)
        self.assertEqual(committed.scores["plate"], "BBB-002")


class TestMatchVoterMinorityNoCommit(unittest.TestCase):
    """When no single plate has ``voting_min_agree`` votes the voter defers."""

    def test_evenly_split_window_no_commit(self) -> None:
        cfg = MatchingConfig(
            voting_enabled=True,
            voting_window_frames=4,
            voting_min_agree=3,
        )
        voter = MatchVoter(cfg)
        # 2-2 split — no plate has 3 votes.
        results = [
            voter.submit("CAM-01", 9, _decision(p))
            for p in ["AAA", "BBB", "AAA", "BBB"]
        ]
        # Once buffer >= min_agree=3 the voter starts trying to commit; all
        # decisions remain None because the top key tops out at 2 votes.
        self.assertTrue(all(r is None for r in results))

    def test_reject_verdicts_do_not_count_toward_commit(self) -> None:
        cfg = MatchingConfig(
            voting_enabled=True,
            voting_window_frames=5,
            voting_min_agree=3,
        )
        voter = MatchVoter(cfg)
        # 2 rejects + 2 confirms for the same plate — confirms top out at 2.
        for plate, verdict in [
            ("ABC", "confirm"),
            ("ABC", "reject"),
            ("ABC", "reject"),
            ("ABC", "confirm"),
        ]:
            result = voter.submit("CAM-01", 3, _decision(plate, verdict=verdict))
        # Only 2 confirms for ABC — below the 3-of-N threshold.
        self.assertIsNone(result)


class TestMatchVoterPlateFlip(unittest.TestCase):
    """A plate flip mid-window must not produce a wrong commit."""

    def test_plate_flip_within_window_does_not_commit_minority(self) -> None:
        cfg = MatchingConfig(
            voting_enabled=True,
            voting_window_frames=5,
            voting_min_agree=3,
        )
        voter = MatchVoter(cfg)
        # Two votes for AAA then three for BBB — once BBB hits 3-of-5, commit.
        for plate in ["AAA", "AAA", "BBB"]:
            self.assertIsNone(voter.submit("CAM-01", 5, _decision(plate)))
        # 4th submission: [AAA, AAA, BBB, BBB] — top key is AAA *or* BBB
        # with count 2 each; below the floor → no commit.
        self.assertIsNone(voter.submit("CAM-01", 5, _decision("BBB")))
        # 5th submission tips it over: [AAA, AAA, BBB, BBB, BBB] → BBB wins.
        committed = voter.submit("CAM-01", 5, _decision("BBB"))
        self.assertIsNotNone(committed)
        self.assertEqual(committed.scores["plate"], "BBB")


class TestMatchVoterStaleCleanup(unittest.TestCase):
    """``cleanup_stale`` drops buffers idle longer than ``max_age_seconds``."""

    def test_cleanup_stale_drops_old_buffers(self) -> None:
        cfg = MatchingConfig(
            voting_enabled=True,
            voting_window_frames=3,
            voting_min_agree=2,
        )
        voter = MatchVoter(cfg)
        voter.submit("CAM-01", 1, _decision("X"))
        voter.submit("CAM-01", 2, _decision("Y"))

        # Backdate one entry so cleanup considers it stale.
        with voter._lock:  # noqa: SLF001 — test-only introspection
            voter._buffers[("CAM-01", 1)].last_seen -= 999.0

        dropped = voter.cleanup_stale(max_age_seconds=10.0)
        self.assertEqual(dropped, 1)
        snap = voter.snapshot()
        self.assertNotIn(("CAM-01", 1), snap)
        self.assertIn(("CAM-01", 2), snap)

    def test_cleanup_stale_keeps_fresh_buffers(self) -> None:
        cfg = MatchingConfig(voting_enabled=True)
        voter = MatchVoter(cfg)
        voter.submit("CAM-01", 7, _decision("A"))
        # Nothing should be dropped right away.
        self.assertEqual(voter.cleanup_stale(max_age_seconds=120.0), 0)


class TestMatchVoterPartitioning(unittest.TestCase):
    """Votes are partitioned per ``(camera_id, track_id)``; cross-key bleed
    would be a correctness regression."""

    def test_votes_do_not_bleed_across_cameras(self) -> None:
        cfg = MatchingConfig(
            voting_enabled=True,
            voting_window_frames=3,
            voting_min_agree=2,
        )
        voter = MatchVoter(cfg)
        # CAM-01 / track 1 — vote ABC once.
        self.assertIsNone(voter.submit("CAM-01", 1, _decision("ABC")))
        # CAM-02 / track 1 — vote ABC once. No bleed.
        self.assertIsNone(voter.submit("CAM-02", 1, _decision("ABC")))
        # CAM-01 / track 1 again — now buffer is 2 ABCs → commit.
        committed = voter.submit("CAM-01", 1, _decision("ABC"))
        self.assertIsNotNone(committed)
        self.assertEqual(committed.scores["plate"], "ABC")

    def test_clear_with_no_args_wipes_everything(self) -> None:
        cfg = MatchingConfig(voting_enabled=True)
        voter = MatchVoter(cfg)
        voter.submit("CAM-01", 1, _decision("X"))
        voter.submit("CAM-02", 2, _decision("Y"))
        wiped = voter.clear()
        self.assertEqual(wiped, 2)
        self.assertEqual(voter.snapshot(), {})


class TestMatchVoterWindowMutation(unittest.TestCase):
    """When ``voting_window_frames`` shrinks live, the deque is rebuilt so
    counts reflect the new horizon. Defensive, since voting parameters are
    usually static, but cheap to verify."""

    def test_shrunk_window_rebuilds_deque(self) -> None:
        cfg = MatchingConfig(
            voting_enabled=True,
            voting_window_frames=5,
            voting_min_agree=2,
        )
        voter = MatchVoter(cfg)
        for _ in range(4):
            voter.submit("CAM-01", 1, _decision("X"))
        # Shrink window to 2 — the next submit should keep only the latest 2.
        cfg.voting_window_frames = 2
        voter.submit("CAM-01", 1, _decision("X"))
        snap = voter.snapshot()
        # Buffer length capped at 2 by the rebuilt deque.
        self.assertEqual(snap[("CAM-01", 1)][0], 2)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()

"""The supervisor's CPU budget: pin only what we own, and size threads for concurrency.

Two bugs, both found in production on 2026-07-15, both locked down here.

1. PINNING AGAINST A QUOTA. `_available_cores` read sched_getaffinity and treated it
   as the budget. That is true for `docker --cpuset-cpus`; under a Kubernetes CPU
   *limit* (a CFS quota) the mask is the whole NODE, so the supervisor pinned 8 groups
   onto host CPUs shared with every other pod. Four groups landed on contended cores
   and ran at 0.003-0.015 fps/camera (5-26 MINUTES per slot flip) while four on quiet
   cores held 1.3-2.0 — and the kernel could not rebalance, because affinity forbade
   it. Raising the pod's CPU limit did nothing, which is the tell: affinity, not quota,
   was the cap.

2. SIZING THREADS TO THE QUOTA. The first fix apportioned exactly `quota` threads,
   giving gate and ground ONE each — dropping them from 1.8/1.6 fps to 0.40/0.20,
   worse than the bug it replaced, while the pod drew only 8.8 of its 14 cores. A
   quota caps CPU-seconds burned; it is not a thread count. These workers block on
   HEVC decode, so quota-many threads leave the quota idle.
"""
import unittest
from unittest import mock

import supervisor as S

# The real production topology (supervisor log, 2026-07-15): 8 groups, 26 cameras.
GROUPS = [
    {"name": "gate", "cams": "CAM-23,CAM-03"},
    {"name": "b1-a", "cams": "CAM-07,CAM-20,CAM-24"},
    {"name": "b1-b", "cams": "CAM-21,CAM-22"},
    {"name": "b1-c", "cams": "CAM-04,CAM-05,CAM-06,CAM-08"},
    {"name": "b2-a", "cams": "CAM-09,CAM-12,CAM-13,CAM-16,CAM-19"},
    {"name": "b2-b", "cams": "CAM-17,CAM-18,CAM-25"},
    {"name": "b2-c", "cams": "CAM-10,CAM-11,CAM-14,CAM-15"},
    {"name": "ground", "cams": "CAM-01,CAM-02,CAM-00"},
]


def _budget(quota, mask, n_groups=len(GROUPS)):
    with mock.patch.object(S, "_quota_cores", return_value=quota), \
         mock.patch.object(S, "_affinity_cores", return_value=mask):
        return S._cpu_budget(n_groups)


class TestPinningDecision(unittest.TestCase):
    """A quota means the cores are SHARED, whatever its value."""

    def test_quota_below_mask_does_not_pin(self):
        self.assertFalse(_budget(12.0, 20)[1])

    def test_quota_ABOVE_mask_still_does_not_pin(self):
        # The trap: the limit was raised past the mask, and pinning "looked" safe
        # again. It never was — the cores were never ours. Keying on quota < mask
        # would silently restore the starvation.
        self.assertFalse(_budget(64.0, 20)[1])

    def test_no_quota_pins(self):
        # cpuset (docker --cpuset-cpus) or a whole box: the mask IS ours, so slicing
        # it per group is real isolation. Unchanged behaviour.
        self.assertTrue(_budget(None, 8)[1])


class TestThreadSizing(unittest.TestCase):
    """A quota caps the BURN. Threads exist to keep it busy across I/O waits."""

    def test_threads_exceed_the_quota(self):
        total, _ = _budget(14.0, 20)
        self.assertGreater(
            total, 14, "quota-many threads leave the quota idle — decode blocks"
        )

    def test_never_exceeds_the_mask(self):
        # More threads than the node has cores buys nothing.
        self.assertLessEqual(_budget(64.0, 20)[0], 20)

    def test_no_group_is_choked_to_one_thread(self):
        # The 0.40/0.20 fps regression: one thread cannot decode a 1080p HEVC
        # stream AND run inference on it.
        total, _ = _budget(14.0, 20)
        for g, cores in zip(GROUPS, S._core_slices(GROUPS, total)):
            self.assertGreaterEqual(len(cores), 2, f"{g['name']} choked to {len(cores)}")

    def test_restores_the_thread_counts_that_worked(self):
        # 14-core quota on the 20-core node must reproduce the per-group counts the
        # PINNED config had when the healthy groups were doing 1.3-2.0 fps — only
        # now unpinned, so no group can be locked out of its cores.
        total, pin = _budget(14.0, 20)
        got = {g["name"]: len(c) for g, c in zip(GROUPS, S._core_slices(GROUPS, total))}
        self.assertEqual(
            got,
            {"gate": 2, "b1-a": 2, "b1-b": 2, "b1-c": 3,
             "b2-a": 4, "b2-b": 2, "b2-c": 3, "ground": 2},
        )
        self.assertFalse(pin)

    def test_tiny_quota_still_gives_every_group_a_floor(self):
        total, _ = _budget(2.0, 20)
        self.assertGreaterEqual(total, 2 * len(GROUPS))


class TestQuotaParsing(unittest.TestCase):
    def test_cgroup_v2_unlimited_reads_as_none(self):
        with mock.patch("builtins.open", mock.mock_open(read_data="max 100000")):
            self.assertIsNone(S._quota_cores())

    def test_cgroup_v2_quota_is_cores(self):
        with mock.patch("builtins.open", mock.mock_open(read_data="1400000 100000")):
            self.assertAlmostEqual(S._quota_cores(), 14.0)

    def test_unreadable_cgroup_reads_as_none(self):
        # No cgroup at all (Windows dev box) must not raise — it means "unlimited".
        with mock.patch("builtins.open", side_effect=OSError):
            self.assertIsNone(S._quota_cores())


if __name__ == "__main__":
    unittest.main()

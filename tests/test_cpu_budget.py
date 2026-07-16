"""The supervisor's CPU budget: pin only what we own, size threads TO the quota.

Three bugs, all found in production 2026-07-15/16, all locked down here.

1. PINNING AGAINST A QUOTA. `_available_cores` read sched_getaffinity and treated it
   as the budget. That is true for `docker --cpuset-cpus`; under a Kubernetes CPU
   *limit* (a CFS quota) the mask is the whole NODE, so the supervisor pinned 8 groups
   onto host CPUs shared with every other pod. Four groups landed on contended cores
   and ran at 0.003-0.015 fps/camera (5-26 MINUTES per slot flip) while four on quiet
   cores held 1.3-2.0 — and the kernel could not rebalance, because affinity forbade
   it. Raising the pod's CPU limit did nothing, which is the tell: affinity, not quota,
   was the cap.

2. SIZING THREADS TO THE QUOTA WITHOUT A PER-GROUP FLOOR. The first fix apportioned
   exactly `quota` threads by camera share, giving gate and ground ONE each —
   dropping them from 1.8/1.6 fps to 0.40/0.20, worse than the bug it replaced. One
   thread cannot decode a 1080p HEVC stream AND run inference on it. The floor
   (_MIN_THREADS_PER_GROUP, enforced per-slice via _core_slices min_size) is the fix.

3. OVERSUBSCRIBING THE QUOTA. Bug 2's fix shipped bundled with 1.5x thread
   oversubscription "to keep the quota busy across decode waits". cpu.stat deltas
   (2026-07-16) showed 57% of CFS periods THROTTLED at both the 14- and 20-core
   limits, ~16 cores burned for ~8 inferences/s — bursting threads blow the period
   budget early and the whole cgroup freezes for the rest of the 100ms, which is how
   an 11ms int8 inference stretched to ~1000ms wall-clock. CFS caps the burn per
   PERIOD, not on average; excess threads buy freezes, not throughput.
"""
import os
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

# The same 26 cameras consolidated per-floor (the quota-regime topology).
FLOOR_GROUPS = [
    {"name": "gate", "cams": "CAM-23,CAM-03"},
    {"name": "b1", "cams": "CAM-07,CAM-20,CAM-24,CAM-21,CAM-22,CAM-04,CAM-05,CAM-06,CAM-08"},
    {"name": "b2", "cams": "CAM-09,CAM-12,CAM-13,CAM-16,CAM-19,CAM-17,CAM-18,CAM-25,CAM-10,CAM-11,CAM-14,CAM-15"},
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
    """Threads are sized TO the quota. Bursting past it freezes the cgroup."""

    def test_sized_to_the_quota(self):
        # 57% of CFS periods throttled was the price of oversubscription.
        self.assertEqual(_budget(18.0, 20)[0], 18)

    def test_never_exceeds_the_mask(self):
        # More threads than the node has cores buys nothing.
        self.assertLessEqual(_budget(64.0, 20)[0], 20)

    def test_tiny_quota_still_gives_every_group_a_floor(self):
        # The floor is the ONLY sanctioned excess over the quota: below 2
        # threads a group cannot decode and infer at once (bug 2).
        self.assertGreaterEqual(_budget(2.0, 20)[0], 2 * len(GROUPS))

    def test_no_group_is_choked_to_one_thread(self):
        # Camera-share apportionment alone would hand the 2-camera gate group a
        # single thread. min_size re-creates bug 2's fix at the slice level.
        total, pin = _budget(14.0, 20)
        self.assertFalse(pin)
        for g, cores in zip(GROUPS, S._core_slices(GROUPS, total, min_size=2)):
            self.assertGreaterEqual(len(cores), 2, f"{g['name']} choked to {len(cores)}")

    def test_floor_groups_get_camera_share_slices(self):
        # The live regime: 20-core quota on the 20-core node, 4 floor groups.
        # Big floors get the cores; small groups keep the 2-thread floor.
        total, pin = _budget(20.0, 20, n_groups=len(FLOOR_GROUPS))
        self.assertEqual(total, 20)
        self.assertFalse(pin)
        got = {
            g["name"]: len(c)
            for g, c in zip(FLOOR_GROUPS, S._core_slices(FLOOR_GROUPS, total, min_size=2))
        }
        self.assertEqual(got, {"gate": 2, "b1": 7, "b2": 9, "ground": 2})


class TestCoreSlices(unittest.TestCase):
    def test_min_size_overshoot_is_clawed_back(self):
        # Lifting small groups to min_size can push the floors past the total;
        # the surplus must come back from the over-granted, never below min_size,
        # and the slices must still be disjoint and cover exactly the total.
        slices = S._core_slices(GROUPS, 16, min_size=2)
        sizes = [len(s) for s in slices]
        self.assertEqual(sum(sizes), 16)
        self.assertTrue(all(n >= 2 for n in sizes), sizes)
        flat = [c for s in slices for c in s]
        self.assertEqual(flat, list(range(16)))  # disjoint, contiguous, complete

    def test_skewed_shares_never_produce_empty_slices(self):
        # Latent overflow: one huge group next to many tiny ones made the floor
        # lifts exceed the total, and later groups got EMPTY ranges.
        skewed = [{"name": "big", "cams": ",".join(f"C{i}" for i in range(20))}] + [
            {"name": f"tiny{i}", "cams": f"T{i}"} for i in range(7)
        ]
        slices = S._core_slices(skewed, 8, min_size=1)
        self.assertTrue(all(len(s) >= 1 for s in slices), [len(s) for s in slices])
        self.assertEqual(sum(len(s) for s in slices), 8)

    def test_impossible_floor_degrades_to_one(self):
        # 8 groups on 10 cores cannot honour min_size=2; degrade rather than lie.
        slices = S._core_slices(GROUPS, 10, min_size=2)
        self.assertEqual(sum(len(s) for s in slices), 10)
        self.assertTrue(all(len(s) >= 1 for s in slices))


class TestChildEnv(unittest.TestCase):
    """Unpinned = quota regime: OMP pools are pure burn, OpenVINO capped at 8.

    The OV cap sits at 8 because that is where 320px int8 thread scaling ends on
    BOTH boxes — but not below it: bench_yolo on the production Xeon (2026-07-16)
    measured 185ms/inf at 2 threads vs 35ms at 8. A tighter cap starves the big
    floor groups; a looser one buys spin, not speed.
    """

    def test_unpinned_kills_omp_spin_and_caps_openvino(self):
        env = S._child_env(list(range(9)), pin=False)  # b2's 9-core slice
        self.assertEqual(env["OMP_NUM_THREADS"], "1")  # PaddleOCR's own warning
        self.assertEqual(env["VA_OV_NUM_THREADS"], "8")  # _OV_THREADS_MAX
        self.assertEqual(env["VA_CV_NUM_THREADS"], "9")  # decode keeps the slice
        self.assertEqual(env["VA_NO_AFFINITY"], "1")
        self.assertNotIn("VA_CPU_LIST", env)

    def test_small_slice_is_not_padded_to_the_cap(self):
        env = S._child_env([0, 1], pin=False)  # gate/ground
        self.assertEqual(env["VA_OV_NUM_THREADS"], "2")

    def test_pinned_keeps_slice_sized_pools(self):
        env = S._child_env([4, 5, 6], pin=True)
        self.assertEqual(env["VA_CPU_LIST"], "4,5,6")
        self.assertEqual(env["OMP_NUM_THREADS"], "3")


class TestGroupingMode(unittest.TestCase):
    """Quota regime consolidates to per-floor; cpuset keeps per-area. Env forces."""

    def test_quota_groups_by_floor(self):
        with mock.patch.object(S, "_quota_cores", return_value=20.0), \
             mock.patch.dict(os.environ):
            os.environ.pop("VA_GROUP_BY", None)
            self.assertEqual(S._grouping_mode(), "floor")

    def test_no_quota_groups_by_area(self):
        with mock.patch.object(S, "_quota_cores", return_value=None), \
             mock.patch.dict(os.environ):
            os.environ.pop("VA_GROUP_BY", None)
            self.assertEqual(S._grouping_mode(), "area")

    def test_env_overrides_without_a_rebuild(self):
        # The escape hatch if the isolated bench proves inference compute-bound
        # (fewer serial loops would then halve throughput).
        with mock.patch.object(S, "_quota_cores", return_value=20.0), \
             mock.patch.dict(os.environ, {"VA_GROUP_BY": "area"}):
            self.assertEqual(S._grouping_mode(), "area")


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

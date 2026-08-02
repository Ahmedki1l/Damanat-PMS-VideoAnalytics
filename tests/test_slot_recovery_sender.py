"""The PMS-AI hand-off for recovered cars.

Naming a slot is half a recovery. These tests pin the other half: that a claim
actually leaves the process, that a refusal is treated as an answer rather than a
failure to retry, and that an unreachable PMS-AI cannot grow the queue without
bound — which is precisely what the undrained list this replaces did.
"""

import io
import json
import unittest
from urllib import error

from src.vehicle_registry.slot_recovery_sender import (
    RECOVERY_PATH,
    SERVICE_KEY_HEADER,
    SlotRecoverySender,
)

CLAIM = {
    "plate_number": "BHD-9990",
    "slot_id": "B13 COO",
    "camera_id": "CAM-24",
    "reid_score": 0.962,
    "reid_margin": 0.902,
    "reid_same_view": True,
    "ocr_text": "39BHOL",
}


class _Response(io.BytesIO):
    status = 200

    def __init__(self, body):
        super().__init__(json.dumps(body).encode("utf-8"))

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


def _opener(body=None, raises=None, capture=None):
    def _open(req, timeout=None):
        del timeout
        if capture is not None:
            capture.append(req)
        if raises is not None:
            raise raises
        return _Response(body if body is not None else {"result": "created"})
    return _open


def _sender(**kw):
    kw.setdefault("opener", _opener())
    return SlotRecoverySender("http://pms-ai:8080", "sk-test", **kw)


class TestDelivery(unittest.TestCase):
    def test_a_claim_is_posted_to_the_recovery_endpoint(self):
        seen = []
        s = _sender(opener=_opener(capture=seen))
        s.enqueue(CLAIM)
        self.assertEqual(s.drain(), 1)
        self.assertEqual(len(seen), 1)
        self.assertTrue(seen[0].full_url.endswith(RECOVERY_PATH))
        self.assertEqual(seen[0].get_method(), "POST")

    def test_the_service_key_is_sent(self):
        seen = []
        s = _sender(opener=_opener(capture=seen))
        s.enqueue(CLAIM)
        s.drain()
        headers = {k.lower(): v for k, v in seen[0].header_items()}
        self.assertEqual(headers[SERVICE_KEY_HEADER.lower()], "sk-test")

    def test_the_payload_carries_both_witnesses(self):
        """Score/margin alone is not what the endpoint accepts — ocr_text is
        required (schemas/slot_recovery.py min_length=1) and same_view changes how
        the evidence is recorded."""
        seen = []
        s = _sender(opener=_opener(capture=seen))
        s.enqueue(CLAIM)
        s.drain()
        body = json.loads(seen[0].data.decode("utf-8"))
        self.assertEqual(body["plate_number"], "BHD-9990")
        self.assertEqual(body["ocr_text"], "39BHOL")
        self.assertTrue(body["reid_same_view"])
        self.assertNotIn("_attempts", body, "internal bookkeeping must not be sent")

    def test_queue_is_emptied_on_success(self):
        s = _sender()
        s.enqueue(CLAIM)
        s.drain()
        self.assertEqual(s.pending(), 0)


class TestRefusalIsAnAnswer(unittest.TestCase):
    def test_rejected_is_settled_not_retried(self):
        """PMS-AI re-reads the live slot and refuses when the world moved on. That
        is the race guard working; re-sending asks it to act on evidence it has
        already judged stale."""
        s = _sender(opener=_opener({"result": "rejected", "reason": "slot vacant"}))
        s.enqueue(CLAIM)
        self.assertEqual(s.drain(), 1)
        self.assertEqual(s.pending(), 0)

    def test_already_open_is_settled(self):
        s = _sender(opener=_opener({"result": "already_open"}))
        s.enqueue(CLAIM)
        s.drain()
        self.assertEqual(s.pending(), 0)

    def test_a_422_is_not_retried(self):
        """A malformed claim sends the same bytes on every attempt."""
        s = _sender(opener=_opener(
            raises=error.HTTPError("u", 422, "unprocessable", {}, None)))
        s.enqueue(CLAIM)
        s.drain()
        self.assertEqual(s.pending(), 0)

    def test_a_409_is_not_retried(self):
        """409 is the feature disabled on the PMS-AI side — a retry cannot fix it."""
        s = _sender(opener=_opener(
            raises=error.HTTPError("u", 409, "conflict", {}, None)))
        s.enqueue(CLAIM)
        s.drain()
        self.assertEqual(s.pending(), 0)


class TestTransientFailure(unittest.TestCase):
    def test_a_500_is_retried_then_given_up_on(self):
        s = _sender(
            opener=_opener(raises=error.HTTPError("u", 500, "boom", {}, None)),
            max_attempts=3,
        )
        s.enqueue(CLAIM)
        s.drain()
        self.assertEqual(s.pending(), 1, "first failure should requeue")
        s.drain()
        self.assertEqual(s.pending(), 1)
        s.drain()
        self.assertEqual(s.pending(), 0, "budget spent — dropped, not spun on")

    def test_a_connection_error_is_retried(self):
        s = _sender(opener=_opener(raises=error.URLError("refused")))
        s.enqueue(CLAIM)
        s.drain()
        self.assertEqual(s.pending(), 1)


class TestTheQueueIsBounded(unittest.TestCase):
    def test_oldest_is_dropped_past_the_cap(self):
        """An unreachable PMS-AI must not turn a best-effort hand-off into an
        unbounded leak — which is exactly what _pending_slot_recoveries was."""
        s = _sender(max_queue=2)
        for i in range(5):
            s.enqueue({**CLAIM, "slot_id": f"B{i}"})
        self.assertEqual(s.pending(), 2)


class TestUrlValidation(unittest.TestCase):
    def test_a_relative_url_is_refused(self):
        with self.assertRaises(ValueError):
            SlotRecoverySender("pms-ai:8080", "k")

    def test_credentials_in_the_url_are_refused(self):
        with self.assertRaises(ValueError):
            SlotRecoverySender("http://u:p@pms-ai:8080", "k")

    def test_redirects_are_not_followed(self):
        """Following a redirect off an authenticated endpoint leaks the key."""
        import inspect
        from src.vehicle_registry import slot_recovery_sender as mod
        src = inspect.getsource(mod)
        self.assertIn("_NoRedirectHandler", src)


class TestTheEngineActuallyHandsOff(unittest.TestCase):
    """The queue existed and nothing drained it. These fail if that returns."""

    def test_both_bind_sites_queue_a_recovery(self):
        import inspect
        from src.core.engine.engine_runtime import ParkingEngineRuntimeMixin
        for fn in (ParkingEngineRuntimeMixin._maybe_bind_reid_solo,
                   ParkingEngineRuntimeMixin._retry_reid_identify):
            self.assertIn("_queue_slot_recovery", inspect.getsource(fn),
                          f"{fn.__name__} binds a recovered car without handing it "
                          "to PMS-AI — the car stays uncounted")

    def test_the_pending_queue_has_a_drainer(self):
        import pathlib
        root = pathlib.Path(__file__).resolve().parent.parent / "src"
        body = "\n".join(p.read_text(encoding="utf-8", errors="ignore")
                         for p in root.rglob("*.py"))
        self.assertGreater(body.count("_pending_slot_recoveries"), 1)
        self.assertIn("_flush_pending_slot_recoveries", body)

    def test_the_sender_is_started_by_the_engine(self):
        import pathlib
        root = pathlib.Path(__file__).resolve().parent.parent / "src"
        body = "\n".join(p.read_text(encoding="utf-8", errors="ignore")
                         for p in root.rglob("*.py"))
        self.assertIn("_start_slot_recovery_sender()", body)
        self.assertIn("_stop_slot_recovery_sender()", body)


if __name__ == "__main__":
    unittest.main()

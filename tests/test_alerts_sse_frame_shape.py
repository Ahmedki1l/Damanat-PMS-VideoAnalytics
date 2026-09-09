"""The alert SSE stream must emit frames sse_starlette can actually encode.

On 2026-09-09 `/api/alerts/stream` raised

    TypeError: ServerSentEvent.__init__() got an unexpected keyword argument
    'is_alert'

19 times and delivered nothing at all. `sse_starlette.event.ensure_bytes`
splats a yielded dict as `ServerSentEvent(**data)`, and that class accepts only
data/event/id/retry/comment/sep — so yielding the alert fields directly blew up
on the FIRST message, the handshake, before any client saw a single event.

These tests pin the encoder contract rather than the string we happen to build,
so they still fail if someone re-flattens the payload later.
"""
import asyncio
import json

import pytest
from sse_starlette.event import ensure_bytes

from src.events.event_bus import EventBus
from src.models.state_machine import SlotEvent

SEP = "\r\n"

HANDSHAKE_FIELDS = {
    "is_alert": False,
    "severity": "info",
    "alert_type": "connection_established",
    "msg": "Real-time alerts stream established",
}


def _wire(payload: dict) -> bytes:
    """Exactly what the server puts on the socket for one yielded value."""
    return ensure_bytes(dict(payload), SEP)


def _gateway_parse(chunk: bytes) -> dict:
    """The API Gateway's decoder, upstream.py:258-266, applied to our bytes."""
    for line in chunk.decode().split(SEP):
        if not line.startswith("data:"):
            continue
        return json.loads(line[5:].strip())
    raise AssertionError("no data: line in %r" % chunk)


# --------------------------------------------------------------------------- #
# The bug itself
# --------------------------------------------------------------------------- #
def test_yielding_alert_fields_directly_is_what_broke_the_stream():
    with pytest.raises(TypeError) as caught:
        _wire(HANDSHAKE_FIELDS)
    assert "is_alert" in str(caught.value)


def test_wrapping_under_data_encodes_cleanly():
    chunk = _wire({"data": json.dumps(HANDSHAKE_FIELDS)})
    assert b"data:" in chunk
    assert _gateway_parse(chunk) == HANDSHAKE_FIELDS


def test_a_real_alert_payload_survives_the_round_trip():
    event = SlotEvent(
        event_type="unauthorized_parking",
        slot_id="B11_CFO",
        track_id=7,
        timestamp="2026-09-09T09:02:56+03:00",
        camera_id="CAM-08",
        floor="B1",
        plate_number="SHR-1198",
        is_alert=True,
        severity="critical",
    )
    assert _gateway_parse(_wire({"data": json.dumps(event.to_dict())})) == event.to_dict()


def test_field_order_is_preserved_through_json_dumps():
    # to_dict() fixes a deliberate field order; the old code yielded the dict to
    # keep it. json.dumps preserves insertion order, so nothing was given up.
    payload = SlotEvent("vehicle_parked", "A1", 1, "2026-09-09T06:00:00+03:00").to_dict()
    chunk = _wire({"data": json.dumps(payload)})
    assert list(_gateway_parse(chunk)) == list(payload)


# --------------------------------------------------------------------------- #
# End to end through the real endpoint
# --------------------------------------------------------------------------- #
def _alerts_stream_endpoint(bus):
    from src.api import create_app

    app = create_app(event_bus=bus)
    for route in app.routes:
        if getattr(route, "path", None) == "/api/alerts/stream":
            return route.endpoint
    raise AssertionError("/api/alerts/stream is not registered")


def test_the_live_handshake_frame_encodes_and_parses():
    bus = EventBus()

    async def first_frame():
        agen = _alerts_stream_endpoint(bus)()
        try:
            return await agen.__anext__()
        finally:
            await agen.aclose()

    frame = asyncio.run(first_frame())
    # The handshake must survive the exact encoder that used to reject it.
    assert _gateway_parse(_wire(frame)) == HANDSHAKE_FIELDS


def test_a_published_alert_reaches_the_stream_encodable():
    bus = EventBus()
    alert = SlotEvent(
        event_type="unauthorized_parking",
        slot_id="G1",
        track_id=3,
        timestamp="2026-09-09T09:19:04+03:00",
        camera_id="CAM-23",
        is_alert=True,
        severity="critical",
    )

    async def second_frame():
        agen = _alerts_stream_endpoint(bus)()
        try:
            await agen.__anext__()          # handshake
            bus.emit(alert)
            return await asyncio.wait_for(agen.__anext__(), timeout=2)
        finally:
            await agen.aclose()

    frame = asyncio.run(second_frame())
    assert _gateway_parse(_wire(frame)) == alert.to_dict()

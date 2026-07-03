"""End-to-end: a car entering (as PMS-AI forwards it) → VA's reaction here.

This drives VA's *real* HTTP surface (`POST /api/anpr/event` on the FastAPI app
built by ``create_app``) with the exact wire payload PMS-AI's
``core_backend_client.notify_pms_anpr`` emits — ``{plate, direction,
image_base64, camera_id}`` — and asserts what VA does in response:

  * a confirmed ``VehicleSession`` is created for the plate, and
  * the durable per-plate gallery folder ``vehicle_images/gallery/<plate>/`` is
    seeded the moment the CAM-03 ``B-entry`` reference arrives (the A1 fix — the
    exact DJS-7842 / LLJ-9005 "no folder was created" gap), and
  * two cars entering sequentially keep their OWN plate + folder (no swap).

The two repos run in separate venvs, so this cannot import PMS-AI in-process.
Instead ``_pms_forward`` reproduces PMS-AI's forward contract byte-for-byte; the
matching PMS-AI test (``tests/test_entry_suppression_forward.py`` over there)
proves PMS emits that same shape. The wire payload is the seam both sides pin.
"""

import base64
import os
import tempfile
from datetime import datetime, timedelta

import cv2
import numpy as np
from fastapi.testclient import TestClient

from src.api import create_app
from src.config import MatchingConfig
from src.vehicle_registry.vehicle_registry import VehicleRegistry
from src.vehicle_registry.gallery_store import safe_plate


class FakeMatcher:
    """Deterministic ReID stand-in — feature encodes the image's mean pixel, so
    two solid-colour cars get distinct, reproducible vectors."""

    backend = "fake"

    def extract_feature(self, image):
        value = float(np.mean(image) / 255.0)
        return np.array([value, 1.0 - value], dtype=np.float32)

    def extract_features_batch(self, images):
        return [self.extract_feature(image) for image in images]

    @staticmethod
    def compute_similarity(feat1, feat2):
        if feat1 is None or feat2 is None:
            return 0.0
        return float(np.dot(feat1, feat2))


class Clock:
    """Manually-advanced clock so a test can space two cars past VA's 3s ANPR
    burst-coalesce window without sleeping (in production the two forwards are
    separated by each car's full approach + ramp-confirm cycle, ~8s+)."""

    def __init__(self, start=None):
        self.t = start or datetime(2026, 7, 3, 10, 0, 0)

    def __call__(self):
        return self.t

    def advance(self, seconds):
        self.t += timedelta(seconds=seconds)


def _make_client(clock=None):
    image_dir = tempfile.mkdtemp(prefix="pms_va_e2e_")
    cfg = MatchingConfig()
    cfg.gallery_persist_enabled = True
    cfg.reid_openvino_model_dir = ""  # tag → "…:default"
    registry = VehicleRegistry(image_dir=image_dir, matching_config=cfg, clock=clock)
    registry._reid_matcher = FakeMatcher()
    app = create_app(vehicle_registry=registry, snapshot_base_dir=image_dir)
    return TestClient(app), registry, image_dir


def _jpeg_b64(value):
    """Encode a solid-colour car crop exactly as PMS-AI does before forwarding
    (cv2 JPEG → base64 ascii)."""
    img = np.full((64, 64, 3), value, dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    return base64.b64encode(buf.tobytes()).decode("ascii")


def _pms_forward(client, plate, direction, value, camera_id="ANPR"):
    """Reproduce ``notify_pms_anpr``'s POST to VA and return the parsed body."""
    resp = client.post(
        "/api/anpr/event",
        json={
            "plate": plate,
            "direction": direction,
            "image_base64": _jpeg_b64(value),
            "camera_id": camera_id,
        },
    )
    assert resp.status_code == 200, resp.text
    return resp.json()


def _plate_dir(image_dir, plate):
    return os.path.join(image_dir, "gallery", safe_plate(plate))


def _sessions_for(registry, plate):
    return [s for s in registry._sessions.values() if s.plate == plate]


# ── The main win: a car enters → VA makes a session + seeds the folder ────────
def test_entry_then_b_entry_creates_session_and_seeds_folder():
    client, registry, image_dir = _make_client()
    plate = "DJS-7842"

    # 1. PMS forwards the gate ANPR read (direction "entry", with the car image).
    r1 = _pms_forward(client, plate, "entry", 120)
    assert r1["image_saved"] is True
    assert _sessions_for(registry, plate), "entry must create a confirmed session"
    # Gate-only shot must NOT have seeded the durable folder yet (gate_reference_only).
    assert not os.path.isdir(_plate_dir(image_dir, plate)), (
        "the wide gate-only ANPR shot must not be persisted to the gallery"
    )

    # 2. PMS forwards the CAM-03 in-garage confirmation (direction "B-entry").
    r2 = _pms_forward(client, plate, "B-entry", 200)
    assert r2["image_saved"] is True

    # VA's reaction: the per-plate gallery folder now exists on disk — the exact
    # gap that left DJS-7842 / LLJ-9005 with no folder.
    pdir = _plate_dir(image_dir, plate)
    assert os.path.isdir(pdir), "B-entry must seed gallery/<plate>/"
    assert os.path.isfile(os.path.join(pdir, "meta.json"))
    assert [f for f in os.listdir(pdir) if f.endswith(".jpg")], "expected a crop"


# ── Anti-swap: two cars entering sequentially keep their OWN identity ─────────
def test_two_cars_spaced_keep_own_plate_and_folder():
    clock = Clock()
    client, registry, image_dir = _make_client(clock=clock)

    # Car A's full entry cycle.
    _pms_forward(client, "DJS-7842", "entry", 200)
    _pms_forward(client, "DJS-7842", "B-entry", 200)

    # Real gap before the next car reaches the gate (> the 3s coalesce window).
    clock.advance(15)

    # Car B's full entry cycle.
    _pms_forward(client, "LLJ-9005", "entry", 60)
    _pms_forward(client, "LLJ-9005", "B-entry", 60)

    plates = sorted(s.plate for s in registry._sessions.values())
    assert plates == ["DJS-7842", "LLJ-9005"], (
        "each car must keep its own plate — neither swapped nor left unknown"
    )
    assert os.path.isdir(_plate_dir(image_dir, "DJS-7842"))
    assert os.path.isdir(_plate_dir(image_dir, "LLJ-9005"))


# ── An exit forward creates nothing (behaviour preserved) ─────────────────────
def test_exit_forward_creates_no_session_no_folder():
    client, registry, image_dir = _make_client()
    plate = "EXIT-01"

    r = _pms_forward(client, plate, "exit", 120)
    assert r["status"] == "ok"
    assert _sessions_for(registry, plate) == []
    assert not os.path.isdir(_plate_dir(image_dir, plate))


# ── B-entry with no prior gate entry does not fabricate a folder ──────────────
def test_b_entry_without_prior_entry_is_noop():
    client, registry, image_dir = _make_client()
    plate = "LATE-03"

    # CAM-03 confirmation arrives but the gate "entry" was never seen.
    r = _pms_forward(client, plate, "B-entry", 200)
    assert r["image_saved"] is False, "no session to attach to → nothing saved"
    assert not os.path.isdir(_plate_dir(image_dir, plate))

"""End-to-end: a car LEAVING under a misread entry plate → VA's reaction here.

The exit sibling of ``test_pms_to_va_entry_e2e.py``, and it exists for the same
reason: the two repos run in separate venvs, so neither can import the other
in-process, and the wire payload is the only seam both sides can pin. That file
covers ``/api/anpr/event``; the whole ReID exit path — ``/api/reid/compare`` and
``/api/reid/rename`` — had no equivalent on either side. PMS-AI's exit suite
stubs VA at the function level, so a field rename here would leave every test in
both repos green and silently break plate correction in production.

The exit path matters more than its size suggests. Every entry burst measured
over ai-logs.txt (8/10–8/16) had ``reads=1`` with nothing discarded, and
HikCentral is fed by the SAME entry LPR, so nothing on the entry side can ever
catch a wrong entry plate. The exit read is the first independent look at the
car, and these two endpoints are how that correction reaches VA. If ``rename``
fails, the correction survives in PMS-AI's tables and dies here: the gallery
stays filed under the misread and the next slot update writes it straight back.

The bodies below are byte-for-byte what ``app/utils/va_reid_client.py`` emits.
The matching half is PMS-AI ``tests/test_va_contract.py``, which pins the same
literals from the sending side — the two files must be edited together.
"""

from __future__ import annotations

import base64
import json
import os
import tempfile

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.api import create_app
from src.config import MatchingConfig
from src.vehicle_registry.gallery_store import safe_plate
from src.vehicle_registry.vehicle_registry import VehicleRegistry

# PMS-AI: app/utils/va_reid_client.py
COMPARE_PATH = "/api/reid/compare"
RENAME_PATH = "/api/reid/rename"
SERVICE_KEY_HEADER = "X-Service-Key"

MISREAD = "SDD-6707"
CORRECT = "SDD-6701"


class FakeMatcher:
    """Deterministic stand-in — the feature encodes the image's mean pixel.

    Same device as the entry E2E: this is about the wire contract, not about
    whether the real embedding can tell two synthetic swatches apart.
    """

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


@pytest.fixture
def va(monkeypatch):
    """VA's real HTTP surface with a deterministic matcher behind it."""
    image_dir = tempfile.mkdtemp(prefix="pms_va_exit_e2e_")
    cfg = MatchingConfig()
    cfg.gallery_persist_enabled = True
    # These tests plant galleries on disk rather than admitting them through
    # `add_reference`, so the verified-admission fields (sha256, provenance) are
    # absent and every ref would be filtered out before scoring. The admission
    # rules are `test_gallery_rename.py`'s subject, not this file's.
    cfg.gallery_strict_admission_enabled = False
    cfg.reid_openvino_model_dir = ""
    registry = VehicleRegistry(image_dir=image_dir, matching_config=cfg)
    registry._reid_matcher = FakeMatcher()

    # `/api/reid/compare` reaches for the module-level singleton rather than the
    # registry's matcher, so patching `registry._reid_matcher` alone would leave
    # the endpoint loading a real OpenVINO model.
    import src.reid_matcher.reid_matcher as reid_module

    monkeypatch.setattr(reid_module, "get_reid_matcher", lambda: FakeMatcher())
    monkeypatch.setattr(
        reid_module.VehicleReIDMatcher,
        "compute_similarity",
        staticmethod(FakeMatcher.compute_similarity),
    )

    app = create_app(vehicle_registry=registry, snapshot_base_dir=image_dir)
    client = TestClient(app)
    client.registry = registry
    client.image_dir = image_dir
    return client


def _plant(registry, plate, count=3, tag=None):
    """Put a gallery on disk for `plate`, as an entry would have."""
    store = registry.gallery_store
    tag = tag or store._model_tag
    folder = os.path.join(str(store._root), safe_plate(plate))
    os.makedirs(folder, exist_ok=True)
    refs = []
    for index in range(count):
        stem = f"crop_{index}"
        np.save(os.path.join(folder, stem + ".npy"), np.ones(2, dtype=np.float32))
        with open(os.path.join(folder, stem + ".jpg"), "wb") as fh:
            fh.write(b"jpeg")
        refs.append({
            "crop": stem + ".jpg",
            "vec": stem + ".npy",
            "model_tag": tag,
            "quality": 0.5 + index / 100.0,
            "camera": "CAM-ENTRY",
            "ts": "2026-08-16T09:00:00",
        })
    with open(os.path.join(folder, "meta.json"), "w", encoding="utf-8") as fh:
        json.dump({"plate": plate, "model_tag": tag, "refs": refs}, fh)
    return folder


def _folder(registry, plate):
    return os.path.join(str(registry.gallery_store._root), safe_plate(plate))


def _crop_b64(value=120):
    """A crop encoded exactly as PMS-AI encodes it: raw file bytes → base64."""
    img = np.full((64, 64, 3), value, dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    return base64.b64encode(buf.tobytes()).decode("ascii")


# --- rename ------------------------------------------------------------------

def test_rename_accepts_the_body_pms_ai_actually_sends(va):
    """`{"from": ..., "to": ...}` — the aliases, not the field names.

    `ReIDRenameRequest` declares `from_plate: str = Field(alias="from")` because
    `from` is a Python keyword. Drop the alias and every correction 422s.
    """
    _plant(va.registry, MISREAD)

    response = va.post(
        RENAME_PATH,
        json={"from": MISREAD, "to": CORRECT},
        headers={SERVICE_KEY_HEADER: "svc-key"},
    )

    assert response.status_code == 200, response.text
    body = response.json()
    # PMS-AI logs `response.text` verbatim; these keys are the whole report.
    assert set(body) == {
        "status", "gallery_renamed", "sessions_updated", "slots_updated",
    }
    assert body["gallery_renamed"] is True
    assert os.path.isdir(_folder(va.registry, CORRECT))
    assert not os.path.isdir(_folder(va.registry, MISREAD))


def test_the_unaliased_field_names_are_also_accepted(va):
    """`populate_by_name` makes the alias tolerated on both spellings.

    Worth pinning because it is the opposite of what the alias suggests, and it
    sets how much the sending side has to worry: a PMS-AI change from `from` to
    `from_plate` would NOT break the correction here. What VA does reject is a
    body carrying neither spelling, which is the case that actually matters —
    PMS-AI swallows non-200s by design, so a rejected rename is invisible from
    over there and only ever shows up as a misread VA keeps re-minting.
    """
    _plant(va.registry, MISREAD)

    accepted = va.post(
        RENAME_PATH,
        json={"from_plate": MISREAD, "to_plate": CORRECT},
        headers={SERVICE_KEY_HEADER: "svc-key"},
    )
    assert accepted.status_code == 200, accepted.text
    assert os.path.isdir(_folder(va.registry, CORRECT))

    refused = va.post(
        RENAME_PATH,
        json={"src": CORRECT, "dst": MISREAD},
        headers={SERVICE_KEY_HEADER: "svc-key"},
    )
    assert refused.status_code == 422
    assert os.path.isdir(_folder(va.registry, CORRECT))


def test_rename_replayed_still_answers_200(va):
    """PMS-AI calls this fire-and-forget and retries it from the exit sweep.

    It cannot tell a lost reply from a failed rename, so the second call must be
    a no-op that still succeeds rather than a 404 or a 500.
    """
    _plant(va.registry, MISREAD)
    first = va.post(RENAME_PATH, json={"from": MISREAD, "to": CORRECT},
                    headers={SERVICE_KEY_HEADER: "svc-key"})
    second = va.post(RENAME_PATH, json={"from": MISREAD, "to": CORRECT},
                     headers={SERVICE_KEY_HEADER: "svc-key"})

    assert first.status_code == 200
    assert second.status_code == 200, second.text
    assert os.path.isdir(_folder(va.registry, CORRECT))


def test_a_plate_with_no_gallery_reports_it_rather_than_failing(va):
    """B2 runs `VA_IDENTITY_DISABLED` — 15 of 35 slots produce no plate signal.

    A correction for a car that was never in a gallery is routine, not an error.
    """
    response = va.post(RENAME_PATH, json={"from": "ZZZ-0001", "to": "ZZZ-0002"},
                       headers={SERVICE_KEY_HEADER: "svc-key"})

    assert response.status_code == 200, response.text
    assert response.json()["gallery_renamed"] is False


def test_renaming_a_plate_to_itself_changes_nothing(va):
    _plant(va.registry, MISREAD)
    response = va.post(RENAME_PATH, json={"from": MISREAD, "to": MISREAD},
                       headers={SERVICE_KEY_HEADER: "svc-key"})

    assert response.status_code == 200
    assert response.json()["sessions_updated"] == 0
    assert os.path.isdir(_folder(va.registry, MISREAD))


# --- compare -----------------------------------------------------------------

def test_compare_accepts_the_body_pms_ai_sends_and_returns_what_it_reads(va):
    """`{image_base64, plates}` in; `query_quality_ok` + `results[]` out.

    `exit_match_service` reads exactly `query_quality_ok`, `results[].plate` and
    `results[].score`. Everything else in the payload is VA's own.
    """
    _plant(va.registry, MISREAD)

    response = va.post(
        COMPARE_PATH,
        json={"image_base64": _crop_b64(), "plates": [MISREAD]},
        headers={SERVICE_KEY_HEADER: "svc-key"},
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert "query_quality_ok" in body and "results" in body
    assert isinstance(body["query_quality_ok"], bool)
    entry, = body["results"]
    assert entry["plate"] == MISREAD
    assert isinstance(entry["score"], float)
    assert entry["refs"] == 3


def test_a_plate_with_no_usable_refs_scores_null_not_zero(va):
    """Absence of evidence, and PMS-AI drops it rather than ranking it last.

    A zero would read as "maximally dissimilar" and let a car with no gallery
    lose a comparison it never actually entered.
    """
    _plant(va.registry, MISREAD)

    response = va.post(
        COMPARE_PATH,
        json={"image_base64": _crop_b64(), "plates": [MISREAD, "NO-GALLERY"]},
        headers={SERVICE_KEY_HEADER: "svc-key"},
    )

    body = response.json()
    scores = {r["plate"]: r["score"] for r in body["results"]}
    assert scores["NO-GALLERY"] is None
    assert scores[MISREAD] is not None
    # Nulls sort last, so a real candidate is never displaced by a missing one.
    assert body["results"][-1]["plate"] == "NO-GALLERY"


def test_refs_under_a_stale_model_tag_are_not_scored(va):
    """A cosine distance across two model contracts means nothing.

    BHD-9990 in production carries 4 refs under one tag and 16 under another.
    Scoring vectors directly, as this endpoint does, must drop the stale ones —
    so a plate whose whole gallery is stale answers `null`, not a number.
    """
    _plant(va.registry, MISREAD, tag="openvino:PS_matcher V2:LONG-GONE")

    response = va.post(
        COMPARE_PATH,
        json={"image_base64": _crop_b64(), "plates": [MISREAD]},
        headers={SERVICE_KEY_HEADER: "svc-key"},
    )

    entry, = response.json()["results"]
    assert entry["score"] is None


def test_compare_scores_at_most_twenty_plates(va):
    """The ceiling PMS-AI's `EXIT_MATCH_SHORTLIST` guard is anchored to.

    `payload.plates[:20]` truncates in silence. PMS-AI asserts its shortlist
    stays under this number; this is the measurement that number refers to, so
    raising one without the other fails on one side or the other.
    """
    plates = [f"AAA-{n:04d}" for n in range(35)]
    _plant(va.registry, plates[0])

    response = va.post(
        COMPARE_PATH,
        json={"image_base64": _crop_b64(), "plates": plates},
        headers={SERVICE_KEY_HEADER: "svc-key"},
    )

    assert len(response.json()["results"]) == 20


def test_an_undecodable_image_is_refused_rather_than_scored(va):
    """PMS-AI reads any non-200 as "no appearance evidence" and refuses to match.

    What must not happen is VA scoring garbage and returning a number for it.
    """
    _plant(va.registry, MISREAD)

    response = va.post(
        COMPARE_PATH,
        json={"image_base64": base64.b64encode(b"not-an-image").decode("ascii"),
              "plates": [MISREAD]},
        headers={SERVICE_KEY_HEADER: "svc-key"},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "undecodable_image"


def test_compare_reads_no_gallery_at_all_as_service_unavailable(va, monkeypatch):
    """Gallery disabled is a 503, not an empty result set.

    An empty `results[]` would be indistinguishable from "none of these cars
    look like it", which is a decision VA is not allowed to make.
    """
    monkeypatch.setattr(va.registry, "_gallery_store", None)
    monkeypatch.setattr(
        type(va.registry), "gallery_store", property(lambda self: None)
    )

    response = va.post(
        COMPARE_PATH,
        json={"image_base64": _crop_b64(), "plates": [MISREAD]},
        headers={SERVICE_KEY_HEADER: "svc-key"},
    )

    assert response.status_code == 503
    assert response.json()["detail"] == "gallery_disabled"

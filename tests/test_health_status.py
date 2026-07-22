"""The /api/health verdict must be COMPUTED, not hardcoded.

Before this, ``status`` was the constant string ``"ok"`` — a stopped engine, a frozen
camera (the CAM-24 stale-stream case), or a downed DB all reported green. These tests
pin the derived verdict and the HTTP-503-on-unhealthy contract.
"""

import time
from datetime import datetime, timedelta

from fastapi.testclient import TestClient

from src.api import create_app
from src.camera_manager import CameraConfig, CameraStream
from src.core.engine.engine import ParkingEngine


class _FakeCam:
    def __init__(self, delivering, stale, down, total):
        self._d = (delivering, list(stale), list(down), total)

    @property
    def active_count(self):
        return self._d[0] + len(self._d[1])  # open sockets, frozen included

    @property
    def total_count(self):
        return self._d[3]

    def stream_health(self, max_age_s=15.0):
        d, s, dn, t = self._d
        return {"delivering": d, "stale": list(s), "down": list(dn), "total": t}


class _FakeDB:
    def __init__(self, ok):
        self._ok = ok

    def ping(self):
        return self._ok


class _FakeLocalEntryBridge:
    def __init__(self, *, healthy, last_error):
        self._metrics = {
            "healthy": healthy,
            "last_error": last_error,
            "queue_saturated": not healthy,
        }

    def metrics(self):
        return dict(self._metrics)


def _engine(*, running=True, age_s=1.0, cam=None, db=True, model=True):
    e = ParkingEngine.__new__(ParkingEngine)  # bypass heavy __init__
    e.is_running = running
    e.model_loaded = model
    e.cam_manager = cam
    e.pipelines = {f"c{i}": None for i in range(cam.total_count)} if cam else {}
    e.last_processed_at = (
        datetime.now() - timedelta(seconds=age_s) if age_s is not None else None
    )
    e.start_time = time.time() - 100
    e._frame_count = 1234
    e.db_manager = _FakeDB(db) if db is not None else None
    e._entry_v2_local_bridge = None
    return e


def _status(**kw):
    return _engine(**kw).get_engine_status()["status"]


# --- verdict matrix -------------------------------------------------------- #

def test_all_green_is_ok():
    assert _status(cam=_FakeCam(20, [], [], 20)) == "ok"


def test_frozen_camera_is_degraded_not_ok():
    # The exact regression: a stream open but not delivering (CAM-24) used to read "ok".
    s = _engine(cam=_FakeCam(19, ["CAM-24"], [], 20)).get_engine_status()
    assert s["status"] == "degraded"
    assert "CAM-24" in s["health_reasons"][0]
    assert s["camera_streams_stale"] == ["CAM-24"]


def test_downed_camera_is_degraded():
    assert _status(cam=_FakeCam(19, [], ["CAM-13"], 20)) == "degraded"


def test_every_camera_dark_is_unhealthy():
    assert _status(cam=_FakeCam(0, [], ["a", "b"], 2)) == "unhealthy"


def test_lagging_loop_is_degraded():
    assert _status(age_s=45, cam=_FakeCam(20, [], [], 20)) == "degraded"


def test_wedged_loop_is_unhealthy():
    assert _status(age_s=120, cam=_FakeCam(20, [], [], 20)) == "unhealthy"


def test_stopped_engine_is_unhealthy():
    assert _status(running=False, cam=_FakeCam(20, [], [], 20)) == "unhealthy"


def test_no_frame_ever_processed_is_unhealthy():
    assert _status(age_s=None, cam=_FakeCam(20, [], [], 20)) == "unhealthy"


def test_db_unreachable_is_unhealthy():
    assert _status(cam=_FakeCam(20, [], [], 20), db=False) == "unhealthy"


def test_no_db_configured_is_degraded_not_failed():
    assert _status(cam=_FakeCam(20, [], [], 20), db=None) == "degraded"


def test_model_not_loaded_is_degraded():
    assert _status(cam=_FakeCam(20, [], [], 20), model=False) == "degraded"


def test_local_entry_queue_saturation_degrades_engine_health():
    engine = _engine(cam=_FakeCam(20, [], [], 20))
    engine._entry_v2_local_bridge = _FakeLocalEntryBridge(
        healthy=False,
        last_error="local_zone_queue_capacity_exceeded",
    )

    status = engine.get_engine_status()

    assert status["status"] == "degraded"
    assert status["entry_v2_local_zone"]["queue_saturated"] is True
    assert any(
        "local_zone_queue_capacity_exceeded" in reason
        for reason in status["health_reasons"]
    )


# --- HTTP contract --------------------------------------------------------- #

def test_unhealthy_returns_503():
    app = create_app(get_engine_status=_engine(running=False, cam=_FakeCam(20, [], [], 20)).get_engine_status)
    r = TestClient(app).get("/api/health")
    assert r.status_code == 503
    assert r.json()["status"] == "unhealthy"


def test_degraded_returns_200():
    app = create_app(get_engine_status=_engine(cam=_FakeCam(19, ["CAM-24"], [], 20)).get_engine_status)
    r = TestClient(app).get("/api/health")
    assert r.status_code == 200
    assert r.json()["status"] == "degraded"


def test_healthy_returns_200_ok():
    app = create_app(get_engine_status=_engine(cam=_FakeCam(20, [], [], 20)).get_engine_status)
    r = TestClient(app).get("/api/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_api_only_without_engine_is_liveness_ok():
    # No engine callback wired: the endpoint is a pure liveness probe and stays 200/ok.
    r = TestClient(create_app()).get("/api/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


# --- camera staleness primitive ------------------------------------------- #

def test_seconds_since_frame_reports_infinity_before_first_frame():
    cfg = CameraConfig(id="CAM-TEST", name="t", floor="B1", ip="127.0.0.1",
                       user="u", password="p", slots_file="")
    stream = CameraStream(cfg, max_grab_fps=8)
    assert stream.seconds_since_frame == float("inf")
    stream._latest_ts = time.time()
    assert stream.seconds_since_frame < 1.0

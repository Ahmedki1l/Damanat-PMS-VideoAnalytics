import numpy as np
import pytest

from src.camera_manager import CameraConfig, CameraStream


class _FakeCapture:
    def __init__(self):
        self.grab_calls = 0
        self.retrieve_calls = 0
        self.read_calls = 0
        self.frame = np.zeros((2, 2, 3), dtype=np.uint8)

    def grab(self):
        self.grab_calls += 1
        return True

    def retrieve(self):
        self.retrieve_calls += 1
        return True, self.frame

    def read(self):
        self.read_calls += 1
        return True, self.frame

    def isOpened(self):
        return True


class _Clock:
    def __init__(self):
        self.value = 0.0

    def now(self):
        return self.value


class _TimedCapture(_FakeCapture):
    def __init__(self, clock, stop_event, source_fps, total_frames):
        super().__init__()
        self.clock = clock
        self.stop_event = stop_event
        self.source_interval = 1.0 / source_fps
        self.total_frames = total_frames

    def grab(self):
        self.clock.value += self.source_interval
        ok = super().grab()
        if self.grab_calls >= self.total_frames:
            self.stop_event.set()
        return ok


class _ClosedCapture(_FakeCapture):
    def set(self, *_args):
        return True

    def isOpened(self):
        return False


def _stream(max_grab_fps):
    config = CameraConfig(
        id="CAM-TEST",
        name="test",
        floor="B1",
        ip="127.0.0.1",
        user="user",
        password="password",
        slots_file="",
    )
    stream = CameraStream(config, max_grab_fps=max_grab_fps)
    stream.cap = _FakeCapture()
    return stream


def test_throttled_capture_drains_with_grab_before_materializing():
    stream = _stream(max_grab_fps=8)

    assert stream._grab_frame()
    assert stream._grab_frame()
    ok, frame = stream._retrieve_frame()

    assert ok
    assert frame is stream.cap.frame
    assert stream.cap.grab_calls == 2
    assert stream.cap.retrieve_calls == 1
    assert stream.cap.read_calls == 0


def test_unthrottled_capture_reads_every_frame():
    stream = _stream(max_grab_fps=0)

    ok, frame = stream._read_frame()

    assert ok
    assert frame is stream.cap.frame
    assert stream.cap.read_calls == 1
    assert stream.cap.grab_calls == 0


@pytest.mark.parametrize("source_fps", [6.0, 8.0])
def test_publish_deadline_is_evaluated_after_blocking_grab(
    monkeypatch, source_fps
):
    stream = _stream(max_grab_fps=8)
    clock = _Clock()
    total_frames = 48
    stream.cap = _TimedCapture(
        clock, stream._stop_event, source_fps, total_frames
    )
    monkeypatch.setattr("src.camera_manager.time.perf_counter", clock.now)

    stream._grabber_loop()

    # A source at or below the cap should publish essentially every frame. If
    # the deadline is checked before grab(), this falls to exactly half.
    assert stream.cap.retrieve_calls >= total_frames * 0.9


def test_capture_failure_invalidates_the_last_frame(monkeypatch):
    stream = _stream(max_grab_fps=8)
    stream._latest_frame = stream.cap.frame
    stream._latest_ts = 123.0
    monkeypatch.setattr(stream.cap, "grab", lambda: False)
    monkeypatch.setattr(stream, "_reconnect", stream._stop_event.set)

    stream._grabber_loop()

    assert stream.read_stamped() == (False, None, 0.0, 0)


def test_initial_open_failure_starts_reconnect_worker(monkeypatch):
    stream = _stream(max_grab_fps=8)
    starts = []
    monkeypatch.setattr("src.camera_manager.cv2.VideoCapture", lambda *_a: _ClosedCapture())
    monkeypatch.setattr(stream, "_start_grabber", lambda: starts.append(True))

    assert stream.open() is False
    assert starts == [True]

import cv2
import numpy as np
import pytest

from src.entry.analyzer import ExistingModelsEvidenceProcessor
from src.entry.domain import EvidenceUnavailable, PlateReadState
from src.entry.settings import EntrySettings


class FakeReID:
    def extract_feature(self, frame):
        assert frame.shape == (20, 30, 3)
        return np.array([3.0, 4.0], dtype=np.float32)


class FakePlateDetector:
    def __init__(self, present=True):
        self.present = present

    def crop_plate(self, frame):
        if not self.present:
            return None
        return frame[10:18, 8:22]


class FakeOCR:
    def __init__(self):
        self.kwargs = None

    def read(self, crop, **kwargs):
        assert crop.shape == (8, 14, 3)
        self.kwargs = kwargs
        return "1234ABC", 0.94


class MatchDecision:
    def __init__(self, detector, ocr):
        self.plate_detector = detector
        self.plate_ocr = ocr


class Registry:
    def __init__(self, detector, ocr):
        self.reid_matcher = FakeReID()
        self.match_decision = MatchDecision(detector, ocr)


def jpeg_bytes():
    frame = np.full((20, 30, 3), 127, dtype=np.uint8)
    ok, encoded = cv2.imencode(".jpg", frame)
    assert ok
    return encoded.tobytes()


def test_existing_model_adapter_reuses_components_and_returns_compact_evidence():
    detector = FakePlateDetector()
    ocr = FakeOCR()
    processor = ExistingModelsEvidenceProcessor(
        Registry(detector, ocr), EntrySettings()
    )
    raw = jpeg_bytes()

    result = processor.analyze(
        event_id="event-1",
        camera_id="CAM-23",
        source_role="primary",
        images=(raw,),
        metadata={"image_roles": ["vehicle"]},
    )

    assert result[0].embedding == pytest.approx((0.6, 0.8))
    assert result[0].plate.state == PlateReadState.READABLE
    assert result[0].plate.key == "1234ABC"
    assert ocr.kwargs == {
        "allow_retry": False,
        "apply_plate_roi": False,
        "apply_hud_mask": False,
    }
    assert not hasattr(result[0], "image")
    assert raw not in repr(result).encode("utf-8")


def test_existing_model_adapter_marks_no_lpd_box_as_no_plate_without_ocr():
    ocr = FakeOCR()
    processor = ExistingModelsEvidenceProcessor(
        Registry(FakePlateDetector(present=False), ocr), EntrySettings()
    )
    result = processor.analyze(
        event_id="event-1",
        camera_id="ANPR",
        source_role="anpr",
        images=(jpeg_bytes(),),
        metadata={},
    )
    assert result[0].plate.state == PlateReadState.NO_PLATE
    assert ocr.kwargs is None


def test_existing_model_adapter_rejects_undecodable_image():
    processor = ExistingModelsEvidenceProcessor(
        Registry(FakePlateDetector(), FakeOCR()), EntrySettings()
    )
    with pytest.raises(EvidenceUnavailable, match="no_decodable_image"):
        processor.analyze(
            event_id="event-1",
            camera_id="ANPR",
            source_role="anpr",
            images=(b"not-a-jpeg",),
            metadata={},
        )


def test_existing_model_adapter_rejects_decompression_bomb_before_decode(monkeypatch):
    processor = ExistingModelsEvidenceProcessor(
        Registry(FakePlateDetector(), FakeOCR()),
        EntrySettings(max_decoded_image_pixels=1_000_000),
    )
    # Minimal JPEG header declaring 60,000 x 60,000 pixels. It is intentionally
    # not a complete image: the dimension guard must reject it before OpenCV is
    # ever asked to allocate a decoded frame.
    encoded = (
        b"\xff\xd8"
        b"\xff\xc0\x00\x0b\x08\xea\x60\xea\x60\x01\x01\x11\x00"
        b"\xff\xd9"
    )
    decode_calls = []
    monkeypatch.setattr(cv2, "imdecode", lambda *args: decode_calls.append(args))

    with pytest.raises(EvidenceUnavailable, match="decoded_image_dimensions_exceeded"):
        processor.analyze(
            event_id="event-bomb",
            camera_id="CAM-23",
            source_role="primary",
            images=(encoded,),
            metadata={},
        )

    assert decode_calls == []

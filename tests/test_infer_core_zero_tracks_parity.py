"""The async inference core must emit raw NMS boxes when the tracker returns nothing.

Parity regression found 2026-07-18 while comparing the async core against the
old serial path. Ultralytics' ``on_predict_postprocess_end`` leaves the RAW NMS
boxes on the results when ``tracker.update`` returns zero rows, and the serial
path's ``_parse_results`` emitted them with ``track_id=-1`` — the reason the
slot assigner has always handled ``-1`` by stamping synthetic ids. The async
core's ``_tracks_to_detections`` returned ``[]`` instead, so in the zero-tracks
regime (a lone car whose track has not activated yet, or was just lost) the new
engine saw NOTHING where the old engine saw the car. The loss was invisible to
every fps metric AND to the [CAMDETS] ``raw`` count, which can only see what
this function returns.

These tests pin the fallback so the regression cannot return silently.
"""
import unittest
from types import SimpleNamespace

import numpy as np
import torch

from src.detection.detector import is_untracked
from src.detection.ov_infer_core import OVInferCore


class _FakeBoxes:
    """Minimal stand-in for ultralytics Boxes: xyxy / conf / cls tensors."""

    def __init__(self, rows):  # rows: [x1, y1, x2, y2, conf, cls]
        t = torch.tensor(rows, dtype=torch.float32)
        self.xyxy, self.conf, self.cls = t[:, :4], t[:, 4], t[:, 5]

    def __len__(self):
        return len(self.conf)


def _core_stub(max_box_area_ratio=0.5):
    """The two methods under test, unbound, on a bare stub — constructing a real
    OVInferCore needs a compiled OpenVINO model, which parity of pure conversion
    logic does not."""
    stub = SimpleNamespace(max_box_area_ratio=max_box_area_ratio)
    stub._raw_boxes_to_detections = (
        lambda results, shape: OVInferCore._raw_boxes_to_detections(stub, results, shape)
    )
    return stub


FRAME = (720, 1280, 3)


class TestZeroTracksFallback(unittest.TestCase):
    def test_tracked_rows_still_win(self):
        tracks = np.array([[100, 100, 300, 300, 7, 0.9, 2, 0]])
        dets = OVInferCore._tracks_to_detections(_core_stub(), tracks, FRAME, results=None)
        self.assertEqual([(d.track_id, d.confidence) for d in dets], [(7, 0.9)])

    def test_zero_tracks_falls_back_to_raw_boxes_as_untracked(self):
        """THE REGRESSION: this returned [] before the fix."""
        res = [SimpleNamespace(boxes=_FakeBoxes([[400, 340, 750, 590, 0.41, 2.0]]))]
        dets = OVInferCore._tracks_to_detections(
            _core_stub(), np.empty((0, 8)), FRAME, results=res
        )
        self.assertEqual(len(dets), 1)
        self.assertTrue(is_untracked(dets[0].track_id))
        self.assertEqual(dets[0].bbox, (400.0, 340.0, 750.0, 590.0))
        self.assertEqual(dets[0].class_id, 2)

    def test_none_tracks_and_no_results_is_safe(self):
        self.assertEqual(
            OVInferCore._tracks_to_detections(_core_stub(), None, FRAME), []
        )

    def test_fallback_applies_the_whole_frame_area_guard(self):
        res = [SimpleNamespace(boxes=_FakeBoxes([[0, 0, 1280, 720, 0.9, 2.0]]))]
        dets = OVInferCore._tracks_to_detections(_core_stub(), None, FRAME, results=res)
        self.assertEqual(dets, [], "a whole-frame box must still be rejected")

    def test_fallback_with_empty_boxes_is_safe(self):
        res = [SimpleNamespace(boxes=None)]
        self.assertEqual(
            OVInferCore._tracks_to_detections(_core_stub(), None, FRAME, results=res),
            [],
        )


if __name__ == "__main__":
    unittest.main()

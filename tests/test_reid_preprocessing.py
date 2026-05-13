import unittest
from unittest.mock import patch

import numpy as np

from src.config import ReIDPreprocessingConfig
from src.reid_matcher import reid_matcher
from src.reid_matcher.reid_matcher import _TorchreidBackend
from src.reid_matcher.reid_preprocessing import normalize_for_reid


class TestReIDPreprocessing(unittest.TestCase):
    """Phase 2 cleanup — the legacy ``USE_LAB_CLAHE`` module global is gone;
    the CLAHE toggle now lives on ``ReIDPreprocessingConfig.enabled`` (which
    feeds ``self.preprocessing_config.enabled`` on the torchreid backend).

    The OpenVINO backend has its own preprocessing pipeline; CLAHE is only
    invoked on the legacy ``_TorchreidBackend`` so we exercise that backend
    directly here.
    """

    def test_normalize_none_returns_none(self):
        self.assertIsNone(normalize_for_reid(None))

    def test_normalize_empty_returns_same_empty_image(self):
        empty = np.empty((0, 0, 3), dtype=np.uint8)

        result = normalize_for_reid(empty)

        self.assertIs(result, empty)
        self.assertEqual(result.size, 0)

    def test_normalize_valid_bgr_preserves_shape_and_dtype(self):
        image = np.zeros((24, 32, 3), dtype=np.uint8)
        image[:, :, 0] = 30
        image[:, :, 1] = 80
        image[:, :, 2] = 160

        result = normalize_for_reid(image)

        self.assertEqual(result.shape, image.shape)
        self.assertEqual(result.dtype, image.dtype)

    def test_normalize_does_not_modify_input_in_place(self):
        image = np.zeros((24, 32, 3), dtype=np.uint8)
        image[:, :, 0] = 30
        image[:, :, 1] = 80
        image[:, :, 2] = 160
        original = image.copy()

        normalize_for_reid(image)

        np.testing.assert_array_equal(image, original)

    def test_preprocess_skips_clahe_when_flag_disabled(self):
        backend = self._make_lightweight_backend(enabled=False)
        image = np.zeros((24, 32, 3), dtype=np.uint8)

        with patch.object(
            reid_matcher,
            "normalize_for_reid",
            wraps=normalize_for_reid,
        ) as normalize_mock:
            tensor = backend._preprocess(image)

        normalize_mock.assert_not_called()
        self.assertEqual(tuple(tensor.shape), (3, 128, 256))

    def test_preprocess_applies_clahe_when_flag_enabled(self):
        backend = self._make_lightweight_backend(enabled=True)
        image = np.zeros((24, 32, 3), dtype=np.uint8)

        with patch.object(
            reid_matcher,
            "normalize_for_reid",
            wraps=normalize_for_reid,
        ) as normalize_mock:
            tensor = backend._preprocess(image)

        normalize_mock.assert_called_once()
        self.assertEqual(tuple(tensor.shape), (3, 128, 256))

    @staticmethod
    def _make_lightweight_backend(*, enabled: bool):
        """Build a partial torchreid backend without invoking ``__init__``
        so torchreid weights are never loaded. Only the attributes
        ``_preprocess`` reads are populated.
        """
        backend = _TorchreidBackend.__new__(_TorchreidBackend)
        backend.input_size = (128, 256)
        backend.norm_mean = [0.485, 0.456, 0.406]
        backend.norm_std = [0.229, 0.224, 0.225]
        # New API: the CLAHE toggle is read from
        # ``self.preprocessing_config.enabled`` rather than the legacy
        # ``reid_matcher.USE_LAB_CLAHE`` module global (Phase 2 cleanup).
        backend.preprocessing_config = ReIDPreprocessingConfig(enabled=enabled)
        return backend


if __name__ == "__main__":
    unittest.main()

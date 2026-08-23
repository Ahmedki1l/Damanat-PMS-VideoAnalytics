"""Per-camera RTSP stream selection.

`processing.stream_channel` is a single engine-wide value AND it is DB-owned --
sync_app_config_from_db rewrites it from the Config table -- so "everything on
main except Ground" cannot be expressed by editing the global. These cover the
YAML-only override map that can, and which survives that sync because the DB has
no column for it.
"""

import pytest

from src.config import (
    ProcessingConfig,
    _parse_processing_overrides,
)


def staged(**overrides):
    config = ProcessingConfig()
    config.stream_channel = 101
    config.camera_overrides = _parse_processing_overrides(overrides)
    return config


class TestResolution:
    def test_an_unlisted_camera_keeps_the_global_stream(self):
        config = staged(**{"CAM-00": {"stream_channel": 102}})
        assert config.resolve_for_camera("CAM-03").stream_channel == 101

    def test_a_listed_camera_gets_its_own_stream(self):
        config = staged(**{"CAM-00": {"stream_channel": 102}})
        assert config.resolve_for_camera("CAM-00").stream_channel == 102

    def test_ground_on_sub_leaves_every_parking_camera_on_main(self):
        config = staged(**{
            "CAM-00": {"stream_channel": 102},
            "CAM-01": {"stream_channel": 102},
            "CAM-02": {"stream_channel": 102},
        })
        ground = ["CAM-00", "CAM-01", "CAM-02"]
        parking = [f"CAM-{n:02d}" for n in range(3, 15)]
        assert all(
            config.resolve_for_camera(c).stream_channel == 102 for c in ground
        )
        assert all(
            config.resolve_for_camera(c).stream_channel == 101 for c in parking
        )

    def test_resolving_never_mutates_the_shared_config(self):
        config = staged(**{"CAM-00": {"stream_channel": 102}})
        config.resolve_for_camera("CAM-00")
        assert config.stream_channel == 101
        assert config.camera_overrides

    def test_the_resolved_copy_carries_no_override_map(self):
        """Otherwise a second resolve off the copy could re-apply an override."""
        config = staged(**{"CAM-00": {"stream_channel": 102}})
        assert config.resolve_for_camera("CAM-00").camera_overrides == {}

    @pytest.mark.parametrize("spelling", ["CAM-00", "cam_00", "cam00", "Cam-00"])
    def test_camera_ids_match_however_they_are_spelled(self, spelling):
        config = staged(**{"CAM-00": {"stream_channel": 102}})
        assert config.resolve_for_camera(spelling).stream_channel == 102


class TestValidation:
    """A typo must RAISE. Failing open leaves a camera silently on the wrong
    stream, which is invisible until someone reads a resolution off the boot
    log -- the same reasoning as _parse_assigner_overrides."""

    def test_an_unknown_key_raises(self):
        with pytest.raises(ValueError, match="unknown key"):
            _parse_processing_overrides({"CAM-00": {"stream_chanel": 102}})

    def test_an_unknown_channel_raises(self):
        with pytest.raises(ValueError, match="101"):
            _parse_processing_overrides({"CAM-00": {"stream_channel": 103}})

    def test_a_non_integer_channel_raises(self):
        with pytest.raises(ValueError, match="integer stream id"):
            _parse_processing_overrides({"CAM-00": {"stream_channel": "main"}})

    def test_a_numeric_string_is_accepted_as_yaml_writes_it(self):
        parsed = _parse_processing_overrides({"CAM-00": {"stream_channel": "102"}})
        assert parsed["CAM00"]["stream_channel"] == 102

    def test_an_empty_entry_is_dropped_rather_than_stored(self):
        assert _parse_processing_overrides({"CAM-00": {}}) == {}

    def test_no_overrides_at_all_is_fine(self):
        assert _parse_processing_overrides({}) == {}
        assert _parse_processing_overrides(None) == {}

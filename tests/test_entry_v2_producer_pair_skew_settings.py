"""Configuration contract for Entry V2 producer-pair deduplication."""

from dataclasses import replace

import pytest

from src.entry.domain import EntryMode
from src.entry.settings import EntrySettings


SKEW_ENV_NAME = "ENTRY_V2_PRODUCER_PAIR_MAX_SKEW_SECONDS"
SCORE_ENV_NAME = "ENTRY_V2_PRODUCER_PAIR_MIN_REID_SCORE"


def _active_settings(**overrides) -> EntrySettings:
    base = EntrySettings(
        mode=EntryMode.SHADOW,
        primary_lines=frozenset({"RAMP-IN"}),
        primary_directions=frozenset({"b-to-a"}),
        pms_base_url="http://pms-ai:8080",
        service_key="secret",
    )
    return replace(base, **overrides)


def test_producer_pair_skew_defaults_to_five_seconds(monkeypatch):
    monkeypatch.delenv(SKEW_ENV_NAME, raising=False)

    assert EntrySettings().producer_pair_max_skew_seconds == pytest.approx(5.0)
    assert EntrySettings.from_env().producer_pair_max_skew_seconds == pytest.approx(5.0)


def test_producer_pair_skew_reads_environment_override(monkeypatch):
    monkeypatch.setenv(SKEW_ENV_NAME, "12.5")

    assert EntrySettings.from_env().producer_pair_max_skew_seconds == pytest.approx(
        12.5
    )


@pytest.mark.parametrize(
    "value", [0.0, -0.001, float("nan"), float("inf"), -float("inf")]
)
def test_active_configuration_rejects_non_positive_or_non_finite_skew(value):
    settings = _active_settings(producer_pair_max_skew_seconds=value)

    assert "producer_pair_max_skew_seconds" in settings.configuration_errors()


@pytest.mark.parametrize("value", [float.fromhex("0x0.0000000000001p-1022"), 5.0])
def test_active_configuration_accepts_any_finite_positive_skew(value):
    settings = _active_settings(producer_pair_max_skew_seconds=value)

    assert "producer_pair_max_skew_seconds" not in settings.configuration_errors()


@pytest.mark.parametrize("raw_value", ["0", "-1", "nan", "inf", "-inf", "five-seconds"])
def test_invalid_environment_skew_fails_active_configuration(monkeypatch, raw_value):
    monkeypatch.setenv(SKEW_ENV_NAME, raw_value)
    settings = replace(
        EntrySettings.from_env(),
        mode=EntryMode.SHADOW,
        primary_lines=frozenset({"RAMP-IN"}),
        primary_directions=frozenset({"b-to-a"}),
        pms_base_url="http://pms-ai:8080",
        service_key="secret",
    )

    assert "producer_pair_max_skew_seconds" in settings.configuration_errors()


def test_producer_pair_reid_score_defaults_to_strict_threshold(monkeypatch):
    monkeypatch.delenv(SCORE_ENV_NAME, raising=False)

    assert EntrySettings().producer_pair_min_reid_score == pytest.approx(0.95)
    assert EntrySettings.from_env().producer_pair_min_reid_score == pytest.approx(0.95)


def test_producer_pair_reid_score_reads_environment_override(monkeypatch):
    monkeypatch.setenv(SCORE_ENV_NAME, "0.97")

    assert EntrySettings.from_env().producer_pair_min_reid_score == pytest.approx(0.97)


@pytest.mark.parametrize(
    "value", [-1.0001, 1.0001, float("nan"), float("inf"), -float("inf")]
)
def test_active_configuration_rejects_out_of_range_or_non_finite_pair_score(value):
    settings = _active_settings(producer_pair_min_reid_score=value)

    assert "producer_pair_min_reid_score" in settings.configuration_errors()


@pytest.mark.parametrize(
    ("event_score", "merge_score", "pair_score", "valid"),
    [
        (0.82, 0.82, 0.8999, False),
        (0.82, 0.82, 0.90, True),
        (0.96, 0.82, 0.9599, False),
        (0.96, 0.82, 0.96, True),
        (0.82, 0.97, 0.9699, False),
        (0.82, 0.97, 0.97, True),
        (0.82, 0.82, 1.0, True),
    ],
)
def test_pair_score_meets_static_and_configured_safety_floors(
    event_score, merge_score, pair_score, valid
):
    settings = _active_settings(
        event_consistency_min_score=event_score,
        merge_min_score=merge_score,
        producer_pair_min_reid_score=pair_score,
    )

    errors = settings.configuration_errors()
    assert ("producer_pair_min_reid_score" not in errors) is valid


@pytest.mark.parametrize(
    "raw_value", ["0.89", "-1", "1.0001", "nan", "inf", "very-high"]
)
def test_invalid_environment_pair_score_fails_active_configuration(
    monkeypatch, raw_value
):
    monkeypatch.setenv(SCORE_ENV_NAME, raw_value)
    settings = replace(
        EntrySettings.from_env(),
        mode=EntryMode.SHADOW,
        primary_lines=frozenset({"RAMP-IN"}),
        primary_directions=frozenset({"b-to-a"}),
        pms_base_url="http://pms-ai:8080",
        service_key="secret",
    )

    assert "producer_pair_min_reid_score" in settings.configuration_errors()


def _set_active_local_zone_environment(monkeypatch, *, mode="shadow") -> None:
    monkeypatch.setenv("ENTRY_V2_MODE", mode)
    monkeypatch.setenv("ENTRY_V2_PRIMARY_CAMERAS", "CAM-23")
    monkeypatch.setenv("ENTRY_V2_PRIMARY_LINES", "Park_Entry")
    monkeypatch.setenv("ENTRY_V2_PRIMARY_DIRECTIONS", "ramp-entry")
    monkeypatch.setenv("PMS_API_URL", "http://pms-ai:8080")
    monkeypatch.setenv("ENTRY_V2_SERVICE_KEY", "secret")
    # The supervisor's co-location attestation is the OTHER way to satisfy this
    # guard, so these tests must start from neither route being available or
    # they would pass for the wrong reason.
    monkeypatch.delenv("VA_ENTRY_HOST", raising=False)
    monkeypatch.delenv("VA_GROUP_CAMERAS", raising=False)
    if mode == "authoritative":
        monkeypatch.setenv("VA_PROCESS_COUNT", "1")


@pytest.mark.parametrize("raw_value", ["1", "true", " TRUE ", "yes", "On"])
@pytest.mark.parametrize("mode", ["shadow", "authoritative"])
def test_canonical_local_zone_accepts_explicit_true_single_process_values(
    monkeypatch, raw_value, mode
):
    _set_active_local_zone_environment(monkeypatch, mode=mode)
    monkeypatch.setenv("VA_SINGLE_PROCESS", raw_value)

    settings = EntrySettings.from_env()

    assert settings.va_single_process is True
    assert (
        "entry_v2_local_zone_requires_single_process_or_gate_group"
        not in settings.configuration_errors()
    )


@pytest.mark.parametrize(
    "raw_value", [None, "", "0", "false", "no", "off", "enabled", "2"]
)
def test_canonical_local_zone_rejects_missing_false_or_unknown_single_process(
    monkeypatch, raw_value
):
    _set_active_local_zone_environment(monkeypatch)
    if raw_value is None:
        monkeypatch.delenv("VA_SINGLE_PROCESS", raising=False)
    else:
        monkeypatch.setenv("VA_SINGLE_PROCESS", raw_value)

    settings = EntrySettings.from_env()

    assert settings.va_single_process is False
    assert (
        "entry_v2_local_zone_requires_single_process_or_gate_group"
        in settings.configuration_errors()
    )


def test_canonical_cam03_fallback_also_requires_single_process(monkeypatch):
    _set_active_local_zone_environment(monkeypatch)
    monkeypatch.setenv("ENTRY_V2_PRIMARY_LINES", "HIKVISION-IN")
    monkeypatch.setenv("ENTRY_V2_FALLBACK_CAMERAS", "CAM-03")
    monkeypatch.setenv("ENTRY_V2_FALLBACK_LINES", "B1_Entrence")
    monkeypatch.setenv("ENTRY_V2_FALLBACK_DIRECTIONS", "b-entry")
    monkeypatch.delenv("VA_SINGLE_PROCESS", raising=False)

    settings = EntrySettings.from_env()

    assert (
        "entry_v2_local_zone_requires_single_process_or_gate_group"
        in settings.configuration_errors()
    )


def test_hikvision_only_path_does_not_require_single_process_attestation(monkeypatch):
    _set_active_local_zone_environment(monkeypatch)
    monkeypatch.setenv("ENTRY_V2_PRIMARY_LINES", "HIKVISION-IN")
    monkeypatch.delenv("VA_SINGLE_PROCESS", raising=False)

    settings = EntrySettings.from_env()

    assert (
        "entry_v2_local_zone_requires_single_process_or_gate_group"
        not in settings.configuration_errors()
    )


# ---------------------------------------------------------------------------
# BUILD 5 co-location attestation.
#
# VA_SINGLE_PROCESS is an ENGINE switch (main.py forces VA_INFER=async and calls
# engine.run_single_process()), not an attestation. On the multi-process
# supervisor it is deliberately empty, so requiring it would have forced
# operators to choose between a working Entry V2 and a working engine. The
# supervisor reports the real property instead: which group serves --api, and
# which cameras that group was given.
# ---------------------------------------------------------------------------


def _attest(monkeypatch, *, api: bool, cameras: str) -> None:
    if api:
        monkeypatch.setenv("VA_ENTRY_HOST", "1")
    else:
        monkeypatch.delenv("VA_ENTRY_HOST", raising=False)
    monkeypatch.setenv("VA_GROUP_CAMERAS", cameras)


@pytest.mark.parametrize("mode", ["shadow", "authoritative"])
def test_gate_group_attestation_satisfies_local_zone_without_single_process(
    monkeypatch, mode
):
    """The BUILD 5 gate worker: --api plus both gate cameras in one process."""
    _set_active_local_zone_environment(monkeypatch, mode=mode)
    monkeypatch.setenv("ENTRY_V2_FALLBACK_CAMERAS", "CAM-03")
    monkeypatch.setenv("ENTRY_V2_FALLBACK_LINES", "1,B1_Entrence")
    monkeypatch.setenv("ENTRY_V2_FALLBACK_DIRECTIONS", "b-entry")
    monkeypatch.delenv("VA_SINGLE_PROCESS", raising=False)
    _attest(monkeypatch, api=True, cameras="CAM-23,CAM-03")

    settings = EntrySettings.from_env()

    assert settings.va_single_process is False
    assert settings.local_zone_is_co_located() is True
    assert (
        "entry_v2_local_zone_requires_single_process_or_gate_group"
        not in settings.configuration_errors()
    )


def test_non_api_worker_is_never_attested(monkeypatch):
    """A floor worker owns no gate camera and serves no HTTP: correctly disabled."""
    _set_active_local_zone_environment(monkeypatch)
    monkeypatch.delenv("VA_SINGLE_PROCESS", raising=False)
    _attest(monkeypatch, api=False, cameras="CAM-09,CAM-10,CAM-11")

    settings = EntrySettings.from_env()

    assert settings.local_zone_is_co_located() is False
    assert (
        "entry_v2_local_zone_requires_single_process_or_gate_group"
        in settings.configuration_errors()
    )


def test_api_worker_missing_a_local_zone_camera_fails_closed(monkeypatch):
    """VA_GATE_CAMERAS emptied: CAM-03 falls back to an area group.

    This is the split that manufactures dropped entries -- two coordinators,
    one witness each, the two-witness rule permanently unsatisfiable. It must
    stay caught, which is the whole reason the group's camera list is verified
    rather than the --api flag being trusted on its own.
    """
    _set_active_local_zone_environment(monkeypatch)
    monkeypatch.setenv("ENTRY_V2_FALLBACK_CAMERAS", "CAM-03")
    monkeypatch.setenv("ENTRY_V2_FALLBACK_LINES", "1,B1_Entrence")
    monkeypatch.setenv("ENTRY_V2_FALLBACK_DIRECTIONS", "b-entry")
    monkeypatch.delenv("VA_SINGLE_PROCESS", raising=False)
    _attest(monkeypatch, api=True, cameras="CAM-23,CAM-08")

    settings = EntrySettings.from_env()

    assert settings.local_zone_cameras() == frozenset({"CAM23", "CAM03"})
    assert settings.local_zone_is_co_located() is False
    assert (
        "entry_v2_local_zone_requires_single_process_or_gate_group"
        in settings.configuration_errors()
    )


def test_attestation_without_group_cameras_fails_closed(monkeypatch):
    """--api alone proves nothing about which cameras this process owns."""
    _set_active_local_zone_environment(monkeypatch)
    monkeypatch.delenv("VA_SINGLE_PROCESS", raising=False)
    monkeypatch.setenv("VA_ENTRY_HOST", "1")
    monkeypatch.delenv("VA_GROUP_CAMERAS", raising=False)

    settings = EntrySettings.from_env()

    assert settings.local_zone_is_co_located() is False
    assert (
        "entry_v2_local_zone_requires_single_process_or_gate_group"
        in settings.configuration_errors()
    )


def test_only_the_enabled_local_zone_camera_must_be_co_located(monkeypatch):
    """CAM-03 reaching Entry V2 over HTTP imposes no topology requirement.

    Its crossings land in the --api process by construction, so only CAM-23's
    RTSP zone constrains which group may host the coordinator.
    """
    _set_active_local_zone_environment(monkeypatch)
    monkeypatch.setenv("ENTRY_V2_FALLBACK_CAMERAS", "CAM-03")
    monkeypatch.setenv("ENTRY_V2_FALLBACK_LINES", "1")  # no B1_Entrence zone
    monkeypatch.setenv("ENTRY_V2_FALLBACK_DIRECTIONS", "b-to-a")
    monkeypatch.delenv("VA_SINGLE_PROCESS", raising=False)
    _attest(monkeypatch, api=True, cameras="CAM-23")

    settings = EntrySettings.from_env()

    assert settings.local_zone_cameras() == frozenset({"CAM23"})
    assert settings.local_zone_is_co_located() is True
    assert (
        "entry_v2_local_zone_requires_single_process_or_gate_group"
        not in settings.configuration_errors()
    )


def test_single_process_still_satisfies_the_guard_on_build_4(monkeypatch):
    """The BUILD 4 route is preserved, not replaced."""
    _set_active_local_zone_environment(monkeypatch)
    monkeypatch.setenv("VA_SINGLE_PROCESS", "1")
    _attest(monkeypatch, api=False, cameras="")

    settings = EntrySettings.from_env()

    assert settings.local_zone_is_co_located() is True
    assert (
        "entry_v2_local_zone_requires_single_process_or_gate_group"
        not in settings.configuration_errors()
    )

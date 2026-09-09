"""AUTHORITATIVE must accept the supervisor's attestation, not a process count.

Regression for the 2026-09-09 outage. ENTRY_V2_MODE was set to authoritative on
a healthy 4-group pod (gate / ground / b1 / b2). supervisor.py OVERWRITES
VA_PROCESS_COUNT in every worker with the group count, so it read 4, the old
gate demanded exactly 1, and VA degraded to DisabledEvidenceProcessor. From
then on every ingest answered

    503 {"detail":"entry_v2_invalid_configuration:entry_v2_requires_single_process_va"}

and PMS-AI returned a camera-facing 503 BEFORE legacy dispatch could run --
with the legacy burst flusher already disabled by authoritative mode, nothing
recorded entries at all for ~4h. Three cars entered; the HikCentral reconciler
opened two of them 33 and 64 minutes late and missed the third entirely.

The property being protected is real (Entry V2 holds identities and crossings
in RAM, so one process must own them), but a process count never measured it.
The supervisor states it directly with VA_ENTRY_HOST on the one --api group.
"""
import pytest

from src.entry.settings import EntrySettings

BASE = {
    "ENTRY_V2_MODE": "authoritative",
    "ENTRY_V2_PRIMARY_LINES": "RAMP-IN",
    "PMS_API_URL": "http://pms-ai:8080",
    "ENTRY_V2_SERVICE_KEY": "secret",
}

TOPOLOGY_KEYS = (
    "VA_PROCESS_COUNT",
    "VA_SINGLE_PROCESS",
    "VA_ENTRY_HOST",
    "VA_GROUP_CAMERAS",
)


def _settings(monkeypatch, **env):
    for key in TOPOLOGY_KEYS:
        monkeypatch.delenv(key, raising=False)
    for key, value in {**BASE, **env}.items():
        monkeypatch.setenv(key, value)
    return EntrySettings.from_env()


# --------------------------------------------------------------------------- #
# The outage
# --------------------------------------------------------------------------- #
def test_the_four_group_gate_worker_is_accepted(monkeypatch):
    """The exact configuration that took entry recording down."""
    cfg = _settings(
        monkeypatch,
        VA_PROCESS_COUNT="4",          # supervisor: len(groups)
        VA_ENTRY_HOST="1",             # supervisor: the one --api group
        VA_GROUP_CAMERAS="CAM23,CAM03",
    )

    errors = cfg.configuration_errors()
    assert "entry_v2_requires_single_process_va" not in errors
    assert "VA_PROCESS_COUNT" not in errors
    assert cfg.coordinator_is_single_hosted()


def test_a_non_api_worker_in_the_same_pod_still_degrades(monkeypatch):
    """Correct: it receives no ingest and cannot host the state anyway."""
    cfg = _settings(
        monkeypatch, VA_PROCESS_COUNT="4", VA_GROUP_CAMERAS="CAM11,CAM12"
    )

    assert "entry_v2_requires_single_process_va" in cfg.configuration_errors()
    assert not cfg.coordinator_is_single_hosted()


def test_the_group_count_alone_never_attests_either_way(monkeypatch):
    """4 groups + VA_ENTRY_HOST passes; 4 groups without it does not."""
    host = _settings(
        monkeypatch, VA_PROCESS_COUNT="4", VA_ENTRY_HOST="1"
    ).coordinator_is_single_hosted()
    bare = _settings(
        monkeypatch, VA_PROCESS_COUNT="4"
    ).coordinator_is_single_hosted()

    assert host is True
    assert bare is False


# --------------------------------------------------------------------------- #
# Nothing that worked before may stop working
# --------------------------------------------------------------------------- #
def test_a_bare_single_process_still_passes_on_its_declared_count(monkeypatch):
    """`python main.py --api` with no supervisor sets neither attestation."""
    cfg = _settings(monkeypatch, VA_PROCESS_COUNT="1")

    assert "entry_v2_requires_single_process_va" not in cfg.configuration_errors()


def test_va_single_process_still_passes(monkeypatch):
    """The BUILD 4 engine switch remains a valid attestation."""
    cfg = _settings(monkeypatch, VA_SINGLE_PROCESS="1", VA_PROCESS_COUNT="4")

    assert "entry_v2_requires_single_process_va" not in cfg.configuration_errors()


def test_shadow_is_unaffected_by_any_of_this(monkeypatch):
    cfg = _settings(
        monkeypatch, ENTRY_V2_MODE="shadow", VA_PROCESS_COUNT="4"
    )

    assert "entry_v2_requires_single_process_va" not in cfg.configuration_errors()


# --------------------------------------------------------------------------- #
# Fail-closed is preserved
# --------------------------------------------------------------------------- #
def test_multi_process_without_any_attestation_still_fails(monkeypatch):
    cfg = _settings(monkeypatch, VA_PROCESS_COUNT="2")

    assert "entry_v2_requires_single_process_va" in cfg.configuration_errors()


@pytest.mark.parametrize("raw_count", [None, "", "two"])
def test_an_unusable_count_with_no_attestation_names_both_faults(
    monkeypatch, raw_count
):
    env = {} if raw_count is None else {"VA_PROCESS_COUNT": raw_count}
    cfg = _settings(monkeypatch, **env)

    errors = cfg.configuration_errors()
    assert "VA_PROCESS_COUNT" in errors
    assert "entry_v2_requires_single_process_va" in errors


def test_a_malformed_count_is_not_mistaken_for_a_single_process(monkeypatch):
    """_env parsing defaults a bad count to 1; that must not attest."""
    cfg = _settings(monkeypatch, VA_PROCESS_COUNT="two")

    assert cfg.va_process_count == 1          # the parsed fallback
    assert cfg.invalid_va_process_count == "two"
    assert not cfg.coordinator_is_single_hosted()


# --------------------------------------------------------------------------- #
# The local-zone check stays strictly stronger
# --------------------------------------------------------------------------- #
def test_local_zone_still_requires_its_cameras_in_this_group(monkeypatch):
    """Co-location keeps the camera-subset test the coordinator check omits.

    HTTP ingest imposes no camera requirement -- it always lands in the --api
    process -- but the RTSP bridge reads cameras THIS worker owns, so a gate
    camera missing from the group must still fail.
    """
    cfg = _settings(
        monkeypatch,
        VA_PROCESS_COUNT="4",
        VA_ENTRY_HOST="1",
        VA_GROUP_CAMERAS="CAM11,CAM12",   # gate camera absent
        ENTRY_V2_PRIMARY_CAMERAS="CAM23",
        ENTRY_V2_PRIMARY_LINES="PARK_ENTRY",
        ENTRY_V2_LOCAL_ZONE_CAM23="1",
    )

    # Asserted unconditionally: a guard here would pass even if the local zone
    # silently stopped being enabled and the test stopped testing anything.
    assert cfg.local_zone_cameras() == frozenset({"CAM23"})
    assert cfg.coordinator_is_single_hosted()
    assert not cfg.local_zone_is_co_located()
    assert (
        "entry_v2_local_zone_requires_single_process_or_gate_group"
        in cfg.configuration_errors()
    )

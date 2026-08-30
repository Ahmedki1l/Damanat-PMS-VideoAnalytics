"""The supervisor's half of the Entry V2 co-location contract.

``src/entry/settings.py`` accepts ``VA_ENTRY_HOST`` + ``VA_GROUP_CAMERAS`` as an
alternative to ``VA_SINGLE_PROCESS``. That is only sound if the supervisor
actually emits them, and emits them for exactly one group -- so both halves are
tested, and the end-to-end pairing is asserted here rather than assumed.
"""

import supervisor
from src.entry.settings import EntrySettings


GATE = {"name": "gate", "cams": "CAM-23,CAM-03", "api": True}
FLOOR = {"name": "b2-1", "cams": "CAM-09,CAM-10", "api": False}


def test_api_group_is_attested_with_its_camera_list():
    env = supervisor._entry_v2_topology_env(GATE)

    assert env["VA_ENTRY_HOST"] == "1"
    assert env["VA_GROUP_CAMERAS"] == "CAM-23,CAM-03"


def test_non_api_group_reports_cameras_but_no_attestation():
    env = supervisor._entry_v2_topology_env(FLOOR)

    assert "VA_ENTRY_HOST" not in env
    assert env["VA_GROUP_CAMERAS"] == "CAM-09,CAM-10"


def test_exactly_one_group_in_a_real_topology_is_attested(monkeypatch):
    """Two attested groups would mean two coordinators racing for one gate."""
    monkeypatch.setenv("VA_STATIC_GROUPS", "1")
    groups = supervisor.resolve_groups()

    attested = [
        g for g in groups if "VA_ENTRY_HOST" in supervisor._entry_v2_topology_env(g)
    ]

    assert len(groups) > 1, "static topology should be multi-group"
    assert len(attested) == 1
    assert "CAM-23" in attested[0]["cams"] and "CAM-03" in attested[0]["cams"]


def test_attestation_satisfies_the_settings_guard_end_to_end(monkeypatch):
    """The pairing that matters: what the supervisor emits unblocks shadow.

    Asserted across the module boundary because each half is individually
    plausible and still useless if the variable names drift apart.
    """
    monkeypatch.setenv("ENTRY_V2_MODE", "shadow")
    monkeypatch.setenv("ENTRY_V2_PRIMARY_CAMERAS", "CAM-23")
    monkeypatch.setenv("ENTRY_V2_PRIMARY_LINES", "1,Park_Entry")
    monkeypatch.setenv("ENTRY_V2_PRIMARY_DIRECTIONS", "ramp-entry")
    monkeypatch.setenv("ENTRY_V2_FALLBACK_CAMERAS", "CAM-03")
    monkeypatch.setenv("ENTRY_V2_FALLBACK_LINES", "1,B1_Entrence")
    monkeypatch.setenv("ENTRY_V2_FALLBACK_DIRECTIONS", "B-to-A,b-entry")
    monkeypatch.setenv("PMS_API_URL", "http://pms-ai:8080")
    monkeypatch.setenv("ENTRY_V2_SERVICE_KEY", "secret")
    monkeypatch.delenv("VA_SINGLE_PROCESS", raising=False)

    for name, value in supervisor._entry_v2_topology_env(GATE).items():
        monkeypatch.setenv(name, value)

    assert EntrySettings.from_env().configuration_errors() == []


def test_floor_worker_is_correctly_left_disabled(monkeypatch):
    """Not a bug: a worker with no gate camera cannot host the coordinator."""
    monkeypatch.setenv("ENTRY_V2_MODE", "shadow")
    monkeypatch.setenv("ENTRY_V2_PRIMARY_CAMERAS", "CAM-23")
    monkeypatch.setenv("ENTRY_V2_PRIMARY_LINES", "1,Park_Entry")
    monkeypatch.setenv("ENTRY_V2_PRIMARY_DIRECTIONS", "ramp-entry")
    monkeypatch.setenv("PMS_API_URL", "http://pms-ai:8080")
    monkeypatch.setenv("ENTRY_V2_SERVICE_KEY", "secret")
    monkeypatch.delenv("VA_SINGLE_PROCESS", raising=False)
    monkeypatch.delenv("VA_ENTRY_HOST", raising=False)

    for name, value in supervisor._entry_v2_topology_env(FLOOR).items():
        monkeypatch.setenv(name, value)

    assert (
        "entry_v2_local_zone_requires_single_process_or_gate_group"
        in EntrySettings.from_env().configuration_errors()
    )

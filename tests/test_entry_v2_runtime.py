from src.entry import runtime
from src.entry.domain import EntryMode
from src.entry.settings import EntrySettings


class _Registry:
    pass


class _ProcessorSpy:
    received = None

    def __init__(self, registry, settings, *, image_dir):
        type(self).received = {
            "registry": registry,
            "settings": settings,
            "image_dir": image_dir,
        }


def test_runtime_passes_configured_image_root_to_evidence_processor(
    monkeypatch, tmp_path
):
    settings = EntrySettings(
        mode=EntryMode.SHADOW,
        primary_lines=frozenset({"PARK_ENTRY"}),
        primary_directions=frozenset({"ramp-entry"}),
        pms_base_url="http://pms-ai:8080",
        service_key="test-key",
        va_single_process=True,
    )
    registry = _Registry()
    configured_image_root = tmp_path / "configured-images"
    monkeypatch.setattr(
        runtime.EntrySettings,
        "from_env",
        staticmethod(lambda: settings),
    )
    monkeypatch.setattr(runtime, "ExistingModelsEvidenceProcessor", _ProcessorSpy)

    coordinator = runtime.build_entry_coordinator(
        registry,
        image_dir=str(configured_image_root),
    )

    assert coordinator.available is True
    assert _ProcessorSpy.received == {
        "registry": registry,
        "settings": settings,
        "image_dir": str(configured_image_root),
    }

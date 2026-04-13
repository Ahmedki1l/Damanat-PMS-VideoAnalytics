import json
import os
from urllib import error, request


def _base_url() -> str:
    return os.getenv("PMS_API_URL", "").rstrip("/")


def _post(path: str, payload: dict) -> None:
    base_url = _base_url()
    if not base_url:
        return

    body = json.dumps(payload).encode("utf-8")
    req = request.Request(
        f"{base_url}{path}",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=5):
            return
    except error.URLError as exc:
        print(f"[PMS API] Failed POST {path}: {exc}")


def bind_slot_session(
    plate_number: str,
    slot_id: str,
    slot_number: str,
    zone_id: str | None,
    zone_name: str | None,
    floor: str | None,
    camera_id: str | None,
    parked_at: str | None,
) -> None:
    _post("/api/v1/internal/parking-sessions/bind-slot", {
        "plate_number": plate_number,
        "slot_id": slot_id,
        "slot_number": slot_number,
        "zone_id": zone_id,
        "zone_name": zone_name,
        "floor": floor,
        "camera_id": camera_id or "UNKNOWN",
        "parked_at": parked_at,
        "snapshot_path": None,
    })


def unbind_slot_session(
    plate_number: str,
    slot_id: str,
    slot_number: str,
    camera_id: str | None,
    left_at: str | None,
) -> None:
    _post("/api/v1/internal/parking-sessions/unbind-slot", {
        "plate_number": plate_number,
        "slot_id": slot_id,
        "slot_number": slot_number,
        "camera_id": camera_id or "UNKNOWN",
        "left_at": left_at,
        "snapshot_path": None,
    })

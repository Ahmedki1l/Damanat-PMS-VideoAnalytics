"""
Named/reserved slot helpers.

All logic is driven by the `reservation_type` and `reserved_for` columns on
the `parking_slots` DB table. Functions accept a ParkingSlot ORM object so
callers never maintain an external mapping.

reservation_type values:
  "GENERAL"  — normal public parking
  "EMPLOYEE" — named slot reserved for a specific person (reserved_for = vehicles.title)
  "SPECIAL"  — special-needs / handicap slot
"""
import re


def _norm_title(value: str | None) -> str:
    """Comparison key for a title / reservation label: lowercase with every
    non-alphanumeric character dropped, so `"C.E.O "` and `"ceo"` both become
    `"ceo"`.

    Must stay byte-identical in behaviour to the Gateway's
    `app/services/alert_auto_resolve.py:_norm()`. That module auto-RESOLVES a
    vehicle_intrusion when the title matches the slot; this engine RAISES one
    when it doesn't. If the two normalise differently, a car whose title
    differs only in case or punctuation gets a critical alert here that the
    Gateway silently clears there.
    """
    if not value:
        return ""
    return re.sub(r"[^a-z0-9]", "", value.lower())


def titles_match(vehicle_title: str | None, reserved_for: str | None) -> bool:
    """True when this vehicle's owner holds the slot's reservation.

    Empty on either side is never a match: a blank `vehicles.title` (the column
    defaults to '') must not silently satisfy an unreserved-looking slot.
    """
    left, right = _norm_title(vehicle_title), _norm_title(reserved_for)
    return bool(left) and left == right


def is_named_slot(slot) -> bool:
    return bool(slot and slot.reservation_type == "EMPLOYEE")


def get_named_slot_title(slot) -> str | None:
    return slot.reserved_for if (slot and slot.reservation_type == "EMPLOYEE") else None


def is_restricted_slot(slot) -> bool:
    if not slot:
        return False
    return slot.reservation_type != "GENERAL" or slot.is_violation_zone


def get_slot_restriction_type(slot) -> str | None:
    if not slot:
        return None
    if slot.reservation_type == "EMPLOYEE":
        return "EMPLOYEE"
    if slot.reservation_type == "SPECIAL":
        return "SPECIAL"
    if slot.is_violation_zone:
        return "violation"
    return None

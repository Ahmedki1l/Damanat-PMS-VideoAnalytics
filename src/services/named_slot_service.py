"""
Named/reserved slot helpers.

All logic is driven by the `parking_category` and `reserved_for` columns on
the `parking_slots` DB table. Functions accept a ParkingSlot ORM object so
callers never maintain an external mapping.

parking_category values:
  "GENERAL"  — normal public parking
  "EMPLOYEE" — named slot reserved for a specific person (reserved_for = vehicles.title)
  "SPECIAL"  — special-needs / handicap slot
"""


def is_named_slot(slot) -> bool:
    return bool(slot and slot.parking_category == "EMPLOYEE")


def get_named_slot_title(slot) -> str | None:
    return slot.reserved_for if (slot and slot.parking_category == "EMPLOYEE") else None


def is_restricted_slot(slot) -> bool:
    if not slot:
        return False
    return slot.parking_category != "GENERAL" or slot.is_violation_zone


def get_slot_restriction_type(slot) -> str | None:
    if not slot:
        return None
    if slot.parking_category == "EMPLOYEE":
        return "EMPLOYEE"
    if slot.parking_category == "SPECIAL":
        return "SPECIAL"
    if slot.is_violation_zone:
        return "violation"
    return None

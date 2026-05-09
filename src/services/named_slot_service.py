"""
Named/reserved slot helpers.

All logic is driven by the `parking_category` and `reserved_for` columns on
the `parking_slots` DB table. Functions accept a ParkingSlot ORM object so
callers never maintain an external mapping.

parking_category values:
  "general"  — normal public parking
  "employee" — named slot reserved for a specific person (reserved_for = vehicles.title)
  "special"  — special-needs / handicap slot
"""


def is_named_slot(slot) -> bool:
    return bool(slot and slot.parking_category == "employee")


def get_named_slot_title(slot) -> str | None:
    return slot.reserved_for if (slot and slot.parking_category == "employee") else None


def is_restricted_slot(slot) -> bool:
    if not slot:
        return False
    return slot.parking_category != "general" or slot.is_violation_zone


def get_slot_restriction_type(slot) -> str | None:
    if not slot:
        return None
    if slot.parking_category == "employee":
        return "employee"
    if slot.parking_category == "special":
        return "special"
    if slot.is_violation_zone:
        return "violation"
    return None

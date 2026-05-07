NAMED_SLOT_TITLE_BY_SLOT_ID = {
    "B10_CTO": "B10_CTO",
    "B1_CRO": "B1_CRO",
    "B3_CEO": "B3_CEO",
    "GMIA": "GMIA",
    "B6_Reserved": "B6_Reserved",
    "B11_CFO": "B11_CFO",
    "B13_COO": "B13_COO",
    "B8_BDM": "B8_BDM",
    "G1": "G1_SN",
}


def get_named_slot_title(slot_id: str) -> str | None:
    return NAMED_SLOT_TITLE_BY_SLOT_ID.get(slot_id)


def is_named_slot(slot_id: str) -> bool:
    return slot_id in NAMED_SLOT_TITLE_BY_SLOT_ID


def is_restricted_slot(slot_id: str) -> bool:
    if not slot_id:
        return False
    return is_named_slot(slot_id) or "violation" in slot_id.lower()


def get_slot_restriction_type(slot_id: str) -> str | None:
    if is_named_slot(slot_id):
        return "named_slot"
    if slot_id and "violation" in slot_id.lower():
        return "violation"
    return None

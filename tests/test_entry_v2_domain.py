from src.entry.domain import canonical_plate, plate_key


def test_digit_first_plate_preserves_character_order_as_a_distinct_identity():
    assert plate_key("1234-abc") == "1234ABC"
    assert plate_key("1234-abc") != plate_key("abc-1234")
    assert canonical_plate("1234-abc") == "1234ABC"

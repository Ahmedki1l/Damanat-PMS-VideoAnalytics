"""Datetime helpers for VA writers.

The shared MSSQL DB stores naive datetimes that the codebase treats as the
**facility wall clock** (KSA wall clock, not UTC). Calling `datetime.utcnow()`
or bare `datetime.now()` from a UTC-tzed K8s pod silently lands rows 3 hours
behind, because the host has no idea the facility is at UTC+3.

Use `facility_now_naive()` everywhere a DB column gets a "now" timestamp.
The offset is hardcoded to UTC+3 (Saudi Arabia / Riyadh, no DST) so the
helper works identically across local dev (Windows), Docker, and K8s with
no env-var configuration required.

This module is intentionally standalone (no project imports) so SQLAlchemy
model files can import it without triggering circular config loads.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

FACILITY_TZ = timezone(timedelta(hours=3))


def facility_tz() -> timezone:
    """`tzinfo` for facility-local datetimes (UTC+3, KSA)."""
    return FACILITY_TZ


def facility_now_naive() -> datetime:
    """Current facility-local datetime, NAIVE (no tzinfo). Drop-in
    replacement for `datetime.utcnow()` / `datetime.now()` at every DB-write
    call site. Works regardless of the host OS / container timezone."""
    return datetime.now(FACILITY_TZ).replace(tzinfo=None)


def normalize_timestamp_for_clock(
    value: datetime,
    clock_sample: datetime,
) -> datetime:
    """Represent ``value`` using a component clock's datetime convention.

    Network timestamps carry an explicit offset, while much of VA still uses a
    naive ``datetime.now`` clock.  Mixing the two makes age/order comparisons
    raise ``TypeError``.  Convert the same instant to the clock's timezone and
    then mirror its aware/naive shape; never merely strip an offset from an
    aware value because that changes the represented instant.

    A naive ``value`` is intentionally interpreted in the clock's timezone.
    Public network boundaries must reject ambiguous naive timestamps before
    calling this helper; the behavior remains useful for internal legacy calls.
    """
    value_is_aware = value.tzinfo is not None and value.utcoffset() is not None
    clock_is_aware = (
        clock_sample.tzinfo is not None and clock_sample.utcoffset() is not None
    )

    if clock_is_aware:
        if value_is_aware:
            return value.astimezone(clock_sample.tzinfo)
        return value.replace(tzinfo=clock_sample.tzinfo)

    if not value_is_aware:
        return value

    # ``datetime.now()`` (the registry default) returns host-local naive time.
    # A no-argument astimezone conversion uses that same local zone and applies
    # the offset in force at ``value`` before the tzinfo is removed.
    return value.astimezone().replace(tzinfo=None)

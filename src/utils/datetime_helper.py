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

from datetime import datetime, timezone, timedelta


FACILITY_TZ = timezone(timedelta(hours=3))


def facility_tz() -> timezone:
    """`tzinfo` for facility-local datetimes (UTC+3, KSA)."""
    return FACILITY_TZ


def facility_now_naive() -> datetime:
    """Current facility-local datetime, NAIVE (no tzinfo). Drop-in
    replacement for `datetime.utcnow()` / `datetime.now()` at every DB-write
    call site. Works regardless of the host OS / container timezone."""
    return datetime.now(FACILITY_TZ).replace(tzinfo=None)

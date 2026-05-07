"""Datetime helpers for VA writers.

The shared MSSQL DB stores naive datetimes that the codebase treats as the
**facility wall clock** (KSA wall clock, not UTC). Calling `datetime.utcnow()`
or bare `datetime.now()` from a UTC-tzed K8s pod silently lands rows 3 hours
behind, because the host has no idea the facility is at UTC+3.

Use `facility_now_naive()` everywhere a DB column gets a "now" timestamp.
The offset is read from the FACILITY_TIMEZONE_OFFSET_HOURS env var (default
3.0 = Saudi Arabia / Riyadh, no DST). Both PMS-AI and the API Gateway read
the same env var; keep them aligned or per-service "today" windows diverge.

This module is intentionally standalone (no project imports) so SQLAlchemy
model files can import it without triggering circular config loads.
"""
from __future__ import annotations

import os
from datetime import datetime, timezone, timedelta


def _facility_offset_hours() -> float:
    raw = os.environ.get("FACILITY_TIMEZONE_OFFSET_HOURS", "3.0")
    try:
        return float(raw)
    except (TypeError, ValueError):
        return 3.0


def facility_tz() -> timezone:
    """`tzinfo` for facility-local datetimes."""
    return timezone(timedelta(hours=_facility_offset_hours()))


def facility_now_naive() -> datetime:
    """Current facility-local datetime, NAIVE (no tzinfo). Drop-in
    replacement for `datetime.utcnow()` / `datetime.now()` at every DB-write
    call site. Works regardless of the host OS / container timezone."""
    return datetime.now(facility_tz()).replace(tzinfo=None)

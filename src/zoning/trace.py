"""
trace.py — opt-in area-transition trace for on-site testing (zoning).

Enabled only when the ``ZONING_TRACE`` env var is truthy (1/true/yes/on), so it
is a zero-overhead no-op in production — just don't set the var at deploy. When
on, it writes a readable ``[AREA]`` line to stdout AND to ``zoning_trace.log``
(a dedicated file so the trace isn't buried under HEVC/YOLO console spam).

Usage:
    ZONING_TRACE=1 python main.py --api --show     # bash
    $env:ZONING_TRACE=1; python main.py --api --show   # PowerShell

Events:
    SETTLE    a car settled into an area (incl. arrival out of transit)
    CROSS     a car crossed a boundary band -> IN_TRANSIT (resolved direction)
    SNAPSHOT  periodic map of every car per area + who is in transit
"""

from __future__ import annotations

import logging
import os
import sys

_ENABLED = os.environ.get("ZONING_TRACE", "").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)
_logger: logging.Logger | None = None


def enabled() -> bool:
    return _ENABLED


def _get_logger() -> logging.Logger:
    global _logger
    if _logger is not None:
        return _logger
    lg = logging.getLogger("zoning.trace")
    lg.setLevel(logging.INFO)
    lg.propagate = False  # keep it off the root handlers / decode-error spam
    fmt = logging.Formatter("%(asctime)s [AREA] %(message)s", "%H:%M:%S")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    lg.addHandler(sh)
    try:
        fh = logging.FileHandler("zoning_trace.log", encoding="utf-8")
        fh.setFormatter(fmt)
        lg.addHandler(fh)
    except Exception:  # pragma: no cover - file handler is best-effort
        pass
    _logger = lg
    return lg


def trace(event: str, **fields) -> None:
    """Log one trace line ``EVENT  k=v k=v`` when ZONING_TRACE is on."""
    if not _ENABLED:
        return
    parts = " ".join(f"{k}={v}" for k, v in fields.items() if v is not None)
    _get_logger().info(f"{event:<8} {parts}")

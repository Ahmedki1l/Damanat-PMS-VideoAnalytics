"""Root logging configuration — the one place a handler is attached.

WHY THIS EXISTS
    Nothing in this service ever configured the ``logging`` module. There is no
    ``basicConfig``, no ``dictConfig``, no logging config file anywhere in
    ``src/``, ``main.py`` or ``supervisor.py``. ``main.py`` calls
    ``uvicorn.run(app, ..., log_level="info")``, which sets levels on the
    ``uvicorn.*`` loggers only and leaves the ROOT logger with no handler at all.

    Records therefore fell through to ``logging.lastResort``, whose level is
    WARNING, and every ``logger.info()`` in ``src/`` was discarded — all 143 of
    them, including the ``[EntryV2][ReID]`` evaluation line that entry
    calibration depends on. What actually reached the logs was the 199 ``print()``
    calls, which is why the service looked like it was logging.

    Calling :func:`configure_logging` once at startup is what makes those lines
    exist. It is a prerequisite for reviewing any Entry V2 shadow run.

WHY STDOUT
    ``supervisor.py`` redirects each child's stdout and stderr to SEPARATE files
    (``logs/va_<group>.out.log`` / ``.err.log``). The 199 ``print()`` calls go to
    stdout. Sending log records to stderr would split one causal story across two
    files with no way to interleave them, so we deliberately use stdout: prints
    and log records land in one stream, in order.
"""
from __future__ import annotations

import logging
import os
import sys
from typing import Optional

# Marks the handler this module owns, so a second call can find it instead of
# attaching a duplicate (which is how one line becomes two).
_HANDLER_FLAG = "_damanat_root_handler"

DEFAULT_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"
DEFAULT_DATEFMT = "%Y-%m-%d %H:%M:%S"

# Third-party loggers that are chatty at INFO. Raising the root logger to INFO
# for the first time turns these on too, and a flood of library noise would bury
# the lines we actually configured logging to see.
_NOISY_LIBRARIES = (
    "urllib3",
    "httpx",
    "httpcore",
    "PIL",
    "matplotlib",
    "asyncio",
    "openvino",
    "paddle",
    "ppocr",
)

# OUR OWN per-frame loggers, capped for the same reason as the libraries above.
#
# Turning the root logger up to INFO for the first time switches on 146
# logger.info() calls at once, and a few of them sit in the frame loop rather
# than on an event — engine_tracking's "[quality] track=... clearance=..."
# fires per detection whenever clearance < 1.0. On a fleet that is already
# CPU-starved, that is a real cost paid on every frame of every camera, and it
# would bury the entry decisions this logging was turned on to see.
#
# These are capped, NOT silenced: raise one deliberately with
# LOG_LEVEL_OVERRIDES when you are actually debugging it.
_PER_FRAME_LOGGERS = (
    "src.core.engine.engine_tracking",
    "src.core.engine.engine_runtime",
    "src.core.motion_scheduler",
    "src.detection.tracker",
)


def _resolve_level(raw: Optional[str]) -> int:
    """Map LOG_LEVEL to a logging level, falling back to INFO.

    A malformed LOG_LEVEL must never stop the engine booting — an operator
    typo is not a reason to take the cameras down, and the fallback is the
    level we would have used anyway.
    """
    if not raw or not raw.strip():
        return logging.INFO
    candidate = raw.strip().upper()
    resolved = logging.getLevelName(candidate)
    if isinstance(resolved, int):
        return resolved
    return logging.INFO


def configure_logging(
    level: Optional[str] = None,
    *,
    stream=None,
    force: bool = False,
) -> logging.Logger:
    """Attach exactly one stdout handler to the root logger. Idempotent.

    Args:
        level: level name; defaults to the ``LOG_LEVEL`` env var, then INFO.
        stream: override the output stream (tests).
        force: replace the handler this module previously installed. Used when
            the stream itself changed; never needed in production.

    Returns:
        The root logger, configured.
    """
    root = logging.getLogger()
    resolved = _resolve_level(level if level is not None else os.getenv("LOG_LEVEL"))

    existing = [h for h in root.handlers if getattr(h, _HANDLER_FLAG, False)]
    if existing and not force:
        # Already configured — a second call (supervisor child re-entry, or the
        # API server thread) must adjust the level, not duplicate the handler.
        root.setLevel(resolved)
        for handler in existing:
            handler.setLevel(logging.NOTSET)
        return root

    for handler in existing:
        root.removeHandler(handler)
        try:
            handler.close()
        except Exception:
            pass

    handler = logging.StreamHandler(stream if stream is not None else sys.stdout)
    handler.setFormatter(logging.Formatter(DEFAULT_FORMAT, datefmt=DEFAULT_DATEFMT))
    handler.setLevel(logging.NOTSET)
    setattr(handler, _HANDLER_FLAG, True)

    root.addHandler(handler)
    root.setLevel(resolved)

    # Only quiet loggers that have not been given an explicit level already,
    # so an operator debugging one of them is not silently overruled here.
    for name in _NOISY_LIBRARIES + _PER_FRAME_LOGGERS:
        noisy = logging.getLogger(name)
        if noisy.level == logging.NOTSET:
            noisy.setLevel(max(resolved, logging.WARNING))

    # Explicit per-logger overrides win over everything above, including the
    # per-frame caps: "src.core.engine.engine_tracking=INFO,src.ocr=DEBUG".
    for name, level in _parse_overrides(os.getenv("LOG_LEVEL_OVERRIDES")).items():
        logging.getLogger(name).setLevel(level)

    return root


def _parse_overrides(raw: Optional[str]) -> dict:
    """Parse "logger=LEVEL,logger=LEVEL". Malformed entries are skipped.

    One bad entry must not cost the others: this runs before anything can
    report the problem, so failing the whole string would silently drop a
    deliberate override an operator is relying on.
    """
    overrides: dict = {}
    for chunk in (raw or "").split(","):
        chunk = chunk.strip()
        if not chunk or "=" not in chunk:
            continue
        name, _, level = chunk.partition("=")
        name, level = name.strip(), level.strip()
        if not name or not level:
            continue
        resolved = logging.getLevelName(level.upper())
        if isinstance(resolved, int):
            overrides[name] = resolved
    return overrides


def is_configured() -> bool:
    """True when this module owns a handler on the root logger."""
    return any(
        getattr(h, _HANDLER_FLAG, False) for h in logging.getLogger().handlers
    )

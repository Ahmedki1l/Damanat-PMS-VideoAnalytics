"""Production factory kept separate from the pure coordinator."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
from typing import Any, Mapping, Optional

from src.matching.decision_log import DecisionLog, prune_old_logs

from .analyzer import DisabledEvidenceProcessor, ExistingModelsEvidenceProcessor
from .callback import (
    DeliveryResult,
    HttpConfirmationClient,
    RetryingConfirmationSink,
)
from .coordinator import EntryCoordinator
from .identity import RegistryIdentityPublisher
from .settings import EntrySettings


logger = logging.getLogger(__name__)
_RETRY_TASK_STATE_KEY = "entry_v2_callback_retry_task"

# Filenames are entry_decisions_<worker>_<YYYY-MM-DD>.jsonl. The prefix keeps
# this corpus distinct from the slot-identity training corpus, which has a
# different lifecycle and its own retention.
ENTRY_DECISION_LOG_PREFIX = "entry_decisions"


class _DisabledSink:
    def deliver(self, payload: Mapping[str, Any]) -> DeliveryResult:
        del payload
        return DeliveryResult(False, 0, "entry_v2_unavailable")


def _prepare_decision_log_dir(settings: EntrySettings) -> None:
    """Create the log directory and prune expired files. Safe in every mode.

    Runs even when Entry V2 is OFF, and deliberately so: the deployment ladder's
    first step ships the image with ENTRY_V2_MODE=off and verifies the log
    directory exists before shadow is ever enabled. If the directory only
    appeared once the pipeline was live, that check could not be made until it
    was too late to be useful.

    Never raises. A log volume that cannot be prepared is a degraded review, not
    a reason to keep the cameras down.
    """
    if not settings.decision_log_dir:
        return
    try:
        os.makedirs(settings.decision_log_dir, exist_ok=True)
    except OSError as exc:
        logger.warning(
            "[EntryV2] decision log directory %s is not usable: %r",
            settings.decision_log_dir,
            exc,
        )
        return
    prune_old_logs(
        settings.decision_log_dir,
        settings.decision_log_retention_days,
        ENTRY_DECISION_LOG_PREFIX,
    )


def build_decision_log(settings: EntrySettings) -> Optional[DecisionLog]:
    """Construct the Entry V2 decision log, or None when it is not configured.

    The worker name comes from VA_GROUP so the supervisor's processes never share
    a file handle (interleaved, corrupt lines are what that avoids). Entry V2
    runs in the single --api/gate group, so in practice this is one file per day.
    """
    _prepare_decision_log_dir(settings)
    if not settings.decision_log_dir:
        return None
    return DecisionLog(
        directory=settings.decision_log_dir,
        worker=os.environ.get("VA_GROUP", "") or f"pid{os.getpid()}",
        max_queue=settings.decision_log_queue_max,
        prefix=ENTRY_DECISION_LOG_PREFIX,
    )


def build_entry_coordinator(
    registry,
    *,
    image_dir: str | None = None,
) -> EntryCoordinator:
    settings = EntrySettings.from_env()
    errors = settings.configuration_errors()
    if errors:
        # Still prepare the directory: an invalid configuration degrades to
        # DisabledEvidenceProcessor, which is safe but records nothing, and the
        # operator checking for the directory needs to be able to tell those
        # apart from a mode that simply is not on yet.
        _prepare_decision_log_dir(settings)
        return EntryCoordinator(
            settings,
            DisabledEvidenceProcessor(),
            _DisabledSink(),
            unavailable_reason="entry_v2_invalid_configuration:" + ",".join(errors),
        )
    if settings.mode.value == "off":
        _prepare_decision_log_dir(settings)
        return EntryCoordinator(
            settings,
            DisabledEvidenceProcessor(),
            _DisabledSink(),
        )

    processor = ExistingModelsEvidenceProcessor(
        registry,
        settings,
        image_dir=image_dir,
    )
    client = HttpConfirmationClient(
        settings.callback_url,
        settings.service_key,
        timeout_seconds=settings.callback_timeout_seconds,
        expected_mode=settings.mode.value,
    )
    sink = RetryingConfirmationSink(
        client,
        max_attempts=settings.callback_max_attempts,
        initial_backoff_seconds=settings.callback_initial_backoff_seconds,
        max_backoff_seconds=settings.callback_max_backoff_seconds,
    )
    return EntryCoordinator(
        settings,
        processor,
        sink,
        identity_publisher=RegistryIdentityPublisher(registry),
        decision_log=build_decision_log(settings),
    )


async def _callback_retry_loop(coordinator: EntryCoordinator) -> None:
    """Drain transient callback failures on an operational cadence, never TTL."""
    interval = coordinator.settings.callback_retry_interval_seconds
    while True:
        await asyncio.sleep(interval)
        try:
            await asyncio.to_thread(coordinator.retry_pending_callbacks)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("[EntryV2] Scheduled callback retry failed")


def entry_callback_retry_lifespan(coordinator: EntryCoordinator):
    """Build the FastAPI lifespan that owns the callback retry worker."""

    @contextlib.asynccontextmanager
    async def lifespan(app):
        if not coordinator.available:
            yield
            return

        current = getattr(app.state, _RETRY_TASK_STATE_KEY, None)
        owns_task = current is None or current.done()
        if owns_task:
            current = asyncio.create_task(
                _callback_retry_loop(coordinator),
                name="entry-v2-callback-retry",
            )
            setattr(app.state, _RETRY_TASK_STATE_KEY, current)
        try:
            yield
        finally:
            task = getattr(app.state, _RETRY_TASK_STATE_KEY, None)
            if owns_task and task is current:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task
                setattr(app.state, _RETRY_TASK_STATE_KEY, None)

    return lifespan

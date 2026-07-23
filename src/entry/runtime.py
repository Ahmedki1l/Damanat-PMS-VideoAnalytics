"""Production factory kept separate from the pure coordinator."""

from __future__ import annotations

import asyncio
import contextlib
import logging
from typing import Any, Mapping

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


class _DisabledSink:
    def deliver(self, payload: Mapping[str, Any]) -> DeliveryResult:
        del payload
        return DeliveryResult(False, 0, "entry_v2_unavailable")


def build_entry_coordinator(
    registry,
    *,
    image_dir: str | None = None,
) -> EntryCoordinator:
    settings = EntrySettings.from_env()
    errors = settings.configuration_errors()
    if errors:
        return EntryCoordinator(
            settings,
            DisabledEvidenceProcessor(),
            _DisabledSink(),
            unavailable_reason="entry_v2_invalid_configuration:" + ",".join(errors),
        )
    if settings.mode.value == "off":
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

"""
event_bus.py — Structured event generation and logging.

Emits JSON-formatted events to stdout and optionally to a log file.
Provides a slot status summary for periodic bulk output.

Event types:
  - vehicle_entering  : A car has been detected in a slot (not yet confirmed).
  - vehicle_parked    : A car has been in the slot long enough to be confirmed.
  - vehicle_leaving   : A car is no longer detected in an occupied slot.
  - slot_vacant       : A slot has been vacant long enough to be confirmed empty.
"""

import json
import sys
from typing import List, Dict, Any, Optional, Callable

from src.models.state_machine import SlotEvent


class EventBus:
    """
    Central event emitter for the parking system.

    Broadcasts events to registered subscribers and optionally logs to a file.
    """

    def __init__(self, log_file: str = ""):
        """
        Args:
            log_file: Path to optional log file. Empty string = stdout only.
        """
        self._log_file = None
        self._subscribers: List[Callable[[SlotEvent], None]] = []
        
        if log_file:
            self._log_file = open(log_file, "a", encoding="utf-8")
            print(f"[INFO] Event log file: {log_file}")

    def subscribe(self, callback: Callable[[SlotEvent], None]) -> None:
        """Register a new subscriber callback."""
        if callback not in self._subscribers:
            self._subscribers.append(callback)

    def unsubscribe(self, callback: Callable[[SlotEvent], None]) -> None:
        """Unregister an existing subscriber callback."""
        if callback in self._subscribers:
            self._subscribers.remove(callback)

    def emit(self, event: SlotEvent) -> None:
        """
        Emit a single event to all subscribers and log file.

        Args:
            event: SlotEvent to emit.
        """
        # Broadcast to all subscribers
        for callback in self._subscribers:
            try:
                callback(event)
            except Exception as e:
                print(f"[ERROR] EventBus subscriber error: {e}")

        # Log to file
        if self._log_file:
            event_json = json.dumps(event.to_dict())
            self._log_file.write(event_json + "\n")
            self._log_file.flush()

    def emit_batch(self, events: List[SlotEvent]) -> None:
        """Emit multiple events."""
        for event in events:
            self.emit(event)

    def emit_status_summary(self, slot_statuses: List[Dict[str, Any]]) -> None:
        """
        Emit a bulk status summary of all slots.
        """
        summary = {
            "type": "status_summary",
            "slots": slot_statuses,
        }
        
        # In a real system, we might want a separate subscriber list for summaries,
        # but for now we'll just focus on SlotEvents for SSE.
        
        if self._log_file:
            summary_json = json.dumps(summary)
            self._log_file.write(summary_json + "\n")
            self._log_file.flush()

    def close(self) -> None:
        """Clean up log file handle."""
        if self._log_file:
            self._log_file.close()
            self._log_file = None
        self._subscribers.clear()


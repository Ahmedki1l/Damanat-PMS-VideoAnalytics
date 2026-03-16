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
from typing import List, Dict, Any, Optional

from src.models.state_machine import SlotEvent


class EventBus:
    """
    Central event emitter for the parking system.

    Prints structured JSON to stdout and optionally writes to a file.
    """

    def __init__(self, log_file: str = ""):
        """
        Args:
            log_file: Path to optional log file. Empty string = stdout only.
        """
        self._log_file = None
        if log_file:
            self._log_file = open(log_file, "a", encoding="utf-8")
            print(f"[INFO] Event log file: {log_file}")

    def emit(self, event: SlotEvent) -> None:
        """
        Emit a single event.

        Args:
            event: SlotEvent to emit.
        """
        event_json = json.dumps(event.to_dict())

        # Always print to stdout
        print(f"[EVENT] {event_json}")

        # Optionally write to file
        if self._log_file:
            self._log_file.write(event_json + "\n")
            self._log_file.flush()

    def emit_batch(self, events: List[SlotEvent]) -> None:
        """Emit multiple events."""
        for event in events:
            self.emit(event)

    def emit_status_summary(self, slot_statuses: List[Dict[str, Any]]) -> None:
        """
        Emit a bulk status summary of all slots.

        Useful for periodic health checks or dashboard updates.

        Args:
            slot_statuses: List of slot status dicts from SlotStateMachine.get_status().
        """
        summary = {
            "type": "status_summary",
            "slots": slot_statuses,
        }
        summary_json = json.dumps(summary)
        #print(f"[STATUS] {summary_json}")

        if self._log_file:
            self._log_file.write(summary_json + "\n")
            self._log_file.flush()

    def close(self) -> None:
        """Clean up log file handle."""
        if self._log_file:
            self._log_file.close()
            self._log_file = None

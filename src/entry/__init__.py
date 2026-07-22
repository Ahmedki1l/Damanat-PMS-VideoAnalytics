"""RAM-only V2 gate-entry validation.

The package deliberately has no dependency on the parking state machine.  Its
pure decision core consumes compact ReID/OCR evidence and emits metadata-only
decisions; model loading and HTTP transport live behind adapters.
"""

from .coordinator import EntryCoordinator
from .domain import EntryMode
from .settings import EntrySettings

__all__ = ["EntryCoordinator", "EntryMode", "EntrySettings"]

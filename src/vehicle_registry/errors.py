"""Typed in-process failures exposed by the vehicle registry."""

from datetime import datetime


class ValidatedEntrySupersededByExit(ValueError):
    """A source-ordered exit won while a validated entry was being published."""

    def __init__(self, latest_exit_at: datetime):
        super().__init__("validated entry superseded by exit")
        self.latest_exit_at = latest_exit_at

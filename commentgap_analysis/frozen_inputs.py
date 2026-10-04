"""Report private saved inputs that a notebook cannot restore."""
from __future__ import annotations

from pathlib import Path


def unavailable(variable: str, required_input: Path | str, reason: str = "") -> None:
    """Name an unrestorable variable and its missing input without fabricating it."""
    message = f"Frozen input unavailable for {variable}: {required_input}."
    if reason:
        message += f" {reason}"
    print(message)

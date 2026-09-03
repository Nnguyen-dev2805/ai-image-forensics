"""Shared exception types for the NPR external adapter."""

from __future__ import annotations

__all__ = ["NPRConfigError", "NPRDeferredError", "NPRRuntimeError"]


class NPRConfigError(Exception):
    """Raised for invalid NPR configuration that must always fail a run."""


class NPRDeferredError(Exception):
    """Raised when an environment condition should defer a run when allowed."""


class NPRRuntimeError(Exception):
    """Raised when the NPR runtime subprocess fails after setup or emits invalid scores."""

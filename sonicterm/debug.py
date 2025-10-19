#!/usr/bin/env python3
#
# SonicTerm
# Copyright (c) 2025 Martynas Jocius
#
"""Centralised debug helpers for SonicTerm."""

from __future__ import annotations

import os
from typing import Any, Callable

_TRUTHY = {"1", "true", "yes", "on"}


def _as_bool(value: Any) -> bool:
    """Return True when ``value`` represents an enabled flag."""
    if isinstance(value, str):
        return value.strip().lower() in _TRUTHY
    return bool(value)


DEBUG_ENABLED: bool = _as_bool(os.getenv("SONICTERM_DEBUG", ""))

try:
    from .utils.debug import (  # type: ignore
        debug_log as _debug_log_impl,
        update_state as _update_state_impl,
        log_operation as _log_operation_impl,
        log_error as _log_error_impl,
        set_debug_mode as _set_debug_mode_impl,
        debug_manager as _debug_manager_impl,
    )
except Exception:  # pragma: no cover - debug utilities missing
    def _noop(*_args: Any, **_kwargs: Any) -> None:
        return None

    _debug_log_impl = _update_state_impl = _log_operation_impl = _log_error_impl = _noop

    def _set_debug_mode_impl(_enabled: bool) -> None:  # type: ignore
        return None

    debug_manager = None  # type: ignore
else:
    debug_manager = _debug_manager_impl
    _set_debug_mode_impl(DEBUG_ENABLED)


def _call_when_enabled(callback: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    if DEBUG_ENABLED:
        return callback(*args, **kwargs)
    return None


def is_debug_enabled() -> bool:
    """Return current debug flag state."""
    return DEBUG_ENABLED


def set_debug_mode(enabled: bool) -> None:
    """Toggle debug output and propagate to the underlying manager."""
    global DEBUG_ENABLED
    DEBUG_ENABLED = bool(enabled)
    _set_debug_mode_impl(DEBUG_ENABLED)


def enable_debug() -> None:
    """Enable debug logging."""
    set_debug_mode(True)


def disable_debug() -> None:
    """Disable debug logging."""
    set_debug_mode(False)


def debug_log(*args: Any, **kwargs: Any) -> None:
    _call_when_enabled(_debug_log_impl, *args, **kwargs)


def update_state(*args: Any, **kwargs: Any) -> None:
    _call_when_enabled(_update_state_impl, *args, **kwargs)


def log_operation(*args: Any, **kwargs: Any) -> None:
    _call_when_enabled(_log_operation_impl, *args, **kwargs)


def log_error(*args: Any, **kwargs: Any) -> None:
    _call_when_enabled(_log_error_impl, *args, **kwargs)


__all__ = [
    "DEBUG_ENABLED",
    "debug_log",
    "update_state",
    "log_operation",
    "log_error",
    "debug_manager",
    "is_debug_enabled",
    "set_debug_mode",
    "enable_debug",
    "disable_debug",
]

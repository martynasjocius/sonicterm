#!/usr/bin/env python3
#
# SonicTerm
# Copyright (c) 2025 Martynas Jocius
#

import importlib
from typing import Any

__all__ = ["TUIManager", "tui_manager"]


def __getattr__(name: str) -> Any:
    if name in __all__:
        module = importlib.import_module("sonicterm.ui.tui")
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module 'sonicterm.ui' has no attribute {name!r}")

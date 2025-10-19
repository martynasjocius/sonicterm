#!/usr/bin/env python3
#
# SonicTerm
# Copyright (c) 2025 Martynas Jocius
#
"""Core scene management exports for SonicTerm."""

import importlib
from typing import Any

__all__ = ['SceneBasedSoundscape', 'SceneFileHandler']


def __getattr__(name: str) -> Any:
    if name in __all__:
        module = importlib.import_module('sonicterm.core.scene')
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module 'sonicterm.core' has no attribute {name!r}")

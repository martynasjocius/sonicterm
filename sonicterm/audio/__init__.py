#!/usr/bin/env python3
#
# SonicTerm
# Copyright (c) 2025 Martynas Jocius
#
"""Audio utilities for SonicTerm."""

import importlib
from typing import Any

__all__ = ['SamplePlayer']


def __getattr__(name: str) -> Any:
    if name == 'SamplePlayer':
        module = importlib.import_module('sonicterm.audio.player')
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module 'sonicterm.audio' has no attribute {name!r}")

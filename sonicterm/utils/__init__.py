#!/usr/bin/env python3
#
# SonicTerm
# Copyright (c) 2025 Martynas Jocius
#
"""Utility helpers shared across SonicTerm."""

import json
import sys
from importlib import import_module
from pathlib import Path
from types import ModuleType
from typing import Any

__all__ = ("list_available_scenes", "webcam")


def list_available_scenes(scenes_dir='scenes'):
    """List all available scene files with their metadata."""
    scenes_path = Path(scenes_dir)
    scenes = []
    
    if not scenes_path.exists():
        return scenes
    
    for scene_file in scenes_path.glob('*.json'):
        try:
            with open(scene_file, 'r') as f:
                scene_data = json.load(f)
            name = scene_data.get('name', scene_file.stem)
            description = scene_data.get('description', 'No description')
            scenes.append({
                'file': scene_file.name,
                'path': str(scene_file),
                'name': name,
                'description': description
            })
        except Exception:
            scenes.append({
                'file': scene_file.name,
                'path': str(scene_file),
                'name': scene_file.stem,
                'description': '(invalid JSON)'
            })
    
    return scenes


def __getattr__(name: str) -> Any:
    if name == "webcam":
        try:
            module = import_module("sonicterm.utils.webcam")
        except ModuleNotFoundError:
            stub_name = "sonicterm.utils.webcam"
            module = sys.modules.get(stub_name)
            if module is None:
                module = ModuleType(stub_name)

                class _StubCameraCapture:
                    """Fallback stub used when Pillow/NumPy dependencies are missing."""

                    last_error = "Camera capture unavailable: Pillow dependency not installed"

                    def capture_image(self, *args, **kwargs):  # noqa: D401
                        """Return no image when webcam support is unavailable."""
                        return None

                    def capture_and_process(self, *args, **kwargs):
                        return None

                module.camera_capture = _StubCameraCapture()
                sys.modules[stub_name] = module
        return module
    raise AttributeError(f"module 'sonicterm.utils' has no attribute {name!r}")

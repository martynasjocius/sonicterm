#
# SonicTerm
# Copyright (c) 2025 Martynas Jocius
#
"""Tests for external visual plugin integration."""

from __future__ import annotations

import json
import sys
import tempfile
import textwrap
from pathlib import Path
from typing import Iterable
import unittest

from sonicterm.cli import validate_scene_file
from sonicterm.core.scene import SceneBasedSoundscape
from sonicterm.ui.tui import tui_manager


class VisualPluginTests(unittest.TestCase):
    """Validate visual plugin loading, rendering, and error handling."""

    def tearDown(self) -> None:
        # Reset plugin configuration in the TUI between tests.
        tui_manager.configure_visual_plugin(None)
        self._purge_visual_modules()

    @staticmethod
    def _purge_visual_modules() -> None:
        keys: Iterable[str] = [key for key in sys.modules if key.startswith("sonicterm_visual_")]
        for key in keys:
            sys.modules.pop(key, None)

    def _write_plugin(self, directory: Path, body: str, filename: str = "plugin.py") -> Path:
        plugin_path = directory / filename
        plugin_path.write_text(textwrap.dedent(body), encoding="utf-8")
        return plugin_path

    def _write_scene(self, directory: Path, plugin_path: Path) -> Path:
        scene_path = directory / "scene.json"
        scene_data = {
            "name": "Test Scene",
            "samples": [
                {"path": str(Path.cwd() / "samples" / "forest.wav"), "name": "Forest"}
            ],
            "visual": {
                "name": "Test Visual",
                "path": str(plugin_path)
            }
        }
        scene_path.write_text(json.dumps(scene_data), encoding="utf-8")
        return scene_path

    def test_validate_scene_accepts_visual_plugin(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            plugin_path = self._write_plugin(
                tmp_path,
                """
                from typing import Any, Dict

                def render(width: int, height: int, context: Dict[str, Any] | None = None):
                    return [[(0, 0, 0) for _ in range(max(1, width))] for _ in range(max(1, height))]
                """,
                filename="visual_valid.py",
            )
            scene_path = self._write_scene(tmp_path, plugin_path)

            ok, info = validate_scene_file(str(scene_path))
            self.assertTrue(ok)
            self.assertIsNotNone(info)

    def test_visual_plugin_loads_and_renders(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            plugin_path = self._write_plugin(
                tmp_path,
                """
                from typing import Any, Dict

                COLORS = [
                    (255, 0, 0),
                    (0, 255, 0),
                    (0, 0, 255),
                ]

                def render(width: int, height: int, context: Dict[str, Any] | None = None):
                    frame = []
                    for row in range(height):
                        color = COLORS[row % len(COLORS)]
                        frame.append([color for _ in range(width)])
                    return frame
                """,
            )
            scene_path = self._write_scene(tmp_path, plugin_path)

            scene = SceneBasedSoundscape(scene_path)
            self.assertTrue(scene.load_scene())
            self.assertIsNotNone(scene.visual_plugin)
            self.assertIn(scene.visual_plugin.key, tui_manager.map_modes)

            self.addCleanup(lambda: [player.stop() for player in scene.players])

            matrix, error = scene.render_visual_plugin(6, 6)
            self.assertIsNone(error)
            self.assertIsNotNone(matrix)

    def test_visual_plugin_render_error_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            plugin_path = self._write_plugin(
                tmp_path,
                """
                def render(width: int, height: int):
                    raise RuntimeError("boom")
                """,
                filename="visual_error.py",
            )
            scene_path = self._write_scene(tmp_path, plugin_path)

            scene = SceneBasedSoundscape(scene_path)
            self.assertTrue(scene.load_scene())

            self.addCleanup(lambda: [player.stop() for player in scene.players])

            matrix, error = scene.render_visual_plugin(4, 4)
            self.assertIsNone(matrix)
            self.assertIsNotNone(error)
            self.assertIn("boom", error)
            self.assertIsNotNone(scene.visual_plugin)
            self.assertIn("boom", scene.visual_plugin.last_error or "")

    def test_visual_plugin_context_includes_logger(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            plugin_path = self._write_plugin(
                tmp_path,
                """
                from typing import Any, Dict

                def render(width: int, height: int, context: Dict[str, Any] | None = None):
                    logger = (context or {}).get("logger")
                    if logger:
                        logger("render frame {}", context.get("frame"))
                    return [[(0, 0, 0) for _ in range(max(1, width))] for _ in range(max(1, height))]
                """,
                filename="visual_logger.py",
            )
            scene_path = self._write_scene(tmp_path, plugin_path)

            captured = []
            original_log = getattr(tui_manager, "log_from_plugin", tui_manager.log)

            def capture_log(message: str) -> None:
                captured.append(message)

            if hasattr(tui_manager, "log_from_plugin"):
                tui_manager.log_from_plugin = capture_log  # type: ignore[assignment]
            else:
                tui_manager.log = capture_log  # type: ignore[assignment]
            try:
                scene = SceneBasedSoundscape(scene_path)
                self.assertTrue(scene.load_scene())
                self.addCleanup(lambda: [player.stop() for player in scene.players])

                matrix, error = scene.render_visual_plugin(4, 4)
                self.assertIsNone(error)
                self.assertIsNotNone(matrix)
            finally:
                if hasattr(tui_manager, "log_from_plugin"):
                    tui_manager.log_from_plugin = original_log  # type: ignore[assignment]
                else:
                    tui_manager.log = original_log  # type: ignore[assignment]

            self.assertTrue(
                any("Visual Plugin" in entry and "render frame" in entry for entry in captured),
                msg=f"Plugin logger output not captured: {captured}",
            )

    def test_octfive_flag_contains_all_colors(self) -> None:
        plugin_path = Path("tmp/octfive.py").resolve()
        if not plugin_path.exists():  # pragma: no cover - guard for CI environments
            self.skipTest("octfive plugin not present")

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            scene_path = tmp_path / "scene.json"
            scene_data = {
                "name": "Flag Scene",
                "samples": [
                    {"path": str(Path.cwd() / "samples" / "forest.wav"), "name": "Forest"}
                ],
                "visual": {
                    "name": "Flag",
                    "path": str(plugin_path),
                },
            }
            scene_path.write_text(json.dumps(scene_data), encoding="utf-8")

            scene = SceneBasedSoundscape(scene_path)
            self.assertTrue(scene.load_scene())
            self.addCleanup(lambda: [player.stop() for player in scene.players])

            width, height = 30, 18
            matrix, error = scene.render_visual_plugin(width, height)
            self.assertIsNone(error)
            self.assertIsNotNone(matrix)

            def within_tolerance(color, target, tolerance=110):
                return all(abs(int(c) - t) <= tolerance for c, t in zip(color, target))

            stripe_count = 3
            stripe_height = max(1, height // stripe_count)
            sample_rows = [
                min(height - 1, stripe_height // 2),
                min(height - 1, stripe_height + stripe_height // 2),
                min(height - 1, 2 * stripe_height + stripe_height // 2),
            ]
            targets = [
                (255, 214, 0),
                (0, 136, 72),
                (220, 0, 0),
            ]

            for target, row_index in zip(targets, sample_rows):
                stripe_color = matrix[row_index][0][:3]
                self.assertTrue(
                    within_tolerance(stripe_color, target),
                    msg=f"Row {row_index} color {stripe_color} not close to {target}",
                )


if __name__ == "__main__":
    unittest.main()

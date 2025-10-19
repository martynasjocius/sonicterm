#
# SonicTerm
# Copyright (c) 2025 Martynas Jocius
#
import sys
import types
import unittest
from unittest import mock

import numpy as np

import sonicterm.ui.tui as tui_module
from sonicterm.ui.tui import TUIManager


class TUIColorMatrixSizingTests(unittest.TestCase):
    def setUp(self):
        self.tui = TUIManager()
        self.tui.visual_mode = True
        self.tui.chronicle_visible = False
        self.tui.players_status = {"Sample": {"state": "Active"}}

    def _install_scene_stub(self):
        module = types.ModuleType("sonicterm.core.scene")
        manager = types.SimpleNamespace(request_webcam_resize=mock.Mock())
        module.scene_manager = manager
        sys.modules["sonicterm.core.scene"] = module
        self.addCleanup(sys.modules.pop, "sonicterm.core.scene", None)
        return manager

    def test_color_matrix_dimensions(self):
        terminal_width = 120
        terminal_height = 40
        panel_padding = (1, 2)

        with mock.patch.object(self.tui, "_get_safe_terminal_size", return_value=(terminal_width, terminal_height)):
            inner_width, inner_height = self.tui._compute_color_matrix_dimensions(panel_padding)

        base_width = terminal_width - 6
        horizontal_pad = tui_module.TUIManager._padding_metrics(panel_padding)[1]
        base_inner_width = max(base_width - horizontal_pad, 8)
        expected_width = base_inner_width + tui_module.COLOR_MATRIX_EXTRA_WIDTH
        base_height = terminal_height - 3 - 0 - 2  # header, footer, margin reservations
        vertical_pad = tui_module.TUIManager._padding_metrics(panel_padding)[0]
        base_inner_height = max(base_height - vertical_pad, 3)
        expected_height = base_inner_height + tui_module.COLOR_MATRIX_EXTRA_HEIGHT

        self.assertEqual(inner_width, expected_width)
        self.assertEqual(inner_height, expected_height)

    def test_update_webcam_matrix_stores_frame_even_when_visual_disabled(self):
        self.tui.visual_mode = False
        frame = np.zeros((4, 4, 3), dtype=np.uint8)

        self.tui.update_webcam_matrix(frame)

        self.assertIs(self.tui.webcam_matrix, frame)

    def test_camera_panel_shows_placeholder_without_frame(self):
        self.tui.map_modes = ["random", "camera"]
        self.tui.current_map_mode = 1
        self.tui.players_status = {}
        manager = self._install_scene_stub()

        with mock.patch.object(self.tui, "_get_safe_terminal_size", return_value=(80, 30)):
            layout = self.tui.create_layout()

        colors_panel = layout["colors"].content
        self.assertEqual(
            tui_module.VISUAL_PANEL_TITLES["camera_matrix"],
            colors_panel.title,
        )
        self.assertIn("Camera feed initializing", str(colors_panel.renderable))
        manager.request_webcam_resize.assert_called()

    def test_camera_panel_renders_frame_when_available(self):
        self.tui.map_modes = ["random", "camera"]
        self.tui.current_map_mode = 1
        self.tui.players_status = {}
        frame = np.zeros((6, 8, 3), dtype=np.uint8)
        frame[..., 0] = 255
        self.tui.webcam_matrix = frame
        manager = self._install_scene_stub()

        with mock.patch.object(self.tui, "_get_safe_terminal_size", return_value=(100, 40)):
            layout = self.tui.create_layout()

        colors_panel = layout["colors"].content
        self.assertEqual(
            tui_module.VISUAL_PANEL_TITLES["camera_matrix"],
            colors_panel.title,
        )
        manager.request_webcam_resize.assert_called()

    def test_camera_panel_uses_scene_manager_fallback(self):
        self.tui.map_modes = ["random", "camera"]
        self.tui.current_map_mode = 1
        self.tui.players_status = {}
        manager = self._install_scene_stub()
        fallback = np.zeros((5, 7, 3), dtype=np.uint8)
        fallback[..., 1] = 128
        manager.color_matrix = fallback
        manager.webcam_last_error = None

        with mock.patch.object(self.tui, "_get_safe_terminal_size", return_value=(90, 35)):
            layout = self.tui.create_layout()

        colors_panel = layout["colors"].content
        self.assertEqual(
            tui_module.VISUAL_PANEL_TITLES["camera_matrix"],
            colors_panel.title,
        )
        manager.request_webcam_resize.assert_called()

    def test_camera_panel_displays_last_error_message(self):
        self.tui.map_modes = ["random", "camera"]
        self.tui.current_map_mode = 1
        self.tui.players_status = {}
        manager = self._install_scene_stub()
        manager.webcam_last_error = "ffmpeg error: permission denied"

        with mock.patch.object(self.tui, "_get_safe_terminal_size", return_value=(80, 30)):
            layout = self.tui.create_layout()

        colors_panel = layout["colors"].content
        self.assertEqual(
            tui_module.VISUAL_PANEL_TITLES["camera_matrix"],
            colors_panel.title,
        )
        self.assertIn("permission denied", str(colors_panel.renderable))

    def test_update_webcam_matrix_triggers_display_refresh_in_camera_mode(self):
        self.tui.map_modes = ["random", "camera"]
        self.tui.current_map_mode = 1
        frame = np.zeros((4, 4, 3), dtype=np.uint8)
        with mock.patch.object(self.tui, "_queue_live_update") as queue_update:
            self.tui.update_webcam_matrix(frame)

        queue_update.assert_called()


if __name__ == "__main__":
    unittest.main()

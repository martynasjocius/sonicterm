#
# SonicTerm
# Copyright (c) 2025 Martynas Jocius
#
import sys
import types
import unittest
from unittest import mock

from sonicterm.ui.tui import TUIManager


class TUIVisualToggleIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.tui = TUIManager()
        self.tui.visual_mode = True

    def _install_scene_stub(self, running=True):
        module = types.ModuleType("sonicterm.core.scene")
        manager = types.SimpleNamespace(
            webcam_running=running,
            camera_device="/dev/video0",
            webcam_update_interval=0.2,
            webcam_matrix_size=(40, 25),
            _start_webcam_capture=mock.Mock(),
            _stop_webcam_capture=mock.Mock(),
        )
        module.scene_manager = manager
        sys.modules["sonicterm.core.scene"] = module
        self.addCleanup(sys.modules.pop, "sonicterm.core.scene", None)
        return manager

    def test_toggle_visual_off_stops_webcam(self):
        manager = self._install_scene_stub(running=True)
        self.tui.map_modes = ["camera"]
        self.tui.current_map_mode = 0
        with mock.patch.object(self.tui, "log"):
            self.tui.toggle_visual()  # toggles to False
        self.assertFalse(self.tui.visual_mode)
        manager._stop_webcam_capture.assert_called_once()
        self.assertIsNone(self.tui.webcam_matrix)

    def test_toggle_visual_on_starts_webcam_if_not_running(self):
        self.tui.visual_mode = False
        manager = self._install_scene_stub(running=False)
        self.tui.map_modes = ["camera"]
        self.tui.current_map_mode = 0
        with mock.patch.object(self.tui, "log"):
            self.tui.toggle_visual()  # toggles to True
        self.assertTrue(self.tui.visual_mode)
        manager._start_webcam_capture.assert_called_once()


    def test_toggle_visual_on_does_not_start_camera_in_non_camera_mode(self):
        manager = self._install_scene_stub(running=False)
        self.tui.map_modes = ["random"]
        self.tui.current_map_mode = 0
        self.tui.visual_mode = False
        with mock.patch.object(self.tui, "log"):
            self.tui.toggle_visual()
        self.assertTrue(self.tui.visual_mode)
        manager._start_webcam_capture.assert_not_called()
if __name__ == '__main__':
    unittest.main()

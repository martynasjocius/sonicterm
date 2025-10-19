#
# SonicTerm
# Copyright (c) 2025 Martynas Jocius
#
import unittest
from unittest import mock

import numpy as np

from sonicterm.ui.tui import TUIManager


class TUIVisualTitleTests(unittest.TestCase):
    def setUp(self):
        self.tui = TUIManager()
        self.tui.chronicle_visible = False
        self.tui.players_status = {"Sample": {"state": "Active"}}

    def _create_layout(self, width=120):
        with mock.patch.object(self.tui, "_get_safe_terminal_size", return_value=(width, 40)):
            return self.tui.create_layout()

    def _panel_title_and_width(self, width=120):
        layout = self._create_layout(width)
        colors_panel = layout["colors"].content
        title = colors_panel.title
        content = str(colors_panel.renderable)
        lines = [line for line in content.split("\n") if line]
        max_width = max((len(line) for line in lines), default=0)
        return title, max_width

    def test_activity_map_title_fits(self):
        self.tui.visual_mode = True
        self.tui.map_modes = ["random"]
        self.tui.current_map_mode = 0

        width = 120
        title, max_width = self._panel_title_and_width(width)
        self.assertLess(len(title), 80)
        self.assertLessEqual(max_width, width)

    def test_camera_title_fits(self):
        self.tui.visual_mode = True
        self.tui.webcam_matrix = np.zeros((20, 20, 3), dtype=int)
        self.tui.map_modes = ["camera"]
        self.tui.current_map_mode = 0
        width = 120
        title, max_width = self._panel_title_and_width(width)
        self.assertLess(len(title), 80)
        self.assertLessEqual(max_width, width)


if __name__ == "__main__":
    unittest.main()

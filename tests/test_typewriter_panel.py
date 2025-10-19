#
# SonicTerm
# Copyright (c) 2025 Martynas Jocius
#
import unittest
from pathlib import Path

from sonicterm.ui.tui import TUIManager


class TypewriterPanelTests(unittest.TestCase):
    def setUp(self):
        self.tui = TUIManager()
        self.tui.enable_tui()

    def test_typewriter_text_progresses(self):
        text = "Hello World"
        self.tui.set_typewriter_from_config({'text': text}, Path('.'))
        self.tui.typewriter_typing_interval_range = (0.0, 0.0)
        self.tui.toggle_typewriter_panel()

        # Force layout render to drive typing
        self.tui.create_layout()
        self.assertEqual(self.tui.typewriter_render_buffer, text)

    def test_typewriter_missing_shows_message(self):
        self.tui.set_typewriter_from_config(None, Path('.'))
        self.tui.toggle_typewriter_panel()
        self.tui.create_layout()
        self.assertIn("No typewriter", self.tui.typewriter_render_buffer)

    def test_typewriter_pace_setting(self):
        self.tui.set_typewriter_from_config({'text': 'Hi', 'pace': 'fast'}, Path('.'))
        self.assertEqual(self.tui.typewriter_typing_interval_range, (0.01, 0.04))


if __name__ == '__main__':
    unittest.main()

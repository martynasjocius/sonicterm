#
# SonicTerm
# Copyright (c) 2025 Martynas Jocius
#
import unittest
from pathlib import Path
from sonicterm.ui.tui import TUIManager


class StoryScrollingTests(unittest.TestCase):
    def setUp(self):
        self.tui = TUIManager()
        self.tui.enable_tui()
        # Disable chronicle panel to avoid buffer line complications in tests
        self.tui.chronicle_visible = False

    def test_scrolling_when_content_exceeds_capacity(self):
        """Test that scrolling works when typewriter content exceeds panel capacity."""
        # Create a long typewriter with multiple lines
        long_typewriter = """Line 1: This is the first line of a very long typewriter.
Line 2: This is the second line of a very long typewriter.
Line 3: This is the third line of a very long typewriter.
Line 4: This is the fourth line of a very long typewriter.
Line 5: This is the fifth line of a very long typewriter.
Line 6: This is the sixth line of a very long typewriter.
Line 7: This is the seventh line of a very long typewriter.
Line 8: This is the eighth line of a very long typewriter.
Line 9: This is the ninth line of a very long typewriter.
Line 10: This is the tenth line of a very long typewriter."""
        
        typewriter_config = {'text': long_typewriter, 'pace': 'fast'}
        self.tui.set_typewriter_from_config(typewriter_config, Path('.'))
        self.tui.toggle_typewriter_panel()
        
        # Set fast typing for testing
        self.tui.typewriter_typing_interval_range = (0.0, 0.0)
        
        # Simulate a small panel (capacity = 3 lines)
        max_height = 5  # 3 content lines + 2 for border
        panel_padding = 1
        
        # Test initial state - should show first lines
        panel, panel_height = self.tui._build_typewriter_panel(panel_padding, max_height)
        self.assertIsNotNone(panel)
        
        # Simulate typing progress
        typewriter_lines = long_typewriter.split('\n')
        
        # Test with partial content (should not scroll yet)
        self.tui.typewriter_visible_chars = len('\n'.join(typewriter_lines[:3]))
        self.tui.typewriter_render_buffer = self.tui.typewriter_full_text[:self.tui.typewriter_visible_chars]
        
        panel, panel_height = self.tui._build_typewriter_panel(panel_padding, max_height)
        content_text = str(panel.renderable)
        visible_lines = content_text.split('\n')
        
        # Should show first 3 lines, no scrolling yet
        self.assertEqual(self.tui.typewriter_scroll_position, 0)
        self.assertIn("Line 1:", visible_lines[0])
        self.assertIn("Line 3:", visible_lines[2])
        
        # Test with more content (should start scrolling)
        self.tui.typewriter_visible_chars = len('\n'.join(typewriter_lines[:6]))
        self.tui.typewriter_render_buffer = self.tui.typewriter_full_text[:self.tui.typewriter_visible_chars]
        
        panel, panel_height = self.tui._build_typewriter_panel(panel_padding, max_height)
        content_text = str(panel.renderable)
        visible_lines = content_text.split('\n')
        
        # Should now be scrolling to show the latest content
        self.assertGreater(self.tui.typewriter_scroll_position, 0)
        # Should show lines 4, 5, 6 (the last 3 lines of current content)
        self.assertIn("Line 4:", visible_lines[0])
        self.assertIn("Line 6:", visible_lines[2])
        
        # Test with full content (should show last lines)
        self.tui.typewriter_visible_chars = self.tui.typewriter_total_chars
        self.tui.typewriter_render_buffer = self.tui.typewriter_full_text
        
        panel, panel_height = self.tui._build_typewriter_panel(panel_padding, max_height)
        content_text = str(panel.renderable)
        visible_lines = content_text.split('\n')
        
        # Should show the last 3 lines of the typewriter
        self.assertIn("Line 8:", visible_lines[0])
        self.assertIn("Line 10:", visible_lines[2])
        
        # Verify scroll position is correct
        expected_scroll = len(typewriter_lines) - 3  # Total lines - visible capacity
        self.assertEqual(self.tui.typewriter_scroll_position, expected_scroll)

    def test_no_scrolling_when_content_fits(self):
        """Test that no scrolling occurs when content fits within panel capacity."""
        short_typewriter = """Line 1: Short typewriter.
Line 2: Only two lines."""
        
        typewriter_config = {'text': short_typewriter, 'pace': 'fast'}
        self.tui.set_typewriter_from_config(typewriter_config, Path('.'))
        self.tui.toggle_typewriter_panel()
        
        # Set fast typing for testing
        self.tui.typewriter_typing_interval_range = (0.0, 0.0)
        
        # Simulate a large panel (capacity = 5 lines)
        max_height = 7  # 5 content lines + 2 for border
        panel_padding = 1
        
        # Test with full content
        self.tui.typewriter_visible_chars = self.tui.typewriter_total_chars
        self.tui.typewriter_render_buffer = self.tui.typewriter_full_text
        
        panel, panel_height = self.tui._build_typewriter_panel(panel_padding, max_height)
        self.assertIsNotNone(panel)
        
        # Should not be scrolling since content fits
        self.assertEqual(self.tui.typewriter_scroll_position, 0)
        
        content_text = str(panel.renderable)
        visible_lines = content_text.split('\n')
        
        # Should show all lines
        self.assertIn("Line 1:", visible_lines[0])
        self.assertIn("Line 2:", visible_lines[1])

    def test_scroll_reset_on_typewriter_change(self):
        """Test that scroll position resets when typewriter content changes."""
        typewriter1 = "Line 1\nLine 2\nLine 3\nLine 4\nLine 5"
        typewriter2 = "New Line 1\nNew Line 2"
        
        # Set first typewriter
        typewriter_config1 = {'text': typewriter1, 'pace': 'fast'}
        self.tui.set_typewriter_from_config(typewriter_config1, Path('.'))
        self.tui.toggle_typewriter_panel()
        
        # Simulate scrolling
        self.tui.typewriter_visible_chars = self.tui.typewriter_total_chars
        self.tui.typewriter_render_buffer = self.tui.typewriter_full_text
        
        max_height = 5  # 3 content lines + 2 for border
        panel_padding = 1
        
        panel, _ = self.tui._build_typewriter_panel(panel_padding, max_height)
        self.assertGreater(self.tui.typewriter_scroll_position, 0)
        
        # Set second typewriter
        typewriter_config2 = {'text': typewriter2, 'pace': 'fast'}
        self.tui.set_typewriter_from_config(typewriter_config2, Path('.'))
        
        # Scroll position should be reset
        self.assertEqual(self.tui.typewriter_scroll_position, 0)
        self.assertEqual(self.tui.typewriter_last_content_height, 0)


if __name__ == '__main__':
    unittest.main()
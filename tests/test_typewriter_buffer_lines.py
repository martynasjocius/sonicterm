#
# SonicTerm
# Copyright (c) 2025 Martynas Jocius
#
import unittest
from pathlib import Path
from sonicterm.ui.tui import TUIManager


class StoryBufferLinesTests(unittest.TestCase):
    def setUp(self):
        self.tui = TUIManager()
        self.tui.enable_tui()

    def test_buffer_lines_when_chronicle_visible_and_typing(self):
        """Test that buffer lines are added when Chronicles panel is visible and typewriter is still typing."""
        # Create a typewriter with multiple lines
        typewriter_text = """Line 1: First line of the typewriter.
Line 2: Second line of the typewriter.
Line 3: Third line of the typewriter.
Line 4: Fourth line of the typewriter.
Line 5: Fifth line of the typewriter.
Line 6: Sixth line of the typewriter."""
        
        typewriter_config = {'text': typewriter_text, 'pace': 'fast'}
        self.tui.set_typewriter_from_config(typewriter_config, Path('.'))
        self.tui.toggle_typewriter_panel()
        
        # Set fast typing for testing
        self.tui.typewriter_typing_interval_range = (0.0, 0.0)
        
        # Simulate a small panel (capacity = 3 lines)
        max_height = 5  # 3 content lines + 2 for border
        panel_padding = 1
        
        # Test with Chronicles panel visible and typewriter still typing
        self.tui.chronicle_visible = True
        self.tui.typewriter_visible_chars = len(typewriter_text) // 2  # Half typed
        self.tui.typewriter_render_buffer = self.tui.typewriter_full_text[:self.tui.typewriter_visible_chars]
        
        panel, panel_height = self.tui._build_typewriter_panel(panel_padding, max_height)
        self.assertIsNotNone(panel)
        
        content_text = str(panel.renderable)
        visible_lines = content_text.split('\n')
        
        # Should have buffer lines at the bottom
        self.assertGreaterEqual(len(visible_lines), 3)  # At least 3 content lines
        
        # Check that buffer lines are empty (last 2 lines should be empty)
        if len(visible_lines) >= 2:
            self.assertEqual(visible_lines[-1], "")  # Last line should be empty
            self.assertEqual(visible_lines[-2], "")    # Second to last should be empty
        
        # Test without Chronicles panel - no buffer lines
        self.tui.chronicle_visible = False
        # Complete the typewriter typing to avoid buffer lines
        self.tui.typewriter_visible_chars = self.tui.typewriter_total_chars
        self.tui.typewriter_render_buffer = self.tui.typewriter_full_text
        
        panel, panel_height = self.tui._build_typewriter_panel(panel_padding, max_height)
        content_text = str(panel.renderable)
        visible_lines = content_text.split('\n')
        
        # Should not have empty buffer lines at the end
        if len(visible_lines) > 0:
            self.assertNotEqual(visible_lines[-1], "")  # Last line should not be empty

    def test_buffer_lines_when_typewriter_complete(self):
        """Test that buffer lines are not added when typewriter typing is complete."""
        typewriter_text = """Line 1: First line.
Line 2: Second line.
Line 3: Third line."""
        
        typewriter_config = {'text': typewriter_text, 'pace': 'fast'}
        self.tui.set_typewriter_from_config(typewriter_config, Path('.'))
        self.tui.toggle_typewriter_panel()
        
        # Set fast typing for testing
        self.tui.typewriter_typing_interval_range = (0.0, 0.0)
        
        # Simulate a small panel (capacity = 2 lines)
        max_height = 4  # 2 content lines + 2 for border
        panel_padding = 1
        
        # Test with Chronicles panel visible but typewriter complete
        self.tui.chronicle_visible = True
        self.tui.typewriter_visible_chars = self.tui.typewriter_total_chars  # Fully typed
        self.tui.typewriter_render_buffer = self.tui.typewriter_full_text
        
        panel, panel_height = self.tui._build_typewriter_panel(panel_padding, max_height)
        self.assertIsNotNone(panel)
        
        content_text = str(panel.renderable)
        visible_lines = content_text.split('\n')
        
        # Should not have buffer lines when typewriter is complete
        if len(visible_lines) > 0:
            self.assertNotEqual(visible_lines[-1], "")  # Last line should not be empty

    def test_paragraph_spacing_preserved(self):
        """Test that paragraph spacing (double newlines) is preserved."""
        typewriter_with_paragraphs = """First paragraph with multiple words that should be wrapped properly when displayed in the terminal.

Second paragraph that is separated from the first by a blank line and also contains many words.

Third paragraph with more content."""
        
        typewriter_config = {'text': typewriter_with_paragraphs, 'pace': 'fast'}
        self.tui.set_typewriter_from_config(typewriter_config, Path('.'))
        self.tui.toggle_typewriter_panel()
        
        # Test word wrapping function directly
        original_lines = typewriter_with_paragraphs.split('\n')
        wrapped_lines = self.tui._wrap_text_lines(original_lines, 50)
        
        # Should preserve empty lines (paragraph separators)
        empty_line_count = sum(1 for line in wrapped_lines if not line.strip())
        self.assertGreater(empty_line_count, 0)  # Should have empty lines
        
        # Verify that empty lines are preserved in the right places
        # Find the positions of empty lines
        empty_positions = [i for i, line in enumerate(wrapped_lines) if not line.strip()]
        self.assertGreater(len(empty_positions), 0)  # Should have at least one empty line


if __name__ == '__main__':
    unittest.main()
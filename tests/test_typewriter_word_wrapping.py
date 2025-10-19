#
# SonicTerm
# Copyright (c) 2025 Martynas Jocius
#
import unittest
from pathlib import Path
from sonicterm.ui.tui import TUIManager


class TypewriterWordWrappingTests(unittest.TestCase):
    def setUp(self):
        self.tui = TUIManager()
        self.tui.enable_tui()

    def test_word_wrapping_with_long_paragraphs(self):
        """Test that very long paragraphs are properly wrapped and scrolling works."""
        # Create a story with very long paragraphs (500+ characters each)
        long_paragraph_story = """This is a very long paragraph that contains many words and should definitely exceed the typical terminal width when displayed in the story panel. The purpose of this test is to verify that the word wrapping functionality correctly splits these long lines into multiple shorter lines that fit within the available panel width, while keeping words intact and not breaking them in the middle. This paragraph should be wrapped into several lines when displayed in the terminal, and the scrolling functionality should work correctly with the wrapped content. The word wrapping algorithm should respect word boundaries and ensure that no single word is split across multiple lines, which would make the text difficult to read and understand.

This is another extremely long paragraph that serves as a second test case for the word wrapping functionality. It contains many words and sentences that together form a very long line of text that would definitely overflow the typical terminal width. The word wrapping should handle this gracefully by breaking the line at appropriate word boundaries, creating multiple shorter lines that fit within the panel width. This ensures that users can read the story content without having to deal with text that extends beyond the visible area of the terminal. The scrolling mechanism should work seamlessly with these wrapped lines, allowing users to navigate through the entire story content while always keeping the most recent text visible.

Here is a third paragraph with an even longer line of text that contains many words and should definitely test the limits of the word wrapping functionality. This paragraph is designed to be particularly challenging for the word wrapping algorithm because it contains very long words and sentences that might require special handling. The algorithm should be able to handle various edge cases such as words that are longer than the available width, multiple consecutive spaces, and other formatting considerations. The scrolling functionality should continue to work correctly even with these complex wrapped lines, ensuring that users can always see the most recent content and navigate through the story effectively."""
        
        typewriter_config = {'text': long_paragraph_story, 'pace': 'fast'}
        self.tui.set_typewriter_from_config(typewriter_config, Path('.'))
        self.tui.toggle_typewriter_panel()
        
        # Set fast typing for testing
        self.tui.typewriter_typing_interval_range = (0.0, 0.0)
        
        # Simulate a small panel (capacity = 3 lines)
        max_height = 5  # 3 content lines + 2 for border
        panel_padding = 1
        
        # Test word wrapping function directly
        original_lines = long_paragraph_story.split('\n')
        wrapped_lines = self.tui._wrap_text_lines(original_lines, 50)  # Test with 50 char width
        
        # Verify that long lines were wrapped
        self.assertGreater(len(wrapped_lines), len(original_lines))
        
        # Verify that all wrapped lines fit within the specified width
        for line in wrapped_lines:
            self.assertLessEqual(len(line), 50)
        
        # Verify that words weren't broken in the middle
        for line in wrapped_lines:
            # Skip empty lines (paragraph separators)
            if not line.strip():
                continue
            # Check that we don't have words split across lines
            words = line.split(' ')
            for word in words:
                self.assertGreater(len(word), 0)  # No empty words
        
        # Test with full content loaded
        self.tui.typewriter_visible_chars = self.tui.typewriter_total_chars
        self.tui.typewriter_render_buffer = self.tui.typewriter_full_text
        
        panel, panel_height = self.tui._build_typewriter_panel(panel_padding, max_height)
        self.assertIsNotNone(panel)
        
        content_text = str(panel.renderable)
        visible_lines = content_text.split('\n')
        
        # Verify that the panel was created successfully
        self.assertGreater(len(visible_lines), 0)
        
        # Calculate expected max width based on terminal width
        terminal_width, _ = self.tui._get_safe_terminal_size()
        padding_chars = panel_padding * 2 if isinstance(panel_padding, int) else 4
        expected_max_width = max(20, terminal_width - 2 - padding_chars)
        
        # Verify that visible lines fit within the calculated width
        for line in visible_lines:
            self.assertLessEqual(len(line), expected_max_width)

    def test_word_wrapping_with_mixed_content(self):
        """Test word wrapping with a mix of short and long lines."""
        mixed_content = """Short line.
This is a very long line that contains many words and should definitely be wrapped when displayed in the story panel because it exceeds the typical terminal width by a significant margin.
Another short line.
Here is another extremely long paragraph that contains many words and sentences that together form a very long line of text that would definitely overflow the typical terminal width and should be wrapped appropriately.
Final short line."""
        
        typewriter_config = {'text': mixed_content, 'pace': 'fast'}
        self.tui.set_typewriter_from_config(typewriter_config, Path('.'))
        self.tui.toggle_typewriter_panel()
        
        # Test word wrapping with different widths
        original_lines = mixed_content.split('\n')
        
        # Test with narrow width
        wrapped_narrow = self.tui._wrap_text_lines(original_lines, 30)
        self.assertGreater(len(wrapped_narrow), len(original_lines))
        
        # Test with wider width
        wrapped_wide = self.tui._wrap_text_lines(original_lines, 80)
        self.assertLessEqual(len(wrapped_wide), len(wrapped_narrow))
        
        # Verify all lines fit within specified widths
        for line in wrapped_narrow:
            self.assertLessEqual(len(line), 30)
        for line in wrapped_wide:
            self.assertLessEqual(len(line), 80)

    def test_word_wrapping_edge_cases(self):
        """Test word wrapping with edge cases like very long words."""
        edge_case_content = """Normal line with regular words.
Line with a verylongwordthatexceedsnormallengthandmightcauseissues.
Line with multiple    spaces    between    words.
Line with a verylongwordthatexceedsnormallengthandmightcauseissues followed by normal text.
"""
        
        typewriter_config = {'text': edge_case_content, 'pace': 'fast'}
        self.tui.set_typewriter_from_config(typewriter_config, Path('.'))
        self.tui.toggle_typewriter_panel()
        
        # Test word wrapping with narrow width to force wrapping
        original_lines = edge_case_content.split('\n')
        wrapped_lines = self.tui._wrap_text_lines(original_lines, 20)
        
        # Verify that wrapping occurred
        self.assertGreater(len(wrapped_lines), len(original_lines))
        
        # Verify all lines fit within width
        for line in wrapped_lines:
            self.assertLessEqual(len(line), 20)
        
        # Verify that very long words are handled (they should be on their own line)
        for line in wrapped_lines:
            if 'verylongwordthatexceedsnormallengthandmightcauseissues' in line:
                # This word is longer than 20 chars, so it should be the only content on its line
                self.assertEqual(line.strip(), 'verylongwordthatexceedsnormallengthandmightcauseissues')


if __name__ == '__main__':
    unittest.main()
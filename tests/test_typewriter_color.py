#
# SonicTerm
# Copyright (c) 2025 Martynas Jocius
#
import unittest
from pathlib import Path
from sonicterm.ui.tui import TUIManager


class TypewriterColorTests(unittest.TestCase):
    def setUp(self):
        self.tui = TUIManager()
        self.tui.enable_tui()
        # Disable chronicle panel to avoid buffer line complications in tests
        self.tui.chronicle_visible = False

    def test_default_color(self):
        """Test that default color is bright_white when no color is specified."""
        typewriter_config = {'text': 'Test typewriter', 'pace': 'fast'}
        self.tui.set_typewriter_from_config(typewriter_config, Path('.'))
        self.tui.toggle_typewriter_panel()
        
        # Set fast typing for testing
        self.tui.typewriter_typing_interval_range = (0.0, 0.0)
        self.tui.typewriter_visible_chars = self.tui.typewriter_total_chars
        self.tui.typewriter_render_buffer = self.tui.typewriter_full_text
        
        panel, panel_height = self.tui._build_typewriter_panel(1, 5)
        self.assertIsNotNone(panel)
        
        # Check that default color is bright_white
        self.assertEqual(self.tui.typewriter_text_color, "bright_white")

    def test_custom_color_from_config(self):
        """Test that custom color is applied from scene configuration."""
        typewriter_config = {
            'text': 'Test typewriter with custom color',
            'pace': 'fast',
            'color': 'red'
        }
        self.tui.set_typewriter_from_config(typewriter_config, Path('.'))
        self.tui.toggle_typewriter_panel()
        
        # Set fast typing for testing
        self.tui.typewriter_typing_interval_range = (0.0, 0.0)
        self.tui.typewriter_visible_chars = self.tui.typewriter_total_chars
        self.tui.typewriter_render_buffer = self.tui.typewriter_full_text
        
        panel, panel_height = self.tui._build_typewriter_panel(1, 5)
        self.assertIsNotNone(panel)
        
        # Check that custom color is applied
        self.assertEqual(self.tui.typewriter_text_color, "red")

    def test_color_case_insensitive(self):
        """Test that color attribute is case insensitive."""
        typewriter_config = {
            'text': 'Test typewriter',
            'pace': 'fast',
            'color': 'BLUE'  # Uppercase
        }
        self.tui.set_typewriter_from_config(typewriter_config, Path('.'))
        
        # Check that color is converted to lowercase
        self.assertEqual(self.tui.typewriter_text_color, "blue")

    def test_extended_color_support(self):
        """Test that extended colors like pink, purple, bright colors work."""
        extended_colors = ["pink", "purple", "orange", "bright_red", "bright_green", "dim_blue", "bright_white"]
        
        for color in extended_colors:
            with self.subTest(color=color):
                typewriter_config = {
                    'text': f'Test typewriter with {color}',
                    'pace': 'fast',
                    'color': color
                }
                self.tui.set_typewriter_from_config(typewriter_config, Path('.'))
                
                # Ensure panel is visible for testing
                self.tui.typewriter_visible = True
                self.tui.typewriter_visible_chars = self.tui.typewriter_total_chars
                self.tui.typewriter_render_buffer = self.tui.typewriter_full_text
                
                panel, panel_height = self.tui._build_typewriter_panel(1, 5)
                self.assertIsNotNone(panel, f"Panel creation failed for color '{color}'")
                self.assertEqual(self.tui.typewriter_text_color, color, f"Color '{color}' not applied correctly")

    def test_color_reset_on_typewriter_change(self):
        """Test that color resets to default when typewriter changes."""
        # Set initial typewriter with custom color
        typewriter_config1 = {
            'text': 'First typewriter',
            'pace': 'fast',
            'color': 'green'
        }
        self.tui.set_typewriter_from_config(typewriter_config1, Path('.'))
        self.assertEqual(self.tui.typewriter_text_color, "green")
        
        # Set new typewriter without color
        typewriter_config2 = {
            'text': 'Second typewriter',
            'pace': 'fast'
        }
        self.tui.set_typewriter_from_config(typewriter_config2, Path('.'))
        
        # Color should reset to default
        self.assertEqual(self.tui.typewriter_text_color, "bright_white")

    def test_color_with_path_based_typewriter(self):
        """Test that color works with path-based stories."""
        # Create a temporary typewriter file
        typewriter_path = Path("temp_typewriter.txt")
        typewriter_path.write_text("This is a test typewriter from file.")
        
        try:
            typewriter_config = {
                'path': 'temp_typewriter.txt',
                'pace': 'fast',
                'color': 'yellow'
            }
            self.tui.set_typewriter_from_config(typewriter_config, Path('.'))
            
            # Check that color is applied
            self.assertEqual(self.tui.typewriter_text_color, "yellow")
            
        finally:
            # Clean up
            if typewriter_path.exists():
                typewriter_path.unlink()

    def test_panel_title_is_typewriter(self):
        """Test that panel title is 'Typewriter' instead of 'Story'."""
        typewriter_config = {'text': 'Test typewriter', 'pace': 'fast'}
        self.tui.set_typewriter_from_config(typewriter_config, Path('.'))
        self.tui.toggle_typewriter_panel()
        
        # Set fast typing for testing
        self.tui.typewriter_typing_interval_range = (0.0, 0.0)
        self.tui.typewriter_visible_chars = self.tui.typewriter_total_chars
        self.tui.typewriter_render_buffer = self.tui.typewriter_full_text
        
        panel, panel_height = self.tui._build_typewriter_panel(1, 5)
        self.assertIsNotNone(panel)
        
        # Check that panel title is "Typewriter"
        # The panel title is not in the renderable content, it's in the panel's title attribute
        self.assertIsNotNone(panel)
        # We can verify the title by checking the panel's title attribute
        self.assertEqual(panel.title, "Typewriter")


if __name__ == '__main__':
    unittest.main()
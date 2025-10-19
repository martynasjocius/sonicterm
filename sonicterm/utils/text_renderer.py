#!/usr/bin/env python3
#
# SonicTerm
# Copyright (c) 2025 Martynas Jocius
#
"""Block-based text rendering utilities for SonicTerm."""

from typing import List, Tuple
import numpy as np


def get_letter_patterns() -> dict:
    """
    Get 5x7 pixel patterns for letters.
    Each letter is represented as a list of 7 rows, each row is 5 bits.
    1 = filled pixel, 0 = empty pixel
    """
    return {
        'S': [
            0b01110,  # .███.
            0b10000,  # █....
            0b10000,  # █....
            0b01110,  # .███.
            0b00001,  # ....█
            0b00001,  # ....█
            0b11110,  # ████.
        ],
        'O': [
            0b01110,  # .███.
            0b10001,  # █...█
            0b10001,  # █...█
            0b10001,  # █...█
            0b10001,  # █...█
            0b10001,  # █...█
            0b01110,  # .███.
        ],
        'N': [
            0b10001,  # █...█
            0b11001,  # ██..█
            0b10101,  # █.█.█
            0b10011,  # █..██
            0b10001,  # █...█
            0b10001,  # █...█
            0b10001,  # █...█
        ],
        'I': [
            0b11111,  # █████
            0b00100,  # ..█..
            0b00100,  # ..█..
            0b00100,  # ..█..
            0b00100,  # ..█..
            0b00100,  # ..█..
            0b11111,  # █████
        ],
        'C': [
            0b01110,  # .███.
            0b10001,  # █...█
            0b10000,  # █....
            0b10000,  # █....
            0b10000,  # █....
            0b10001,  # █...█
            0b01110,  # .███.
        ],
        'T': [
            0b11111,  # █████
            0b00100,  # ..█..
            0b00100,  # ..█..
            0b00100,  # ..█..
            0b00100,  # ..█..
            0b00100,  # ..█..
            0b00100,  # ..█..
        ],
        'E': [
            0b11111,  # █████
            0b10000,  # █....
            0b10000,  # █....
            0b11110,  # ████.
            0b10000,  # █....
            0b10000,  # █....
            0b11111,  # █████
        ],
        'R': [
            0b11110,  # ████.
            0b10001,  # █...█
            0b10001,  # █...█
            0b11110,  # ████.
            0b10100,  # █.█..
            0b10010,  # █..█.
            0b10001,  # █...█
        ],
        'M': [
            0b10001,  # █...█
            0b11011,  # ██.██
            0b10101,  # █.█.█
            0b10101,  # █.█.█
            0b10001,  # █...█
            0b10001,  # █...█
            0b10001,  # █...█
        ],
        ' ': [
            0b00000,  # .....
            0b00000,  # .....
            0b00000,  # .....
            0b00000,  # .....
            0b00000,  # .....
            0b00000,  # .....
            0b00000,  # .....
        ],
    }


def render_text_to_matrix(text: str, matrix_width: int, matrix_height: int, 
                         color_palette: List[Tuple[int, int, int]]) -> np.ndarray:
    """
    Render text to a color matrix using block characters.
    
    Args:
        text: Text to render (will be converted to uppercase)
        matrix_width: Width of the target matrix
        matrix_height: Height of the target matrix  
        color_palette: List of RGB tuples to use for coloring
        
    Returns:
        numpy array of shape (matrix_height, matrix_width, 3) with RGB values
    """
    text = text.upper()
    patterns = get_letter_patterns()
    
    # Calculate text dimensions
    letter_width = 5
    letter_height = 7
    letter_spacing = 1  # Space between letters
    
    # Calculate total text width
    total_text_width = len(text) * letter_width + (len(text) - 1) * letter_spacing
    
    # Calculate scaling to fit the matrix
    scale_x = max(1, matrix_width // (total_text_width + 4))  # +4 for padding
    scale_y = max(1, matrix_height // (letter_height + 2))    # +2 for padding
    scale = min(scale_x, scale_y, 4)  # Limit maximum scale to 4 for readability
    
    # Calculate scaled dimensions
    scaled_letter_width = letter_width * scale
    scaled_letter_height = letter_height * scale
    scaled_spacing = letter_spacing * scale
    scaled_text_width = len(text) * scaled_letter_width + (len(text) - 1) * scaled_spacing
    
    # Calculate centering offsets
    start_x = (matrix_width - scaled_text_width) // 2
    start_y = max(0, (matrix_height - scaled_letter_height) // 2 - 3)
    
    # Create the matrix
    matrix = np.zeros((matrix_height, matrix_width, 3), dtype=np.uint8)
    
    # Choose colors from palette (use brighter colors for text)
    if len(color_palette) >= 20:
        # Use colors from the latter part of the palette (brighter)
        text_colors = color_palette[-8:]
    else:
        # Use all available colors
        text_colors = color_palette
    
    # Render each letter
    current_x = start_x
    for char_idx, char in enumerate(text):
        if char not in patterns:
            char = ' '  # Use space for unknown characters
            
        pattern = patterns[char]
        
        # Choose color for this letter (cycle through available colors)
        color = text_colors[char_idx % len(text_colors)]
        
        # Render the letter pattern
        for row in range(letter_height):
            if start_y + row * scale < 0 or start_y + row * scale >= matrix_height:
                continue
                
            pattern_row = pattern[row]
            for col in range(letter_width):
                if current_x + col * scale < 0 or current_x + col * scale >= matrix_width:
                    continue
                    
                # Check if this pixel should be filled
                if (pattern_row >> (letter_width - 1 - col)) & 1:
                    # Fill a scaled block
                    for sy in range(scale):
                        for sx in range(scale):
                            y_pos = start_y + row * scale + sy
                            x_pos = current_x + col * scale + sx
                            
                            if (0 <= y_pos < matrix_height and 
                                0 <= x_pos < matrix_width):
                                matrix[y_pos, x_pos] = color
        
        # Move to next letter position
        current_x += scaled_letter_width + scaled_spacing
    
    return matrix


def create_splash_matrix(matrix_width: int, matrix_height: int, 
                        color_palette: List[Tuple[int, int, int]]) -> np.ndarray:
    """
    Create a splash screen matrix with "SONICTERM" text.
    
    Args:
        matrix_width: Width of the target matrix
        matrix_height: Height of the target matrix
        color_palette: List of RGB tuples to use for coloring
        
    Returns:
        numpy array of shape (matrix_height, matrix_width, 3) with RGB values
    """
    return render_text_to_matrix("SONICTERM", matrix_width, matrix_height, color_palette)

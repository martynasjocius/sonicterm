#!/usr/bin/env python3
#
# SonicTerm
# Copyright (c) 2025 Martynas Jocius
#
"""Shared color palette helpers for SonicTerm."""

import random
from typing import List, Tuple


def get_dark_nature_palette() -> List[Tuple[int, int, int]]:
    """
    Get unified dark nature-inspired color palette.
    
    Returns RGB tuples for dark, natural colors including:
    - Various greens (forest, moss, pine)
    - Dark blues (deep water, night sky)
    - Dark browns (earth, bark, soil)
    - Blacks and very dark grays
    - Dark purples (deep forest shadows)
    """
    return [
        # Very dark colors (blacks and near-blacks)
        (0, 0, 0),           # Pure black
        (10, 10, 10),        # Very dark gray
        (15, 20, 15),        # Very dark forest green
        (20, 15, 25),        # Very dark purple
        
        # Dark greens (forest, moss, pine)
        (0, 40, 0),          # Very dark green
        (10, 50, 10),        # Dark forest green
        (20, 60, 20),        # Dark moss green
        (15, 45, 25),        # Dark pine green
        (25, 55, 15),        # Dark olive green
        
        # Dark blues (deep water, night)
        (0, 20, 40),         # Very dark blue
        (10, 25, 45),        # Dark navy blue
        (5, 30, 50),         # Dark ocean blue
        (15, 35, 55),        # Dark slate blue
        
        # Dark browns (earth, bark, soil)
        (40, 25, 10),        # Very dark brown
        (45, 30, 15),        # Dark bark brown
        (50, 35, 20),        # Dark earth brown
        (35, 20, 10),        # Very dark soil
        
        # Dark purples (deep shadows)
        (30, 15, 40),        # Dark purple
        (35, 20, 45),        # Dark violet
        (25, 10, 35),        # Very dark purple
        
        # Medium-dark nature colors
        (30, 70, 30),        # Medium dark green
        (40, 80, 40),        # Forest green
        (20, 60, 80),        # Dark teal
        (60, 40, 25),        # Medium dark brown
        (45, 25, 55),        # Medium dark purple
    ]


def get_warm_camera_palette() -> List[Tuple[int, int, int]]:
    """Return a warm palette of ambers, yellows, and oranges for camera visuals."""

    return [
        (8, 4, 0),      # Deep ember
        (16, 6, 0),     # Dark amber
        (22, 10, 0),    # Burnt orange base
        (30, 14, 0),    # Rich ember glow
        (42, 20, 0),    # Toasted orange
        (54, 28, 0),    # Copper tone
        (66, 34, 2),    # Warm copper
        (80, 40, 4),    # Dark tangerine
        (96, 48, 6),    # Sunset orange
        (112, 56, 8),   # Golden orange
        (128, 64, 10),  # Bright amber
        (144, 72, 12),  # Goldenrod
        (164, 84, 16),  # Warm honey
        (184, 96, 20),  # Soft marigold
        (200, 110, 24), # Bright marigold
        (212, 122, 30), # Golden yellow
        (224, 134, 36), # Sunlit gold
        (236, 148, 42), # Warm amber
        (244, 160, 50), # Soft golden glow
        (252, 174, 56), # Gentle golden highlight
    ]


def get_rich_color_names() -> List[str]:
    """
    Get Rich color names that correspond to our dark nature palette.
    Used for random mode generation.
    """
    return [
        # Very dark colors
        'black',
        'dim white',  # Very dark gray
        'dim green',
        'dim magenta',
        
        # Dark greens
        'dim green',
        'green',
        'dim green',
        'green',
        'dim green',
        
        # Dark blues  
        'dim blue',
        'dim cyan',
        'dim blue',
        'dim cyan',
        
        # Dark browns (using dim yellow/magenta as approximations)
        'dim yellow',
        'dim yellow',
        'dim yellow', 
        'dim yellow',
        
        # Dark purples
        'dim magenta',
        'dim magenta',
        'dim magenta',
        
        # Medium-dark colors
        'green',
        'green',
        'cyan',
        'dim yellow',
        'magenta',
    ]


def get_random_dark_nature_color() -> str:
    """Get a random Rich color name from the dark nature palette."""
    colors = get_rich_color_names()
    return random.choice(colors)


def map_rich_color_to_rgb(color_name: str) -> Tuple[int, int, int]:
    """
    Map Rich color names to RGB values for our dark nature palette.
    """
    color_map = {
        # Very dark colors
        'black': (0, 0, 0),
        'dim white': (15, 15, 15),  # Very dark gray
        
        # Greens
        'dim green': (10, 50, 10),   # Dark forest green
        'green': (30, 70, 30),       # Medium dark green
        'bright_green': (40, 80, 40), # Forest green (not too bright)
        
        # Blues
        'dim blue': (0, 20, 40),     # Very dark blue
        'blue': (20, 40, 60),        # Dark blue (not too bright)
        'dim cyan': (5, 30, 50),     # Dark ocean blue
        'cyan': (20, 60, 80),        # Dark teal
        'bright_cyan': (25, 65, 85), # Slightly brighter teal
        
        # Browns (using yellow variations)
        'dim yellow': (40, 25, 10),  # Very dark brown
        'yellow': (60, 40, 25),      # Medium dark brown
        'bright_yellow': (70, 45, 30), # Lighter brown (not bright yellow)
        
        # Purples
        'dim magenta': (30, 15, 40), # Dark purple
        'magenta': (45, 25, 55),     # Medium dark purple
        'bright_magenta': (50, 30, 60), # Slightly brighter purple
        
        # Whites (pure white)
        'white': (255, 255, 255),     # Pure white
        'bright_white': (255, 255, 255), # Pure white (same as white)
    }
    
    return color_map.get(color_name, (20, 20, 20))  # Default to dark gray

#!/usr/bin/env python3
#
# SonicTerm
# Copyright (c) 2025 Martynas Jocius
#

import os
import random
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import psutil
from rich.align import Align
from rich.console import Console, Group
from rich.live import Live
from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .. import (
    STATE_ACTIVE,
    STATE_WAITING,
    STATE_LOADING,
    UI_APP_NAME,
    UI_PANEL_SCENE,
    UI_PANEL_CHRONICLE,
    UI_COLUMN_SOUND,
    UI_COLUMN_PROGRESS,
    UI_COLUMN_STATUS,
    UI_NO_ACTIVE_SOUNDS,
    VERSION,
    CAMERA_UPDATE_INTERVAL,
)

# Progress bar constants
SHORT_SAMPLE_THRESHOLD = (
    8.0  # Seconds - samples shorter than this show pulsing full-width bars
)
COLOR_MATRIX_EXTRA_WIDTH = 4  # Expand color matrix width to use additional panel space
COLOR_MATRIX_EXTRA_HEIGHT = (
    2  # Expand color matrix height to use additional panel space
)

DEFAULT_PANEL_PADDING = (1, 2)

VISUAL_PANEL_TITLES = {
    "activity_map": "Activity Map",
    "camera_matrix": "Camera Matrix",
}

TYPEWRITER_PACE_RANGES = {
    "default": (0.02, 0.08),
    "slow": (0.05, 0.14),
    "medium": (0.015, 0.06),
    "fast": (0.01, 0.04),
}

TYPEWRITER_PUNCTUATION_PAUSES = {
    ".": 0.25,
    "!": 0.25,
    "?": 0.25,
    ",": 0.12,
    ";": 0.12,
    ":": 0.12,
    "\n": 0.18,
    "…": 0.3,
}

# Debug support
from ..debug import debug_log, update_state, is_debug_enabled

# Unified color palette
from ..utils.colors import get_dark_nature_palette

try:
    import numpy as np  # type: ignore
except ImportError:  # pragma: no cover - numpy is optional for plugin rendering
    np = None


@dataclass
class LayoutContext:
    panel_padding: Tuple[int, int]
    sample_count: int
    terminal_width: int
    terminal_height: int
    current_mode: str
    camera_layout_active: bool
    header_height: int
    display_footer: bool
    footer_min_height: int
    visual_min_total: int
    visual_available_total: int
    available_main_height: int
    main_panel_height: int
    margin_reserved: int
    max_colors_height: int




@dataclass
class HelpDialogBuilder:
    app_name: str
    version: str
    keymaps: Sequence[Tuple[str, str]] = (
        ("?", "Show help"),
        ("q", "Quit SonicTerm"),
        ("Ctrl+C", "Force quit"),
        ("Space", "Pause / resume"),
        ("v", "Toggle visuals"),
        ("c", "Toggle log panel"),
        ("m", "Switch map mode"),
        ("t", "Toggle typewriter"),
        ("Esc", "Close dialog"),
    )
    website_url: str = "https://sonicterm.org"
    repo_url: str = "https://github.com/martynasjocius/sonicterm"

    def build(self) -> "Group":
        title, tagline, authors, divider = self._build_header()
        help_table = self._build_table()
        footer_items = self._build_footer()

        content: List = [
            Align.center(title),
            Text(""),
            Align.center(tagline),
            Text(""),
            Align.center(divider),
            Text(""),
            help_table,
            Text(""),
            Align.center(divider),
            Text(""),
            Align.center(authors),
            Text(""),
        ]
        content.extend(footer_items)
        return Group(*content)

    def _build_header(self) -> Tuple[Text, Text, Text, Text]:
        title_text = Text()
        title_text.append(f"{self.app_name} {self.version}", style="bold bright_cyan")

        tagline_text = Text()
        tagline_text.append(
            "Terminal soundscape engine for installations", style="white"
        )

        authors_text = Text()
        authors_text.append(
            "Created and maintained by Martynas Jocius • Licensed under MIT",
            style="white",
        )

        divider_text = Text("─" * 60, style="dim white")
        return title_text, tagline_text, authors_text, divider_text

    def _build_table(self) -> Table:
        help_table = Table(
            show_header=True, header_style="bold cyan", box=None, padding=(0, 2)
        )
        help_table.add_column("Key", style="bold yellow", width=10)
        help_table.add_column("Action", style="white", width=25, justify="left")
        help_table.add_column("Key", style="bold yellow", width=10)
        help_table.add_column("Action", style="white", width=25, justify="left")

        rows: List[Tuple[str, str, str, str]] = []
        keymap_list = list(self.keymaps)
        for i in range(0, len(keymap_list), 2):
            first_key, first_action = keymap_list[i]
            if i + 1 < len(keymap_list):
                second_key, second_action = keymap_list[i + 1]
            else:
                second_key, second_action = "", ""
            rows.append(
                (
                    first_key,
                    f"  {first_action}" if first_action else "",
                    second_key,
                    f"  {second_action}" if second_action else "",
                )
            )

        for row in rows:
            help_table.add_row(*row)

        return help_table

    def _build_footer(self) -> List:
        footer_text = Text()
        footer_text.append("Press ", style="dim white")
        footer_text.append("q", style="bold yellow")
        footer_text.append(" or ", style="dim white")
        footer_text.append("Esc", style="bold yellow")
        footer_text.append(" to close this dialog", style="dim white")

        learn_more_text = Text()
        learn_more_text.append("Learn more at ", style="dim white")
        learn_more_text.append(self.website_url, style="cyan")

        report_issues_text = Text()
        report_issues_text.append("Report issues at ", style="dim white")
        report_issues_text.append(self.repo_url, style="cyan")

        divider_text = Text("─" * 60, style="dim white")

        return [
            Align.center(learn_more_text),
            Text(""),
            Align.center(report_issues_text),
            Text(""),
            Align.center(divider_text),
            Text(""),
            Align.center(footer_text),
        ]

@dataclass
class ColorMatrixPanelBuilder:
    manager: "TUIManager"
    layout: Layout
    panel_padding: Tuple[int, int]

    def build(self) -> None:
        self._compute_dimensions()
        content, title = self._render_content()
        self._apply_panel(content, title)

    def _compute_dimensions(self) -> None:
        inner_width, inner_height = self.manager._compute_color_matrix_dimensions(
            self.panel_padding
        )
        self.inner_width = inner_width
        self.inner_height = inner_height

        panel_size = getattr(self.layout["colors"], "size", None)
        vertical_pad, horizontal_pad = self.manager._padding_metrics(
            self.panel_padding
        )

        if panel_size is not None:
            available_width = max(1, panel_size.width - horizontal_pad - 2)
            available_height = max(1, panel_size.height - vertical_pad - 2)
        else:
            available_width = inner_width
            available_height = inner_height

        target_width = max(1, min(inner_width, available_width))
        target_height = max(1, min(inner_height, available_height))
        usable_height = max(1, target_height)

        dims = (target_width, target_height)
        manager = self.manager
        measured_inner = (target_width, usable_height)

        if dims != manager._last_visual_dimensions or manager._measured_visual_inner != measured_inner:
            manager._record_visual_panel_size(
                target_width, target_height, interior=usable_height
            )
            manager._measured_visual_inner = measured_inner
            manager._cached_plugin_frame = None
            manager._last_plugin_frame_time = 0.0
        manager._last_visual_dimensions = dims

        self.target_width = target_width
        self.target_height = target_height
        self.usable_height = usable_height

    def _render_content(self) -> Tuple[Text, str]:
        mode = self.manager.get_current_map_mode()
        if self.manager.plugin_map_mode and mode == self.manager.plugin_map_mode:
            return self._render_plugin_matrix()
        if mode == "camera":
            return self._render_camera_matrix()
        return self._render_activity_matrix(mode)

    def _render_camera_matrix(self) -> Tuple[Text, str]:
        manager = self.manager
        try:
            from ..core.scene import scene_manager
        except ImportError:
            scene_manager = None

        if scene_manager and hasattr(scene_manager, "request_webcam_resize"):
            try:
                scene_manager.request_webcam_resize(
                    self.inner_width, self.inner_height
                )
            except Exception:
                pass

            self.last_camera_error = getattr(scene_manager, "webcam_last_error", None)
        else:
            self.last_camera_error = None

        if manager.webcam_matrix is None and scene_manager is not None:
            fallback_matrix = getattr(scene_manager, "color_matrix", None)
            if fallback_matrix is not None:
                manager.webcam_matrix = fallback_matrix

        content = Text()
        matrix = manager.webcam_matrix
        title = VISUAL_PANEL_TITLES["camera_matrix"]

        if matrix is not None:
            height, width, _ = matrix.shape
            scale_x = max(1, self.target_width // max(width, 1))
            scale_y = max(1, self.usable_height // max(height, 1))

            if is_debug_enabled():
                debug_log(
                    "WEB_CAM_RENDER_SCALE",
                    "Prepared webcam render scaling",
                    {
                        "scale_x": scale_x,
                        "scale_y": scale_y,
                        "inner_width": self.inner_width,
                        "inner_height": self.inner_height,
                        "matrix_shape": (height, width),
                    },
                )

            while scale_x * width > self.inner_width and scale_x > 1:
                scale_x -= 1
            while scale_y * height > self.inner_height and scale_y > 1:
                scale_y -= 1

            total_rows = height * scale_y
            rendered_rows = 0
            for row in range(height):
                for _ in range(scale_y):
                    line = Text()
                    for col in range(width):
                        r, g, b = matrix[row, col]
                        line.append("█" * scale_x, style=f"rgb({r},{g},{b})")
                    padding = self.target_width - len(line.plain)
                    if padding > 0:
                        line.append(" " * padding)
                    content.append(line)
                    rendered_rows += 1
                    if rendered_rows < total_rows:
                        content.append("\n")
            return content, title

        placeholder = "Camera feed initializing…"
        if self.last_camera_error:
            placeholder = f"Camera feed error: {self.last_camera_error}"
        content.append(placeholder, style="dim white")
        return content, title

    def _render_plugin_matrix(self) -> Tuple[Text, str]:
        manager = self.manager
        content = Text()
        title = manager.plugin_display_name or "Visual Plugin"
        matrix_result, error_message = manager._request_visual_plugin_frame(
            self.target_width, self.usable_height
        )

        if matrix_result is None:
            placeholder = error_message or "Visual plugin inactive"
            content.append(placeholder, style="dim white")
            return content, title

        debug_log(
            "TUI_PLUGIN_DIMENSIONS",
            "Visual plugin render request",
            {
                "inner_width": self.inner_width,
                "inner_height": self.inner_height,
                "mode": manager.get_current_map_mode(),
            },
        )

        if np is not None:
            try:
                matrix_array = np.asarray(matrix_result, dtype=np.uint8)
                use_numpy = True
            except Exception:
                matrix_array = matrix_result
                use_numpy = False
        else:
            matrix_array = matrix_result
            use_numpy = False

        try:
            if use_numpy:
                plugin_height, plugin_width = matrix_array.shape[:2]
            else:
                plugin_height = len(matrix_array)
                plugin_width = len(matrix_array[0]) if plugin_height else 0
        except Exception:
            plugin_height = plugin_width = 0

        if plugin_height <= 0 or plugin_width <= 0:
            content.append("Visual plugin returned invalid frame", style="dim white")
            return content, title

        col_indices = self._sample_indices(plugin_width, self.target_width)
        row_indices = self._sample_indices(plugin_height, self.usable_height)

        for ridx_idx, row_index in enumerate(row_indices):
            line = Text()
            for col_index in col_indices:
                if use_numpy:
                    r, g, b = matrix_array[row_index, col_index][:3]
                else:
                    r, g, b = matrix_array[row_index][col_index][:3]
                line.append("█", style=f"rgb({int(r)},{int(g)},{int(b)})")
            padding = self.target_width - len(line.plain)
            if padding > 0:
                line.append(" " * padding)
            content.append(line)
            if ridx_idx < len(row_indices) - 1:
                content.append("\n")

        return content, title

    def _render_activity_matrix(self, mode: str) -> Tuple[Text, str]:
        manager = self.manager
        grid_width = max(8, self.target_width)
        grid_height = max(3, self.usable_height)

        current_time = time.time()
        grid_size = grid_width * grid_height

        if not hasattr(manager, "color_grid") or len(manager.color_grid) != grid_size:
            manager.color_grid = manager._generate_color_grid(grid_width, grid_height)
        elif current_time - manager.last_activity_time < 1.0:
            manager._update_color_grid_partial(grid_width, grid_height)

        content = Text()
        for row in range(grid_height):
            for col in range(grid_width):
                idx = row * grid_width + col
                r, g, b = manager.color_grid[idx]
                content.append("█", style=f"rgb({r},{g},{b})")
            if row < grid_height - 1:
                content.append("\n")

        if mode == "random":
            title = VISUAL_PANEL_TITLES["activity_map"]
        else:
            title = f"Map ({mode.title()}, {grid_width}×{grid_height} cells)"
        return content, title

    def _apply_panel(self, content: Text, title: str) -> None:
        centered_content = Align.center(content, vertical="top")
        self.layout["colors"].update(
            Panel(
                centered_content,
                title=title,
                border_style="magenta",
                padding=self.panel_padding,
                expand=True,
            )
        )

        actual_size = getattr(self.layout["colors"], "size", None)
        manager = self.manager
        if actual_size is not None:
            vertical_pad, horizontal_pad = manager._padding_metrics(self.panel_padding)
            actual_width = max(1, actual_size.width - horizontal_pad - 2)
            actual_height = max(1, actual_size.height - vertical_pad - 2)
            actual_inner = max(1, actual_height)
            actual_dims = (actual_width, actual_height)
            if (
                actual_dims != manager._last_visual_dimensions
                or (actual_width, actual_inner) != manager._measured_visual_inner
            ):
                manager._record_visual_panel_size(
                    actual_width, actual_height, interior=actual_inner
                )
                manager._measured_visual_inner = (actual_width, actual_inner)
                manager._cached_plugin_frame = None
                manager._last_plugin_frame_time = 0.0
                manager._last_visual_dimensions = actual_dims

    @staticmethod
    def _sample_indices(size: int, target: int) -> List[int]:
        target = max(1, min(target, size))
        if target == size:
            return list(range(size))
        step = size / target
        return [min(size - 1, int(i * step)) for i in range(target)]



class TUIManager:
    """Manages the terminal user interface for SonicTerm."""

    @staticmethod
    def _padding_metrics(padding):
        """Calculate vertical and horizontal padding totals."""
        if isinstance(padding, tuple):
            if len(padding) == 2:
                return padding[0] * 2, padding[1] * 2
            if len(padding) == 4:
                top, right, bottom, left = padding
                return top + bottom, left + right
        return 0, 0

    @staticmethod
    def _panel_vertical_chrome(padding):
        """Return panel chrome height (padding + borders)."""
        vertical_pad, _ = TUIManager._padding_metrics(padding)
        return vertical_pad + 2  # Top and bottom borders consume two rows

    def _compute_color_matrix_dimensions(self, panel_padding):
        """Compute inner dimensions for the color matrix panel."""
        vertical_pad, horizontal_pad = self._padding_metrics(panel_padding)
        terminal_width, terminal_height = self._get_safe_terminal_size()
        current_mode = self.get_current_map_mode()
        camera_layout_active = self.visual_mode and current_mode == "camera"

        base_width = terminal_width - 6
        base_width = max(base_width, 20)

        header_reserved = 3
        footer_reserved = 10 if self.chronicle_visible else 0
        margin_reserved = 2
        base_height = (
            terminal_height - header_reserved - footer_reserved - margin_reserved
        )
        base_height = max(base_height, 5)

        if self._max_colors_height is not None:
            base_height = min(base_height, self._max_colors_height)

        inner_width_base = max(base_width - horizontal_pad, 8)
        inner_width = inner_width_base + COLOR_MATRIX_EXTRA_WIDTH

        inner_height = max(base_height - vertical_pad, 3) + COLOR_MATRIX_EXTRA_HEIGHT

        return inner_width, inner_height

    def __init__(self):
        self.console = Console()
        self.log_buffer = deque(maxlen=50)  # Keep last 50 log messages
        self.log_file_handle = None
        self.log_file_path = None
        self.players_status = {}  # Track player status
        self.scene_name = "Unknown Scene"
        self.live_display = None
        self.tui_enabled = False
        self.chronicle_visible = True  # Processes panel visible by default
        self.color_grid = []  # Store color rectangles
        self.webcam_matrix = None  # Store webcam color matrix
        self.last_activity_time = time.time()
        self.progress_animation_index = 0
        self.is_quitting = False
        self.is_recording = False
        self.is_stopped = False
        self.record_indicator_visible = False
        self.last_record_indicator_toggle = time.time()
        self.progress_chars = ["◐", "◓", "◑", "◒"]  # Circular progress animation
        self.loading_chars = ["○", "●"]  # Loading animation: empty and filled circles
        self.loading_animation_index = 0

        # Typewriter panel state
        self.typewriter_visible = False
        self.typewriter_available = False
        self.typewriter_full_text = ""
        self.typewriter_render_buffer = ""
        self.typewriter_visible_chars = 0
        self.typewriter_total_chars = 0
        self.typewriter_next_char_time = time.time()
        self.typewriter_typing_interval_range = TYPEWRITER_PACE_RANGES["default"]
        self.typewriter_unavailable_message = (
            "No typewriter text in scene configuration."
        )
        self.typewriter_completion_time = 0  # Time when typing completed
        # Typewriter panel scrolling state
        self.typewriter_scroll_position = 0  # Current scroll position (line index)
        self.typewriter_last_content_height = (
            0  # Track previous content height for scroll adjustment
        )
        self.typewriter_text_color = (
            "bright_white"  # Default text color for typewriter panel
        )
        self.typewriter_configured_color = (
            "bright_white"  # Store the configured color separately
        )
        self.start_time = time.time()  # Track application start time
        self.last_cpu_check = 0  # Track last CPU measurement time
        self.cached_cpu_percent = 0.0  # Cache CPU percentage
        self.base_map_modes = ["random", "camera"]
        self.map_modes = list(self.base_map_modes)
        self.current_map_mode = 0  # Index into map_modes (Random is default)
        self.plugin_map_mode = None
        self.plugin_display_name = None

        # Help dialog state
        self.help_dialog_visible = False

        # Pause/resume state
        self.is_paused = False
        self.is_loading = True  # Default to loading until scene is loaded
        self.global_control_active = False  # Global control state
        self.sort_config = None  # Sort configuration from scene

        # Visual mode toggle state
        self.visual_mode = False  # Visual panel hidden by default

        # Tmux compatibility
        self.is_tmux = os.environ.get("TMUX") is not None
        self.last_update_time = 0
        self.update_throttle = 0.1 if self.is_tmux else 0.05
        self.cached_terminal_size = None
        self.last_size_check = 0
        self.pending_terminal_size = None
        self.pending_size_time = 0
        self.size_history = deque(maxlen=5)

        self._last_visual_dimensions = None
        self._last_plugin_request_dims = None
        self._max_colors_height = None
        self._colors_width_ratio = None
        self._last_layout_metrics = None

        # Mode transition stabilization
        self.mode_transition_time = None
        self.transition_settle_duration = (
            5.0 if self.is_tmux else 2.0
        )  # Reasonable stabilization period

    def _get_safe_terminal_size(self):
        """Get terminal size with caching and tmux compatibility"""
        current_time = time.time()

        # Cache terminal size for a short period to reduce overhead
        cache_duration = 0.6 if self.is_tmux else 0.5
        if (
            self.cached_terminal_size is not None
            and current_time - self.last_size_check < cache_duration
        ):
            return self.cached_terminal_size

        try:
            size = self.console.size
            raw_width, raw_height = size.width, size.height
            width, height = raw_width, raw_height

            if self.is_tmux:
                if width <= 0 or height <= 0:
                    if self.cached_terminal_size is not None:
                        return self.cached_terminal_size
                    width, height = 80, 24

                self.size_history.append((width, height))

                if len(self.size_history) >= 3:
                    sorted_widths = sorted(w for w, _ in self.size_history)
                    sorted_heights = sorted(h for _, h in self.size_history)
                    width = sorted_widths[len(sorted_widths) // 2]
                    height = sorted_heights[len(sorted_heights) // 2]
                elif self.cached_terminal_size is not None:
                    return self.cached_terminal_size

                width = max(width, 40)
                height = max(height, 12)
                target_size = (width, height)

                if self.cached_terminal_size is not None:
                    if target_size == self.cached_terminal_size:
                        self.last_size_check = current_time
                        return self.cached_terminal_size

                    if self.pending_terminal_size == target_size:
                        if current_time - self.pending_size_time >= 0.2:
                            self.pending_terminal_size = None
                        else:
                            self.last_size_check = current_time
                            return self.cached_terminal_size
                    else:
                        self.pending_terminal_size = target_size
                        self.pending_size_time = current_time
                        self.last_size_check = current_time
                        return self.cached_terminal_size
                else:
                    self.pending_terminal_size = None
                    self.pending_size_time = 0

                self.cached_terminal_size = target_size
                self.last_size_check = current_time
                return self.cached_terminal_size

            self.cached_terminal_size = (width, height)
            self.last_size_check = current_time

            return self.cached_terminal_size

        except Exception as e:
            # Fallback dimensions if size detection fails
            return (80, 24)

    def _emit_log_event(self, message):
        timestamped = f"{time.strftime('%H:%M:%S')} {message}"

        if self.log_file_handle:
            try:
                self.log_file_handle.write(timestamped + "\n")
                self.log_file_handle.flush()
            except Exception:
                pass

        if self.tui_enabled:
            self.log_buffer.append(timestamped)
        else:
            print(timestamped)

    def log_from_plugin(self, message):
        """Record log output from visual plugins without triggering immediate re-render."""
        self._emit_log_event(message)

    def _record_visual_panel_size(self, width, height, interior=None):
        if interior is not None and interior >= 0:
            message = (
                f"Visual panel size: {width}×{height} characters (inner≈{interior})"
            )
        else:
            message = f"Visual panel size: {width}×{height} characters"
        if is_debug_enabled():
            debug_log(
                "TUI_VISUAL_DIMENSIONS",
                "Visual panel measured",
                {
                    "width": width,
                    "height": height,
                    "mode": self.get_current_map_mode(),
                },
            )

        self._emit_log_event(message)

    def _log_layout_metrics(self, metrics):
        key = tuple(sorted(metrics.items()))
        if self._last_layout_metrics == key:
            return

        self._last_layout_metrics = key

        term_w, term_h = metrics.get("terminal", (0, 0))
        orientation = metrics.get("orientation", "vertical")
        main_h = metrics.get("main_height")
        colors_h = metrics.get("colors_height")
        footer_h = metrics.get("footer_height")
        footer_visible = metrics.get("footer_visible")

        orientation_tag = "vert" if orientation == "vertical" else "horiz"
        log_tag = "Y" if self.chronicle_visible else "N"
        writer_tag = "Y" if self.typewriter_visible else "N"

        message = (
            f"Layout: term={term_w}x{term_h} {orientation_tag} "
            f"main≈{main_h} colors≈{colors_h} footer={footer_h} "
            f"footer={'Y' if footer_visible else 'N'} log={log_tag} typewriter={writer_tag}"
        )

        self._emit_log_event(message)

    def _analyze_frequency_character(self):
        """Analyze the frequency characteristics of currently playing samples"""
        if not self.players_status:
            return 0.5  # Neutral if no samples

        playing_samples = [
            status
            for status in self.players_status.values()
            if status.get("state") == STATE_ACTIVE
        ]

        if not playing_samples:
            return 0.5  # Neutral if nothing playing

        # Use volume as the primary factor for color brightness
        volume_values = []
        for status in playing_samples:
            volume = status.get("volume", 0.5)  # Default mid-range
            volume_values.append(volume)

        # Calculate average volume - this indicates intensity
        avg_volume = sum(volume_values) / len(volume_values)

        # Higher volume = brighter colors
        # Lower volume = darker colors
        return avg_volume

    def _generate_color_grid(self, width, height):
        """Generate a grid of colored rectangles using unified dark nature palette with direct RGB values"""
        # Calculate activity level based on playing samples
        playing = [
            status
            for status in self.players_status.values()
            if status.get("state") == STATE_ACTIVE
        ]
        total_samples = len(self.players_status)

        if total_samples == 0:
            activity_level = 0.0
            frequency_character = 0.5  # Neutral
        else:
            playing_ratio = len(playing) / total_samples
            volumes = (
                [status.get("volume", 0.0) for status in playing] if playing else []
            )
            avg_volume = sum(volumes) / len(volumes) if volumes else 0.0
            activity_level = min(1.0, avg_volume)

            # Analyze frequency characteristics of playing samples
            frequency_character = self._analyze_frequency_character()

        # Get the dark nature palette (RGB tuples)
        dark_palette = get_dark_nature_palette()

        # Generate colors using direct RGB values from the dark nature palette
        # Use frequency analysis to bias toward different parts of the palette
        colors = []
        for _ in range(width * height):
            rand = random.random()

            # Use frequency character to bias color selection within the dark palette
            if (
                frequency_character > 0.7
            ):  # High frequencies - use slightly brighter colors from palette
                # Use colors from the latter half of the palette (indices 12-24)
                palette_range = dark_palette[12:]
            elif frequency_character < 0.3:  # Low frequencies - use darkest colors
                # Use colors from the first half of the palette (indices 0-12)
                palette_range = dark_palette[:13]
            else:  # Mid frequencies - use full palette
                palette_range = dark_palette

            # Select random color from the appropriate range
            rgb_color = random.choice(palette_range)
            colors.append(rgb_color)

        return colors

    def _update_color_grid_partial(self, width, height):
        """Update only about 10% of the color grid pixels for subtle animation"""
        # Calculate how many pixels to update (about 10%)
        total_pixels = width * height
        pixels_to_update = max(1, int(total_pixels * 0.08))

        # Randomly select pixels to update
        indices_to_update = random.sample(
            range(total_pixels), min(pixels_to_update, total_pixels)
        )

        # Get the dark nature palette
        dark_palette = get_dark_nature_palette()

        # Update selected pixels with new RGB colors from unified dark nature palette
        for idx in indices_to_update:
            rgb_color = random.choice(dark_palette)
            self.color_grid[idx] = rgb_color

    def set_quitting(self, quitting=True):
        """Set the quitting state to change the progress animation"""
        self.is_quitting = quitting

    def get_progress_symbol(self):
        """Get the current progress symbol"""
        if self.is_quitting:
            return "⏹"  # Stop symbol when quitting
        else:
            # Cycle through progress animation
            symbol = self.progress_chars[self.progress_animation_index]
            self.progress_animation_index = (self.progress_animation_index + 1) % len(
                self.progress_chars
            )
            return symbol

    def get_loading_symbol(self):
        """Get the current loading symbol"""
        symbol = self.loading_chars[self.loading_animation_index]
        self.loading_animation_index = (self.loading_animation_index + 1) % len(
            self.loading_chars
        )
        return symbol

    def enable_tui(self):
        """Enable TUI mode"""
        self.tui_enabled = True

    def disable_tui(self):
        """Disable TUI mode"""
        self.tui_enabled = False
        if self.live_display:
            self.live_display.stop()
            self.live_display = None

    def log(self, message):
        """Add a log message to both TUI and stdout"""
        timestamped = f"{time.strftime('%H:%M:%S')} {message}"

        if self.log_file_handle:
            try:
                self.log_file_handle.write(timestamped + "\n")
                self.log_file_handle.flush()
            except Exception:
                pass

        if self.tui_enabled:
            self.log_buffer.append(timestamped)
            if self.chronicle_visible:
                self.update_display()
        else:
            print(message)

    def start_file_logging(self, log_path):
        """Enable file logging to the specified path."""
        try:
            if self.log_file_handle:
                self.stop_file_logging()
            self.log_file_handle = open(log_path, "w", encoding="utf-8")
            self.log_file_path = log_path
        except Exception as exc:
            self.log_file_handle = None
            self.log_file_path = None
            print(f"Warning: unable to start file logging ({exc})")

    def stop_file_logging(self):
        """Close file logging if active."""
        if self.log_file_handle:
            try:
                self.log_file_handle.flush()
                self.log_file_handle.close()
            except Exception:
                pass
            finally:
                self.log_file_handle = None
                self.log_file_path = None

    def update_scene_name(self, name):
        """Update the scene name"""
        self.scene_name = name

    def update_player_status(self, player_name, status):
        """Update player status for TUI display"""
        if self.tui_enabled:
            old_status = self.players_status.get(player_name, {})
            self.players_status[player_name] = status

            # Check if a sample just started playing (state changed to 'Living')
            if (
                old_status.get("state") != STATE_ACTIVE
                and status.get("state") == STATE_ACTIVE
            ):
                self.last_activity_time = time.time()  # Trigger color update

    def toggle_chronicle(self):
        """Toggle Processes panel visibility"""
        self.chronicle_visible = not self.chronicle_visible
        state_label = "shown" if self.chronicle_visible else "hidden"
        self.log(f"{UI_PANEL_CHRONICLE} {state_label}. Press c to toggle.")

    def toggle_map_mode(self):
        """Toggle between available map modes (random, camera)"""
        if not self.visual_mode:
            self.log("Visual panel hidden; enable it before changing map mode.")
            return

        old_mode = self.map_modes[self.current_map_mode]
        self.current_map_mode = (self.current_map_mode + 1) % len(self.map_modes)
        new_mode = self.map_modes[self.current_map_mode]

        # Mark mode transition time for stabilization
        self.mode_transition_time = time.time()

        debug_log(
            "TUI_MAP_MODE_TOGGLE",
            f"Map mode toggled: {old_mode} -> {new_mode}",
            {
                "old_mode": old_mode,
                "new_mode": new_mode,
                "mode_index": self.current_map_mode,
            },
        )
        update_state("tui", "map_mode", new_mode, f"User toggled map mode")

        if new_mode == self.plugin_map_mode and self.plugin_display_name:
            mode_label = self.plugin_display_name
        else:
            mode_label = new_mode
        self.log(f"Map mode set to {mode_label}. Press m to toggle.")

        # Notify scene manager about mode change
        from ..core.scene import scene_manager

        if scene_manager and hasattr(scene_manager, "set_map_mode"):
            scene_manager.set_map_mode(new_mode)
        else:
            debug_log(
                "TUI_MAP_MODE_ERROR", "Scene manager not available for map mode change"
            )

        self._sync_camera_capture()

    def get_current_map_mode(self):
        """Get the current map mode"""
        return self.map_modes[self.current_map_mode]

    def _sync_camera_capture(self):
        try:
            from ..core.scene import scene_manager
        except ImportError:
            scene_manager = None

        current_mode = self.get_current_map_mode()

        if current_mode == "camera" and self.visual_mode:
            if scene_manager and not getattr(scene_manager, "webcam_running", False):
                device = getattr(scene_manager, "camera_device", "/dev/video0")
                interval = getattr(
                    scene_manager, "webcam_update_interval", CAMERA_UPDATE_INTERVAL
                )
                size = getattr(scene_manager, "webcam_matrix_size", (40, 25))
                try:
                    scene_manager._start_webcam_capture(device, interval, size)
                except Exception:
                    pass
        else:
            if scene_manager and getattr(scene_manager, "webcam_running", False):
                try:
                    scene_manager._stop_webcam_capture()
                except Exception:
                    pass

    def _request_visual_plugin_frame(self, width, height):
        try:
            from ..core.scene import scene_manager
        except ImportError:
            scene_manager = None

        if not scene_manager or not hasattr(scene_manager, "render_visual_plugin"):
            return None, "Visual plugin support unavailable"

        dims = (width, height)
        if is_debug_enabled() and self._last_plugin_request_dims != dims:
            debug_log(
                "TUI_PLUGIN_REQUEST",
                "Requesting visual plugin frame",
                {
                    "width": width,
                    "height": height,
                },
            )
        self._last_plugin_request_dims = dims
        return scene_manager.render_visual_plugin(width, height)

    def toggle_pause(self):
        """Toggle pause/resume for all audio playback"""
        import pygame

        if self.is_paused:
            pygame.mixer.unpause()
            self.is_paused = False
            self.log("Audio resumed. Space to pause.")
            debug_log("AUDIO_RESUMED", "Audio playback resumed by user")

            # Start recording when playback resumes
            if (
                hasattr(self, "start_recording_callback")
                and self.start_recording_callback
            ):
                self.start_recording_callback()
        else:
            pygame.mixer.pause()
            self.is_paused = True
            self.log("Audio paused. Space to resume.")
            debug_log("AUDIO_PAUSED", "Audio playback paused by user")

        update_state("tui", "is_paused", self.is_paused, "User toggled pause/resume")

    def toggle_visual(self):
        """Toggle visual mode on/off"""
        self.visual_mode = not self.visual_mode

        if self.visual_mode:
            self.log("Visual panel shown. Press V to hide.")
            debug_log("VISUAL_MODE_ON", "Visual panel enabled by user")
        else:
            self.log("Visual panel hidden. Press V to show.")
            debug_log("VISUAL_MODE_OFF", "Visual panel disabled by user")

        # Reset Typewriter scroll position when visual mode changes to ensure typing cursor remains visible
        if self.typewriter_visible:
            self.typewriter_scroll_position = 0
            self.typewriter_last_content_height = 0

        self._sync_camera_capture()
        update_state("tui", "visual_mode", self.visual_mode, "User toggled visual mode")

    def pause_aware_sleep(self, duration):
        """Sleep that pauses when global pause is active"""

        elapsed_unpaused_time = 0.0
        sleep_interval = 0.05  # 50ms intervals

        while elapsed_unpaused_time < duration:
            if not self.is_paused:
                # Sleep for a short interval and track the time
                actual_sleep = min(sleep_interval, duration - elapsed_unpaused_time)
                time.sleep(actual_sleep)
                elapsed_unpaused_time += actual_sleep
            else:
                # When paused, just sleep briefly and check again
                # Don't count this time toward the duration
                time.sleep(0.1)

    def show_help_dialog(self):
        """Show the help/keymaps dialog"""
        self.help_dialog_visible = True
        debug_log("HELP_DIALOG_SHOW", "Help dialog opened")

    def hide_help_dialog(self):
        """Hide the help/keymaps dialog"""
        if self.help_dialog_visible:
            self.help_dialog_visible = False
            debug_log("HELP_DIALOG_HIDE", "Help dialog closed")

    def toggle_help_dialog(self):
        """Toggle the help/keymaps dialog visibility"""
        if self.help_dialog_visible:
            self.hide_help_dialog()
        else:
            self.show_help_dialog()

    def _create_help_content(self):
        """Create the help dialog content with all available keymaps."""
        return HelpDialogBuilder(UI_APP_NAME, VERSION).build()

    def remove_player_status(self, player_name):
        """Remove player from status tracking"""
        if self.tui_enabled and player_name in self.players_status:
            del self.players_status[player_name]

    def update_webcam_matrix(self, color_matrix):
        """Update the webcam color matrix for display."""
        self.webcam_matrix = color_matrix

        if not self.visual_mode:
            if is_debug_enabled():
                debug_log(
                    "WEB_CAM_FRAME_STORED",
                    "Stored webcam matrix while visual panel hidden",
                    {
                        "has_matrix": color_matrix is not None,
                        "matrix_shape": getattr(color_matrix, "shape", None),
                    },
                )
            return

        if self.get_current_map_mode() == "camera" and color_matrix is not None:
            if is_debug_enabled():
                debug_log(
                    "WEB_CAM_FRAME_APPLIED",
                    "Applied webcam matrix to visual panel",
                    {"matrix_shape": getattr(color_matrix, "shape", None)},
                )
            self._queue_live_update(force=True)
        elif (
            self.get_current_map_mode() == "camera"
            and color_matrix is None
            and is_debug_enabled()
        ):
            debug_log("WEB_CAM_FRAME_RESET", "Cleared webcam matrix", {})
            self._queue_live_update(force=True)

    def create_layout(self) -> Layout:
        """Create the TUI layout."""
        context = self._build_layout_context()
        layout = Layout()
        self._apply_layout_split(layout, context)
        self._log_layout_metrics_for_context(context)
        self._populate_header(layout, context)
        self._populate_main_panel(layout, context)
        if self.visual_mode:
            self._create_color_matrix_panel(layout, context.panel_padding)
        if context.display_footer:
            self._populate_footer(layout, context)
        if self.help_dialog_visible:
            return self._build_help_overlay(layout, context)
        return layout

    def _build_layout_context(self) -> LayoutContext:
        panel_padding = DEFAULT_PANEL_PADDING
        sample_count = len(self.players_status)
        terminal_width, terminal_height = self._get_safe_terminal_size()
        current_mode = self.get_current_map_mode()
        camera_layout_active = self.visual_mode and current_mode == "camera"
        header_height = 3

        display_footer = self.chronicle_visible or self.typewriter_visible

        footer_min_height = 0
        if self.chronicle_visible:
            footer_min_height += 10 if self.visual_mode else 6
        if self.typewriter_visible:
            footer_min_height += 6

        visual_panel_chrome = self._panel_vertical_chrome(panel_padding)
        visual_min_total = 0
        if self.visual_mode:
            visual_min_height = 6 if display_footer else 8
            visual_min_total = visual_min_height + visual_panel_chrome

        margin_reserved = 2
        colors_reserved = visual_min_total if self.visual_mode else 0

        visual_available_total = (
            terminal_height - header_height - footer_min_height - margin_reserved
        )
        available_main_height = max(1, visual_available_total - colors_reserved)

        if self.webcam_matrix is not None:
            min_main_height = 8
            desired_main_height = 8 + sample_count * 2
        else:
            min_main_height = 12
            desired_main_height = 12 + sample_count * 2

        if available_main_height < min_main_height:
            main_panel_height = available_main_height
        else:
            main_panel_height = min(desired_main_height, available_main_height)

        main_panel_height = max(1, int(main_panel_height))

        if self.visual_mode:
            visual_available = max(1, visual_available_total)
            min_colors_height = max(1, visual_min_total)
            max_main_for_visual = max(1, visual_available - min_colors_height)
            main_panel_height = min(main_panel_height, max_main_for_visual)

        used_height = (
            header_height + main_panel_height + footer_min_height + margin_reserved
        )
        remaining_height = max(0, terminal_height - used_height)
        if self.visual_mode:
            max_colors_height = max(visual_min_total or 5, remaining_height)
        else:
            max_colors_height = max(5, remaining_height)
        self._max_colors_height = max_colors_height

        return LayoutContext(
            panel_padding=panel_padding,
            sample_count=sample_count,
            terminal_width=terminal_width,
            terminal_height=terminal_height,
            current_mode=current_mode,
            camera_layout_active=camera_layout_active,
            header_height=header_height,
            display_footer=display_footer,
            footer_min_height=footer_min_height,
            visual_min_total=visual_min_total,
            visual_available_total=visual_available_total,
            available_main_height=available_main_height,
            main_panel_height=main_panel_height,
            margin_reserved=margin_reserved,
            max_colors_height=max_colors_height,
        )

    def _apply_layout_split(self, layout: Layout, context: LayoutContext) -> None:
        header_height = context.header_height
        footer_min_height = context.footer_min_height
        visual_min_total = context.visual_min_total
        camera_layout_active = context.camera_layout_active
        main_panel_height = context.main_panel_height

        if context.display_footer:
            if self.visual_mode:
                    layout.split_column(
                        Layout(name="header", size=header_height),
                        Layout(name="main", size=main_panel_height),
                        Layout(
                            name="colors",
                            ratio=1,
                            minimum_size=max(1, visual_min_total),
                        ),
                        Layout(
                            name="footer",
                            ratio=1,
                            minimum_size=footer_min_height or 3,
                        ),
                    )
            else:
                layout.split_column(
                    Layout(name="header", size=header_height),
                    Layout(name="main"),
                    Layout(
                        name="footer",
                        ratio=1,
                        minimum_size=footer_min_height or 3,
                    ),
                )
        else:
            if self.visual_mode:
                layout.split_column(
                    Layout(name="header", size=header_height),
                    Layout(name="main", size=main_panel_height),
                    Layout(
                        name="colors",
                        ratio=1,
                        minimum_size=max(1, visual_min_total),
                    ),
                )
            else:
                layout.split_column(
                    Layout(name="header", size=header_height),
                    Layout(name="main"),
                )

    def _log_layout_metrics_for_context(self, context: LayoutContext) -> None:
        if self.visual_mode:
            main_height_metric = context.main_panel_height
            colors_height_metric = max(
                0, context.visual_available_total - context.main_panel_height
            )
        else:
            main_height_metric = (
                context.terminal_height
                - context.header_height
                - (context.footer_min_height if context.display_footer else 0)
                - context.margin_reserved
            )
            colors_height_metric = 0

        metrics = {
            "terminal": (context.terminal_width, context.terminal_height),
            "orientation": "vertical",
            "main_height": int(main_height_metric),
            "main_desired": int(context.main_panel_height),
            "colors_height": int(colors_height_metric),
            "footer_height": int(
                context.footer_min_height if context.display_footer else 0
            ),
            "footer_visible": context.display_footer,
        }
        self._log_layout_metrics(metrics)

    def _populate_header(self, layout: Layout, context: LayoutContext) -> None:
        progress_symbol = self.get_progress_symbol()
        terminal_width = context.terminal_width

        if self.is_recording:
            now = time.time()
            if now - self.last_record_indicator_toggle >= 1.0:
                self.record_indicator_visible = not self.record_indicator_visible
                self.last_record_indicator_toggle = now

        app_text = Text()
        app_text.append(f"{UI_APP_NAME} {VERSION}", style="bold white")

        help_text = Text()
        help_text.append("Press ", style="bold white")
        help_text.append("?", style="bold white")
        help_text.append(" for help", style="bold white")

        indicator_symbol = self._get_playback_indicator(progress_symbol)
        playback_state = self._get_playback_state()
        metrics_text = self._build_system_metrics_text(indicator_symbol, playback_state)

        if terminal_width >= 70:
            header_table = Table.grid(expand=True)
            header_table.add_column(justify="left")
            header_table.add_column(justify="center")
            header_table.add_column(justify="right")
            header_table.add_row(app_text, metrics_text, help_text)
            header_content = header_table
        elif terminal_width >= 50:
            header_table = Table.grid(expand=True)
            header_table.add_column(justify="left")
            header_table.add_column(justify="right")
            header_table.add_row(app_text, metrics_text)
            header_table.add_row(help_text, "")
            header_content = header_table
        else:
            fallback_text = Text()
            fallback_text.append_text(app_text.copy())
            fallback_text.append("\n")
            fallback_text.append_text(metrics_text.copy())
            fallback_text.append("\n")
            fallback_text.append_text(help_text.copy())
            header_content = Align.left(fallback_text)

        layout["header"].update(
            Panel(header_content, border_style="green", padding=(0, 2))
        )

    def _build_system_metrics_text(
        self, indicator_symbol: str, playback_state: str
    ) -> Text:
        metrics_text = Text()

        try:
            current_time = time.time()
            if current_time - self.last_cpu_check > 1.0:
                self.cached_cpu_percent = psutil.cpu_percent(interval=None)
                self.last_cpu_check = current_time

            memory = psutil.virtual_memory()
            ram_percent = memory.percent

            if self.is_recording:
                if self.record_indicator_visible:
                    metrics_text.append("● Rec", style="bright_red")
                else:
                    metrics_text.append("     ", style="bright_red")
            else:
                if playback_state == "Stopped":
                    metrics_text.append("Stopped", style="bright_white")
                else:
                    metrics_text.append(
                        f"{indicator_symbol} {playback_state}", style="bright_white"
                    )
            metrics_text.append("  ", style="dim white")
            metrics_text.append(f"CPU: {int(self.cached_cpu_percent)}%", style="yellow")
            metrics_text.append("  ", style="dim white")
            metrics_text.append(f"RAM: {int(ram_percent)}%", style="green")

            try:
                battery = psutil.sensors_battery()
            except Exception:
                battery = None

            if battery and battery.percent is not None:
                metrics_text.append("  ", style="dim white")
                metrics_text.append(f"Batt: {int(battery.percent)}%", style="cyan")
        except Exception:
            if self.is_recording:
                if self.record_indicator_visible:
                    metrics_text.append("● Rec", style="bright_red")
                else:
                    metrics_text.append("     ", style="bright_red")
            else:
                if playback_state == "Stopped":
                    metrics_text.append("Stopped", style="bright_white")
                else:
                    metrics_text.append(
                        f"{indicator_symbol} {playback_state}", style="bright_white"
                    )
            metrics_text.append("  System metrics unavailable", style="dim red")

        return metrics_text

    def _populate_main_panel(self, layout: Layout, context: LayoutContext) -> None:
        panel_padding = context.panel_padding
        terminal_width = context.terminal_width
        panel_width = max(terminal_width - 6, 40)
        sound_width = max(34, min(42, int(panel_width * 3.5 / 11.5)))
        status_width = max(12, min(18, int(panel_width * 1.5 / 11.5)))
        vol_width = 10

        remaining_width = panel_width - sound_width - status_width - vol_width - 8
        progress_width = max(10, remaining_width)

        table = Table(
            show_header=True,
            header_style="bold white",
            expand=True,
            box=None,
            padding=(0, 1),
            pad_edge=False,
        )
        table.add_column(UI_COLUMN_SOUND, style="cyan", width=sound_width)
        table.add_column("Volume", width=vol_width, justify="right")
        table.add_column(UI_COLUMN_STATUS, width=status_width)
        table.add_column(UI_COLUMN_PROGRESS, width=progress_width)

        total_samples = len(self.players_status)
        if total_samples > 0:
            table.add_row("", "", "", "", style="dim")

        number_width = len(str(total_samples)) if total_samples > 0 else 1

        if self.sort_config:
            sorted_players = self._sort_players_by_config(self.players_status.items())
        else:
            sorted_players = sorted(
                self.players_status.items(), key=lambda item: item[0].lower()
            )

        for index, (player_name, status) in enumerate(sorted_players, 1):
            combined_bar = self._create_combined_progress_bar(status, progress_width)
            enumerated_name = Text()
            enumerated_name.append(
                f"{str(index).rjust(number_width)}. ", style="dim white"
            )

            max_name_width = sound_width - number_width - 3
            if len(player_name) > max_name_width:
                truncated_name = player_name[: max_name_width - 3] + "..."
                enumerated_name.append(truncated_name, style="cyan")
            else:
                enumerated_name.append(player_name, style="cyan")

            status_text = status.get("state", "Unknown")
            metrics_text = self._format_volume(status)
            table.add_row(enumerated_name, metrics_text, status_text, combined_bar)

        if total_samples == 0:
            table.add_row("", "", Text(UI_NO_ACTIVE_SOUNDS, style="dim"), "")

        scene_title = self.scene_name.strip() or UI_PANEL_SCENE
        layout["main"].update(
            Panel(table, title=scene_title, border_style="blue", padding=panel_padding)
        )

    def _populate_footer(self, layout: Layout, context: LayoutContext) -> None:
        panel_padding = context.panel_padding
        terminal_width = context.terminal_width
        footer_min_height = context.footer_min_height
        main_panel_height = context.main_panel_height
        margin_reserved = context.margin_reserved

        if self.visual_mode:
            footer_height_budget = max(
                footer_min_height or 10,
                context.terminal_height
                - context.header_height
                - main_panel_height
                - 8
                - margin_reserved,
            )
        else:
            footer_height_budget = max(
                footer_min_height or 6,
                context.terminal_height
                - context.header_height
                - main_panel_height
                - margin_reserved,
            )

        typewriter_panel = None
        log_panel = None
        typewriter_share = 0
        log_share = 0

        if self.typewriter_visible and self.chronicle_visible:
            share = max(6, footer_height_budget // 2)
            typewriter_share = log_share = share
        elif self.typewriter_visible:
            typewriter_share = max(8, footer_height_budget)
        elif self.chronicle_visible:
            log_share = max(6, footer_height_budget)

        if self.typewriter_visible:
            typewriter_inner = max(1, (typewriter_share or footer_height_budget) - 2)
            typewriter_panel, _ = self._build_typewriter_panel(
                panel_padding, typewriter_inner + 2
            )
            typewriter_share = typewriter_inner + 2

        if self.chronicle_visible:
            log_inner = max(4, (log_share or footer_height_budget) - 2)
            log_lines = list(self.log_buffer)
            visible_lines = log_lines[-log_inner:]
            if len(visible_lines) < log_inner:
                visible_lines = [""] * (log_inner - len(visible_lines)) + visible_lines

            max_log_width = max(20, terminal_width - 4)
            padded_lines = [
                f"{line[:max_log_width]:<{max_log_width}}" for line in visible_lines
            ]
            log_text = "\n".join(padded_lines)

            log_panel = Panel(
                log_text,
                title=UI_PANEL_CHRONICLE,
                border_style="yellow",
                padding=panel_padding,
            )
            log_share = log_inner + 2

        footer_layout = layout["footer"]
        if typewriter_panel and log_panel:
            footer_layout.split_row(
                Layout(typewriter_panel, ratio=typewriter_share),
                Layout(log_panel, ratio=log_share),
            )
        elif typewriter_panel:
            footer_layout.update(typewriter_panel)
        elif log_panel:
            footer_layout.update(log_panel)

    def _build_help_overlay(self, base_layout: Layout, context: LayoutContext) -> Layout:
        dialog_content = HelpDialogBuilder(UI_APP_NAME, VERSION).build()
        _ = base_layout  # Base layout not rendered while help overlay is shown

        content_height = 28
        dialog_height = min(content_height, context.terminal_height - 4)
        dialog_width = min(80, context.terminal_width - 8)

        top_space = max(1, (context.terminal_height - dialog_height) // 2)
        bottom_space = max(1, context.terminal_height - dialog_height - top_space)
        side_margin = max(2, (context.terminal_width - dialog_width) // 2)

        overlay = Layout()
        overlay.split_column(
            Layout(name="overlay_top", size=top_space),
            Layout(name="overlay_center", size=dialog_height),
            Layout(name="overlay_bottom", size=bottom_space),
        )

        overlay_center = overlay["overlay_center"]
        overlay_center.split_row(
            Layout(name="overlay_left", size=side_margin),
            Layout(name="overlay_dialog", size=dialog_width),
            Layout(name="overlay_right", size=side_margin),
        )

        overlay["overlay_dialog"].update(
            Panel(
                dialog_content,
                title="Help & Keybindings",
                border_style="bright_cyan",
                padding=(1, 2),
            )
        )

        overlay["overlay_top"].update("")
        overlay["overlay_bottom"].update("")
        overlay["overlay_left"].update("")
        overlay["overlay_right"].update("")

        return overlay

    def _get_playback_state(self) -> str:
        """Return global playback state label."""
        if self.is_stopped:
            return "Stopped"
        if self.is_quitting:
            return "Stopping"
        if self.is_loading:
            return "Loading"
        if self.is_paused:
            if self._has_waiting_samples():
                return "Waiting"
            return "Paused"
        return "Playing"

    def _has_loading_samples(self) -> bool:
        """Check if any samples are in loading state."""
        from .. import STATE_LOADING

        return any(status.get("state") == STATE_LOADING for status in self.players_status.values())

    def _has_waiting_samples(self) -> bool:
        """Check if any samples are in waiting state."""
        return any(status.get("state") == STATE_WAITING for status in self.players_status.values())

    def set_loading_state(self, loading: bool = True) -> None:
        """Set or clear the loading state for global control."""
        self.is_loading = loading
        debug_log("TUI_STATE", f"Loading state set to: {loading}")

    def has_global_control(self) -> bool:
        """Return whether global playback control is active."""
        return getattr(self, "global_control_active", False)

    def set_global_control_active(self, active: bool = True) -> None:
        """Set global playback control flag."""
        self.global_control_active = active
        debug_log("TUI_STATE", f"Global control active set to: {active}")

    def set_recording_callback(self, callback) -> None:
        """Set the recording start callback."""
        self.start_recording_callback = callback

    def set_sort_config(self, sort_config) -> None:
        """Set the player sort configuration provided by the scene."""
        self.sort_config = sort_config
        debug_log("TUI_SORT", f"Sort config set to: {sort_config}")

    def configure_visual_plugin(self, metadata) -> None:
        """Register or clear an external visual plugin mode."""
        previous_mode = self.get_current_map_mode()

        self.plugin_map_mode = None
        self.plugin_display_name = None
        self.map_modes = list(self.base_map_modes)

        if metadata:
            mode_key = metadata.get("mode_key")
            display_name = metadata.get("display_name") or "Visual Plugin"
            if mode_key:
                self.plugin_map_mode = mode_key
                self.plugin_display_name = display_name
                if mode_key not in self.map_modes:
                    self.map_modes.append(mode_key)
                    debug_log(
                        "TUI_PLUGIN_REGISTER",
                        "Visual plugin mode registered",
                        {
                            "mode_key": mode_key,
                            "display_name": display_name,
                        },
                    )
                self.current_map_mode = self.map_modes.index(mode_key)
                update_state(
                    "tui", "map_mode", mode_key, "Visual plugin mode activated"
                )
                self.log(f"Map mode set to {display_name}. Press m to toggle.")
                return

        if previous_mode in self.map_modes:
            self.current_map_mode = self.map_modes.index(previous_mode)
        else:
            self.current_map_mode = 0
            if previous_mode != "random":
                self.log("Visual plugin mode removed; falling back to random visuals.")

    def _sort_players_by_config(self, players_items):
        """Sort players based on scene sort configuration."""
        if not self.sort_config:
            return sorted(players_items, key=lambda item: item[0].lower())

        sort_key = self.sort_config.lower()
        reverse = False

        if sort_key.startswith("-"):
            reverse = True
            sort_key = sort_key[1:]

        if sort_key not in ["name", "length"]:
            debug_log(
                "TUI_SORT", f"Invalid sort key '{sort_key}', using default sorting"
            )
            return sorted(players_items, key=lambda item: item[0].lower())

        try:
            if sort_key == "name":
                sorted_items = sorted(
                    players_items, key=lambda item: item[0].lower(), reverse=reverse
                )
                debug_log(
                    "TUI_SORT",
                    f"Players sorted by name ({'descending' if reverse else 'ascending'})",
                )
                return sorted_items

            if sort_key == "length":
                return sorted(
                    players_items, key=lambda item: item[0].lower(), reverse=reverse
                )
        except Exception as exc:
            debug_log("TUI_SORT", f"Error sorting players: {exc}")
            return sorted(players_items, key=lambda item: item[0].lower())

        return sorted(players_items, key=lambda item: item[0].lower())

    def _get_playback_indicator(self, progress_symbol):
        """Select an indicator glyph based on global playback state."""
        if self.is_quitting:
            return progress_symbol
        if self.is_loading:
            return self.get_loading_symbol()
        if self.is_paused:
            if self._has_waiting_samples():
                return "⏳"
            return "❚❚"
        return progress_symbol
    def _create_color_matrix_panel(self, layout, panel_padding):
        """Create the color matrix visualization panel."""
        ColorMatrixPanelBuilder(self, layout, panel_padding).build()


    def _create_live_indicator(self, status, available_width):
        """Create a live indicator for line-in samples"""
        base_color = self._get_progress_bar_color(status, "playback")

        # Create animated "LIVE" indicator
        live_text = "● LIVE"

        # Calculate available space
        if available_width is None:
            available_width = 20

        # Center the live text
        padding = max(0, available_width - len(live_text))
        left_padding = padding // 2
        right_padding = padding - left_padding

        # Create the indicator
        indicator = Text()
        indicator.append(" " * left_padding, style="dim")
        indicator.append(live_text, style=f"bold {base_color}")
        indicator.append(" " * right_padding, style="dim")

        return indicator

    def _create_progress_bar(
        self, current, total, color, available_width=None, show_duration=False
    ):
        """Create a visual progress bar with dynamic width based on available space"""
        # Calculate available width for the bar (reserve 6 chars for percentage " 100%" + safety)
        if available_width is None:
            # Fallback to a reasonable default
            bar_width = 20
        else:
            bar_width = max(8, available_width - 5)  # Reserve 5 chars for " 100%"

        if total <= 0:
            # Show empty blocks when inactive with softer styling
            empty_bar = Text("░" * bar_width, style=f"dim {color}")
            percentage_text = Text("   0%", style=f"dim {color}")
            bar_text = Text()
            bar_text.append(empty_bar)
            bar_text.append(percentage_text)
            return bar_text

        if show_duration and total < SHORT_SAMPLE_THRESHOLD:
            dark_color = f"dim {color}"
            duration_text = f"{total:.1f}s"
            bar_text = Text()
            bar_text.append(Text("█" * bar_width, style=dark_color))
            bar_text.append(f" {duration_text}", style=dark_color)
            return bar_text

        # Normal progress bar for longer samples (>= 8 seconds)
        progress = min(current / total, 1.0)
        filled = int(progress * bar_width)
        empty = bar_width - filled

        # Create progress bar with softer contrast
        # Use filled blocks for progress and slightly dimmer background
        filled_part = Text("█" * filled, style=color)
        background_part = Text(
            "░" * empty, style=f"dim {color}"
        )  # Dimmed version of same color
        percentage = f"{progress * 100:3.0f}%"

        # Combine parts with main color for percentage
        bar_text = Text()
        bar_text.append(filled_part)
        bar_text.append(background_part)
        bar_text.append(f" {percentage}", style=color)

        return bar_text

    def _create_combined_progress_bar(self, status, available_width):
        """Create a single combined progress bar that shows playing (green) or waiting (purple)"""
        state = status.get("state", "Unknown")

        if state == STATE_ACTIVE:
            # Playing - show playback progress in green (with dynamic brightness)
            current = status.get("playback_progress", 0)
            total = status.get("playback_total", 1)

            # Check if this is a line-in sample (infinite duration)
            if total == float("inf"):
                # Show "Live" indicator for line-in samples
                return self._create_live_indicator(status, available_width)

            base_color = self._get_progress_bar_color(status, "playback")
            show_duration = total < SHORT_SAMPLE_THRESHOLD
            return self._create_progress_bar(
                current, total, base_color, available_width, show_duration=show_duration
            )

        elif state == STATE_WAITING:
            # Waiting - show waiting progress in purple (with dynamic brightness)
            current = status.get("waiting_progress", 0)
            total = status.get("waiting_total", 1)
            base_color = self._get_progress_bar_color(status, "waiting")
            show_duration = total < SHORT_SAMPLE_THRESHOLD
            return self._create_progress_bar(
                current, total, base_color, available_width, show_duration=show_duration
            )

        else:
            # Other states (Waking Up, Initial Wait, etc.) - show as dim bar
            return self._create_progress_bar(0, 1, "dim white", available_width)

    def _format_volume(self, status):
        """Render compact volume with source icon."""
        volume = status.get("volume")
        icon = "✦"  # default icon (spark)
        display_value = " --"

        if volume is not None:
            display_value = f"{float(volume):.2f}".rjust(5)

        control = status.get("volume_control_source")
        if control in ("shell", "camera"):
            icon = "☼"
        elif control == "math":
            icon = "∿"
        else:
            icon = "✦"

        return f"{display_value} {icon}".rstrip()

    def set_recording(self, recording):
        """Update recording indicator state."""
        self.is_recording = bool(recording)
        if self.is_recording:
            self.is_stopped = False
            self.record_indicator_visible = True
            self.last_record_indicator_toggle = time.time()
        else:
            self.record_indicator_visible = False

    def set_stopped(self, stopped):
        """Update stopped indicator state."""
        self.is_stopped = bool(stopped)
        if self.is_stopped:
            self.is_recording = False
            self.record_indicator_visible = False

    def set_typewriter_from_config(self, typewriter_config, base_path: Path):
        """Configure typewriter panel content from scene configuration."""
        self.typewriter_available = False
        self.typewriter_full_text = ""
        self.typewriter_unavailable_message = (
            "No typewriter text in scene configuration."
        )
        self.typewriter_typing_interval_range = TYPEWRITER_PACE_RANGES["default"]

        if not typewriter_config or not isinstance(typewriter_config, dict):
            self._reset_typewriter_progress()
            return

        try:
            pace_key = str(typewriter_config.get("pace", "default")).lower()
            self.typewriter_typing_interval_range = TYPEWRITER_PACE_RANGES.get(
                pace_key, TYPEWRITER_PACE_RANGES["default"]
            )

            # Parse color attribute
            color_key = str(typewriter_config.get("color", "bright_white")).lower()
            self.typewriter_text_color = color_key
            self.typewriter_configured_color = color_key

            if "text" in typewriter_config and typewriter_config["text"]:
                text = str(typewriter_config["text"])
            elif "path" in typewriter_config and typewriter_config["path"]:
                raw_path = typewriter_config["path"]
                typewriter_path = Path(raw_path)
                if not typewriter_path.is_absolute():
                    typewriter_path = Path(base_path) / typewriter_path
                text = typewriter_path.read_text(encoding="utf-8")
            else:
                self.typewriter_unavailable_message = (
                    "Typewriter configuration missing 'text' or 'path'."
                )
                self._reset_typewriter_progress()
                return

            self.typewriter_full_text = text.replace("\r\n", "\n")
            self.typewriter_total_chars = len(self.typewriter_full_text)
            self.typewriter_available = self.typewriter_total_chars > 0
            if not self.typewriter_available:
                self.typewriter_unavailable_message = "Typewriter text is empty."
        except Exception as exc:
            self.typewriter_available = False
            self.typewriter_unavailable_message = f"Typewriter load error: {exc}"
            self.typewriter_full_text = ""
            self.typewriter_total_chars = 0

        self._reset_typewriter_progress()
        # Restore the configured color after reset
        self.typewriter_text_color = self.typewriter_configured_color

    def toggle_typewriter_panel(self):
        """Toggle visibility of the typewriter panel."""
        self.typewriter_visible = not self.typewriter_visible
        state_label = "shown" if self.typewriter_visible else "hidden"
        self.log(f"Typewriter panel {state_label}. Press t to toggle.")
        if self.typewriter_visible:
            self._reset_typewriter_progress()
            # Restore the configured color after reset
            self.typewriter_text_color = self.typewriter_configured_color
        if self.tui_enabled:
            self.update_display(force=True)

    def _reset_typewriter_progress(self):
        self.typewriter_visible_chars = 0
        self.typewriter_render_buffer = (
            "" if self.typewriter_available else self.typewriter_unavailable_message
        )
        self.typewriter_next_char_time = time.time()
        self.typewriter_scroll_position = 0
        self.typewriter_last_content_height = 0
        self.typewriter_text_color = (
            self.typewriter_configured_color
        )  # Restore configured color
        self.typewriter_completion_time = 0  # Reset completion timer

    def _update_typewriter_progress(self):
        if not self.typewriter_visible:
            return

        if not self.typewriter_available:
            self.typewriter_render_buffer = self.typewriter_unavailable_message
            return

        current_time = time.time()
        while (
            self.typewriter_visible_chars < self.typewriter_total_chars
            and current_time >= self.typewriter_next_char_time
        ):
            self.typewriter_visible_chars += 1
            self.typewriter_render_buffer = self.typewriter_full_text[
                : self.typewriter_visible_chars
            ]

            char = self.typewriter_full_text[self.typewriter_visible_chars - 1]
            interval = max(random.uniform(*self.typewriter_typing_interval_range), 0.0)
            delay = TYPEWRITER_PUNCTUATION_PAUSES.get(char, 0.0)
            self.typewriter_next_char_time = current_time + interval + delay
            current_time = time.time()

        if self.typewriter_visible_chars >= self.typewriter_total_chars:
            self.typewriter_render_buffer = self.typewriter_full_text
            # Set completion time if not already set
            if self.typewriter_completion_time == 0:
                self.typewriter_completion_time = time.time()

            # Clear text after 8 seconds
            if time.time() - self.typewriter_completion_time > 8.0:
                self._reset_typewriter_progress()

    def _update_typewriter_scroll(self, content_rows, content_capacity):
        """Update scroll position to keep the typing cursor line visible when content exceeds capacity."""
        # Add buffer lines for live typing visibility when Chronicles panel is open
        buffer_lines = (
            2
            if self.chronicle_visible
            and self.typewriter_visible_chars < self.typewriter_total_chars
            else 0
        )
        effective_capacity = content_capacity - buffer_lines

        if len(content_rows) <= effective_capacity:
            # Content fits, no scrolling needed
            self.typewriter_scroll_position = 0
            self.typewriter_last_content_height = len(content_rows)
            # Add buffer lines at the bottom if needed
            if buffer_lines > 0:
                return content_rows + [""] * buffer_lines
            return content_rows

        # Content exceeds capacity, need to scroll
        current_content_height = len(content_rows)

        # Calculate the maximum scroll position
        max_scroll = max(0, current_content_height - effective_capacity)

        # Always scroll to show the last line (typing cursor) when content changes
        if current_content_height != self.typewriter_last_content_height:
            # Scroll to show the latest content (typing cursor line)
            self.typewriter_scroll_position = max_scroll

        # If we're still typing and the current scroll position doesn't show the typing cursor,
        # adjust it to ensure the typing cursor is visible
        if (
            self.typewriter_visible_chars < self.typewriter_total_chars
            and self.typewriter_scroll_position < max_scroll
        ):
            self.typewriter_scroll_position = max_scroll

        # Ensure scroll position is valid
        self.typewriter_scroll_position = min(
            self.typewriter_scroll_position, max_scroll
        )

        # Update the tracked content height
        self.typewriter_last_content_height = current_content_height

        # Return the visible portion of content with buffer lines
        start_idx = self.typewriter_scroll_position
        end_idx = start_idx + effective_capacity
        visible_content = content_rows[start_idx:end_idx]

        # Add buffer lines at the bottom if needed
        if buffer_lines > 0:
            visible_content = visible_content + [""] * buffer_lines

        return visible_content

    def _wrap_text_lines(self, lines, max_width):
        """Wrap long lines to fit within the specified width, keeping words intact and preserving paragraph spacing."""
        wrapped_lines = []

        for i, line in enumerate(lines):
            # Preserve empty lines (paragraph separators)
            if not line.strip():
                wrapped_lines.append("")
                continue

            if len(line) <= max_width:
                # Line fits, no wrapping needed
                wrapped_lines.append(line)
            else:
                # Line is too long, need to wrap it
                # Split by spaces and filter out empty strings from multiple spaces
                words = [word for word in line.split(" ") if word]
                current_line = ""

                for word in words:
                    # Handle very long words that exceed max_width
                    if len(word) > max_width:
                        # If current line has content, add it first
                        if current_line:
                            wrapped_lines.append(current_line)
                            current_line = ""

                        # Split the very long word across multiple lines
                        remaining_word = word
                        while len(remaining_word) > max_width:
                            wrapped_lines.append(remaining_word[:max_width])
                            remaining_word = remaining_word[max_width:]

                        # Add the remaining part as the start of the next line
                        current_line = remaining_word
                    else:
                        # Normal word processing
                        # If adding this word would exceed the width, start a new line
                        if (
                            current_line
                            and len(current_line) + 1 + len(word) > max_width
                        ):
                            wrapped_lines.append(current_line)
                            current_line = word
                        else:
                            # Add word to current line
                            if current_line:
                                current_line += " " + word
                            else:
                                current_line = word

                # Add the last line if it's not empty
                if current_line:
                    wrapped_lines.append(current_line)

        return wrapped_lines

    def _build_typewriter_panel(self, panel_padding, max_height):
        if not self.typewriter_visible or max_height < 3:
            return None, 0

        self._update_typewriter_progress()
        if self.typewriter_render_buffer:
            rows = self.typewriter_render_buffer.split("\n")
        else:
            rows = [self.typewriter_unavailable_message]

        # Calculate available width for word wrapping
        terminal_width, _ = self._get_safe_terminal_size()

        # Account for panel padding and borders
        # Panel has 2 characters for borders (left + right)
        # Plus padding on each side
        padding_chars = panel_padding * 2 if isinstance(panel_padding, int) else 4
        available_width = max(20, terminal_width - 2 - padding_chars)

        # Apply word wrapping to long lines
        wrapped_rows = self._wrap_text_lines(rows, available_width)

        # Ensure content capacity is at least 1 line to show typing cursor (even with minimal height)
        content_capacity = max(1, max_height - 2)

        # Use the new scrolling logic to get the visible rows
        visible_rows = self._update_typewriter_scroll(wrapped_rows, content_capacity)

        content = Text("\n".join(visible_rows), style=self.typewriter_text_color)
        panel_height = len(visible_rows) + 2
        panel = Panel(
            content, title="Typewriter", border_style="magenta", padding=panel_padding
        )
        return panel, panel_height

    def _get_progress_bar_color(self, status, bar_type):
        """Get dynamic color for progress bars based on sample parameters"""
        # Check if this is a line-in sample (infinite duration)
        playback_total = status.get("playback_total", 1)
        is_linein = playback_total == float("inf")

        # Colors for different states
        if bar_type == "playback":
            if is_linein:
                # Light red/pink colors for line-in samples
                default_color = "red"
                bright_color = "bright_red"
                dim_color = "dim red"
            else:
                default_color = "green"
                bright_color = "bright_green"
                dim_color = "dim green"
        else:  # waiting
            default_color = "magenta"  # Purple for waiting
            bright_color = "bright_magenta"
            dim_color = "dim magenta"

        # Get sample parameters from status
        volume = status.get("volume", 0.5)  # Default mid volume
        state = status.get("state", "Unknown")

        # Only apply dynamic coloring when sample is active
        if state not in [STATE_ACTIVE, STATE_WAITING]:
            return default_color

        # Calculate brightness based on volume
        # Volume ranges from 0.0 to 1.0 typically
        brightness_factor = volume

        if brightness_factor > 0.7:  # High volume
            return bright_color
        elif brightness_factor < 0.3:  # Low volume
            return dim_color
        else:
            return default_color

    def start_live_display(self):
        """Start the live TUI display"""
        if not self.tui_enabled:
            return

        try:
            # Use higher refresh rate for smoother progress bars
            refresh_rate = (
                5 if self.is_tmux else 10
            )  # 10Hz = 0.1s updates for smooth progress
            self.live_display = Live(
                self.create_layout(),
                console=self.console,
                refresh_per_second=refresh_rate,
                screen=True,
            )
            self.live_display.start()
        except Exception as e:
            self.log(f"Failed to start TUI: {e}")
            self.tui_enabled = False

    def _queue_live_update(self, force=False):
        """Schedule a live display refresh from any thread."""
        if not self.tui_enabled or not self.live_display:
            return

        try:
            self.console.call_from_thread(lambda: self.update_display(force=force))
        except Exception:
            # Fallback to direct call if call_from_thread unavailable
            self.update_display(force=force)

    def update_display(self, force=False):
        """Update the live display with throttling for tmux compatibility"""
        if not self.tui_enabled or not self.live_display:
            return

        current_time = time.time()

        # Extra throttling during mode transitions in tmux
        throttle_time = self.update_throttle
        if self.is_tmux and self.mode_transition_time:
            time_since_transition = current_time - self.mode_transition_time
            if time_since_transition < self.transition_settle_duration:
                # Slower updates during transition settling
                throttle_time = 0.2  # Update only every 200ms during transition
            else:
                # Transition settled, clear the transition time
                self.mode_transition_time = None

        # Throttle updates to prevent tmux glitching
        if not force and current_time - self.last_update_time < throttle_time:
            return

        try:
            self.live_display.update(self.create_layout())
            self.last_update_time = current_time
        except Exception as e:
            # Silently handle tmux rendering errors to prevent log spam
            pass

    def stop_live_display(self):
        """Stop the live TUI display"""
        if self.live_display:
            try:
                self.live_display.stop()
            except Exception:
                pass
            finally:
                self.live_display = None
                # Force terminal reset
                import sys

                try:
                    sys.stdout.write("\033[?25h")  # Show cursor
                    sys.stdout.flush()
                except:
                    pass


# Global TUI manager instance
tui_manager = TUIManager()

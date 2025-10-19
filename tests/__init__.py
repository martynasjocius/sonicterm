#
# SonicTerm
# Copyright (c) 2025 Martynas Jocius
#
"""Test helpers for SonicTerm."""

import sys
import types


def _install_pygame_stub():
    if "pygame" in sys.modules:
        return

    pygame = types.ModuleType("pygame")
    mixer = types.ModuleType("pygame.mixer")

    class DummyChannel:
        def __init__(self):
            self._busy = False

        def get_busy(self):
            return self._busy

        def stop(self):
            self._busy = False

    class DummySound:
        def __init__(self, *_args, **_kwargs):
            self._length = 1.0
            self._volume = 1.0

        def get_length(self):
            return self._length

        def set_volume(self, volume):
            self._volume = volume

        def play(self):
            return DummyChannel()

    mixer.Sound = DummySound
    pygame.mixer = mixer
    sys.modules["pygame"] = pygame
    sys.modules["pygame.mixer"] = mixer


def _install_watchdog_stub():
    if "watchdog.observers" in sys.modules:
        return

    observers = types.ModuleType("watchdog.observers")
    events = types.ModuleType("watchdog.events")

    class DummyObserver:
        def schedule(self, *args, **kwargs):
            return None

        def start(self):
            return None

        def stop(self):
            return None

        def join(self, *_args, **_kwargs):
            return None

    class FileSystemEventHandler:
        pass

    observers.Observer = DummyObserver
    events.FileSystemEventHandler = FileSystemEventHandler

    sys.modules["watchdog"] = types.ModuleType("watchdog")
    sys.modules["watchdog.observers"] = observers
    sys.modules["watchdog.events"] = events


def _install_psutil_stub():
    if "psutil" in sys.modules:
        return

    psutil = types.ModuleType("psutil")

    def cpu_percent(interval=None):
        return 0.0

    class Memory:
        percent = 0.0

    def virtual_memory():
        return Memory()

    psutil.cpu_percent = cpu_percent
    psutil.virtual_memory = virtual_memory
    sys.modules["psutil"] = psutil


def _install_rich_stub():
    # Always install stub for tests, even if rich is already imported
    # This ensures consistent behavior across all test runs

    # Remove any existing rich modules to ensure clean installation
    modules_to_remove = [key for key in sys.modules.keys() if key.startswith("rich")]
    for module_name in modules_to_remove:
        del sys.modules[module_name]

    rich = types.ModuleType("rich")
    console_module = types.ModuleType("rich.console")
    live_module = types.ModuleType("rich.live")
    layout_module = types.ModuleType("rich.layout")
    panel_module = types.ModuleType("rich.panel")
    progress_module = types.ModuleType("rich.progress")
    text_module = types.ModuleType("rich.text")
    table_module = types.ModuleType("rich.table")
    align_module = types.ModuleType("rich.align")

    class Console:
        def __init__(self, *args, **kwargs):
            self._size = types.SimpleNamespace(width=120, height=40)

        @property
        def size(self):
            return self._size

        def print(self, *_args, **_kwargs):
            return None

    class Live:
        def __init__(self, *args, **kwargs):
            pass

        def start(self):
            return None

        def stop(self):
            return None

        def update(self, *_args, **_kwargs):
            return None

    class Panel:
        def __init__(
            self,
            renderable,
            title=None,
            border_style=None,
            padding=(0, 0),
            height=None,
            expand=False,
        ):
            self.renderable = renderable
            self.title = title
            self.border_style = border_style
            self.padding = padding
            self.height = height
            self.expand = expand

    class Layout:
        def __init__(self, name=None, size=None, ratio=None, minimum_size=None):
            self.name = name
            self.size = size
            self.ratio = ratio
            self.minimum_size = minimum_size
            self.children = []
            self._mapping = {}
            self.content = None

        def _register(self, layouts):
            self.children = list(layouts)
            for child in layouts:
                if child.name:
                    self._mapping[child.name] = child

        def split_column(self, *layouts):
            self._register(layouts)

        def split_row(self, *layouts):
            self._register(layouts)

        def __getitem__(self, name):
            if name in self._mapping:
                return self._mapping[name]
            for child in self.children:
                if hasattr(child, "__getitem__"):
                    try:
                        return child[name]
                    except KeyError:
                        continue
            raise KeyError(name)

        def keys(self):
            names = set(self._mapping.keys())
            for child in self.children:
                if hasattr(child, "keys"):
                    names.update(child.keys())
            return list(names)

        def update(self, content):
            self.content = content

    class Text:
        def __init__(self, text="", style=None):
            self.parts = [text] if text else []

        def append(self, text, style=None):
            self.parts.append(str(text))

        def append_text(self, other):
            if isinstance(other, Text):
                self.parts.append(other.plain)
            else:
                self.parts.append(str(other))

        def copy(self):
            new = Text()
            new.parts = list(self.parts)
            return new

        @property
        def plain(self):
            return "".join(self.parts)

        def __str__(self):
            return self.plain

    class Table:
        def __init__(self, *args, **kwargs):
            self.columns = []
            self.rows = []

        @classmethod
        def grid(cls, expand=False):
            return cls(expand=expand)

        def add_column(self, header=None, **_kwargs):
            column = types.SimpleNamespace(name=header, width=_kwargs.get("width"))
            self.columns.append(column)

        def add_row(self, *row, **_kwargs):
            self.rows.append(row)

    class Align:
        @staticmethod
        def center(renderable, **_kwargs):
            return renderable

    class Progress:
        pass

    class BarColumn:
        pass

    class TextColumn:
        pass

    class TimeRemainingColumn:
        pass

    class Group:
        def __init__(self, *renderables):
            self.renderables = renderables

    console_module.Console = Console
    live_module.Live = Live
    layout_module.Layout = Layout
    panel_module.Panel = Panel
    progress_module.Progress = Progress
    progress_module.BarColumn = BarColumn
    progress_module.TextColumn = TextColumn
    progress_module.TimeRemainingColumn = TimeRemainingColumn
    text_module.Text = Text
    table_module.Table = Table
    align_module.Align = Align
    console_module.Group = Group

    sys.modules["rich"] = rich
    sys.modules["rich.console"] = console_module
    sys.modules["rich.live"] = live_module
    sys.modules["rich.layout"] = layout_module
    sys.modules["rich.panel"] = panel_module
    sys.modules["rich.progress"] = progress_module
    sys.modules["rich.text"] = text_module
    sys.modules["rich.table"] = table_module
    sys.modules["rich.align"] = align_module
    sys.modules["rich.console"].Group = Group


def _install_stubs():
    _install_pygame_stub()
    _install_watchdog_stub()
    _install_psutil_stub()
    _install_rich_stub()


def _silence_tui_logs():
    try:
        from sonicterm.ui.tui import tui_manager
    except Exception:
        return

    def _noop_log(*_args, **_kwargs):
        return None

    targets = [tui_manager]

    try:
        from sonicterm.ui.tui import TUIManager
    except Exception:
        TUIManager = None  # type: ignore
    else:
        targets.append(TUIManager)

    for target in targets:
        for attr in ("log", "log_from_plugin", "_emit_log_event"):
            original = getattr(target, attr, None)
            if callable(original):
                setattr(target, attr, _noop_log)


_install_stubs()
_silence_tui_logs()

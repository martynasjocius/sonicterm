#!/usr/bin/env python3
#
# SonicTerm
# Copyright (c) 2025 Martynas Jocius
#
"""Scene orchestration logic for SonicTerm."""

from __future__ import annotations

import importlib.util
import inspect
import json
import random
import re
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pygame
from watchdog.events import FileSystemEventHandler

if TYPE_CHECKING:  # pragma: no cover - typing helpers
    from watchdog.events import FileSystemEvent
from watchdog.observers import Observer

from .. import CAMERA_UPDATE_INTERVAL, STATE_ACTIVE, STATE_LOADING
from ..audio import SamplePlayer
from ..audio.player import AUDIO_INPUT_AVAILABLE, LineInPlayer
from ..debug import debug_log, is_debug_enabled, log_error, log_operation, update_state
from ..ui.tui import tui_manager

MOTION_SENSITIVITY = 16.0

# Optional camera support
try:  # pragma: no cover - hardware dependent
    from ..utils.webcam import camera_capture

    WEBCAM_AVAILABLE = True
except ImportError as exc:  # pragma: no cover - optional dependency
    tui_manager.log(f"Camera support not available: {exc}")
    WEBCAM_AVAILABLE = False
    camera_capture = None  # type: ignore


class SceneFileHandler(FileSystemEventHandler):
    """React to scene file changes by triggering selective reloads."""

    def __init__(self, soundscape: "SceneBasedSoundscape") -> None:
        super().__init__()
        self.soundscape = soundscape

    def on_modified(self, event: "FileSystemEvent") -> None:
        if event.is_directory:
            return
        if Path(event.src_path) != self.soundscape.scene_file:
            return

        tui_manager.log(f"Scene file modified: {event.src_path}")
        tui_manager.log("Applying selective updates while keeping current playback")
        self.soundscape.reload_scene()


@dataclass
class VisualPluginRuntime:
    """Runtime metadata for a scene-defined visual plugin."""

    name: str
    key: str
    path: Path
    module: ModuleType
    render: Callable[..., Any]
    requires_context: bool
    state: Dict[str, Any] = field(default_factory=dict)
    start_time: float = field(default_factory=time.time)
    frame_index: int = 0
    last_error: Optional[str] = None


@dataclass
class PlayerManager:
    """Encapsulate creation and maintenance of sample players."""

    scene: "SceneBasedSoundscape"

    def create_players(
        self,
        sample_config: Dict[str, Any],
        global_config: Dict[str, Any],
        sample_index: int,
    ) -> List[SamplePlayer]:
        """Create player instances from a sample config, handling multiply parameter."""

        players: List[SamplePlayer] = []
        active = sample_config.get("active", True)
        multiply = sample_config.get("multiply", 1)

        if not isinstance(multiply, (int, float)) or multiply < 0 or multiply > 8:
            tui_manager.log(
                "Invalid multiply value %s for sample %s; using 1"
                % (
                    multiply,
                    sample_config.get("name", sample_config.get("path", "<unknown>")),
                )
            )
            multiply = 1

        multiply = max(1, int(multiply))

        for instance_num in range(multiply):
            instance_config = sample_config.copy()

            if multiply > 1:
                base_name = sample_config.get(
                    "name", Path(sample_config.get("path", "sample")).stem
                )
                instance_config["name"] = f"{base_name} #{instance_num + 1}"

                timings = instance_config.get("timings")
                if timings:
                    base_wait = instance_config.get("wait", 0)
                    random_timing = random.choice(timings)
                    instance_config["wait"] = base_wait + random_timing
                    debug_log(
                        "MULTIPLY_TIMING",
                        f"Added random timing for {instance_config['name']}",
                        {
                            "base_wait": base_wait,
                            "random_timing": random_timing,
                            "final_wait": instance_config["wait"],
                        },
                    )

            player = self._build_player(instance_config, global_config)
            player.active = active  # Preserve original active flag
            players.append(player)

            debug_log(
                "SAMPLE_PLAYER_CREATED",
                f"Created player for {instance_config['path']}",
                {
                    "index": sample_index,
                    "instance": instance_num + 1,
                    "total_instances": multiply,
                    "volume": instance_config.get("volume", "default"),
                    "player_name": getattr(player, "name", "unknown"),
                    "wait_time": instance_config.get("wait", 0),
                },
            )

        return players

    def cleanup_duplicate_linein_players(self) -> None:
        scene = self.scene
        roland_players = [player for player in scene.players if "Roland" in player.name]

        if len(roland_players) <= 1:
            return

        working_player = None
        failed_player = None
        for player in roland_players:
            if "[failed]" in player.name:
                failed_player = player
            else:
                working_player = player

        if working_player and failed_player:
            tui_manager.log(
                f"[{failed_player.name}] Removing duplicate (keeping working line-in)"
            )
            try:
                scene.players.remove(failed_player)
            except ValueError:
                pass
            tui_manager.log("Cleaned up 1 duplicate line-in player")

    def _build_player(
        self, sample_config: Dict[str, Any], global_config: Dict[str, Any]
    ) -> SamplePlayer:
        sample_path = sample_config.get("path", "")

        if sample_path.startswith("line-in://"):
            if not AUDIO_INPUT_AVAILABLE:
                tui_manager.log(
                    "Warning: Line-in not available (missing pyaudio/numpy). "
                    f"Creating fallback player for {sample_path}"
                )
                player = SamplePlayer(sample_config, global_config)
                player.is_linein_fallback = True  # type: ignore[attr-defined]
                return player

            try:
                return LineInPlayer(sample_config, global_config)
            except Exception as exc:  # pragma: no cover - hardware specific
                tui_manager.log(
                    f"Line-in player failed to initialize ({exc}); falling back to sample"
                )
                player = SamplePlayer(sample_config, global_config)
                player.is_linein_fallback = True  # type: ignore[attr-defined]
                return player

        return SamplePlayer(sample_config, global_config)


@dataclass
class VisualPluginManager:
    """Handle loading, configuration, and rendering of visual plugins."""

    scene: "SceneBasedSoundscape"

    def configure(self) -> None:
        scene = self.scene
        visual_config = None
        if scene.scene_config:
            visual_config = scene.scene_config.get("visual")

        metadata = None

        if visual_config is None:
            scene.visual_plugin = None
            scene.visual_plugin_config = None
        elif not isinstance(visual_config, dict):
            tui_manager.log("Warning: visual configuration must be an object; ignoring")
            log_error(
                "VISUAL_PLUGIN_INVALID",
                "scene_manager",
                "Visual config is not a mapping",
                {"value": visual_config},
            )
            scene.visual_plugin = None
            scene.visual_plugin_config = None
        else:
            try:
                scene.visual_plugin = self._load(visual_config)
                scene.visual_plugin_config = visual_config
                metadata = {
                    "mode_key": scene.visual_plugin.key,
                    "display_name": scene.visual_plugin.name,
                }
                tui_manager.log(f"🎨 Visual plugin loaded: {scene.visual_plugin.name}")
                debug_log(
                    "VISUAL_PLUGIN_LOADED",
                    "Visual plugin configured",
                    {
                        "name": scene.visual_plugin.name,
                        "mode_key": scene.visual_plugin.key,
                        "path": str(scene.visual_plugin.path),
                    },
                )
                scene.set_map_mode(scene.visual_plugin.key)
            except Exception as exc:
                scene.visual_plugin = None
                scene.visual_plugin_config = None
                message = f"Visual plugin failed to load ({exc})"
                tui_manager.log(message)
                log_error(
                    "VISUAL_PLUGIN_LOAD_FAILED",
                    "scene_manager",
                    message,
                    {"exception": repr(exc)},
                )

        tui_manager.configure_visual_plugin(metadata)

    def render(self, width: int, height: int) -> Tuple[Optional[Any], Optional[str]]:
        scene = self.scene
        plugin = scene.visual_plugin
        if not plugin:
            return None, "No visual plugin configured"

        context = {
            "frame": plugin.frame_index,
            "elapsed": time.time() - plugin.start_time,
            "state": plugin.state,
            "width": width,
            "height": height,
            "display_width": width,
            "display_height": height,
            "logger": self._build_logger(plugin),
        }

        try:
            if plugin.requires_context:
                frame = plugin.render(width, height, context)
            else:
                frame = plugin.render(width, height)
            plugin.frame_index += 1
            plugin.last_error = None
        except Exception as exc:  # pragma: no cover - plugin defined by user
            plugin.last_error = str(exc)
            log_error(
                "VISUAL_PLUGIN_RENDER_FAIL",
                "scene_manager",
                f"Render failed for plugin '{plugin.name}'",
                {"exception": repr(exc)},
            )
            return None, f"Visual plugin error: {exc}"

        frame_array: Optional[Any]
        if np is not None:
            try:
                frame_array = np.asarray(frame, dtype=np.uint8)
            except Exception:
                frame_array = None
        else:
            frame_array = None

        if frame_array is None:
            frame_array = frame

        try:
            if np is not None and isinstance(frame_array, np.ndarray):
                if frame_array.ndim != 3 or frame_array.shape[2] < 3:
                    raise ValueError("Plugin frame must be an array of shape (H, W, 3)")
                return frame_array, None

            rows = len(frame_array)
            cols = len(frame_array[0]) if rows else 0
            if rows == 0 or cols == 0:
                raise ValueError("Plugin frame must contain pixel data")
            for row in frame_array:
                if len(row) != cols:
                    raise ValueError("Plugin frame rows are not uniform in length")
                for pixel in row:
                    if len(pixel) < 3:
                        raise ValueError("Plugin frame pixels must provide RGB values")
            return frame_array, None

        except Exception as exc:
            plugin.last_error = str(exc)
            log_error(
                "VISUAL_PLUGIN_FRAME_INVALID",
                "scene_manager",
                f"Invalid plugin frame: {exc}",
                {"exception": repr(exc)},
            )
            return None, f"Visual plugin output invalid: {exc}"

    def _slugify(self, label: str, fallback: str) -> str:
        base = label or fallback
        slug = re.sub(r"[^a-z0-9]+", "-", base.lower()).strip("-")
        if not slug:
            slug = fallback.lower() or "visual"
        return f"visual:{slug}"

    def _load(self, visual_config: Dict[str, Any]) -> VisualPluginRuntime:
        path_value = visual_config.get("path")
        if not path_value:
            raise ValueError("Visual plugin config missing 'path'")

        plugin_path = Path(path_value)
        if not plugin_path.is_absolute():
            plugin_path = (self.scene.scene_file.parent / plugin_path).resolve()

        if not plugin_path.exists():
            raise FileNotFoundError(f"Visual plugin file not found: {plugin_path}")

        display_name = visual_config.get("name") or plugin_path.stem
        mode_key = self._slugify(display_name, plugin_path.stem)

        spec = importlib.util.spec_from_file_location(
            f"sonicterm_visual_{mode_key.replace(':', '_')}", plugin_path
        )
        if spec is None or spec.loader is None:
            raise ImportError(f"Unable to load visual plugin from {plugin_path}")

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        render = getattr(module, "render", None)
        if not callable(render):
            raise AttributeError(
                "Visual plugin must define callable render(width, height, [context])"
            )

        signature = inspect.signature(render)
        parameters = list(signature.parameters.values())
        if len(parameters) < 2:
            raise TypeError(
                "Visual plugin render function must accept at least width and height"
            )

        requires_context = len(parameters) >= 3

        runtime = VisualPluginRuntime(
            name=display_name,
            key=mode_key,
            path=plugin_path,
            module=module,
            render=render,
            requires_context=requires_context,
        )
        runtime.start_time = time.time()
        runtime.frame_index = 0
        runtime.state = {}
        runtime.last_error = None
        return runtime

    def _build_logger(self, plugin: VisualPluginRuntime) -> Callable[..., None]:
        prefix = f"[Visual Plugin:{plugin.name}] "

        def _log(message: Any, *args: Any, **kwargs: Any) -> None:
            level = str(kwargs.pop("level", "info")).lower()
            text = str(message)

            if args or kwargs:
                try:
                    text = text.format(*args, **kwargs)
                except Exception:
                    try:
                        text = text % args
                    except Exception:
                        extras = []
                        if args:
                            extras.append(f"args={args}")
                        if kwargs:
                            extras.append(f"kwargs={kwargs}")
                        text = f"{message} {' '.join(extras)}"

            line = f"{prefix}{text}"
            if hasattr(tui_manager, "log_from_plugin"):
                tui_manager.log_from_plugin(line)
            else:
                tui_manager.log(line)

            if is_debug_enabled():
                debug_log(
                    "VISUAL_PLUGIN_LOG",
                    line,
                    {"plugin": plugin.name, "level": level},
                )

        return _log


@dataclass
class WebcamController:
    """Manage webcam capture lifecycle for the scene."""

    scene: "SceneBasedSoundscape"

    def setup_mapping(self) -> None:
        if not WEBCAM_AVAILABLE:
            tui_manager.log(
                "❌ Camera support not available, falling back to random mapping"
            )
            self.scene.color_matrix = None
            update_state("scene_manager", "camera_active", False)
            return

        try:
            device = "/dev/video0"
            target_fps = 24.0
            matrix_size = (40, 25)
            tui_manager.log(
                "Setting up camera mapping from "
                f"{device} (target {target_fps:.0f}fps, initial size {matrix_size[0]}x{matrix_size[1]})"
            )
            self.start_capture(device, target_fps, matrix_size)
        except Exception as exc:  # pragma: no cover - hardware specific
            tui_manager.log(f"❌ Camera mapping setup error: {exc}")
            self.scene.color_matrix = None

    def start_capture(
        self,
        device: str,
        target_fps: float = 24.0,
        matrix_size: Tuple[int, int] = (48, 36),
    ) -> None:
        scene = self.scene
        if scene.webcam_running:
            return

        scene.webcam_running = True
        scene.webcam_matrix_size = matrix_size
        scene.camera_device = device
        scene.webcam_target_fps = target_fps

        update_interval = 1.0 / target_fps if target_fps > 0 else CAMERA_UPDATE_INTERVAL
        scene.webcam_update_interval = update_interval

        stream_width = max(160, matrix_size[0] * 4)
        stream_height = max(120, matrix_size[1] * 4)

        try:
            scene.webcam_stream = camera_capture.start_stream(
                device, stream_width, stream_height, target_fps
            )
            tui_manager.log(
                f"Started camera stream at {int(target_fps)}fps ({stream_width}x{stream_height})"
            )
        except Exception as exc:
            scene.webcam_stream = None
            scene.webcam_update_interval = CAMERA_UPDATE_INTERVAL
            tui_manager.log(
                f"Camera streaming unavailable ({exc}); falling back to periodic capture"
            )

        scene.webcam_thread = threading.Thread(
            target=self._capture_loop,
            args=(device, target_fps, matrix_size),
            daemon=True,
        )
        scene.webcam_thread.start()
        mode_label = (
            f"{int(target_fps)}fps stream"
            if scene.webcam_stream is not None
            else f"snapshot every {scene.webcam_update_interval:.2f}s"
        )
        tui_manager.log(
            f"Started camera capture ({mode_label}, matrix {matrix_size[0]}x{matrix_size[1]})"
        )
        update_state("scene_manager", "camera_active", True)

    def stop_capture(self) -> None:
        scene = self.scene
        if not scene.webcam_running:
            return

        scene.webcam_running = False
        camera_capture.stop_stream()
        scene.webcam_stream = None
        scene.webcam_target_fps = 0.0
        if scene.webcam_thread and scene.webcam_thread.is_alive():
            scene.webcam_thread.join(timeout=2.0)
        tui_manager.log("Stopped camera capture")
        update_state("scene_manager", "camera_active", False)

    def request_resize(self, width: int, height: int) -> None:
        scene = self.scene
        if not scene.webcam_running:
            return

        new_size = (width, height)
        if new_size == scene.webcam_matrix_size:
            return

        scene.webcam_matrix_size = new_size
        tui_manager.log(f"Camera matrix resized to {width}x{height} to match panel")

    def _capture_loop(
        self, device: str, target_fps: float, matrix_size: Tuple[int, int]
    ) -> None:
        scene = self.scene
        fallback_interval = CAMERA_UPDATE_INTERVAL
        frame_interval = 1.0 / target_fps if target_fps > 0 else fallback_interval

        next_stream_retry = time.time()

        while scene.webcam_running:
            loop_start = time.time()
            try:
                current_matrix_size = scene.webcam_matrix_size
                color_matrix = None

                if scene.webcam_stream is not None:
                    frame = camera_capture.read_stream_frame()
                    if frame is not None:
                        color_matrix = camera_capture.convert_frame_to_color_matrix(
                            frame,
                            current_matrix_size,
                            brightness=scene.camera_brightness,
                            contrast=scene.camera_contrast,
                        )
                    else:
                        camera_capture.stop_stream()
                        scene.webcam_stream = None

                if color_matrix is None and scene.webcam_stream is None:
                    fallback_w = max(160, current_matrix_size[0] * 4)
                    fallback_h = max(120, current_matrix_size[1] * 4)
                    color_matrix = camera_capture.capture_and_process(  # type: ignore[union-attr]
                        device,
                        current_matrix_size,
                        (fallback_w, fallback_h),
                        brightness=scene.camera_brightness,
                        contrast=scene.camera_contrast,
                    )

                if color_matrix is not None:
                    scene.color_matrix = color_matrix
                    self._update_camera_motion(color_matrix)
                    tui_manager.update_webcam_matrix(color_matrix)
                    height, width, _ = color_matrix.shape
                    if getattr(tui_manager, "debug_mode", False):
                        tui_manager.log(
                            f"Camera updated: {width}x{height} matrix ready for display"
                        )
                    scene.webcam_last_error = None
                    if is_debug_enabled():
                        debug_log(
                            "WEB_CAM_CAPTURE_OK",
                            "Webcam capture loop delivered frame",
                            {
                                "matrix_shape": (height, width),
                                "mode": "stream" if scene.webcam_stream else "snapshot",
                                "interval": frame_interval if scene.webcam_stream else fallback_interval,
                            },
                        )
                else:
                    scene.webcam_last_error = getattr(
                        camera_capture, "last_error", None
                    )
                    if scene.webcam_last_error:
                        tui_manager.log(
                            f"Camera capture failed; {scene.webcam_last_error}"
                        )
                        if is_debug_enabled():
                            debug_log(
                                "WEB_CAM_CAPTURE_FAIL",
                                "Webcam capture loop failed",
                                {
                                    "last_error": scene.webcam_last_error,
                                    "matrix_size": current_matrix_size,
                                },
                            )
                    else:
                        tui_manager.log("Camera capture failed; keeping previous matrix")

            except Exception as exc:  # pragma: no cover - hardware specific
                scene.webcam_last_error = str(exc)
                tui_manager.log(f"❌ Camera capture error: {exc}")
                if is_debug_enabled():
                    debug_log(
                        "WEB_CAM_CAPTURE_EXCEPTION",
                        "Unhandled webcam capture exception",
                        {"error": str(exc)},
                    )

            if (
                scene.webcam_stream is None
                and target_fps > 0
                and scene.webcam_running
                and time.time() >= next_stream_retry
            ):
                try:
                    retry_width = max(160, current_matrix_size[0] * 8)
                    retry_height = max(120, current_matrix_size[1] * 8)
                    scene.webcam_stream = camera_capture.start_stream(
                        device, retry_width, retry_height, target_fps
                    )
                    scene.webcam_update_interval = 1.0 / target_fps
                    tui_manager.log(
                        f"Camera stream resumed at {int(target_fps)}fps ({retry_width}x{retry_height})"
                    )
                except Exception:
                    next_stream_retry = time.time() + 5.0
                else:
                    next_stream_retry = time.time() + 5.0

            desired_interval = frame_interval if scene.webcam_stream else fallback_interval
            loop_elapsed = time.time() - loop_start
            sleep_remaining = max(0.0, desired_interval - loop_elapsed)
            end_time = time.time() + sleep_remaining
            while scene.webcam_running and time.time() < end_time:
                time.sleep(0.02)

    def _update_camera_motion(self, color_matrix: np.ndarray) -> None:
        scene = self.scene
        if scene._previous_camera_frame is None:
            scene._previous_camera_frame = color_matrix
            scene.camera_motion = 0.0
            return

        try:
            diff = np.abs(color_matrix.astype(int) - scene._previous_camera_frame.astype(int))
            scene.camera_motion = float(np.mean(diff))
            scene._previous_camera_frame = color_matrix
            update_state(
                "scene_manager",
                "camera_motion",
                scene.camera_motion,
                "Updated camera motion metric",
            )
        except Exception as exc:  # pragma: no cover - safeguard
            debug_log(
                "WEB_CAM_CAPTURE_EXCEPTION",
                f"Camera motion update failed: {exc}",
            )


@dataclass
class GlobalControlManager:
    """Manage global playback control logic and triggers."""

    scene: "SceneBasedSoundscape"

    def start_monitoring(self) -> None:
        if self.scene.global_control_config:
            self.scene.global_control_start_time = time.time()
            debug_log(
                "GLOBAL_CONTROL",
                "Global control monitoring started",
                {"start_time": self.scene.global_control_start_time},
            )

    def check_controls(self) -> None:
        self._check_global_playback_control()

    def _check_global_playback_control(self) -> None:
        scene = self.scene
        config = scene.global_control_config
        if not config:
            return

        current_time = time.time()
        if not scene.global_control_initialized:
            if scene.global_control_start_time == 0.0:
                return
            if current_time - scene.global_control_start_time < 2.0:
                return
            scene.global_control_initialized = True
            debug_log(
                "GLOBAL_CONTROL", "Global control initialized, starting monitoring"
            )
        else:
            if current_time - scene.last_global_control_check < 0.5:
                return

        scene.last_global_control_check = current_time

        try:
            trigger_spec = config.get("trigger", {})
            if trigger_spec:
                self._check_trigger_commands(trigger_spec)
                return

            control_spec = config.get("control", {})
            command = control_spec.get("shell")
            if not command:
                return

            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=1.0,
            )
            if result.returncode != 0:
                return

            output = result.stdout.strip()
            debug_log(
                "GLOBAL_CONTROL",
                f"Shell command output: '{output}'",
                {"command": command, "output": output},
            )

            if output == "1" and tui_manager.is_paused:
                tui_manager.toggle_pause()
                tui_manager.set_loading_state(False)
                tui_manager.log("Global control: Playback resumed (output='1')")
                debug_log("GLOBAL_CONTROL", "Resumed playback due to '1' output")
                if scene.start_recording_callback:
                    scene.start_recording_callback()
                self._update_players_from_loading_to_active()
            elif output == "0" and not tui_manager.is_paused:
                tui_manager.toggle_pause()
                tui_manager.log("Global control: Playback paused (output='0')")
                debug_log("GLOBAL_CONTROL", "Paused playback due to '0' output")
                self._update_players_from_loading_to_active()
            else:
                debug_log(
                    "GLOBAL_CONTROL",
                    f"No state change needed: output='{output}', is_paused={tui_manager.is_paused}",
                )

        except subprocess.TimeoutExpired:
            debug_log("GLOBAL_CONTROL", "Shell command timeout", {"command": command})
        except Exception as exc:  # pragma: no cover - shell interaction
            debug_log("GLOBAL_CONTROL", f"Shell command error: {exc}", {"command": command})

    def _check_trigger_commands(self, trigger_spec: Dict[str, Any]) -> None:
        scene = self.scene

        if "start" in trigger_spec and not scene.trigger_start_executed:
            start_spec = trigger_spec["start"]
            command = start_spec.get("shell")
            if command:
                try:
                    result = subprocess.run(
                        command,
                        shell=True,
                        capture_output=True,
                        text=True,
                        timeout=3600.0,
                    )
                    if result.returncode == 0:
                        output = result.stdout.strip()
                        debug_log(
                            "TRIGGER_START",
                            f"Start trigger command output: '{output}'",
                            {"command": command, "output": output},
                        )
                        if output == "1" or result.returncode == 0:
                            self._execute_start_trigger()
                            scene.trigger_start_executed = True
                            debug_log(
                                "TRIGGER_START",
                                "Start trigger executed successfully",
                            )
                    else:
                        debug_log(
                            "TRIGGER_START",
                            f"Start trigger command failed: {result.stderr}",
                            {"command": command, "returncode": result.returncode},
                        )
                except subprocess.TimeoutExpired:
                    debug_log(
                        "TRIGGER_START", "Start trigger command timeout", {"command": command}
                    )
                except Exception as exc:  # pragma: no cover - shell interaction
                    debug_log(
                        "TRIGGER_START",
                        f"Start trigger command error: {exc}",
                        {"command": command},
                    )

        if "pause" in trigger_spec and not scene.trigger_pause_executed:
            pause_spec = trigger_spec["pause"]
            command = pause_spec.get("shell")
            if command:
                try:
                    result = subprocess.run(
                        command,
                        shell=True,
                        capture_output=True,
                        text=True,
                        timeout=3600.0,
                    )
                    if result.returncode == 0:
                        output = result.stdout.strip()
                        debug_log(
                            "TRIGGER_PAUSE",
                            f"Pause trigger command output: '{output}'",
                            {"command": command, "output": output},
                        )
                        if output == "0" or result.returncode == 0:
                            self._execute_pause_trigger()
                            scene.trigger_pause_executed = True
                            debug_log(
                                "TRIGGER_PAUSE",
                                "Pause trigger executed successfully",
                            )
                    else:
                        debug_log(
                            "TRIGGER_PAUSE",
                            f"Pause trigger command failed: {result.stderr}",
                            {"command": command, "returncode": result.returncode},
                        )
                except subprocess.TimeoutExpired:
                    debug_log(
                        "TRIGGER_PAUSE", "Pause trigger command timeout", {"command": command}
                    )
                except Exception as exc:  # pragma: no cover - shell interaction
                    debug_log(
                        "TRIGGER_PAUSE",
                        f"Pause trigger command error: {exc}",
                        {"command": command},
                    )

    def _execute_start_trigger(self) -> None:
        if tui_manager.is_paused:
            tui_manager.toggle_pause()
            tui_manager.set_loading_state(False)
            tui_manager.log("Trigger: Playback started")
            debug_log("TRIGGER_START", "Resumed playback due to start trigger")
            if self.scene.start_recording_callback:
                self.scene.start_recording_callback()
            self._update_players_from_loading_to_active()
        else:
            debug_log("TRIGGER_START", "Start trigger ignored - already playing")

    def _execute_pause_trigger(self) -> None:
        if not tui_manager.is_paused:
            tui_manager.toggle_pause()
            tui_manager.log("Trigger: Playback paused")
            debug_log("TRIGGER_PAUSE", "Paused playback due to pause trigger")
            self._update_players_from_loading_to_active()
        else:
            debug_log("TRIGGER_PAUSE", "Pause trigger ignored - already paused")

    def _update_players_from_loading_to_active(self) -> None:
        for player in self.scene.players:
            if not player.active:
                continue
            current_status = tui_manager.players_status.get(player.name, {})
            if current_status.get("state") != STATE_LOADING:
                continue
            tui_manager.update_player_status(
                player.name,
                {
                    "state": STATE_ACTIVE,
                    "playback_progress": 0,
                    "playback_total": 0,
                    "waiting_progress": 0,
                    "waiting_total": 0,
                    "active": True,
                },
            )
            debug_log(
                "GLOBAL_CONTROL",
                f"Updated {player.name} from Loading to Active",
            )


class SceneBasedSoundscape:
    """Manages scene-based soundscape playback with live reloading."""

    def __init__(self, scene_file: Path | str) -> None:
        self.scene_file = Path(scene_file)
        self.scene_config: Optional[Dict[str, Any]] = None
        self.players: List[SamplePlayer] = []
        self.file_observer: Optional[Observer] = None
        self.reload_lock = threading.Lock()
        self.color_matrix: Optional[Any] = None
        self.webcam_thread: Optional[threading.Thread] = None
        self.webcam_running = False
        self.webcam_update_interval = CAMERA_UPDATE_INTERVAL
        self.webcam_matrix_size: Tuple[int, int] = (48, 36)
        self.camera_device: Optional[str] = None
        self.webcam_stream = None
        self.webcam_target_fps = 0.0
        self.current_map_mode = "random"
        self.camera_motion = 0.0
        self._previous_camera_frame: Optional[Any] = None
        self.webcam_last_error: Optional[str] = None
        self.visual_plugin: Optional[VisualPluginRuntime] = None
        self.visual_plugin_config: Optional[Dict[str, Any]] = None

        # Global control state
        self.global_control_config: Optional[Dict[str, Any]] = None
        self.last_global_control_check = 0.0
        self.global_control_initialized = False
        self.global_control_start_time = 0.0

        # Trigger state
        self.trigger_start_executed = False
        self.trigger_pause_executed = False

        # Recording callback
        self.start_recording_callback: Optional[Callable[[], None]] = None

        # Camera processing parameters
        self.camera_brightness = 1.3
        self.camera_contrast = 1.4

        # Helper managers
        self.player_manager = PlayerManager(self)
        self.visual_manager = VisualPluginManager(self)
        self.webcam_controller = WebcamController(self)
        self.global_control_manager = GlobalControlManager(self)

        debug_log(
            "SCENE_INIT",
            "Scene manager initialized",
            {"scene_file": str(self.scene_file), "default_map_mode": self.current_map_mode},
        )
        update_state("scene_manager", "scene_file", str(self.scene_file))
        update_state("scene_manager", "map_mode", self.current_map_mode)

    # ------------------------------------------------------------------
    # Compatibility helpers for legacy callers/tests
    # ------------------------------------------------------------------
    def _create_players_from_sample_config(
        self, sample_config: Dict[str, Any], global_config: Dict[str, Any], sample_index: int
    ) -> List[SamplePlayer]:
        return self.player_manager.create_players(sample_config, global_config, sample_index)

    def _cleanup_duplicate_linein_players(self) -> None:
        self.player_manager.cleanup_duplicate_linein_players()

    def _start_global_control_monitoring(self) -> None:
        self.global_control_manager.start_monitoring()

    def _check_global_playback_control(self) -> None:
        self.global_control_manager._check_global_playback_control()

    # ------------------------------------------------------------------
    # Scene loading and reloading
    # ------------------------------------------------------------------
    def load_scene(self) -> bool:
        """Load scene configuration from JSON file."""

        debug_log("SCENE_LOAD_START", "Starting scene load", {"file": str(self.scene_file)})

        try:
            with open(self.scene_file, "r", encoding="utf-8") as handle:
                self.scene_config = json.load(handle)

            scene_name = self.scene_config.get("name", "Unnamed")
            scene_desc = self.scene_config.get("description", "No description")

            tui_manager.set_typewriter_from_config(
                self.scene_config.get("typewriter"), self.scene_file.parent
            )
            self.visual_manager.configure()

            debug_log(
                "SCENE_CONFIG_LOADED",
                "Scene configuration loaded",
                {
                    "name": scene_name,
                    "description": scene_desc,
                    "has_global_config": "global" in self.scene_config,
                    "sample_count": len(self.scene_config.get("samples", [])),
                },
            )

            tui_manager.log(f"Loaded scene: {scene_name}")
            tui_manager.log(f"Description: {scene_desc}")
            tui_manager.update_scene_name(scene_name)

            update_state("scene", "name", scene_name)
            update_state("scene", "description", scene_desc)

            global_config = self.scene_config.get("global", {})
            self.global_control_config = global_config.get("playback")
            if self.global_control_config:
                debug_log(
                    "GLOBAL_CONTROL_INIT",
                    "Global playback control enabled",
                    {"config": self.global_control_config},
                )
                tui_manager.log("Global playback control enabled")
                tui_manager.set_global_control_active(True)
                debug_log(
                    "GLOBAL_CONTROL_INIT",
                    "Loading state maintained due to global control",
                )
            else:
                tui_manager.set_loading_state(False)
                debug_log(
                    "GLOBAL_CONTROL_INIT", "Loading state cleared - no global control"
                )

            samples = self.scene_config.get("samples", [])
            if not samples:
                tui_manager.log("No samples defined in scene!")
                debug_log(
                    "SCENE_LOAD_ERROR", "No samples defined in scene", {"file": str(self.scene_file)}
                )
                return False

            sort_config = global_config.get("sort") if isinstance(global_config, dict) else None
            if sort_config:
                samples = self._sort_samples(samples, sort_config)

            self.players = []
            for index, sample_config in enumerate(samples):
                if "path" not in sample_config:
                    tui_manager.log(f"Skipping sample without path: {sample_config}")
                    continue
                self.players.extend(
                    self.player_manager.create_players(sample_config, global_config, index)
                )

            update_state("scene", "player_count", len(self.players))
            update_state("scene", "loaded", True)

            tui_manager.log(f"Loaded {len(self.players)} samples from scene")
            debug_log(
                "SCENE_LOAD_SUCCESS",
                "Scene loaded successfully",
                {"player_count": len(self.players), "scene_name": scene_name},
            )
            return True

        except FileNotFoundError:
            error_msg = f"Scene file not found: {self.scene_file}"
            tui_manager.log(error_msg)
            log_error(
                "FILE_NOT_FOUND", "scene_manager", error_msg, {"file": str(self.scene_file)}
            )
            return False
        except json.JSONDecodeError as exc:
            error_msg = f"Invalid JSON in scene file: {exc}"
            tui_manager.log(error_msg)
            log_error(
                "JSON_DECODE_ERROR",
                "scene_manager",
                error_msg,
                {"file": str(self.scene_file), "json_error": str(exc)},
            )
            return False
        except Exception as exc:
            error_msg = f"Error loading scene: {exc}"
            tui_manager.log(error_msg)
            log_error(
                "SCENE_LOAD_ERROR",
                "scene_manager",
                error_msg,
                {"file": str(self.scene_file), "exception": str(exc)},
            )
            return False

    def reload_scene(self) -> None:
        """Reload scene configuration with selective updates."""

        with self.reload_lock:
            try:
                with open(self.scene_file, "r", encoding="utf-8") as handle:
                    new_scene_config = json.load(handle)

                tui_manager.log(
                    f"Reloading scene: {new_scene_config.get('name', 'Unnamed')}"
                )

                new_samples = new_scene_config.get("samples", [])
                new_global_config = new_scene_config.get("global", {})

                if not new_samples:
                    tui_manager.log("❌ No samples in new configuration, keeping current")
                    return

                has_multiply = any(sample.get("multiply", 1) != 1 for sample in new_samples)
                current_samples = self.scene_config.get("samples", []) if self.scene_config else []
                current_has_multiply = any(
                    sample.get("multiply", 1) != 1 for sample in current_samples
                )

                if has_multiply or current_has_multiply:
                    tui_manager.log("Multiply parameter detected; performing full reload")
                    self._perform_full_reload(new_scene_config)
                    return

                current_players_by_path: Dict[str, List[SamplePlayer]] = {}
                for player in self.players:
                    path = player.config["path"]
                    current_players_by_path.setdefault(path, []).append(player)

                players_to_keep: List[SamplePlayer] = []
                players_to_start: List[SamplePlayer] = []

                for new_sample_config in new_samples:
                    if "path" not in new_sample_config:
                        tui_manager.log(
                            f"Sample missing 'path' field: {new_sample_config}"
                        )
                        continue

                    sample_path = new_sample_config["path"]
                    if sample_path in current_players_by_path:
                        existing_players = current_players_by_path[sample_path]
                        if sample_path.startswith("line-in://") and len(existing_players) > 1:
                            working_player = next(
                                (player for player in existing_players if "[failed]" not in player.name),
                                None,
                            )
                            failed_player = next(
                                (player for player in existing_players if "[failed]" in player.name),
                                None,
                            )
                            if working_player and failed_player:
                                tui_manager.log(
                                    f"[{failed_player.name}] Removing duplicate (keeping working line-in)"
                                )
                                failed_player.stop()
                                players_to_keep.append(working_player)
                                del current_players_by_path[sample_path]
                                continue

                        current_player = existing_players[0]
                        comparison_player = self.player_manager._build_player(
                            new_sample_config, new_global_config
                        )
                        if current_player.get_config_hash() != comparison_player.get_config_hash():
                            current_player.update_config(
                                new_sample_config, new_global_config
                            )
                            tui_manager.log(
                                f"[{current_player.name}] ⚡ Config updated live"
                            )
                        else:
                            tui_manager.log(
                                f"[{current_player.name}] ✓ No changes, continuing"
                            )

                        players_to_keep.append(current_player)
                        del current_players_by_path[sample_path]
                    else:
                        new_player = self.player_manager._build_player(
                            new_sample_config, new_global_config
                        )
                        new_player.active = new_sample_config.get("active", True)
                        players_to_start.append(new_player)
                        players_to_keep.append(new_player)

                players_to_stop: List[SamplePlayer] = []
                for path_players in current_players_by_path.values():
                    players_to_stop.extend(path_players)

                for player in players_to_stop:
                    tui_manager.log(
                        f"[{player.name}] Stopping (removed from scene)"
                    )
                    player.stop()

                for player in players_to_start:
                    if player.active:
                        tui_manager.log(
                            f"[{player.name}] Starting (new in scene)"
                        )
                        player.start()
                    else:
                        tui_manager.log(
                            f"[{player.name}] Inactive (new but not started)"
                        )
                        tui_manager.update_player_status(
                            player.name,
                            {
                                "state": "Inactive",
                                "playback_progress": 0,
                                "playback_total": 0,
                                "waiting_progress": 0,
                                "waiting_total": 0,
                                "active": False,
                            },
                        )

                self.players = players_to_keep
                self.scene_config = new_scene_config
                self.global_control_config = new_global_config.get("playback")

                tui_manager.set_typewriter_from_config(
                    self.scene_config.get("typewriter"), self.scene_file.parent
                )
                self.visual_manager.configure()

                tui_manager.log(
                    "✅ Scene updated: "
                    f"{len(players_to_keep)} active, {len(players_to_start)} started, {len(players_to_stop)} stopped"
                )

            except FileNotFoundError:
                tui_manager.log(f"❌ Scene file not found: {self.scene_file}")
            except json.JSONDecodeError as exc:
                tui_manager.log(f"❌ Invalid JSON in scene file: {exc}")
            except Exception as exc:
                tui_manager.log(f"❌ Error during selective scene reload: {exc}")

    def _perform_full_reload(self, new_scene_config: Dict[str, Any]) -> None:
        for player in self.players:
            tui_manager.log(f"[{player.name}] stopping (full reload)")
            player.stop()

        self.players = []
        self.scene_config = new_scene_config

        new_samples = new_scene_config.get("samples", [])
        new_global_config = new_scene_config.get("global", {})

        for index, sample_config in enumerate(new_samples):
            if "path" not in sample_config:
                continue
            new_players = self.player_manager.create_players(
                sample_config, new_global_config, index
            )
            self.players.extend(new_players)

        active_count = 0
        for player in self.players:
            if player.active:
                tui_manager.log(f"[{player.name}] Starting (full reload)")
                player.start()
                active_count += 1
            else:
                tui_manager.log(f"[{player.name}] Inactive (ready but not started)")
                tui_manager.update_player_status(
                    player.name,
                    {
                        "state": "Inactive",
                        "playback_progress": 0,
                        "playback_total": 0,
                        "waiting_progress": 0,
                        "waiting_total": 0,
                        "active": False,
                    },
                )

        tui_manager.log(
            f"✅ Full reload complete: {active_count}/{len(self.players)} active players"
        )

    # ------------------------------------------------------------------
    # File watching helpers
    # ------------------------------------------------------------------
    def start_file_watching(self) -> None:
        if self.file_observer is not None:
            return

        event_handler = SceneFileHandler(self)
        observer = Observer()
        observer.schedule(event_handler, str(self.scene_file.parent), recursive=False)
        observer.start()
        self.file_observer = observer
        tui_manager.log(f"Watching scene file: {self.scene_file}")

    def stop_file_watching(self) -> None:
        if not self.file_observer:
            return
        self.file_observer.stop()
        self.file_observer.join()
        self.file_observer = None

    # ------------------------------------------------------------------
    # Visual handling
    # ------------------------------------------------------------------
    def render_visual_plugin(
        self, width: int, height: int
    ) -> Tuple[Optional[Any], Optional[str]]:
        return self.visual_manager.render(width, height)

    def set_map_mode(self, mode: str) -> None:
        debug_log(
            "MAP_MODE_CHANGE_REQUEST",
            f"Map mode change requested: {self.current_map_mode} -> {mode}",
        )

        if mode == self.current_map_mode:
            debug_log(
                "MAP_MODE_NO_CHANGE",
                f"Map mode already set to {mode}, no change needed",
            )
            return

        old_mode = self.current_map_mode
        self.current_map_mode = mode

        update_state("scene_manager", "map_mode", mode, f"Map mode changed from {old_mode}")
        tui_manager.log(f"Map mode changed from {old_mode} to {mode}")
        debug_log("MAP_MODE_CHANGED", f"Map mode changed from {old_mode} to {mode}")

        self._apply_map_mode()

    def _apply_map_mode(self) -> None:
        debug_log("MAP_MODE_APPLY_START", f"Applying map mode: {self.current_map_mode}")

        if self.current_map_mode == "camera":
            log_operation("setup_camera_mapping", "scene_manager")
            self.webcam_controller.setup_mapping()
        elif self.current_map_mode == "random":
            log_operation("setup_random_mapping", "scene_manager")
            self.webcam_controller.stop_capture()
            self.color_matrix = None
            tui_manager.update_webcam_matrix(None)
            tui_manager.log("Using random map generation")
            update_state("scene_manager", "camera_active", False)
        elif self.visual_plugin and self.current_map_mode == self.visual_plugin.key:
            self.webcam_controller.stop_capture()
            self.color_matrix = None
            tui_manager.update_webcam_matrix(None)
            tui_manager.log(f"🎨 Visual plugin active: {self.visual_plugin.name}")
            update_state("scene_manager", "camera_active", False)
            update_state("scene_manager", "visual_plugin", self.visual_plugin.name)
            tui_manager.configure_visual_plugin(
                {
                    "mode_key": self.visual_plugin.key,
                    "display_name": self.visual_plugin.name,
                }
            )
        else:
            log_error(
                "UNKNOWN_MAP_MODE",
                "scene_manager",
                f"Unknown map mode: {self.current_map_mode}",
            )
            tui_manager.log(
                f"Unknown map mode '{self.current_map_mode}'; using random"
            )
            self.current_map_mode = "random"
            self.webcam_controller.stop_capture()
            self.color_matrix = None
            tui_manager.update_webcam_matrix(None)
            tui_manager.log("Using random map generation")
            update_state("scene_manager", "camera_active", False)

        debug_log("MAP_MODE_APPLY_COMPLETE", f"Map mode applied: {self.current_map_mode}")

    # ------------------------------------------------------------------
    # Global control helpers
    # ------------------------------------------------------------------
    def check_global_controls(self) -> None:
        self.global_control_manager.check_controls()

    # ------------------------------------------------------------------
    # Sorting helper
    # ------------------------------------------------------------------
    def _sort_samples(self, samples: List[Dict[str, Any]], sort_config: str) -> List[Dict[str, Any]]:
        if not sort_config:
            return samples

        sort_key = sort_config.lower()
        reverse = False

        if sort_key.startswith("-"):
            reverse = True
            sort_key = sort_key[1:]

        if sort_key not in {"name", "length"}:
            tui_manager.log(
                f"Warning: Invalid sort key '{sort_key}'. Supported: 'name', 'length'"
            )
            return samples

        try:
            if sort_key == "name":
                return sorted(
                    samples,
                    key=lambda sample: sample.get(
                        "name", Path(sample.get("path", "")).stem
                    ).lower(),
                    reverse=reverse,
                )

            def sample_length(sample: Dict[str, Any]) -> float:
                try:
                    sample_path = Path(sample.get("path", ""))
                    if sample_path.exists():
                        import librosa  # type: ignore

                        return float(
                            librosa.get_duration(filename=str(sample_path))
                        )
                except Exception:
                    pass
                return 0.0

            return sorted(samples, key=sample_length, reverse=reverse)

        except Exception as exc:
            tui_manager.log(f"Warning: Failed to sort samples: {exc}")
            return samples

    # ------------------------------------------------------------------
    # Soundscape lifecycle
    # ------------------------------------------------------------------
    def start_soundscape(
        self,
        enable_tui: bool = False,
        start_recording_callback: Optional[Callable[[], None]] = None,
    ) -> None:
        if not self.players:
            return

        if enable_tui and not tui_manager.tui_enabled:
            tui_manager.enable_tui()
            tui_manager.start_live_display()

        tui_manager.set_stopped(False)

        self.current_map_mode = tui_manager.get_current_map_mode()
        self._apply_map_mode()

        self.start_recording_callback = start_recording_callback

        global_config = self.scene_config.get("global", {}) if self.scene_config else {}
        fade_in_time = global_config.get("fade_in_time", 2.0)

        active_players = [player for player in self.players if player.active]

        tui_manager.log(
            f"\nStarting soundscape with {len(active_players)}/{len(self.players)} active samples..."
        )
        tui_manager.log("Press 'q' to quit or Ctrl+C to stop")
        if enable_tui:
            tui_manager.log("TUI enabled with live progress tracking")
            tui_manager.log(
                "Shortcuts: c=Processes panel, l=layout, m=map mode, "
                "Space=pause/resume, v=visual panel, ?=help, q=quit"
            )
        tui_manager.log(
            "Scene file will be monitored for changes with selective live updates\n"
        )

        self.start_file_watching()

        has_global_control = self.global_control_config is not None
        if has_global_control and not tui_manager.is_paused:
            tui_manager.log("Global control detected - starting in paused mode")
            tui_manager.toggle_pause()
            debug_log(
                "GLOBAL_CONTROL",
                "Started in paused mode due to global control config",
            )
        elif not has_global_control and self.start_recording_callback:
            self.start_recording_callback()

        per_player_fade = fade_in_time / len(active_players) if active_players else 0
        for index, player in enumerate(active_players):
            if per_player_fade > 0:
                tui_manager.log(
                    f"[{player.name}] Starting (fade-in {per_player_fade:.2f}s)"
                )
            else:
                tui_manager.log(f"[{player.name}] Starting")
            player.start()

            if has_global_control:
                tui_manager.update_player_status(
                    player.name,
                    {
                        "state": STATE_LOADING,
                        "playback_progress": 0,
                        "playback_total": 0,
                        "waiting_progress": 0,
                        "waiting_total": 0,
                        "active": True,
                    },
                )

            if index < len(active_players) - 1 and per_player_fade > 0:
                time.sleep(per_player_fade)

        self.global_control_manager.start_monitoring()
        self.player_manager.cleanup_duplicate_linein_players()

        inactive_players = [player for player in self.players if not player.active]
        for player in inactive_players:
            tui_manager.update_player_status(
                player.name,
                {
                    "state": "Inactive",
                    "playback_progress": 0,
                    "playback_total": 0,
                    "waiting_progress": 0,
                    "waiting_total": 0,
                    "active": False,
                },
            )

    def stop_soundscape(self) -> None:
        global_config = self.scene_config.get("global", {}) if self.scene_config else {}
        fade_out_time = global_config.get("fade_out_time", 3.0)

        tui_manager.log(f"\nStopping soundscape (fade out: {fade_out_time}s)...")

        self.webcam_controller.stop_capture()
        self.stop_file_watching()

        for player in self.players:
            player.stop()

        try:
            pygame.mixer.stop()
        except pygame.error:  # pragma: no cover - mixer state dependent
            pass
        time.sleep(min(fade_out_time, 2.0))

        tui_manager.set_stopped(True)
        tui_manager.set_recording(False)
        tui_manager.set_loading_state(False)

    # ------------------------------------------------------------------
    # Webcam helpers
    # ------------------------------------------------------------------
    def request_webcam_resize(self, width: int, height: int) -> None:
        self.webcam_controller.request_resize(width, height)


# Global scene manager instance
scene_manager: Optional[SceneBasedSoundscape] = None

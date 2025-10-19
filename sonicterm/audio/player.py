#!/usr/bin/env python3
#
# SonicTerm
# Copyright (c) 2025 Martynas Jocius
#
"""Audio sample playback helpers for SonicTerm."""

import pygame
import time
import random
import threading
import os
import json
import hashlib
import subprocess
import math
from pathlib import Path

from ..debug import debug_log, debug_manager, log_error, is_debug_enabled
from ..ui.tui import tui_manager

# Try to import audio input libraries
try:
    import pyaudio
    import numpy as np

    AUDIO_INPUT_AVAILABLE = True
except ImportError:
    AUDIO_INPUT_AVAILABLE = False
    pyaudio = None
    np = None

from .. import (
    STATE_ACTIVE,
    STATE_WAITING,
    STATE_STARTING,
    STATE_SLEEPING,
    STATE_LOADING,
)

# Constants for better code readability
DEFAULT_VOLUME_MIN = 0.3
DEFAULT_VOLUME_MAX = 0.8
DEFAULT_PAN_MIN = -1.0
DEFAULT_PAN_MAX = 1.0
DEFAULT_GAIN = 1.0
STATUS_UPDATE_INTERVAL = 0.05  # 50ms intervals for progress updates
FALLBACK_MAX_DURATION = 10.0  # Max seconds for fallback duration
CONTROL_POLL_INTERVAL = 1.0  # Seconds between control command executions
CONTROL_COMMAND_TIMEOUT = 1.5  # Fail safe for external control commands


class SamplePlayer:
    """Handles loading, processing, and playback of individual audio samples."""

    def __init__(self, sample_config, global_config):
        self.config = sample_config
        self.global_config = global_config
        self.name = sample_config.get("name", Path(sample_config["path"]).stem)
        self.active = sample_config.get("active", True)  # Active state from config
        self.is_playing = False
        self.thread = None
        self.control_cache = {}
        self.math_control_state = {}

        # Envelope state tracking
        self.envelope_config = sample_config.get("envelope", {})
        self.envelope_attack_time = self.envelope_config.get("attack", 0.0)
        self.envelope_release_time = self.envelope_config.get("release", 0.0)
        self.envelope_start_time = 0.0
        self.envelope_end_time = 0.0
        self.envelope_active = False

    def start_envelope(self):
        """Start envelope attack phase"""
        if self.envelope_attack_time > 0 or self.envelope_release_time > 0:
            self.envelope_start_time = time.time()
            self.envelope_active = True

    def stop_envelope(self):
        """Start envelope release phase"""
        if self.envelope_active and self.envelope_release_time > 0:
            self.envelope_end_time = time.time()

    def _calculate_envelope_multiplier(self):
        """Calculate current envelope multiplier (0.0 to 1.0)"""
        if not self.envelope_active:
            return 1.0

        current_time = time.time()

        # Attack phase
        if self.envelope_start_time > 0 and self.envelope_attack_time > 0:
            attack_elapsed = current_time - self.envelope_start_time
            if attack_elapsed < self.envelope_attack_time:
                # Linear attack from 0 to 1
                return min(1.0, attack_elapsed / self.envelope_attack_time)

        # Sustain phase - full volume
        if self.envelope_end_time == 0:
            return 1.0

        # Release phase
        if self.envelope_end_time > 0 and self.envelope_release_time > 0:
            release_elapsed = current_time - self.envelope_end_time
            if release_elapsed < self.envelope_release_time:
                # Linear release from 1 to 0
                return max(0.0, 1.0 - (release_elapsed / self.envelope_release_time))
            else:
                # Release complete
                self.envelope_active = False
                return 0.0

        return 1.0

    def get_config_hash(self):
        """Generate a hash of the current configuration for change detection"""
        config_str = json.dumps(
            {"sample": self.config, "global": self.global_config}, sort_keys=True
        )
        return hashlib.md5(config_str.encode()).hexdigest()

    def update_config(self, new_sample_config, new_global_config):
        """Update configuration without stopping playback"""
        self.config = new_sample_config
        self.global_config = new_global_config
        tui_manager.log(f"[{self.name}] Configuration updated; playback continues")

    def load_and_process_sample(self):
        """Load sample using pygame"""
        try:
            file_path = self.config["path"]
            if not os.path.exists(file_path):
                tui_manager.log(f"File not found: {file_path}")
                return None, 0.0

            # Load audio file via pygame for consistent decoding performance
            sound = pygame.mixer.Sound(file_path)

            # Get basic duration info (pygame handles all format conversions internally)
            # For duration, we'll use a simple approximation or default
            duration = sound.get_length()  # Returns duration in seconds

            # Generate random effect parameters from config for display/logging
            volume_config = self.config.get(
                "volume", {"min": DEFAULT_VOLUME_MIN, "max": DEFAULT_VOLUME_MAX}
            )
            volume_control_source = self._determine_volume_control_source(volume_config)

            volume = self._resolve_controlled_value(
                volume_config, "volume", DEFAULT_VOLUME_MIN, DEFAULT_VOLUME_MAX
            )

            pan_config = self.config.get(
                "pan", {"min": DEFAULT_PAN_MIN, "max": DEFAULT_PAN_MAX}
            )
            pan_value = random.uniform(pan_config["min"], pan_config["max"])

            gain = self.config.get("gain", DEFAULT_GAIN)

            sound.set_volume(1.0)

            parameters = {
                "pan": pan_value,
                "volume": volume,
                "gain": gain,
                "volume_control_source": volume_control_source,
                "envelope_attack": self.envelope_attack_time,
                "envelope_release": self.envelope_release_time,
            }

            self._apply_channel_mix(None, parameters)

            if is_debug_enabled():
                debug_log(
                    "SAMPLE_LOADED",
                    f"{self.name} loaded",
                    {
                        "duration": duration,
                        "volume": volume,
                        "gain": gain,
                        "pan": parameters.get("pan"),
                        "final_volume": parameters.get("final_volume"),
                        "envelope_attack": self.envelope_attack_time,
                        "envelope_release": self.envelope_release_time,
                        "path": file_path,
                    },
                )

            return sound, duration, parameters

        except Exception as e:
            tui_manager.log(f"Error loading {self.config['path']}: {e}")
            return None, 0.0, {}

    def _resolve_controlled_value(self, config, name, default_min, default_max):
        """Resolve parameter value, running control command if configured."""
        min_val = float(config.get("min", default_min))
        max_val = float(config.get("max", default_max))
        control_type = None
        control_spec = config.get("control")
        control_cmd = None
        camera_target = None
        math_spec = None
        if isinstance(control_spec, dict):
            if "shell" in control_spec:
                control_type = "shell"
                control_cmd = control_spec.get("shell")
            elif "camera" in control_spec:
                control_type = "camera"
                camera_target = control_spec.get("camera")
            elif "math" in control_spec:
                control_type = "math"
                math_spec = control_spec.get("math")

        if control_type == "shell" and control_cmd:
            cache = self.control_cache.get(name)
            now = time.time()

            if cache is None or (now - cache["timestamp"]) >= CONTROL_POLL_INTERVAL:
                try:
                    result = subprocess.run(
                        control_cmd,
                        shell=True,
                        capture_output=True,
                        text=True,
                        timeout=CONTROL_COMMAND_TIMEOUT,
                    )
                    if result.returncode != 0:
                        raise RuntimeError(
                            result.stderr.strip() or f"exit code {result.returncode}"
                        )
                    output = result.stdout.strip()
                    value = float(output)
                    if is_debug_enabled():
                        debug_log(
                            "CONTROL_COMMAND",
                            f"{self.name} control value updated",
                            {
                                "parameter": name,
                                "value": value,
                                "command": control_cmd,
                            },
                        )
                        if debug_manager and getattr(debug_manager, "enabled", False):
                            tui_manager.log(
                                f"[{self.name}] control '{name}' -> {value:.4f}"
                            )
                    cache = {"value": value, "timestamp": now}
                    self.control_cache[name] = cache
                except Exception as exc:
                    tui_manager.log(
                        f"[{self.name}] Control command for '{name}' failed ({exc}); using configured range instead."
                    )
                    cache = {"value": None, "timestamp": now}
                    self.control_cache[name] = cache

            value = cache.get("value") if cache else None

            if value is None:
                return random.uniform(min_val, max_val)

            return max(0.0, min(1.0, value))

        if control_type == "camera" and camera_target:
            value = self._get_camera_control_value(camera_target)
            if value is not None:
                return value
            return random.uniform(min_val, max_val)

        if control_type == "math" and math_spec:
            return self._evaluate_math_control(name, min_val, max_val, math_spec)

        return random.uniform(min_val, max_val)

    @staticmethod
    def _determine_volume_control_source(volume_config):
        """Determine the declared control type for volume configuration."""
        if isinstance(volume_config, dict):
            control_spec = volume_config.get("control")
            if isinstance(control_spec, dict):
                if "shell" in control_spec:
                    return "shell"
                if "camera" in control_spec:
                    return "camera"
                if "math" in control_spec:
                    return "math"
        return "range"

    def _refresh_controlled_parameters(self, channel, parameters):
        """Update controlled parameters during playback."""

        for key, defaults in (("volume", (DEFAULT_VOLUME_MIN, DEFAULT_VOLUME_MAX)),):
            param_config = self.config.get(key)
            if not isinstance(param_config, dict):
                continue
            control_spec = param_config.get("control")
            if not isinstance(control_spec, dict):
                continue

            if (
                "shell" in control_spec
                or "camera" in control_spec
                or "math" in control_spec
            ):
                new_value = self._resolve_controlled_value(
                    param_config,
                    key,
                    defaults[0],
                    defaults[1],
                )
                parameters[key] = new_value

                if key == "volume":
                    if "shell" in control_spec:
                        parameters["volume_control_source"] = "shell"
                    elif "camera" in control_spec:
                        parameters["volume_control_source"] = "camera"
                    elif "math" in control_spec:
                        parameters["volume_control_source"] = "math"

        self._apply_channel_mix(channel, parameters)

    def _apply_channel_mix(self, channel, parameters):
        """Apply pan-aware volume to the playback channel and cache metrics."""

        gain = parameters.get("gain", self.config.get("gain", DEFAULT_GAIN))
        volume = float(parameters.get("volume", DEFAULT_VOLUME_MIN))
        volume = max(0.0, min(1.0, volume))

        # Apply envelope multiplier to volume
        envelope_multiplier = self._calculate_envelope_multiplier()
        final_volume = max(0.0, min(1.0, volume * gain * envelope_multiplier))

        pan = float(parameters.get("pan", 0.0))
        pan = max(-1.0, min(1.0, pan))

        left = final_volume * (1.0 - pan) / 2.0
        right = final_volume * (1.0 + pan) / 2.0

        if channel:
            channel.set_volume(left, right)

        parameters["final_volume"] = final_volume
        parameters["pan"] = pan
        parameters["volume_left"] = left
        parameters["volume_right"] = right

    def _log_playback_start(self, parameters):
        vol = parameters.get("final_volume")
        if vol is None:
            gain = parameters.get("gain", self.config.get("gain", DEFAULT_GAIN))
            base = float(parameters.get("volume", DEFAULT_VOLUME_MIN))
            vol = max(0.0, min(1.0, base * gain))
        pan = float(parameters.get("pan", 0.0))
        source = parameters.get("volume_control_source")
        if source == "shell":
            source_note = " via control"
        else:
            source_note = ""
        tui_manager.log(
            f"[{self.name}] Playing (vol {vol:.2f}, pan {pan:+.2f}{source_note})"
        )

    def _log_playback_finish(self, duration, wait_time, parameters):
        source = parameters.get("volume_control_source")
        if source == "shell":
            source_note = " (control)"
        else:
            source_note = ""
        tui_manager.log(
            f"[{self.name}] Finished ({duration:.1f}s); waiting {wait_time}s{source_note}"
        )

    def _get_camera_control_value(self, mode):
        if str(mode).lower() != "motion":
            return None
        try:
            from ..core.scene import scene_manager as manager
        except ImportError:
            return None
        if not manager:
            return None

        value = getattr(manager, "camera_motion", None)
        if value is None:
            return None
        try:
            return max(0.0, min(1.0, float(value)))
        except (TypeError, ValueError):
            return None

    def _evaluate_math_control(self, name, min_val, max_val, math_spec):
        """Evaluate math-based control specified in sample config."""
        if not isinstance(math_spec, dict):
            return random.uniform(min_val, max_val)

        function = str(math_spec.get("function", "sine")).lower()

        if function == "sine":
            cycle_raw = math_spec.get("time", math_spec.get("cycle", 10.0))
            period = self._parse_duration_seconds(cycle_raw, default=10.0)
            if period <= 0:
                period = 10.0

            state = self.math_control_state.setdefault(name, {})
            start_time = state.get("start_time")
            if start_time is None:
                start_time = time.time()
                state["start_time"] = start_time

            elapsed = time.time() - start_time
            phase = (elapsed % period) / period
            value_normalized = 0.5 - 0.5 * math.cos(2 * math.pi * phase)
            return min_val + (max_val - min_val) * value_normalized

        return random.uniform(min_val, max_val)

    @staticmethod
    def _parse_duration_seconds(value, default=1.0):
        """Parse duration expressions like '30s' into seconds."""
        if value is None:
            return default

        if isinstance(value, (int, float)):
            try:
                return float(value)
            except (TypeError, ValueError):
                return default

        if isinstance(value, str):
            text = value.strip().lower()
            multiplier = 1.0
            if text.endswith("ms"):
                multiplier = 0.001
                text = text[:-2]
            elif text.endswith("s"):
                text = text[:-1]

            try:
                return float(text) * multiplier
            except ValueError:
                return default

        return default

    def play_loop(self):
        """Play sample in a loop with configured wait times"""
        timings = self.config.get("timings", [1, 2, 4, 8, 16])
        initial_wait = self.config.get("wait", 0)

        # Wait before first playback if configured
        if initial_wait > 0:
            tui_manager.log(
                f"[{self.name}] initial wait {initial_wait}s before first play"
            )
            self._update_status(STATE_SLEEPING, 0, 0, 0, initial_wait)

            # Progress through initial wait with pause support
            start_wait_time = time.time()
            while time.time() - start_wait_time < initial_wait:
                if not self.is_playing:
                    break
                elapsed_wait = time.time() - start_wait_time
                self._update_status(STATE_WAITING, 0, 0, elapsed_wait, initial_wait)
                tui_manager.pause_aware_sleep(STATUS_UPDATE_INTERVAL)

        while self.is_playing:
            try:
                # Load and process sample with new random effects
                self._update_status(STATE_STARTING, 0, 1, 0, 0)

                # Time the audio processing for debugging
                process_start = time.time()
                result = self.load_and_process_sample()
                process_time = time.time() - process_start

                if is_debug_enabled():
                    try:
                        debug_log(
                            "AUDIO_PROCESSING",
                            f"{self.name} audio processing completed",
                            {
                                "processing_time": process_time,
                                "sample_name": self.name,
                            },
                        )
                    except Exception as debug_err:
                        tui_manager.log(
                            f"[{self.name}] Debug logging error: {debug_err}"
                        )

                if result[0] is None:
                    break

                sound, sample_duration, parameters = result

                # Sound object is already created by load_and_process_sample
                # Play the sound
                self._update_status(STATE_ACTIVE, 0, sample_duration, 0, 0, parameters)
                channel = sound.play()

                # Start envelope attack phase
                self.start_envelope()

                self._apply_channel_mix(channel, parameters)

                # Debug: Check if we got a channel
                if channel is None:
                    tui_manager.log(
                        f"[{self.name}] No audio channel available; sound skipped"
                    )

                # Wait for sound to finish with progress tracking
                if channel:
                    self._log_playback_start(parameters)
                    start_time = time.time()
                    while channel.get_busy():
                        if not self.is_playing:
                            channel.stop()
                            break

                        # Check if we need to start release phase before sample ends
                        elapsed = time.time() - start_time
                        if (
                            self.envelope_release_time > 0
                            and elapsed >= sample_duration - self.envelope_release_time
                            and self.envelope_end_time == 0
                        ):
                            self.stop_envelope()

                        self._refresh_controlled_parameters(channel, parameters)

                        self._update_status(
                            STATE_ACTIVE, elapsed, sample_duration, 0, 0, parameters
                        )
                        tui_manager.pause_aware_sleep(STATUS_UPDATE_INTERVAL)
                else:
                    # Fallback: use calculated duration with progress and pause support
                    duration = min(sample_duration, FALLBACK_MAX_DURATION)
                    start_time = time.time()
                    while time.time() - start_time < duration:
                        if not self.is_playing:
                            break

                        # Check if we need to start release phase before sample ends
                        elapsed = time.time() - start_time
                        if (
                            self.envelope_release_time > 0
                            and elapsed >= duration - self.envelope_release_time
                            and self.envelope_end_time == 0
                        ):
                            self.stop_envelope()

                        self._refresh_controlled_parameters(None, parameters)
                        self._update_status(
                            STATE_ACTIVE, elapsed, duration, 0, 0, parameters
                        )
                        tui_manager.pause_aware_sleep(STATUS_UPDATE_INTERVAL)

                # Sample has finished playing, now apply timing wait
                if self.is_playing:
                    wait_time = random.choice(timings)
                    self._log_playback_finish(sample_duration, wait_time, parameters)

                    # Progress through wait time with pause support - keep playback bar full, show waiting progress
                    # During this time, continue applying envelope release if active
                    start_wait_time = time.time()
                    while time.time() - start_wait_time < wait_time:
                        if not self.is_playing:
                            break

                        # Continue applying envelope during wait phase (for release)
                        self._refresh_controlled_parameters(None, parameters)

                        elapsed_wait = time.time() - start_wait_time
                        self._update_status(
                            STATE_WAITING,
                            sample_duration,
                            sample_duration,
                            elapsed_wait,
                            wait_time,
                            parameters,
                        )
                        tui_manager.pause_aware_sleep(STATUS_UPDATE_INTERVAL)

            except Exception as e:
                tui_manager.log(f"Error playing {self.name}: {e}")
                break

        # Clean up TUI status when stopping
        tui_manager.remove_player_status(self.name)

    def _update_status(
        self,
        state,
        playback_progress,
        playback_total,
        waiting_progress,
        waiting_total,
        parameters=None,
    ):
        """Update TUI status for this player"""
        status = {
            "state": state,
            "playback_progress": playback_progress,
            "playback_total": playback_total,
            "waiting_progress": waiting_progress,
            "waiting_total": waiting_total,
            "active": self.active,  # Include active state in status
            "volume_control_source": self._determine_volume_control_source(
                self.config.get("volume")
            ),
        }

        # Special handling for line-in fallback samples
        if hasattr(self, "is_linein_fallback") and self.is_linein_fallback:
            # For line-in fallback, set infinite duration and proper volume control
            status["playback_total"] = float("inf")

            # Get current volume and control source for line-in fallback
            volume_config = self.config.get(
                "volume", {"min": DEFAULT_VOLUME_MIN, "max": DEFAULT_VOLUME_MAX}
            )
            current_volume = self._resolve_controlled_value(
                volume_config, "volume", DEFAULT_VOLUME_MIN, DEFAULT_VOLUME_MAX
            )
            volume_control_source = self._determine_volume_control_source(volume_config)

            status.update(
                {
                    "volume": current_volume,
                    "volume_control_source": volume_control_source,
                }
            )

        # Add parameters if provided (this will override the volume_control_source above if present)
        if parameters:
            status.update(parameters)

        # Debug logging for sample state tracking
        if is_debug_enabled():
            playback_percent = (
                (playback_progress / playback_total * 100) if playback_total > 0 else 0
            )
            waiting_percent = (
                (waiting_progress / waiting_total * 100) if waiting_total > 0 else 0
            )

            debug_log(
                "SAMPLE_STATUS",
                f"{self.name} state updated",
                {
                    "sample_name": self.name,
                    "state": state,
                    "playback_progress": playback_progress,
                    "playback_total": playback_total,
                    "playback_percent": playback_percent,
                    "waiting_progress": waiting_progress,
                    "waiting_total": waiting_total,
                    "waiting_percent": waiting_percent,
                    "has_parameters": parameters is not None,
                },
            )

        # Handle Loading state transition
        current_status = tui_manager.players_status.get(self.name, {})
        if current_status.get("state") == STATE_LOADING:
            # If we're transitioning from Loading to a playable state, change to Active
            if state in [STATE_ACTIVE, STATE_WAITING, STATE_STARTING]:
                status["state"] = STATE_ACTIVE  # Transition to Active
                debug_log(
                    "SAMPLE_STATE", f"{self.name} transitioned from Loading to Active"
                )

                # Clear global loading state when first sample becomes active
                if tui_manager.is_loading:
                    tui_manager.set_loading_state(False)
                    debug_log(
                        "TUI_STATE", "Cleared global loading state - samples are ready"
                    )
            else:
                # Keep the Loading state for other transitions
                status["state"] = STATE_LOADING

        tui_manager.update_player_status(self.name, status)

        # Trigger display update to show progress changes
        if tui_manager.tui_enabled:
            tui_manager.update_display()

    def start(self):
        """Start playing sample in background thread"""
        if not self.is_playing:
            self.is_playing = True
            self.thread = threading.Thread(target=self.play_loop, daemon=True)
            self.thread.start()

    def stop(self):
        """Stop playing sample"""
        self.is_playing = False
        if self.thread:
            self.thread.join(timeout=1.0)
        tui_manager.log(f"[{self.name}] Stopped")
        tui_manager.remove_player_status(self.name)


class LineInPlayer(SamplePlayer):
    """Player for live audio input (line-in/microphone)"""

    def __init__(self, config, global_config):
        try:
            super().__init__(config, global_config)
        except Exception as e:
            raise

        self.audio = None
        self.stream = None
        self.is_recording = False
        self.audio_buffer = []
        self.buffer_size = config.get(
            "buffer_size", 1024
        )  # Use config buffer size or default to 1024
        self.sample_rate = config.get(
            "sample_rate", 44100
        )  # Use config sample rate or default to 44100
        self.channels = config.get("channels", 1)  # Use config channels or default to 1
        self.input_device = None

        if not AUDIO_INPUT_AVAILABLE:
            raise ImportError(
                "pyaudio and numpy are required for line-in functionality"
            )

    def _determine_volume_control_source(self, volume_config):
        """Override to always return 'math' for line-in samples to show wave icon (∿)"""
        return "math"

    def load_and_process_sample(self):
        """Setup audio input instead of loading file"""
        try:
            # Initialize PyAudio
            self.audio = pyaudio.PyAudio()

            # Get input device info
            self.input_device = self._get_input_device()

            # Setup audio stream
            self.stream = self.audio.open(
                format=pyaudio.paFloat32,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                input_device_index=self.input_device,
                frames_per_buffer=self.buffer_size,
                stream_callback=self._audio_callback,
            )

            # Generate parameters for display/logging (similar to file-based samples)
            volume_config = self.config.get(
                "volume", {"min": DEFAULT_VOLUME_MIN, "max": DEFAULT_VOLUME_MAX}
            )
            volume_control_source = self._determine_volume_control_source(volume_config)
            volume = self._resolve_controlled_value(
                volume_config, "volume", DEFAULT_VOLUME_MIN, DEFAULT_VOLUME_MAX
            )

            pan_config = self.config.get(
                "pan", {"min": DEFAULT_PAN_MIN, "max": DEFAULT_PAN_MAX}
            )
            pan_value = random.uniform(pan_config["min"], pan_config["max"])

            gain = self.config.get("gain", DEFAULT_GAIN)

            parameters = {
                "pan": pan_value,
                "volume": volume,
                "gain": gain,
                "volume_control_source": volume_control_source,
                "envelope_attack": self.envelope_attack_time,
                "envelope_release": self.envelope_release_time,
            }

            # Log successful setup
            device_name = (
                self._get_device_name(self.input_device)
                if self.input_device
                else "default"
            )
            tui_manager.log(
                f"[{self.name}] Line-in setup: {device_name} @ {self.sample_rate}Hz"
            )

            debug_log(
                "LINE_IN_SETUP",
                f"Line-in player setup complete",
                {
                    "device": device_name,
                    "sample_rate": self.sample_rate,
                    "channels": self.channels,
                    "buffer_size": self.buffer_size,
                },
            )

            return parameters, float("inf")  # Infinite duration for live input

        except Exception as e:
            error_msg = f"Failed to setup line-in: {e}"
            tui_manager.log(error_msg)
            log_error(
                "LINE_IN_ERROR", "line_in_player", error_msg, {"config": self.config}
            )

            # Add [failed] suffix to the player name to indicate initialization failure
            if not self.name.endswith(" [failed]"):
                self.name = f"{self.name} [failed]"

            return None, 0.0

    def _get_input_device(self):
        """Get input device index from config or use default"""
        device_name = self.config.get("input_device", "default")

        if device_name == "default":
            return None  # Use default device

        # Try to find device by name
        for i in range(self.audio.get_device_count()):
            device_info = self.audio.get_device_info_by_index(i)
            if device_info["maxInputChannels"] > 0:  # It's an input device
                if device_name.lower() in device_info["name"].lower():
                    return i

        # Fallback to default if not found
        tui_manager.log(
            f"Warning: Input device '{device_name}' not found, using default"
        )
        return None

    def _get_device_name(self, device_index):
        """Get device name by index"""
        if device_index is None:
            return "default"
        try:
            device_info = self.audio.get_device_info_by_index(device_index)
            return device_info["name"]
        except:
            return f"device_{device_index}"

    def _audio_callback(self, in_data, frame_count, time_info, status):
        """Callback for audio input stream"""
        if not self.is_recording:
            return (None, pyaudio.paComplete)

        try:
            # Convert input data to numpy array
            audio_data = np.frombuffer(in_data, dtype=np.float32)

            # Apply SonicTerm's controls
            processed_audio = self._apply_linein_controls(audio_data)

            # Convert back to bytes for pygame
            processed_bytes = processed_audio.astype(np.float32).tobytes()

            # Play through pygame mixer
            self._play_audio_chunk(processed_bytes)

            return (None, pyaudio.paContinue)

        except Exception as e:
            debug_log("LINE_IN_CALLBACK_ERROR", f"Audio callback error: {e}")
            return (None, pyaudio.paComplete)

    def _apply_linein_controls(self, audio_data):
        """Apply SonicTerm's volume, pan, and envelope controls to live audio"""
        try:
            # Get current parameters
            volume_config = self.config.get(
                "volume", {"min": DEFAULT_VOLUME_MIN, "max": DEFAULT_VOLUME_MAX}
            )
            current_volume = self._resolve_controlled_value(
                volume_config, "volume", DEFAULT_VOLUME_MIN, DEFAULT_VOLUME_MAX
            )

            pan_config = self.config.get(
                "pan", {"min": DEFAULT_PAN_MIN, "max": DEFAULT_PAN_MAX}
            )
            current_pan = self._resolve_controlled_value(
                pan_config, "pan", DEFAULT_PAN_MIN, DEFAULT_PAN_MAX
            )

            gain = self.config.get("gain", DEFAULT_GAIN)

            # Apply envelope if active
            envelope_multiplier = self._calculate_envelope_multiplier()

            # Apply volume and gain
            processed_audio = audio_data * current_volume * gain * envelope_multiplier

            # Apply panning (convert mono to stereo)
            if self.channels == 1:
                left_gain = (1 - current_pan) / 2.0
                right_gain = (1 + current_pan) / 2.0
                # For now, just apply volume scaling (full stereo implementation would be more complex)
                processed_audio = processed_audio * ((left_gain + right_gain) / 2.0)

            return processed_audio

        except Exception as e:
            debug_log("LINE_IN_CONTROLS_ERROR", f"Error applying controls: {e}")
            return audio_data  # Return unprocessed audio on error

    def _play_audio_chunk(self, audio_bytes):
        """Play audio chunk through pygame mixer"""
        try:
            # Create pygame sound from audio data
            sound = pygame.mixer.Sound(audio_bytes)

            # Play the sound
            channel = pygame.mixer.find_channel()
            if channel:
                channel.play(sound)

        except Exception as e:
            debug_log("LINE_IN_PLAY_ERROR", f"Error playing audio chunk: {e}")

    def start(self):
        """Start line-in recording and playback"""

        # Update status with initial parameters FIRST (even if audio stream fails)
        volume_config = self.config.get(
            "volume", {"min": DEFAULT_VOLUME_MIN, "max": DEFAULT_VOLUME_MAX}
        )
        pan_config = self.config.get(
            "pan", {"min": DEFAULT_PAN_MIN, "max": DEFAULT_PAN_MAX}
        )
        current_volume = self._resolve_controlled_value(
            volume_config, "volume", DEFAULT_VOLUME_MIN, DEFAULT_VOLUME_MAX
        )
        current_pan = self._resolve_controlled_value(
            pan_config, "pan", DEFAULT_PAN_MIN, DEFAULT_PAN_MAX
        )
        gain = self.config.get("gain", DEFAULT_GAIN)

        volume_control_source = self._determine_volume_control_source(volume_config)

        parameters = {
            "volume": current_volume,
            "pan": current_pan,
            "gain": gain,
            "volume_control_source": volume_control_source,
            "envelope_attack": self.envelope_attack_time,
            "envelope_release": self.envelope_release_time,
        }

        self._update_status(STATE_ACTIVE, 0, float("inf"), 0, 0, parameters)

        if not self.stream:
            # Add [failed] suffix if not already present
            if not self.name.endswith(" [failed]"):
                self.name = f"{self.name} [failed]"
            tui_manager.log(
                f"[{self.name}] Line-in not initialized - audio stream setup failed"
            )
            # Still update status even if audio stream failed
            return

        self.is_recording = True
        self.is_playing = True

        # Start the audio stream
        self.stream.start_stream()

        # Start envelope if configured
        if self.envelope_active:
            self.start_envelope()

        tui_manager.log(f"[{self.name}] Line-in started")

    def stop(self):
        """Stop line-in recording and playback"""
        self.is_recording = False
        self.is_playing = False

        if self.stream:
            self.stream.stop_stream()

        if self.audio:
            self.audio.terminate()

        tui_manager.log(f"[{self.name}] Line-in stopped")
        tui_manager.remove_player_status(self.name)

    def play_loop(self):
        """Override play_loop for line-in (handled by audio callback)"""
        # For line-in, the actual playback is handled by the audio callback
        # This method just keeps the thread alive and updates status
        while self.is_playing:
            time.sleep(0.1)

            # Update status periodically with current parameters
            if self.is_recording:
                # Get current volume and pan values
                volume_config = self.config.get(
                    "volume", {"min": DEFAULT_VOLUME_MIN, "max": DEFAULT_VOLUME_MAX}
                )
                current_volume = self._resolve_controlled_value(
                    volume_config, "volume", DEFAULT_VOLUME_MIN, DEFAULT_VOLUME_MAX
                )

                pan_config = self.config.get(
                    "pan", {"min": DEFAULT_PAN_MIN, "max": DEFAULT_PAN_MAX}
                )
                current_pan = self._resolve_controlled_value(
                    pan_config, "pan", DEFAULT_PAN_MIN, DEFAULT_PAN_MAX
                )

                gain = self.config.get("gain", DEFAULT_GAIN)

                volume_control_source = self._determine_volume_control_source(
                    volume_config
                )

                # Create parameters dict for status update
                parameters = {
                    "volume": current_volume,
                    "pan": current_pan,
                    "gain": gain,
                    "volume_control_source": volume_control_source,
                }

                self._update_status(STATE_ACTIVE, 0, float("inf"), 0, 0, parameters)

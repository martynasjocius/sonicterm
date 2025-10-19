#!/usr/bin/env python3
#
# SonicTerm
# Copyright (c) 2025 Martynas Jocius
#

import argparse
import sys
import time
import threading
import signal
import atexit
import json
import shutil
import subprocess
from pathlib import Path

from sonicterm import VERSION, UI_PANEL_CHRONICLE
from sonicterm.utils import list_available_scenes
from sonicterm.debug import debug_log, set_debug_mode, debug_manager

__all__ = [
    "get_pulse_monitor_source",
    "setup_input_handler",
    "restore_terminal_settings",
    "validate_scene_file",
    "run_health_checks",
    "main",
]

try:
    import pygame  # type: ignore
except ImportError:  # pragma: no cover - handled at runtime when dependencies missing
    pygame = None

try:
    from sonicterm.core import SceneBasedSoundscape  # type: ignore
except ImportError:  # pragma: no cover
    SceneBasedSoundscape = None

try:
    from sonicterm.ui.tui import tui_manager  # type: ignore
except ImportError:  # pragma: no cover
    tui_manager = None


def get_pulse_monitor_source():
    """Return the PulseAudio/PipeWire monitor source for the default sink."""
    try:
        result = subprocess.run(
            ["pactl", "info"], capture_output=True, text=True, timeout=2
        )
        if result.returncode == 0:
            for line in result.stdout.splitlines():
                if line.lower().startswith("default sink:"):
                    sink = line.split(":", 1)[1].strip()
                    if sink:
                        return f"{sink}.monitor"
    except Exception:
        pass
    return "default"


def setup_input_handler(quit_requested, stop_recording_callback=None):
    """Set up keyboard input handler for TUI mode."""
    if tui_manager is None:
        raise RuntimeError("TUI components unavailable (psutil/rich missing?)")
    try:
        import termios
        import tty

        original_settings = termios.tcgetattr(sys.stdin)

        def input_handler():
            try:
                tty.setcbreak(sys.stdin.fileno())

                while not quit_requested.is_set():
                    try:
                        char = sys.stdin.read(1)
                        if char.lower() == "c":
                            tui_manager.toggle_chronicle()
                            tui_manager.update_display()
                        elif char.lower() == "m":
                            tui_manager.toggle_map_mode()
                            tui_manager.update_display()
                        elif char.lower() == "t":
                            tui_manager.toggle_typewriter_panel()
                        elif char == "?":
                            tui_manager.toggle_help_dialog()
                            tui_manager.update_display()
                        elif char == " ":  # Spacebar
                            handled = False
                            if stop_recording_callback:
                                handled = stop_recording_callback()
                            if not handled:
                                # Check if global control is active
                                if tui_manager.has_global_control():
                                    tui_manager.log(
                                        "⚠️  Spacebar control overridden by global playback control settings"
                                    )
                                tui_manager.toggle_pause()
                                tui_manager.update_display()
                        elif char.lower() == "v":  # Visual mode toggle
                            tui_manager.toggle_visual()
                            tui_manager.update_display()
                        elif char.lower() == "q":
                            if tui_manager.help_dialog_visible:
                                tui_manager.hide_help_dialog()
                                tui_manager.update_display()
                            else:
                                tui_manager.set_quitting(True)
                                tui_manager.log("Quit requested by user (q)")
                                quit_requested.set()
                                break
                        elif ord(char) == 27:
                            if tui_manager.help_dialog_visible:
                                tui_manager.hide_help_dialog()
                                tui_manager.update_display()
                        elif ord(char) == 3:
                            tui_manager.set_quitting(True)
                            quit_requested.set()
                            break
                    except:
                        break
            except:
                pass

        input_thread = threading.Thread(target=input_handler, daemon=True)
        input_thread.start()
        tui_manager.log(
            f"Input handler active. Shortcuts: q=quit, c=toggle {UI_PANEL_CHRONICLE}, l=layout, m=map mode, t=typewriter panel, Space=pause/resume"
        )

        return original_settings, input_thread

    except Exception as e:
        if tui_manager is not None:
            tui_manager.log(f"Input handler disabled; not a terminal: {e}")
            tui_manager.log("Use Ctrl+C to quit")
        return None, None


def restore_terminal_settings(original_settings):
    """Simple terminal restoration"""
    if original_settings is not None:
        try:
            import termios

            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, original_settings)
        except:
            pass


def validate_scene_file(scene_file):
    """Validate scene file before initializing heavy components."""
    from pathlib import Path

    scene_path = Path(scene_file)

    # Check if file exists
    if not scene_path.exists():
        print(f"Error: Scene file not found: {scene_file}")
        if scene_file == "scenes/default.json":
            print(
                "The default scene file is missing. Please check your SonicTerm installation."
            )
        else:
            print("Please check the file path and try again.")
        return False, None

    # Check if file is readable
    if not scene_path.is_file():
        print(f"Error: Path is not a file: {scene_file}")
        return False, None

    # Try to parse JSON
    try:
        with open(scene_path, "r", encoding="utf-8") as f:
            scene_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Scene file not found: {scene_file}")
        return False, None
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in scene file: {scene_file}")
        print(f"   JSON error at line {e.lineno}, column {e.colno}: {e.msg}")
        print("Please check the JSON syntax and try again.")
        return False, None
    except UnicodeDecodeError as e:
        print(f"Error: Unable to read scene file (encoding issue): {scene_file}")
        print(f"   {e}")
        return False, None
    except Exception as e:
        print(f"Error: Unable to read scene file: {scene_file}")
        print(f"   {e}")
        return False, None

    # Validate basic scene structure
    if not isinstance(scene_data, dict):
        print(
            f"Error: Scene file must contain a JSON object, not {type(scene_data).__name__}: {scene_file}"
        )
        return False, None

    # Check for required fields
    if "samples" not in scene_data:
        print(f"Error: Scene file missing required 'samples' field: {scene_file}")
        print("The scene file must have a 'samples' array with at least one sample.")
        return False, None

    if not isinstance(scene_data["samples"], list):
        print(f"Error: 'samples' field must be an array: {scene_file}")
        return False, None

    if len(scene_data["samples"]) == 0:
        print(f"Error: Scene file has no samples defined: {scene_file}")
        print("A scene must have at least one sample to play.")
        return False, None

    # Validate each sample has required fields
    for i, sample in enumerate(scene_data["samples"]):
        if not isinstance(sample, dict):
            print(f"Error: Sample {i + 1} must be a JSON object: {scene_file}")
            return False, None

        if "path" not in sample:
            print(f"Error: Sample {i + 1} missing required 'path' field: {scene_file}")
            return False, None

        # Validate multiply parameter if present
        if "multiply" in sample:
            multiply = sample["multiply"]
            if not isinstance(multiply, (int, float)) or multiply < 0 or multiply > 8:
                print(
                    f"Warning: Sample {i + 1} has invalid multiply value ({multiply}). Must be 0-8."
                )
                print("Invalid multiply values will be ignored and default to 1.")

    # Validate playback configuration if present
    if "playback" in scene_data:
        playback_config = scene_data["playback"]
        if not isinstance(playback_config, dict):
            print(f"Error: 'playback' field must be an object: {scene_file}")
            return False, None

        # Validate trigger configuration if present
        if "trigger" in playback_config:
            trigger_config = playback_config["trigger"]
            if not isinstance(trigger_config, dict):
                print(f"Error: 'trigger' field must be an object: {scene_file}")
                return False, None

            # Validate start trigger if present
            if "start" in trigger_config:
                start_config = trigger_config["start"]
                if not isinstance(start_config, dict):
                    print(f"Error: 'start' trigger must be an object: {scene_file}")
                    return False, None
                if "shell" not in start_config:
                    print(f"Error: 'start' trigger missing 'shell' field: {scene_file}")
                    return False, None
                if not isinstance(start_config["shell"], str):
                    print(
                        f"Error: 'start' trigger 'shell' field must be a string: {scene_file}"
                    )
                    return False, None

            # Validate pause trigger if present
            if "pause" in trigger_config:
                pause_config = trigger_config["pause"]
                if not isinstance(pause_config, dict):
                    print(f"Error: 'pause' trigger must be an object: {scene_file}")
                    return False, None
                if "shell" not in pause_config:
                    print(f"Error: 'pause' trigger missing 'shell' field: {scene_file}")
                    return False, None
                if not isinstance(pause_config["shell"], str):
                    print(
                        f"Error: 'pause' trigger 'shell' field must be a string: {scene_file}"
                    )
                    return False, None

        # Validate control configuration if present
        if "control" in playback_config:
            control_config = playback_config["control"]
            if not isinstance(control_config, dict):
                print(f"Error: 'control' field must be an object: {scene_file}")
                return False, None
            if "shell" not in control_config:
                print(f"Error: 'control' missing 'shell' field: {scene_file}")
                return False, None
            if not isinstance(control_config["shell"], str):
                print(f"Error: 'control' 'shell' field must be a string: {scene_file}")
                return False, None

    # Validate visual plugin configuration if present
    if "visual" in scene_data:
        visual_config = scene_data["visual"]
        if not isinstance(visual_config, dict):
            print(f"Error: 'visual' field must be an object: {scene_file}")
            return False, None

        plugin_path_value = visual_config.get("path")
        if not plugin_path_value or not isinstance(plugin_path_value, str):
            print(f"Error: 'visual' config missing string 'path' field: {scene_file}")
            return False, None

        plugin_path = Path(plugin_path_value)
        if not plugin_path.is_absolute():
            plugin_path = scene_path.parent / plugin_path

        if not plugin_path.exists():
            print(f"Error: Visual plugin file not found: {plugin_path}")
            return False, None

        plugin_name = visual_config.get("name")
        if plugin_name is not None and not isinstance(plugin_name, str):
            print(f"Error: 'visual' name must be a string when provided: {scene_file}")
            return False, None

    # All validation passed
    scene_name = scene_data.get("name", "Unnamed Scene")
    scene_desc = scene_data.get("description", "No description")
    sample_count = len(scene_data["samples"])

    return True, {
        "name": scene_name,
        "description": scene_desc,
        "sample_count": sample_count,
        "data": scene_data,
    }


def run_health_checks(device="/dev/video0", matrix_size=(95, 44), enable_debug=False):
    """Run basic system diagnostics and report camera capture status."""
    if enable_debug:
        set_debug_mode(True)
        debug_log(
            "HEALTH_START",
            "Running health diagnostics",
            {
                "device": device,
                "matrix_size": matrix_size,
            },
        )

    print("== SonicTerm Health Check ==")
    print(f"Camera device: {device}")

    try:
        from sonicterm.utils.webcam import camera_capture
    except Exception as exc:
        print(f"❌ Unable to load camera support: {exc}")
        return 1

    success = True

    print("[1/2] Raw frame capture test…", flush=True)
    image_path = camera_capture.capture_image(device)
    if image_path:
        print(f"   ✅ Captured frame: {image_path}")
    else:
        print(
            f"   ❌ Capture failed: {camera_capture.last_error or 'No additional details'}"
        )
        success = False

    print("[2/2] Processing pipeline test…", flush=True)
    color_matrix = camera_capture.capture_and_process(device, matrix_size)
    if color_matrix is not None:
        try:
            mean_value = float(color_matrix.mean())
            min_value = int(color_matrix.min())
            max_value = int(color_matrix.max())
        except Exception:
            mean_value = min_value = max_value = "n/a"
        print(
            f"   ✅ Matrix generated: shape={getattr(color_matrix, 'shape', 'unknown')}, "
            f"avg≈{mean_value}, min={min_value}, max={max_value}"
        )
    else:
        print(
            f"   ❌ Processing failed: {camera_capture.last_error or 'No additional details'}"
        )
        success = False

    if success:
        print("Health check status: OK")
        return 0

    print("Health check status: FAILED")
    return 1


def main():
    """Main entry point for SonicTerm."""
    parser = argparse.ArgumentParser(
        description="SonicTerm - Terminal-Based Generative Soundscape Player",
        epilog=f"SonicTerm v{VERSION}",
    )
    parser.add_argument(
        "scene",
        nargs="?",
        default="scenes/default.json",
        help="Path to scene JSON file (default: scenes/default.json)",
    )
    parser.add_argument(
        "--list-scenes", action="store_true", help="List available scene files"
    )
    parser.add_argument(
        "--no-tui",
        action="store_true",
        help="Disable Terminal User Interface (TUI is enabled by default)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode with detailed state information",
    )
    parser.add_argument(
        "--health",
        action="store_true",
        help="Run environment diagnostics (camera capture pipeline) and exit",
    )
    parser.add_argument(
        "--disable-logging",
        action="store_true",
        help="Disable per-session log file writes (logs directory)",
    )
    parser.add_argument(
        "--record-session",
        action="store_true",
        help="Capture system audio using ffmpeg into recordings/ directory",
    )
    parser.add_argument("--version", action="version", version=f"SonicTerm {VERSION}")

    args = parser.parse_args()

    if args.health:
        return run_health_checks(enable_debug=args.debug)

    if args.list_scenes:
        print("Available scenes:")
        scenes = list_available_scenes()
        if scenes:
            for scene in scenes:
                print(f"  {scene['file']}: {scene['name']} - {scene['description']}")
        else:
            print("  No scenes directory found")
        return

    # Validate scene file early, before initializing pygame and other components
    print(f"Validating scene file: {args.scene}")
    is_valid, scene_info = validate_scene_file(args.scene)

    if not is_valid:
        print("\nSonicTerm startup cancelled due to configuration errors.")
        return 1

    print(f"✅ Scene file validated: {scene_info['name']}")
    print(f"   Description: {scene_info['description']}")
    print(f"   Samples: {scene_info['sample_count']}")

    # Check for multiply parameter usage
    samples_with_multiply = [
        s for s in scene_info["data"]["samples"] if s.get("multiply", 1) > 1
    ]
    if samples_with_multiply:
        total_instances = sum(
            s.get("multiply", 1) for s in scene_info["data"]["samples"]
        )
        print(
            f"   Multiply feature detected: {len(samples_with_multiply)} samples will create {total_instances} total instances"
        )

    print()

    scene_path = Path(args.scene)
    timestamp = time.strftime("%Y%m%d-%H%M%S")

    log_file_path = None
    if not args.disable_logging:
        try:
            logs_dir = Path("logs")
            logs_dir.mkdir(parents=True, exist_ok=True)
            log_file_path = logs_dir / f"{scene_path.stem}-{timestamp}.log"
            tui_manager.start_file_logging(log_file_path)
            tui_manager.log(f"Session log: {log_file_path}")
        except Exception as exc:
            print(f"Warning: could not initialize file logging ({exc})")
    else:
        print("File logging disabled.")

    # All components already imported at the top

    soundscape = None
    original_terminal_settings = None
    cleanup_registered = False
    recording_process = None

    def cleanup_and_exit(signum=None, frame=None):
        """Clean up resources and exit gracefully."""
        nonlocal cleanup_registered, recording_process
        if cleanup_registered:
            return
        cleanup_registered = True

        try:
            if soundscape:
                soundscape.stop_soundscape()
        except:
            pass

        try:
            if recording_process and recording_process.poll() is None:
                recording_process.terminate()
                try:
                    recording_process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    recording_process.kill()
        except Exception:
            pass
        finally:
            recording_process = None
            tui_manager.set_recording(False)

        try:
            tui_manager.disable_tui()
        except:
            pass
        try:
            tui_manager.stop_file_logging()
        except:
            pass

        try:
            restore_terminal_settings(original_terminal_settings)
        except:
            pass

        try:
            subprocess.run(["reset"], timeout=2, capture_output=True)
        except:
            pass

        try:
            pygame.mixer.quit()
        except:
            pass

        if signum:
            print(f"\nReceived signal {signum}, exiting...")

    atexit.register(cleanup_and_exit)
    signal.signal(signal.SIGINT, cleanup_and_exit)
    signal.signal(signal.SIGTERM, cleanup_and_exit)

    recordings_path = None
    recording_process = None
    monitor_source = None

    def start_recording():
        """Start recording when playback begins"""
        nonlocal recording_process
        if recordings_path and not recording_process:
            try:
                recording_process = subprocess.Popen(
                    [
                        ffmpeg_path,
                        "-nostdin",
                        "-loglevel",
                        "warning",
                        "-f",
                        "pulse",
                        "-i",
                        monitor_source,
                        "-ac",
                        "2",
                        "-ar",
                        "44100",
                        str(recordings_path),
                    ],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                tui_manager.log(f"Session recording started: {recordings_path}")
                tui_manager.set_recording(True)
                return True
            except Exception as exc:
                print(f"Warning: failed to start session recording ({exc})")
                recording_process = None
                tui_manager.set_recording(False)
                return False
        return False

    if args.record_session:
        ffmpeg_path = shutil.which("ffmpeg")
        if not ffmpeg_path:
            print("Warning: ffmpeg not found. Session recording disabled.")
        else:
            try:
                monitor_source = get_pulse_monitor_source()
                recordings_dir = Path("recordings")
                recordings_dir.mkdir(parents=True, exist_ok=True)
                recordings_path = recordings_dir / f"{scene_path.stem}-{timestamp}.wav"
                print(f"Recording will start when playback begins: {recordings_path}")
                tui_manager.log(f"Session recording prepared: {recordings_path}")
                tui_manager.log(f"Recording source: {monitor_source}")
                # Don't start recording yet - wait for actual playback
            except Exception as exc:
                print(f"Warning: failed to prepare session recording ({exc})")
                recordings_path = None

    try:
        if args.debug:
            set_debug_mode(True)
            debug_log(
                "STARTUP",
                "SonicTerm starting with debug mode enabled",
                {
                    "scene_file": args.scene,
                    "tui_enabled": not args.no_tui,
                    "version": VERSION,
                },
            )

        enable_tui = not args.no_tui
        if enable_tui:
            # Check terminal compatibility first
            if not sys.stdin.isatty() or not sys.stdout.isatty():
                print("Non-interactive terminal detected; disabling TUI")
                enable_tui = False
            else:
                try:
                    if args.debug:
                        debug_log("TUI_START", "Starting TUI components")
                    tui_manager.enable_tui()
                    if args.debug:
                        debug_log("TUI_ENABLED", "TUI enabled, starting live display")

                    # Test TUI startup with a timeout mechanism
                    def timeout_handler(signum, frame):
                        raise TimeoutError("TUI startup timeout")

                    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
                    signal.alarm(3)  # 3 second timeout

                    try:
                        tui_manager.start_live_display()
                        signal.alarm(0)  # Cancel timeout
                        if args.debug:
                            debug_log(
                                "TUI_LIVE_STARTED", "Live display started successfully"
                            )
                    finally:
                        signal.signal(signal.SIGALRM, old_handler)

                except (Exception, TimeoutError) as e:
                    print(f"Error starting TUI: {e}")
                    if args.debug:
                        import traceback

                        traceback.print_exc()
                    print("Falling back to no-TUI mode")
                    enable_tui = False
                    try:
                        tui_manager.disable_tui()
                    except:
                        pass

        pygame.mixer.pre_init(frequency=44100, size=-16, channels=2, buffer=1024)
        pygame.mixer.init()

        # Set number of mixing channels to support more simultaneous sounds
        pygame.mixer.set_num_channels(32)  # Support up to 32 simultaneous sounds

        if args.debug:
            debug_log(
                "PYGAME_INIT",
                "Pygame mixer initialized",
                {
                    "frequency": 44100,
                    "size": -16,
                    "channels": 2,
                    "buffer": 1024,
                    "num_channels": pygame.mixer.get_num_channels(),
                },
            )

        soundscape = SceneBasedSoundscape(args.scene)

        from sonicterm.core.scene import scene_manager
        import sonicterm.core.scene as scene_module

        scene_module.scene_manager = soundscape

        # Scene already validated, so load_scene should succeed
        # But check anyway in case of unexpected errors
        try:
            if not soundscape.load_scene():
                print("Error: Scene loading failed after validation passed")
                return 1
        except Exception as e:
            print(f"Error during scene loading: {e}")
            if args.debug:
                import traceback

                traceback.print_exc()
            return 1

        quit_requested = threading.Event()
        input_thread = None

        def stop_recording_action():
            nonlocal recording_process, recordings_path
            if recording_process and recording_process.poll() is None:
                tui_manager.log("Stopping session recording (spacebar)")
                try:
                    recording_process.terminate()
                    try:
                        recording_process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        recording_process.kill()
                except Exception as exc:
                    tui_manager.log(f"Recording termination error: {exc}")
                finally:
                    recording_process = None
                    tui_manager.set_stopped(True)
                    try:
                        tui_manager.update_display(force=True)
                    except Exception:
                        pass
                    if recordings_path:
                        tui_manager.log(f"Recording saved to {recordings_path}")
                quit_requested.set()
                return True
            return False

        try:
            if enable_tui:
                callback = stop_recording_action if args.record_session else None
                original_terminal_settings, input_thread = setup_input_handler(
                    quit_requested, callback
                )

            # Set up recording callback for TUI manager
            if args.record_session:
                tui_manager.set_recording_callback(start_recording)

            soundscape.start_soundscape(
                enable_tui=enable_tui,
                start_recording_callback=start_recording
                if args.record_session
                else None,
            )

            while not quit_requested.is_set():
                time.sleep(0.05)
                # Ensure display updates continue even when paused
                if enable_tui:
                    tui_manager.update_display()

                # Check global controls (playback, etc.)
                soundscape.check_global_controls()

        except KeyboardInterrupt:
            tui_manager.set_quitting(True)
            tui_manager.log("Interrupted by user (Ctrl+C)")
            quit_requested.set()

        soundscape.stop_soundscape()

        if args.debug:
            debug_log("SHUTDOWN", "SonicTerm shutting down")
            if debug_manager:
                debug_manager.print_state_summary()

    except Exception as e:
        tui_manager.log(f"Error: {e}")
        return 1
    finally:
        restore_terminal_settings(original_terminal_settings)

        pygame.mixer.quit()
        tui_manager.log("Soundscape stopped.")
        tui_manager.stop_file_logging()

        if recording_process and recording_process.poll() is None:
            try:
                recording_process.terminate()
                recording_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                recording_process.kill()
            except Exception:
                pass
            finally:
                recording_process = None
                tui_manager.set_recording(False)

    return 0


if __name__ == "__main__":
    sys.exit(main())

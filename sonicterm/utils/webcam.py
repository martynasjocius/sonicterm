#!/usr/bin/env python3
#
# SonicTerm
# Copyright (c) 2025 Martynas Jocius
#
"""Camera capture and processing helpers for SonicTerm."""

import os
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from PIL import Image

from ..ui.tui import tui_manager
from .colors import get_warm_camera_palette


@dataclass
class CameraStream:
    process: subprocess.Popen
    width: int
    height: int
    frame_size: int
    fps: float


def _select_resample_filter(image_module):
    """Select a resampling filter compatible with the available Pillow API."""
    resampling_attr = getattr(image_module, "Resampling", None)
    preferred_order = ("LANCZOS", "ANTIALIAS", "BICUBIC", "BILINEAR", "NEAREST")

    for name in preferred_order:
        if resampling_attr is not None:
            candidate = getattr(resampling_attr, name, None)
            if candidate is not None:
                return candidate

        candidate = getattr(image_module, name, None)
        if candidate is not None:
            return candidate

    return 0


class CameraCapture:
    """Handles camera capture and image processing for visual mapping."""
    
    def __init__(self):
        self.temp_dir = Path(tempfile.gettempdir()) / "sonicterm_camera"
        self.temp_dir.mkdir(exist_ok=True)
        self.last_error = None
        self.active_stream: Optional[CameraStream] = None
        self._stream_thread: Optional[threading.Thread] = None
        self._stream_stop = threading.Event()
        self._stream_lock = threading.Lock()
        self._latest_frame: Optional[np.ndarray] = None
        self._latest_timestamp: float = 0.0
        
    def capture_image(self, device: str = "/dev/video0", width: int = 1280, height: int = 960) -> Optional[Path]:
        """Capture a single frame from camera using ffmpeg."""
        try:
            output_path = self.temp_dir / f"camera_capture_{os.getpid()}.jpg"
            
            # Use ffmpeg to capture a single frame
            cmd = [
                "ffmpeg", "-y",  # -y to overwrite existing file
                "-f", "v4l2",    # Video4Linux2 input format
                "-i", device,    # Input device
                "-vframes", "1", # Capture only 1 frame
                "-s", f"{width}x{height}",  # Resolution
                "-q:v", "2",     # High quality
                "-loglevel", "error",  # Reduce ffmpeg output
                str(output_path)
            ]
            
            if getattr(tui_manager, 'debug_mode', False):
                tui_manager.log(f"Capturing camera image from {device}...")
            
            # Run ffmpeg command with shorter timeout
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                timeout=5  # Reduced timeout
            )
            
            if result.returncode != 0:
                error_msg = result.stderr.strip() or "ffmpeg returned non-zero exit code"
                self.last_error = f"ffmpeg error: {error_msg}"
                tui_manager.log(self.last_error)
                return None
                
            if not output_path.exists():
                self.last_error = "Captured image file not found"
                tui_manager.log(self.last_error)
                return None
                
            if getattr(tui_manager, 'debug_mode', False):
                tui_manager.log(f"Camera image captured: {output_path}")
            self.last_error = None
            return output_path
            
        except subprocess.TimeoutExpired:
            self.last_error = "Camera capture timeout (5s)"
            tui_manager.log(self.last_error)
            return None
        except FileNotFoundError:
            self.last_error = "ffmpeg not found. Please install ffmpeg for camera capture."
            tui_manager.log(self.last_error)
            return None
        except Exception as e:
            self.last_error = f"Camera capture error: {e}"
            tui_manager.log(self.last_error)
            return None
    
    def _apply_gain_array(
        self, frame: np.ndarray, brightness: float, contrast: float
    ) -> np.ndarray:
        frame_f = frame.astype(np.float32)
        frame_f = (frame_f - 128.0) * contrast + 128.0
        frame_f *= brightness
        np.clip(frame_f, 0, 255, out=frame_f)
        return frame_f.astype(np.uint8)
    
    def convert_to_color_matrix(
        self, image_path: Path, target_size: Tuple[int, int] = (60, 45)
    ) -> Optional[np.ndarray]:
        """Convert an image on disk to a color matrix suitable for the TUI."""

        try:
            with Image.open(image_path) as img:
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
                matrix = self._image_to_color_matrix(img, target_size)
                if matrix is not None:
                    self.last_error = None
                return matrix

        except Exception as e:
            self.last_error = f"Color matrix conversion error: {e}"
            tui_manager.log(self.last_error)
            return None

    def convert_frame_to_color_matrix(
        self,
        frame: np.ndarray,
        target_size: Tuple[int, int],
        brightness: float = 1.0,
        contrast: float = 1.0,
    ) -> Optional[np.ndarray]:
        """Convert an in-memory RGB frame to a color matrix."""

        try:
            if frame.ndim != 3 or frame.shape[2] != 3:
                raise ValueError("Frame must be an RGB array of shape (H, W, 3)")

            adjusted = self._apply_gain_array(frame, brightness, contrast)
            img = Image.fromarray(adjusted, mode='RGB').transpose(Image.FLIP_LEFT_RIGHT)
            matrix = self._image_to_color_matrix(img, target_size)
            if matrix is not None:
                self.last_error = None
            return matrix

        except Exception as e:
            self.last_error = f"Frame conversion error: {e}"
            tui_manager.log(self.last_error)
            return None

    def _image_to_color_matrix(
        self, img: Image.Image, target_size: Tuple[int, int]
    ) -> Optional[np.ndarray]:
        try:
            orig_width, orig_height = img.size
            orig_aspect_ratio = orig_width / orig_height if orig_height else 1.0

            target_width, target_height = target_size
            panel_aspect_ratio = (target_width * 0.5) / target_height

            if orig_aspect_ratio > panel_aspect_ratio:
                new_width = int(orig_height * panel_aspect_ratio)
                new_height = orig_height
                crop_x = (orig_width - new_width) // 2
                crop_y = 0
            else:
                new_width = orig_width
                new_height = int(orig_width / panel_aspect_ratio)
                crop_x = 0
                crop_y = (orig_height - new_height) // 2

            img = img.crop((crop_x, crop_y, crop_x + new_width, crop_y + new_height))

            resample_filter = _select_resample_filter(Image)
            img = img.resize(target_size, resample=resample_filter)

            img_array = np.array(img)
            gray = np.dot(img_array[...,:3], [0.2989, 0.5870, 0.1140])

            color_palette = get_warm_camera_palette()
            color_matrix = np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)

            for i in range(target_size[1]):
                for j in range(target_size[0]):
                    gray_value = gray[i, j]
                    palette_idx = min(
                        int(gray_value / 255 * (len(color_palette) - 1)),
                        len(color_palette) - 1,
                    )
                    color_matrix[i, j] = color_palette[palette_idx]

            if getattr(tui_manager, 'debug_mode', False):
                tui_manager.log(
                    f"Cropped camera to aspect {panel_aspect_ratio:.2f} and converted to {target_size[0]}x{target_size[1]} matrix"
                )
            return color_matrix

        except Exception as e:
            self.last_error = f"Image processing error: {e}"
            tui_manager.log(self.last_error)
            return None
    
    def cleanup_temp_files(self):
        """Clean up temporary camera files."""
        try:
            for file_path in self.temp_dir.glob("camera_capture_*.jpg"):
                file_path.unlink()
            for file_path in self.temp_dir.glob("*.enhanced.jpg"):
                file_path.unlink()
            if getattr(tui_manager, 'debug_mode', False):
                tui_manager.log("Cleaned up camera temporary files")
        except Exception as e:
            tui_manager.log(f"Temporary file cleanup error: {e}")
    
    def capture_and_process(
        self,
        device: str = "/dev/video0",
        matrix_size: Tuple[int, int] = (60, 45),
        capture_resolution: Optional[Tuple[int, int]] = None,
        brightness: float = 1.0,
        contrast: float = 1.0,
    ) -> Optional[np.ndarray]:
        """Complete camera capture and processing pipeline with configurable resolution."""

        if capture_resolution is None:
            capture_width = max(640, matrix_size[0] * 8)
            capture_height = max(480, matrix_size[1] * 8)
        else:
            capture_width, capture_height = capture_resolution

        image_path = self.capture_image(
            device, width=capture_width, height=capture_height
        )
        if not image_path:
            return None
        
        try:
            with Image.open(image_path) as img:
                frame = np.array(img.convert('RGB'))
        except Exception as exc:
            self.last_error = f"Image load error: {exc}"
            tui_manager.log(self.last_error)
            return None

        adjusted = self._apply_gain_array(frame, brightness, contrast)
        img = Image.fromarray(adjusted, mode='RGB').transpose(Image.FLIP_LEFT_RIGHT)

        color_matrix = self._image_to_color_matrix(img, matrix_size)

        # Cleanup temp files
        self.cleanup_temp_files()
        
        if color_matrix is None and self.last_error is None:
            self.last_error = "Color matrix conversion returned None"

        return color_matrix

    # ------------------------------------------------------------------
    # Streaming helpers
    # ------------------------------------------------------------------
    def start_stream(
        self,
        device: str = "/dev/video0",
        width: int = 640,
        height: int = 480,
        fps: float = 24.0,
    ) -> CameraStream:
        """Start a continuous camera stream using ffmpeg rawvideo output."""

        self.stop_stream()

        cmd = [
            "ffmpeg",
            "-loglevel",
            "error",
            "-fflags",
            "nobuffer",
            "-flags",
            "low_delay",
            "-probesize",
            "32",
            "-analyzeduration",
            "0",
            "-f",
            "v4l2",
            "-framerate",
            str(int(fps)),
            "-video_size",
            f"{width}x{height}",
            "-i",
            device,
            "-vf",
            f"scale={width}:{height}",
            "-pix_fmt",
            "rgb24",
            "-f",
            "rawvideo",
            "pipe:1",
        ]

        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                bufsize=0,
            )
        except FileNotFoundError as exc:
            raise RuntimeError("ffmpeg not found for camera streaming") from exc

        if not process.stdout:
            process.kill()
            raise RuntimeError("Failed to start camera stream")

        stream = CameraStream(
            process=process,
            width=width,
            height=height,
            frame_size=width * height * 3,
            fps=fps,
        )

        self.active_stream = stream
        self._stream_stop.clear()
        self._latest_frame = None
        self._latest_timestamp = 0.0

        self._stream_thread = threading.Thread(
            target=self._stream_reader,
            args=(stream,),
            daemon=True,
        )
        self._stream_thread.start()
        return stream

    def read_stream_frame(self) -> Optional[np.ndarray]:
        """Return the latest frame produced by the stream reader, dropping older ones."""

        if not self.active_stream:
            return None

        with self._stream_lock:
            frame = self._latest_frame
            timestamp = self._latest_timestamp
            self._latest_frame = None

        if frame is None:
            return None

        if timestamp and (time.time() - timestamp) > 1.0:
            return None

        return frame

    def stop_stream(self) -> None:
        """Stop and clean up the active camera stream, if any."""

        stream = self.active_stream
        if not stream:
            return

        self._stream_stop.set()
        if self._stream_thread and self._stream_thread.is_alive():
            self._stream_thread.join(timeout=1.0)
        self._stream_thread = None

        process = stream.process
        if process.poll() is None:
            try:
                process.terminate()
                process.wait(timeout=1.0)
            except Exception:
                process.kill()

        self.active_stream = None
        self._latest_frame = None
        self._latest_timestamp = 0.0

    def _stream_reader(self, stream: CameraStream) -> None:
        process = stream.process
        stdout = process.stdout
        if stdout is None:
            return

        try:
            while not self._stream_stop.is_set() and process.poll() is None:
                frame_data = stdout.read(stream.frame_size)
                if not frame_data or len(frame_data) < stream.frame_size:
                    break

                while not self._stream_stop.is_set():
                    try:
                        peek = stdout.peek(stream.frame_size)
                    except (AttributeError, OSError):
                        break
                    if not peek or len(peek) < stream.frame_size:
                        break
                    frame_data = stdout.read(stream.frame_size)
                    if not frame_data or len(frame_data) < stream.frame_size:
                        break

                frame = np.frombuffer(frame_data, dtype=np.uint8)
                frame = frame.reshape((stream.height, stream.width, 3))
                with self._stream_lock:
                    self._latest_frame = frame
                    self._latest_timestamp = time.time()

        finally:
            if process.poll() is None and not self._stream_stop.is_set():
                try:
                    process.terminate()
                except Exception:
                    pass



# Global camera capture instance
camera_capture = CameraCapture()

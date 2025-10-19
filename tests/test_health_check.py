#
# SonicTerm
# Copyright (c) 2025 Martynas Jocius
#
import io
import unittest
from contextlib import redirect_stdout
from unittest import mock

import numpy as np

from sonicterm.cli import run_health_checks


class HealthCheckTests(unittest.TestCase):
    def test_health_success(self):
        fake_capture = mock.Mock()
        fake_capture.capture_image.return_value = "/tmp/test.jpg"
        fake_capture.capture_and_process.return_value = np.zeros((44, 95, 3), dtype=np.uint8)
        fake_capture.last_error = None

        with mock.patch("sonicterm.utils.webcam.camera_capture", fake_capture):
            buffer = io.StringIO()
            with redirect_stdout(buffer):
                exit_code = run_health_checks(device="/dev/video-test", matrix_size=(95, 44), enable_debug=False)

        self.assertEqual(exit_code, 0)
        output = buffer.getvalue()
        self.assertIn("Health check status: OK", output)
        fake_capture.capture_image.assert_called_once()
        fake_capture.capture_and_process.assert_called_once()

    def test_health_failure_reports_errors(self):
        fake_capture = mock.Mock()

        def fail_capture(device):
            fake_capture.last_error = "raw capture failed"
            return None

        def fail_process(device, size):
            fake_capture.last_error = "processing failed"
            return None

        fake_capture.capture_image.side_effect = fail_capture
        fake_capture.capture_and_process.side_effect = fail_process
        fake_capture.last_error = "raw capture failed"

        with mock.patch("sonicterm.utils.webcam.camera_capture", fake_capture):
            buffer = io.StringIO()
            with redirect_stdout(buffer):
                exit_code = run_health_checks(device="/dev/video-test", matrix_size=(95, 44), enable_debug=False)

        self.assertEqual(exit_code, 1)
        output = buffer.getvalue()
        self.assertIn("Health check status: FAILED", output)
        self.assertIn("raw capture failed", output)
        self.assertIn("processing failed", output)
        fake_capture.capture_image.assert_called_once()
        fake_capture.capture_and_process.assert_called_once()


if __name__ == "__main__":
    unittest.main()

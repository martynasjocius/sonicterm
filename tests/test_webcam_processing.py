#
# SonicTerm
# Copyright (c) 2025 Martynas Jocius
#
import os
import unittest
from unittest import mock

try:
    from PIL import Image
except ImportError:  # pragma: no cover - test skipped when Pillow missing
    Image = None

if Image is not None:
    from sonicterm.utils import webcam
else:  # pragma: no cover - module requires Pillow
    webcam = None


@unittest.skipUnless(webcam, "Pillow not available")
class CameraCaptureConversionTests(unittest.TestCase):
    def setUp(self):
        self.capture = webcam.CameraCapture()
        self.test_path = self.capture.temp_dir / "test_camera_image.jpg"
        img = Image.new("RGB", (10, 10), color=(128, 64, 32))
        img.save(self.test_path, quality=95)

    def tearDown(self):
        if self.test_path.exists():
            try:
                os.remove(self.test_path)
            except OSError:
                pass
        enhanced = self.test_path.with_suffix('.enhanced.jpg')
        if enhanced.exists():
            try:
                os.remove(enhanced)
            except OSError:
                pass

    def test_convert_to_color_matrix_returns_expected_shape(self):
        matrix = self.capture.convert_to_color_matrix(self.test_path, (8, 6))
        self.assertIsNotNone(matrix)
        self.assertEqual(matrix.shape, (6, 8, 3))

    def test_select_resample_filter_handles_missing_attributes(self):
        fake_image_module = mock.Mock()
        fake_image_module.Resampling = mock.Mock()
        fake_image_module.Resampling.LANCZOS = None
        fake_image_module.Resampling.ANTIALIAS = None
        fake_image_module.Resampling.BICUBIC = None
        fake_image_module.Resampling.BILINEAR = None
        fake_image_module.Resampling.NEAREST = 1
        fake_image_module.LANCZOS = None
        fake_image_module.ANTIALIAS = None
        fake_image_module.BICUBIC = 3
        result = webcam._select_resample_filter(fake_image_module)
        self.assertEqual(result, 3)


if __name__ == "__main__":
    unittest.main()

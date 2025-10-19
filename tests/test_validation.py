#
# SonicTerm
# Copyright (c) 2025 Martynas Jocius
#
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from sonicterm.cli import validate_scene_file


class ValidateSceneFileTests(unittest.TestCase):
    def test_valid_scene_returns_metadata(self):
        data = {
            "name": "Test Scene",
            "description": "Unit test scene",
            "samples": [
                {"path": "samples/example.wav"},
                {"path": "samples/another.wav", "volume": {"min": 0.2, "max": 0.6}},
            ],
        }

        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tmp:
            json.dump(data, tmp)
            tmp.flush()
            scene_path = tmp.name

        try:
            ok, info = validate_scene_file(scene_path)
        finally:
            Path(scene_path).unlink(missing_ok=True)

        self.assertTrue(ok)
        self.assertIsNotNone(info)
        self.assertEqual(info["name"], "Test Scene")
        self.assertEqual(info["sample_count"], 2)
        self.assertEqual(info["data"], data)

    def test_missing_sample_path_fails_validation(self):
        data = {"samples": [{"volume": {"min": 0.1, "max": 0.4}}]}

        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tmp:
            json.dump(data, tmp)
            tmp.flush()
            scene_path = tmp.name

        try:
            with redirect_stdout(io.StringIO()):
                ok, info = validate_scene_file(scene_path)
        finally:
            Path(scene_path).unlink(missing_ok=True)

        self.assertFalse(ok)
        self.assertIsNone(info)


if __name__ == "__main__":
    unittest.main()

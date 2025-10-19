#
# SonicTerm
# Copyright (c) 2025 Martynas Jocius
#
import unittest
from unittest import mock

from sonicterm.core.scene import SceneBasedSoundscape


class ScenePlayerCreationTests(unittest.TestCase):
    def test_multiply_creates_unique_players_with_adjusted_wait(self):
        scene = SceneBasedSoundscape("scenes/default.json")

        sample_config = {
            "path": "samples/example.wav",
            "name": "Ocean",
            "multiply": 3,
            "wait": 2,
            "timings": [1, 2, 3],
        }
        global_config = {"effects": {}}

        with mock.patch("sonicterm.core.scene.random.choice", return_value=1.5):
            players = scene.player_manager.create_players(sample_config, global_config, 0)

        self.assertEqual(len(players), 3)
        names = [p.config["name"] for p in players]
        self.assertEqual(names, ["Ocean #1", "Ocean #2", "Ocean #3"])

        waits = [p.config["wait"] for p in players]
        self.assertEqual(waits, [3.5, 3.5, 3.5])
        # Ensure original config not mutated
        self.assertEqual(sample_config.get("name"), "Ocean")
        self.assertEqual(sample_config.get("wait"), 2)

    def test_invalid_multiply_defaults_to_one(self):
        scene = SceneBasedSoundscape("scenes/default.json")
        sample_config = {"path": "samples/example.wav", "multiply": 99}
        global_config = {}

        with mock.patch("sonicterm.core.scene.tui_manager.log") as mock_log:
            players = scene.player_manager.create_players(sample_config, global_config, 0)

        self.assertEqual(len(players), 1)
        mock_log.assert_called()


if __name__ == "__main__":
    unittest.main()

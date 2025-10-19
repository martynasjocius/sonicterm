#
# SonicTerm
# Copyright (c) 2025 Martynas Jocius
#
import unittest
from unittest import mock

from sonicterm.audio.player import SamplePlayer


class SamplePlayerHashTests(unittest.TestCase):
    def setUp(self):
        self.sample_config = {"path": "samples/demo.wav", "volume": {"min": 0.3, "max": 0.7}}
        self.global_config = {"effects": {}}

    def test_config_hash_changes_when_configuration_changes(self):
        player = SamplePlayer(self.sample_config.copy(), self.global_config.copy())
        original_hash = player.get_config_hash()

        with mock.patch("sonicterm.audio.player.tui_manager.log"):
            player.update_config(self.sample_config.copy(), self.global_config.copy())

        self.assertEqual(player.get_config_hash(), original_hash)

        new_config = self.sample_config.copy()
        new_config["volume"] = {"min": 0.1, "max": 0.2}

        with mock.patch("sonicterm.audio.player.tui_manager.log"):
            player.update_config(new_config, self.global_config.copy())

        self.assertNotEqual(player.get_config_hash(), original_hash)


if __name__ == "__main__":
    unittest.main()

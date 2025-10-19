#
# SonicTerm
# Copyright (c) 2025 Martynas Jocius
#
import unittest
from unittest import mock

from sonicterm.audio.player import SamplePlayer


class LoggingTests(unittest.TestCase):
    def setUp(self):
        config = {
            'path': 'samples/test.wav',
            'name': 'TestSample',
            'volume': {'min': 0.0, 'max': 1.0}
        }
        self.player = SamplePlayer(config, {})

    @mock.patch('sonicterm.audio.player.tui_manager.log')
    def test_log_playback_start(self, log_mock):
        params = {'volume': 0.6, 'gain': 0.5, 'pan': -0.2, 'volume_control_source': 'camera'}
        self.player._log_playback_start(params)
        message = log_mock.call_args[0][0]
        self.assertIn('[TestSample]', message)
        self.assertIn('Playing', message)
        self.assertNotIn('▶', message)

    @mock.patch('sonicterm.audio.player.tui_manager.log')
    def test_log_playback_finish(self, log_mock):
        params = {'volume_control_source': 'shell'}
        self.player._log_playback_finish(3.5, 2, params)
        message = log_mock.call_args[0][0]
        self.assertIn('[TestSample]', message)
        self.assertIn('Finished', message)
        self.assertNotIn('⏹', message)
        self.assertIn('2s', message)
        self.assertIn('control', message)


if __name__ == '__main__':
    unittest.main()

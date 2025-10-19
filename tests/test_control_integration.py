#
# SonicTerm
# Copyright (c) 2025 Martynas Jocius
#
import unittest
from unittest import mock

from sonicterm.audio.player import SamplePlayer, CONTROL_POLL_INTERVAL


class ControlIntegrationPlaybackTests(unittest.TestCase):
    def setUp(self):
        self.sample_config = {
            'path': 'samples/test.wav',
            'name': 'TestSample',
            'volume': {'min': 0.0, 'max': 1.0, 'control': {'shell': 'echo 0.2'}},
        }
        self.player = SamplePlayer(self.sample_config, {})

    @mock.patch('sonicterm.audio.player.subprocess.run')
    def test_control_repolled_with_debug(self, run_mock):
        run_mock.return_value = mock.Mock(returncode=0, stdout='0.2', stderr='')

        log_entries = []

        def debug_side_effect(event, message, data=None):
            if event == 'CONTROL_COMMAND':
                log_entries.append((message, data))

        with mock.patch('sonicterm.audio.player.is_debug_enabled', return_value=True), \
             mock.patch('sonicterm.audio.player.debug_log', side_effect=debug_side_effect), \
             mock.patch('sonicterm.audio.player.debug_manager') as debug_manager_mock, \
             mock.patch('sonicterm.audio.player.tui_manager.log') as tui_log_mock, \
             mock.patch('time.time', side_effect=[0.0, CONTROL_POLL_INTERVAL + 0.1]):
            debug_manager_mock.enabled = True
            self.player._resolve_controlled_value(self.sample_config['volume'], 'volume', 0.0, 1.0)
            self.player._resolve_controlled_value(self.sample_config['volume'], 'volume', 0.0, 1.0)

        self.assertEqual(run_mock.call_count, 2)
        self.assertEqual(len(log_entries), 2)
        self.assertGreaterEqual(tui_log_mock.call_count, 2)


if __name__ == '__main__':
    unittest.main()

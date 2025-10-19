#
# SonicTerm
# Copyright (c) 2025 Martynas Jocius
#
import sys
import types
import unittest
from unittest import mock

from sonicterm.audio.player import SamplePlayer, CONTROL_POLL_INTERVAL


class ControlParameterTests(unittest.TestCase):
    def setUp(self):
        self.sample_config = {
            'path': 'samples/test.wav',
            'name': 'TestSample',
            'volume': {'min': 0.0, 'max': 1.0},
        }
        self.player = SamplePlayer(self.sample_config, {})

    @mock.patch('sonicterm.audio.player.subprocess.run')
    def test_control_value_uses_command_result(self, run_mock):
        self.sample_config['volume']['control'] = {"shell": "echo 0.42"}
        self.player.control_cache.clear()
        run_mock.return_value = mock.Mock(returncode=0, stdout='0.42\n', stderr='')

        with mock.patch('time.time', side_effect=[0.0, 0.0]):
            value = self.player._resolve_controlled_value(self.sample_config['volume'], 'volume', 0.0, 1.0)

        self.assertAlmostEqual(value, 0.42)
        run_mock.assert_called_once()

    @mock.patch('sonicterm.audio.player.subprocess.run')
    def test_control_value_cached_within_interval(self, run_mock):
        self.sample_config['volume']['control'] = {"shell": "echo 0.3"}
        self.player.control_cache.clear()
        run_mock.return_value = mock.Mock(returncode=0, stdout='0.3', stderr='')

        with mock.patch('time.time', side_effect=[0.0, 0.0, 0.0]):
            first = self.player._resolve_controlled_value(self.sample_config['volume'], 'volume', 0.0, 1.0)
            second = self.player._resolve_controlled_value(self.sample_config['volume'], 'volume', 0.0, 1.0)

        self.assertEqual(run_mock.call_count, 1)
        self.assertAlmostEqual(first, second)

    @mock.patch('sonicterm.audio.player.subprocess.run')
    def test_control_value_not_clamped_to_config_range(self, run_mock):
        self.sample_config['volume']['min'] = 0.0
        self.sample_config['volume']['max'] = 0.5
        self.sample_config['volume']['control'] = {"shell": "echo 0.61"}
        self.player.control_cache.clear()
        run_mock.return_value = mock.Mock(returncode=0, stdout='0.61', stderr='')

        with mock.patch('time.time', return_value=0.0):
            value = self.player._resolve_controlled_value(self.sample_config['volume'], 'volume', 0.0, 1.0)

        self.assertAlmostEqual(value, 0.61)

    @mock.patch('sonicterm.audio.player.random.uniform', return_value=0.22)
    @mock.patch('sonicterm.audio.player.SamplePlayer._get_camera_control_value', return_value=None)
    def test_camera_control_falls_back_when_no_manager(self, camera_mock, uniform_mock):
        self.sample_config['volume']['control'] = {'camera': 'motion'}
        self.player.control_cache.clear()

        value = self.player._resolve_controlled_value(self.sample_config['volume'], 'volume', 0.0, 1.0)

        self.assertEqual(value, 0.22)
        camera_mock.assert_called_once_with('motion')

    @mock.patch('sonicterm.audio.player.SamplePlayer._get_camera_control_value', return_value=0.73)
    def test_camera_control_reads_motion(self, camera_mock):
        self.sample_config['volume']['control'] = {'camera': 'motion'}
        self.player.control_cache.clear()

        value = self.player._resolve_controlled_value(self.sample_config['volume'], 'volume', 0.0, 1.0)

        self.assertAlmostEqual(value, 0.73)
        camera_mock.assert_called_once_with('motion')

    def test_get_camera_control_value_reads_scene_manager(self):
        module = types.ModuleType('sonicterm.core.scene')
        module.scene_manager = types.SimpleNamespace(camera_motion=0.42)
        with mock.patch.dict(sys.modules, {'sonicterm.core.scene': module}):
            value = self.player._get_camera_control_value('motion')
        self.assertAlmostEqual(value, 0.42)

    @mock.patch('sonicterm.audio.player.random.uniform', return_value=0.55)
    @mock.patch('sonicterm.audio.player.tui_manager.log')
    @mock.patch('sonicterm.audio.player.subprocess.run')
    def test_control_failure_falls_back(self, run_mock, log_mock, uniform_mock):
        self.sample_config['volume']['control'] = {"shell": "echo invalid"}
        self.player.control_cache.clear()
        run_mock.side_effect = RuntimeError('failure')

        with mock.patch('time.time', return_value=0.0):
            value = self.player._resolve_controlled_value(self.sample_config['volume'], 'volume', 0.0, 1.0)

        self.assertEqual(value, 0.55)
        log_mock.assert_called()

    @mock.patch('sonicterm.audio.player.subprocess.run')
    def test_control_repolled_after_interval(self, run_mock):
        self.sample_config['volume']['control'] = {"shell": "echo 0.2"}
        self.player.control_cache.clear()
        run_mock.return_value = mock.Mock(returncode=0, stdout='0.2', stderr='')

        with mock.patch('time.time', side_effect=[0.0, CONTROL_POLL_INTERVAL + 0.1]):
            self.player._resolve_controlled_value(self.sample_config['volume'], 'volume', 0.0, 1.0)
            self.player._resolve_controlled_value(self.sample_config['volume'], 'volume', 0.0, 1.0)

        self.assertEqual(run_mock.call_count, 2)

    @mock.patch('sonicterm.audio.player.SamplePlayer._resolve_controlled_value')
    def test_refresh_controlled_parameters_updates_channel(self, resolve_mock):
        self.sample_config['volume']['control'] = {"shell": "echo 0.1"}
        self.player.control_cache.clear()

        resolve_mock.side_effect = [0.3, 0.4]

        parameters = {'volume': 0.2, 'gain': 0.5, 'pan': 0.1}
        channel = mock.Mock()

        self.player._refresh_controlled_parameters(channel, parameters)

        channel.set_volume.assert_called_once()
        left, right = channel.set_volume.call_args[0]
        expected_final = 0.3 * 0.5
        expected_left = expected_final * (1 - 0.1) / 2.0
        expected_right = expected_final * (1 + 0.1) / 2.0
        self.assertAlmostEqual(left, expected_left)
        self.assertAlmostEqual(right, expected_right)
        self.assertEqual(parameters['volume_control_source'], 'shell')

    @mock.patch('sonicterm.audio.player.SamplePlayer._resolve_controlled_value')
    def test_refresh_controlled_parameters_camera_source(self, resolve_mock):
        self.sample_config['volume']['control'] = {'camera': 'motion'}
        self.player.control_cache.clear()

        resolve_mock.side_effect = [0.6]

        parameters = {'volume': 0.2, 'gain': 0.5, 'pan': 0.0}
        channel = mock.Mock()

        self.player._refresh_controlled_parameters(channel, parameters)

        self.assertEqual(parameters['volume_control_source'], 'camera')
        channel.set_volume.assert_called_once()


if __name__ == '__main__':
    unittest.main()

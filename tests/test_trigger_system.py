#!/usr/bin/env python3
#
# SonicTerm
# Copyright (c) 2025 Martynas Jocius
#
"""
Test script for trigger system functionality
"""

import unittest
import unittest.mock as mock
import subprocess
import time
import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from sonicterm.core.scene import SceneBasedSoundscape


class TriggerSystemTests(unittest.TestCase):
    """Test trigger system functionality"""
    
    def setUp(self):
        """Set up test environment"""
        self.scene_manager = SceneBasedSoundscape("scenes/test.json")
        self.scene_manager.global_control_config = {
            "trigger": {
                "start": {
                    "shell": "echo 1"
                }
            }
        }
        self.scene_manager.global_control_initialized = True
        self.scene_manager.global_control_start_time = time.time()
        self.scene_manager.last_global_control_check = 0.0
        
        # Reset trigger state
        self.scene_manager.trigger_start_executed = False
        self.scene_manager.trigger_pause_executed = False
    
    def test_start_trigger_executes_once(self):
        """Test that start trigger executes only once"""
        with mock.patch('sonicterm.core.scene.tui_manager') as mock_tui:
            mock_tui.is_paused = True
            mock_tui.toggle_pause = mock.Mock()
            mock_tui.set_loading_state = mock.Mock()
            mock_tui.log = mock.Mock()
            mock_tui.update_player_status = mock.Mock()
            
            # First call should execute
            self.scene_manager._check_global_playback_control()
            
            # Verify trigger was executed
            self.assertTrue(self.scene_manager.trigger_start_executed)
            mock_tui.toggle_pause.assert_called_once()
            mock_tui.set_loading_state.assert_called_once_with(False)
            mock_tui.log.assert_called_with("Trigger: Playback started")
    
    def test_start_trigger_ignores_when_already_playing(self):
        """Test that start trigger is ignored when already playing"""
        with mock.patch('sonicterm.core.scene.tui_manager') as mock_tui:
            mock_tui.is_paused = False  # Already playing
            mock_tui.toggle_pause = mock.Mock()
            mock_tui.set_loading_state = mock.Mock()
            mock_tui.log = mock.Mock()
            mock_tui.update_player_status = mock.Mock()
            
            self.scene_manager._check_global_playback_control()
            
            # Verify trigger was marked as executed but no action taken
            self.assertTrue(self.scene_manager.trigger_start_executed)
            mock_tui.toggle_pause.assert_not_called()
            mock_tui.set_loading_state.assert_not_called()
            mock_tui.log.assert_not_called()
    
    def test_trigger_command_failure_does_not_execute(self):
        """Test that trigger does not execute when command fails"""
        self.scene_manager.global_control_config = {
            "trigger": {
                "start": {
                    "shell": "exit 1"  # Command that fails
                }
            }
        }
        
        with mock.patch('sonicterm.core.scene.tui_manager') as mock_tui:
            mock_tui.is_paused = True
            mock_tui.toggle_pause = mock.Mock()
            mock_tui.set_loading_state = mock.Mock()
            mock_tui.log = mock.Mock()
            mock_tui.update_player_status = mock.Mock()
            
            self.scene_manager._check_global_playback_control()
            
            # Verify trigger was not executed
            self.assertFalse(self.scene_manager.trigger_start_executed)
            mock_tui.toggle_pause.assert_not_called()
            mock_tui.set_loading_state.assert_not_called()
            mock_tui.log.assert_not_called()
    
    def test_trigger_priority_over_control(self):
        """Test that trigger takes priority over control configuration"""
        self.scene_manager.global_control_config = {
            "trigger": {
                "start": {
                    "shell": "echo 1"
                }
            },
            "control": {
                "shell": "echo 0"  # This should be ignored
            }
        }
        
        with mock.patch('sonicterm.core.scene.tui_manager') as mock_tui:
            mock_tui.is_paused = True
            mock_tui.toggle_pause = mock.Mock()
            mock_tui.set_loading_state = mock.Mock()
            mock_tui.log = mock.Mock()
            mock_tui.update_player_status = mock.Mock()
            
            self.scene_manager._check_global_playback_control()
            
            # Verify trigger was executed, not control
            self.assertTrue(self.scene_manager.trigger_start_executed)
            mock_tui.toggle_pause.assert_called_once()
            mock_tui.log.assert_called_with("Trigger: Playback started")


class TriggerIntegrationTests(unittest.TestCase):
    """Integration tests for trigger system"""
    
    def setUp(self):
        """Set up integration test environment"""
        self.scene_manager = SceneBasedSoundscape("scenes/test.json")
        
    def test_trigger_monitoring_starts_correctly(self):
        """Test that trigger monitoring starts correctly"""
        self.scene_manager.global_control_config = {
            "trigger": {
                "start": {
                    "shell": "echo 1"
                }
            }
        }
        
        # Start monitoring
        self.scene_manager._start_global_control_monitoring()
        
        # Verify monitoring was started
        self.assertGreater(self.scene_manager.global_control_start_time, 0)
    
    def test_trigger_monitoring_without_config(self):
        """Test that trigger monitoring does not start without config"""
        self.scene_manager.global_control_config = None
        
        # Start monitoring
        self.scene_manager._start_global_control_monitoring()
        
        # Verify monitoring was not started
        self.assertEqual(self.scene_manager.global_control_start_time, 0)


if __name__ == '__main__':
    unittest.main()
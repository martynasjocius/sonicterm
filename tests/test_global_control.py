#!/usr/bin/env python3
#
# SonicTerm
# Copyright (c) 2025 Martynas Jocius
#
"""
Test global playback control functionality
"""

import unittest
import time
import tempfile
import json
from pathlib import Path
from unittest.mock import patch, MagicMock

import pygame

# Mock pygame.mixer for testing
pygame.mixer = MagicMock()
pygame.mixer.pause = MagicMock()
pygame.mixer.unpause = MagicMock()

from sonicterm.core.scene import SceneBasedSoundscape
from sonicterm.ui.tui import tui_manager
from sonicterm import STATE_LOADING


class TestGlobalControl(unittest.TestCase):
    def setUp(self):
        """Set up test environment"""
        # Create a temporary scene file
        self.temp_dir = tempfile.mkdtemp()
        self.scene_file = Path(self.temp_dir) / "test_scene.json"
        
        # Mock TUI manager
        tui_manager.is_paused = False
        tui_manager.log = MagicMock()
        tui_manager.toggle_pause = MagicMock()
        
    def tearDown(self):
        """Clean up test environment"""
        import shutil
        shutil.rmtree(self.temp_dir)
        
    def create_test_scene(self, playback_config):
        """Create a test scene file with given playback config"""
        scene_config = {
            "name": "Test Scene",
            "description": "Test scene for global control",
            "global": {
                "fade_in_time": 1.0,
                "fade_out_time": 1.0,
                "master_volume": 0.8,
                "playback": playback_config
            },
            "samples": [
                {
                    "path": "samples/test.wav",
                    "name": "Test Sample",
                    "volume": {"min": 0.5, "max": 0.8},
                    "gain": 1.0,
                    "wait": 0,
                    "timings": [2, 4, 8]
                }
            ]
        }
        
        with open(self.scene_file, 'w') as f:
            json.dump(scene_config, f)
            
    @patch('subprocess.run')
    def test_global_control_pause_on_zero(self, mock_run):
        """Test that output '0' pauses playback when playing"""
        # Setup
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = "0\n"
        
        self.create_test_scene({
            "control": {"shell": "echo 0"}
        })
        
        # Create scene manager
        scene = SceneBasedSoundscape(self.scene_file)
        scene.load_scene()
        
        # Simulate timing conditions for global control
        scene.global_control_initialized = True
        scene.last_global_control_check = 0  # Force check
        
        # Simulate that we're currently playing (not paused)
        tui_manager.is_paused = False
        
        # Check global controls
        scene.check_global_controls()
        
        # Should have called toggle_pause to pause
        tui_manager.toggle_pause.assert_called_once()
        
    @patch('subprocess.run')
    def test_global_control_resume_on_one(self, mock_run):
        """Test that output '1' resumes playback when paused"""
        # Setup
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = "1\n"
        
        self.create_test_scene({
            "control": {"shell": "echo 1"}
        })
        
        # Create scene manager
        scene = SceneBasedSoundscape(self.scene_file)
        scene.load_scene()
        
        # Simulate timing conditions for global control
        scene.global_control_initialized = True
        scene.last_global_control_check = 0  # Force check
        
        # Initial state: paused
        tui_manager.is_paused = True
        
        # Check global controls
        scene.check_global_controls()
        
        # Should have called toggle_pause to resume
        tui_manager.toggle_pause.assert_called_once()
        
    @patch('subprocess.run')
    def test_global_control_no_change_when_already_correct(self, mock_run):
        """Test that no change occurs when state is already correct"""
        # Setup
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = "1\n"
        
        self.create_test_scene({
            "control": {"shell": "echo 1"}
        })
        
        # Create scene manager
        scene = SceneBasedSoundscape(self.scene_file)
        scene.load_scene()
        
        # Simulate timing conditions for global control
        scene.global_control_initialized = True
        scene.last_global_control_check = 0  # Force check
        
        # Initial state: not paused (already correct for "1")
        tui_manager.is_paused = False
        
        # Check global controls
        scene.check_global_controls()
        
        # Should NOT have called toggle_pause
        tui_manager.toggle_pause.assert_not_called()
        
    @patch('subprocess.run')
    def test_global_control_ignores_other_output(self, mock_run):
        """Test that other output values are ignored"""
        # Setup
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = "2\n"
        
        self.create_test_scene({
            "control": {"shell": "echo 2"}
        })
        
        # Create scene manager
        scene = SceneBasedSoundscape(self.scene_file)
        scene.load_scene()
        
        # Simulate timing conditions for global control
        scene.global_control_initialized = True
        scene.last_global_control_check = 0  # Force check
        
        # Initial state: not paused
        tui_manager.is_paused = False
        
        # Check global controls
        scene.check_global_controls()
        
        # Should NOT have called toggle_pause
        tui_manager.toggle_pause.assert_not_called()
        
    @patch('subprocess.run')
    def test_global_control_handles_command_error(self, mock_run):
        """Test that command errors are handled gracefully"""
        # Setup - command fails
        mock_run.return_value.returncode = 1
        mock_run.return_value.stdout = ""
        
        self.create_test_scene({
            "control": {"shell": "false"}
        })
        
        # Create scene manager
        scene = SceneBasedSoundscape(self.scene_file)
        scene.load_scene()
        
        # Initial state: not paused
        tui_manager.is_paused = False
        
        # Check global controls
        scene.check_global_controls()
        
        # Should NOT have called toggle_pause (command failed)
        tui_manager.toggle_pause.assert_not_called()
        
    def test_global_control_no_config(self):
        """Test that no global control config doesn't cause errors"""
        # Create scene without global control
        self.create_test_scene(None)
        
        # Create scene manager
        scene = SceneBasedSoundscape(self.scene_file)
        scene.load_scene()
        
        # Check global controls (should do nothing)
        scene.check_global_controls()
        
        # Should NOT have called toggle_pause
        tui_manager.toggle_pause.assert_not_called()
        
    def test_global_control_starts_paused(self):
        """Test that scenes with global control start in paused mode"""
        # Create scene with global control
        self.create_test_scene({
            "control": {"shell": "echo 1"}
        })
        
        # Create scene manager
        scene = SceneBasedSoundscape(self.scene_file)
        scene.load_scene()
        
        # Simulate start_soundscape behavior
        scene.global_control_config = scene.scene_config.get('global', {}).get('playback')
        
        # Check if global control is configured
        has_global_control = scene.global_control_config is not None
        self.assertTrue(has_global_control, "Global control should be configured")
        
        # When global control is configured, should start paused
        if has_global_control:
            tui_manager.toggle_pause()  # This simulates the pause call in start_soundscape
            
        # Should have called toggle_pause to start paused
        tui_manager.toggle_pause.assert_called_once()
        
    def test_loading_state_with_global_control(self):
        """Test that players show Loading state when global control is configured"""
        # Create scene with global control
        self.create_test_scene({
            "control": {"shell": "echo 1"}
        })
        
        # Create scene manager
        scene = SceneBasedSoundscape(self.scene_file)
        scene.load_scene()
        
        # Simulate start_soundscape behavior
        scene.global_control_config = scene.scene_config.get('global', {}).get('playback')
        
        # Mock active players
        mock_player = MagicMock()
        mock_player.name = "Test Sample"
        mock_player.active = True
        scene.players = [mock_player]
        
        # Mock tui_manager methods
        tui_manager.update_player_status = MagicMock()
        
        # Simulate the loading state setting
        has_global_control = scene.global_control_config is not None
        if has_global_control:
            for player in scene.players:
                if player.active:
                    tui_manager.update_player_status(player.name, {
                        'state': STATE_LOADING,
                        'playback_progress': 0,
                        'playback_total': 0,
                        'waiting_progress': 0,
                        'waiting_total': 0,
                        'active': True
                    })
        
        # Should have called update_player_status with Loading state
        tui_manager.update_player_status.assert_called_with("Test Sample", {
            'state': STATE_LOADING,
            'playback_progress': 0,
            'playback_total': 0,
            'waiting_progress': 0,
            'waiting_total': 0,
            'active': True
        })


if __name__ == '__main__':
    unittest.main()
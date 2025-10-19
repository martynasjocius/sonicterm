#!/usr/bin/env python3
#
# SonicTerm
# Copyright (c) 2025 Martynas Jocius
#
"""Debug utilities for SonicTerm."""

import time
import json
from typing import Dict, Any, Optional
from .. import DEBUG_MODE


class DebugManager:
    """Manages debug output and state tracking for AI tools."""
    
    def __init__(self):
        self.enabled = False
        self.start_time = time.time()
        self.state_changes = []
        self.current_state = {}
        
    def enable(self):
        """Enable debug mode."""
        self.enabled = True
        self.debug_log("DEBUG_MODE_ENABLED", "Debug mode activated for AI tool integration")
        
    def disable(self):
        """Disable debug mode."""
        if self.enabled:
            self.debug_log("DEBUG_MODE_DISABLED", "Debug mode deactivated")
        self.enabled = False
        
    def debug_log(self, event_type: str, message: str, data: Optional[Dict[str, Any]] = None):
        """Log debug information with structured format for AI tools."""
        if not self.enabled:
            return
            
        timestamp = time.time()
        runtime = timestamp - self.start_time
        
        debug_entry = {
            "timestamp": timestamp,
            "runtime_seconds": round(runtime, 3),
            "event_type": event_type,
            "message": message
        }
        
        if data:
            debug_entry["data"] = data
            
        # Store state change
        self.state_changes.append(debug_entry)
        
        # Use TUI logging if available, otherwise print to stdout
        try:
            from ..ui.tui import tui_manager
            if tui_manager.tui_enabled and hasattr(tui_manager, 'live_display'):
                # Use TUI logging to avoid screen glitches
                tui_manager.log(f"[DEBUG:{runtime:7.3f}s] {event_type}: {message}")
                if data:
                    for key, value in data.items():
                        tui_manager.log(f"[DEBUG:{runtime:7.3f}s]   {key}: {value}")
            else:
                # Print to stdout when TUI is not enabled or not ready
                print(f"[DEBUG:{runtime:7.3f}s] {event_type}: {message}")
                if data:
                    for key, value in data.items():
                        print(f"[DEBUG:{runtime:7.3f}s]   {key}: {value}")
        except ImportError:
            # Fallback to print if TUI is not available
            print(f"[DEBUG:{runtime:7.3f}s] {event_type}: {message}")
            if data:
                for key, value in data.items():
                    print(f"[DEBUG:{runtime:7.3f}s]   {key}: {value}")
                
    def update_state(self, component: str, key: str, value: Any, description: str = ""):
        """Update global state and log the change."""
        if not self.enabled:
            return
            
        if component not in self.current_state:
            self.current_state[component] = {}
            
        old_value = self.current_state[component].get(key, None)
        self.current_state[component][key] = value
        
        change_desc = f"{description} " if description else ""
        self.debug_log(
            "STATE_CHANGE",
            f"{change_desc}{component}.{key}: {old_value} -> {value}",
            {
                "component": component,
                "key": key,
                "old_value": old_value,
                "new_value": value
            }
        )
        
    def log_operation(self, operation: str, component: str, details: Dict[str, Any] = None):
        """Log an operation with optional details."""
        if not self.enabled:
            return
            
        self.debug_log(
            "OPERATION",
            f"{component}: {operation}",
            details or {}
        )
        
    def log_error(self, error_type: str, component: str, error_msg: str, details: Dict[str, Any] = None):
        """Log an error with context."""
        if not self.enabled:
            return
            
        error_data = {
            "component": component,
            "error_message": error_msg
        }
        if details:
            error_data.update(details)
            
        self.debug_log(
            "ERROR",
            f"{component}: {error_type} - {error_msg}",
            error_data
        )
        
    def get_current_state(self) -> Dict[str, Any]:
        """Get current global state snapshot."""
        return {
            "runtime_seconds": round(time.time() - self.start_time, 3),
            "debug_enabled": self.enabled,
            "state": self.current_state.copy(),
            "total_state_changes": len(self.state_changes)
        }
        
    def print_state_summary(self):
        """Print a comprehensive state summary for AI tools."""
        if not self.enabled:
            return
            
        state = self.get_current_state()
        runtime = state["runtime_seconds"]
        
        # Use TUI logging if available, otherwise print to stdout
        try:
            from ..ui.tui import tui_manager
            if tui_manager.tui_enabled and hasattr(tui_manager, 'live_display'):
                # Use TUI logging to avoid screen glitches
                tui_manager.log(f"[DEBUG:{runtime:7.3f}s] === STATE SUMMARY ===")
                tui_manager.log(f"[DEBUG:{runtime:7.3f}s] Runtime: {runtime}s")
                tui_manager.log(f"[DEBUG:{runtime:7.3f}s] Total state changes: {state['total_state_changes']}")
                
                for component, component_state in state["state"].items():
                    tui_manager.log(f"[DEBUG:{runtime:7.3f}s] {component}:")
                    for key, value in component_state.items():
                        tui_manager.log(f"[DEBUG:{runtime:7.3f}s]   {key}: {value}")
                        
                tui_manager.log(f"[DEBUG:{runtime:7.3f}s] === END STATE SUMMARY ===")
            else:
                # Print to stdout when TUI is not enabled or not ready
                print(f"\n[DEBUG:{runtime:7.3f}s] === STATE SUMMARY ===")
                print(f"[DEBUG:{runtime:7.3f}s] Runtime: {runtime}s")
                print(f"[DEBUG:{runtime:7.3f}s] Total state changes: {state['total_state_changes']}")
                
                for component, component_state in state["state"].items():
                    print(f"[DEBUG:{runtime:7.3f}s] {component}:")
                    for key, value in component_state.items():
                        print(f"[DEBUG:{runtime:7.3f}s]   {key}: {value}")
                        
                print(f"[DEBUG:{runtime:7.3f}s] === END STATE SUMMARY ===\n")
        except ImportError:
            # Fallback to print if TUI is not available
            print(f"\n[DEBUG:{runtime:7.3f}s] === STATE SUMMARY ===")
            print(f"[DEBUG:{runtime:7.3f}s] Runtime: {runtime}s")
            print(f"[DEBUG:{runtime:7.3f}s] Total state changes: {state['total_state_changes']}")
            
            for component, component_state in state["state"].items():
                print(f"[DEBUG:{runtime:7.3f}s] {component}:")
                for key, value in component_state.items():
                    print(f"[DEBUG:{runtime:7.3f}s]   {key}: {value}")
                    
            print(f"[DEBUG:{runtime:7.3f}s] === END STATE SUMMARY ===\n")


# Global debug manager instance
debug_manager = DebugManager()


def set_debug_mode(enabled: bool):
    """Set global debug mode state."""
    global DEBUG_MODE
    DEBUG_MODE = enabled
    if enabled:
        debug_manager.enable()
    else:
        debug_manager.disable()


def debug_log(event_type: str, message: str, data: Optional[Dict[str, Any]] = None):
    """Convenience function for debug logging."""
    debug_manager.debug_log(event_type, message, data)


def update_state(component: str, key: str, value: Any, description: str = ""):
    """Convenience function for state updates."""
    debug_manager.update_state(component, key, value, description)


def log_operation(operation: str, component: str, details: Dict[str, Any] = None):
    """Convenience function for operation logging."""
    debug_manager.log_operation(operation, component, details)


def log_error(error_type: str, component: str, error_msg: str, details: Dict[str, Any] = None):
    """Convenience function for error logging."""
    debug_manager.log_error(error_type, component, error_msg, details)

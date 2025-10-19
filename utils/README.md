# Sonicterm Utils

This folder contains utility scripts to help control sonicterm via system commands or sensors.

## Scripts

### wait_for_time

Waits for a specific time and outputs control signals for sonicterm's global playback control.

**Usage:**

```bash
./wait_for_time YYYY-MM-DD HH:MM:SS
```

**Example:**

```bash
./wait_for_time 2025-09-28 12:10:00
```

**Output:**

- `0` - If target time has not been reached yet
- `1` - If target time has been reached or passed

**Integration with Sonicterm:**
Add to your scene configuration:

```json
{
  "global": {
    "playback": {
      "control": {
        "shell": "./utils/wait_for_time 2025-09-28 12:10:00"
      }
    }
  }
}
```

**Features:**

- Validates datetime format
- Handles both future and past time targets
- Outputs only single character (0 or 1)
- Exits immediately after output
- Compatible with sonicterm's global control system

**Error Handling:**

- Invalid datetime format
- Missing arguments
- Clear error messages with examples

## Future

This collection of utility collection is planned to grow with new scripts:

- Sensor-based control (temperature, light, motion)
- Network-based triggers
- File system monitoring
- System resource monitoring
- Time-based automation


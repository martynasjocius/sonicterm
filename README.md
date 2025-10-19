# SonicTerm

SonicTerm is a generative samples player for sound installations.

## Quick Start

```bash
./sonicterm.sh your-scene.json

```

## Features

- **Terminal UI**: Progress bars, status panels, and optional visualisations
- **Signal controls**: Per-sample gain, panning, envelopes, and external automation hooks
- **External control**: Change sample volume or start/stop playback based on external shell command output.

## Scene Configuration

Scenes are defined in JSON files with complete control over each sample:

```json
{
  "name": "My Soundscape",
  "description": "Custom layered scene",
  "global": {
    "fade_in_time": 2.0,
    "fade_out_time": 3.0,
    "master_volume": 0.8
  },
  "samples": [
    {
      "path": "samples/ambient.wav",
      "name": "Ambient Texture",
      "volume": {"min": 0.3, "max": 0.8, "control": {"shell": "cat /tmp/sensor1"}},
      "gain": 1.2,
      "pan": {"min": -0.8, "max": 0.8},
      "envelope": {"attack": 0.5, "release": 1.0},
      "timings": [1, 2, 4, 8, 16],
      "wait": 2.0
    }
  ]
}
```

## Sample Parameters

Each sample supports these parameters:

- **path**: Path to audio file (WAV format recommended)
- **name**: Display name for the sample
- **volume**: Min/max volume levels (0.0-1.0)
- **gain**: Gain multiplier (0.0-10.0, where 1.0 = 100%, 10.0 = 1000%)
- **pan**: Min/max stereo panning (-1.0 = left, 1.0 = right)
- **envelope**: Attack and release times in seconds
- **timings**: Array of wait times (seconds) to randomly choose from
- **wait**: Initial wait time before first playback in seconds (default: 0)

## Terminal User Interface

The TUI is **enabled by default**. Use `--no-tui` to run without it.

**Panel overview:**

- **Scene header**: Scene metadata and run state
- **Progress bars**: 20fps meters per active sample
- **Color matrix**: Optional visualisation tied to activity
- **Process log**: Append-only event feed

## Installation

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Run: `./sonicterm.sh`

## License

MIT License - see LICENSE file for details.

## Contributing

Feel free to submit issues and pull requests.

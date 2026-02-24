# Project Context: MIDI Drum Humanizer

## Overview
The **MIDI Drum Humanizer** is a Python utility that transforms robotic, quantized MIDI drum tracks into expressive, human-sounding performances. It achieves this by applying algorithmic variations to note timing and velocity based on configurable "drummer profiles."

## Core Functionality
- **Timing Humanization**: Shifts note start times (ticks) to simulate groove, rushing/dragging, and natural imperfections.
- **Velocity Humanization**: Adjusts note velocities to add dynamics, accents, and ghost notes.
- **Styles**: Supports different drummer personas (e.g., "balanced", "lazy", "tight").
- **Visualization**: Can generate plots comparing the original vs. processed MIDI data.

## Architecture

### Entry Point
- **`humanize-drums.py`**: The main execution script. It wraps `src/drums_midi_humanizer/cli.py`.

### Source Code (`src/drums_midi_humanizer/`)
- **`cli.py`**: Handles argument parsing and initiates the `DrumHumanizer`.
- **`core/humanizer.py`**: Contains the `DrumHumanizer` class.
  - **Logic**: Iterates through MIDI tracks, identifies drum types (Kick, Snare, etc.), and applies specific timing/velocity rules defined by the `DrummerProfile`.
  - **Key Methods**: `process_file`, `humanize_timings`, `humanize_velocity`.
- **`config/`**: Contains drum mappings (GM, Addictive Drums, etc.) and drummer profile definitions.
- **`visualization/`**: Logic for generating comparison graphs (matplotlib).

### Utilities
- **`midi-tempo-analysis.py`**: A standalone script for detecting tempo and time signatures in MIDI files lacking explicit metadata.
- **`midi_visualizer.py`**: Contains the plotting logic used for generating comparison images.

## Key Classes & Data Structures

### `HumanizerConfig` (Dataclass)
Holds runtime configuration:
- `timing_variation` (int): Max ticks for timing jitter.
- `velocity_variation` (int): Max velocity fluctuation.
- `ghost_note_prob` (float): Probability of generating ghost notes.
- `drummer_style` (str): Key for the drummer profile to use.

### `DrumHumanizer`
The primary processor. It maintains state for:
- `ticks_per_beat`: Derived from the input MIDI.
- `drum_map`: Mapping of MIDI note numbers to instrument types.
- `profile`: The active `DrummerProfile`.

## Development Guidelines
- **Dependencies**: `mido` (MIDI I/O), `numpy` (math), `matplotlib` (viz), `pytest` (tests).
- **Testing**: Run `pytest` to execute tests in `tests/`.
- **Style**: Follows standard Python practices. Type hinting is used in newer modules.

## Common Tasks
1.  **Adding a new Drummer Profile**: Update `src/drums_midi_humanizer/config/drums.py`.
2.  **Supporting a new Drum Library**: Add mapping to `src/drums_midi_humanizer/config/drums.py`.
3.  **Tweaking Algorithms**: Modify `_apply_timing_variation` or `_apply_velocity_variation` in `core/humanizer.py`.

## Usage Example
```bash
python humanize-drums.py "input.mid" --style tight --timing 5 --visualize
```
"""Configuration for drum maps and drummer profiles.

This module defines the data structures and constants used to configure the
drum humanizer. It includes:
- Drum mappings for various popular drum libraries (GM, Addictive Drums, etc.).
- Drummer profiles that define different playing styles (Jazz, Rock, Metal, etc.).
- Rudiment definitions for pattern detection.
"""

from dataclasses import dataclass
from typing import Dict, Any, Set, Tuple, List

@dataclass
class DrummerProfile:
    """Profile defining a drummer's playing characteristics.

    Attributes:
        timing_bias (float): Base timing offset in ticks (positive = late/lazy, negative = early/rushed).
        velocity_emphasis (float): Multiplier for velocity dynamics (1.0 = standard, >1.0 = more dynamic).
        ghost_multiplier (float): Velocity multiplier for ghost notes (0.0-1.0).
        kick_timing_tightness (float): How tight the kick drum is to the grid (higher = tighter).
        hihat_variation (float): Amount of timing variation for hi-hats.
        rushing_factor (float): Tendency to rush (positive) or drag (negative) tempo.
        groove_consistency (float): How consistent the timing variations are (0.0-1.0).
        rudiment_sensitivity (float): Probability threshold for applying rudiment logic (0.0-1.0).
    """
    timing_bias: float
    velocity_emphasis: float
    ghost_multiplier: float
    kick_timing_tightness: float
    hihat_variation: float
    rushing_factor: float
    groove_consistency: float
    rudiment_sensitivity: float = 0.7  # How likely to detect and apply rudiments (0.0-1.0)

@dataclass
class DrumMap:
    """Mapping of MIDI notes to drum types.

    Attributes:
        kick_notes (set[int]): MIDI note numbers for kick drums.
        snare_notes (set[int]): MIDI note numbers for snare drums.
        hihat_notes (set[int]): MIDI note numbers for hi-hats (open, closed, pedal).
        tom_notes (set[int]): MIDI note numbers for toms.
        cymbal_notes (set[int]): MIDI note numbers for cymbals (crash, ride, splash, china).
    """
    kick_notes: set[int]
    snare_notes: set[int]
    hihat_notes: set[int]
    tom_notes: set[int]
    cymbal_notes: set[int]

    def get_note_groups(self) -> Tuple[Set[int], Set[int], Set[int], Set[int], Set[int]]:
        """Get all note groups as a tuple of sets.
        
        Returns:
            Tuple[Set[int], ...]: A tuple containing sets of note numbers for:
                (kick_notes, snare_notes, hihat_notes, tom_notes, cymbal_notes).
        """
        return (
            self.kick_notes,
            self.snare_notes,
            self.hihat_notes,
            self.tom_notes,
            self.cymbal_notes
        )

# Drum maps for different libraries.
# Keys represent the library identifier used in the CLI.
DRUM_MAPS = {
    "gm": DrumMap(  # General MIDI drum map
        kick_notes={35, 36},  # Acoustic Bass Drum, Bass Drum 1
        snare_notes={38, 40},  # Acoustic Snare, Electric Snare
        hihat_notes={42, 44, 46},  # Closed HH, Pedal HH, Open HH
        tom_notes={41, 43, 45, 47, 48, 50},  # Low/Mid/High Floor/Tom
        cymbal_notes={49, 51, 52, 53, 55, 57, 59}  # Crash/Ride cymbals
    ),
    "ad2": DrumMap(  # Addictive Drums 2
        kick_notes={36},
        snare_notes={38, 40},
        hihat_notes={42, 44, 46, 54},  # Including extra articulations
        tom_notes={41, 43, 45, 47, 48, 50},
        cymbal_notes={49, 51, 52, 53, 55, 57, 59, 56}  # Including extra crashes
    ),
    "sd3": DrumMap(  # Superior Drummer 3
        kick_notes={35, 36},
        snare_notes={38, 40, 37},  # Including rim shots
        hihat_notes={42, 44, 46, 54, 56},  # More hihat articulations
        tom_notes={41, 43, 45, 47, 48, 50},
        cymbal_notes={49, 51, 52, 53, 55, 57, 59, 56, 58}  # More cymbal options
    ),
    "ez2": DrumMap(  # EZdrummer 2
        kick_notes={35, 36},
        snare_notes={38, 40, 37},
        hihat_notes={42, 44, 46, 54},
        tom_notes={41, 43, 45, 47, 48, 50},
        cymbal_notes={49, 51, 52, 53, 55, 57, 59}
    ),
    "ssd5": DrumMap(  # Steven Slate Drums 5
        kick_notes={35, 36},
        snare_notes={38, 40, 37, 39},  # Including additional articulations
        hihat_notes={42, 44, 46, 54, 56},
        tom_notes={41, 43, 45, 47, 48, 50},
        cymbal_notes={49, 51, 52, 53, 55, 57, 59, 56, 58}
    ),
    "mtpk2": DrumMap(  # MT Power Kit 2
        kick_notes={36},
        snare_notes={38, 40},
        hihat_notes={42, 44, 46},
        tom_notes={41, 43, 45, 47},
        cymbal_notes={49, 51, 52, 53, 55}
    )
}

# Drummer style profiles.
# Keys represent the style name used in the CLI.
DRUMMER_PROFILES: Dict[str, Dict[str, float]] = {
    "balanced": {
        "timing_bias": 0,  # Neutral timing
        "velocity_emphasis": 1.0,  # Normal dynamics
        "ghost_multiplier": 1.0,  # Standard ghost notes
        "kick_timing_tightness": 1.0,  # Standard kick timing
        "hihat_variation": 1.0,  # Standard hi-hat variation
        "rushing_factor": 0,  # No tendency to rush or drag
        "groove_consistency": 0.7,  # Moderately consistent groove
        "rudiment_sensitivity": 0.7,  # Standard rudiment detection
    },
    "jazzy": {
        "timing_bias": -2,
        "velocity_emphasis": 1.2,
        "ghost_multiplier": 1.5,
        "kick_timing_tightness": 0.7,
        "hihat_variation": 1.3,
        "rushing_factor": -0.3,
        "groove_consistency": 0.6,
        "rudiment_sensitivity": 0.9,  # High rudiment detection for jazz style
    },
    "rock": {
        "timing_bias": 2,
        "velocity_emphasis": 1.3,
        "ghost_multiplier": 0.8,
        "kick_timing_tightness": 1.2,
        "hihat_variation": 0.9,
        "rushing_factor": 0.2,
        "groove_consistency": 0.8,
        "rudiment_sensitivity": 0.6,  # Moderate rudiment detection
    },
    "precise": {
        "timing_bias": 0,
        "velocity_emphasis": 0.8,
        "ghost_multiplier": 0.5,
        "kick_timing_tightness": 1.5,
        "hihat_variation": 0.6,
        "rushing_factor": 0,
        "groove_consistency": 0.9,
        "rudiment_sensitivity": 0.8,  # High rudiment detection for precise playing
    },
    "loose": {
        "timing_bias": 0,
        "velocity_emphasis": 1.4,
        "ghost_multiplier": 1.7,
        "kick_timing_tightness": 0.6,
        "hihat_variation": 1.5,
        "rushing_factor": 0.1,
        "groove_consistency": 0.5,
        "rudiment_sensitivity": 0.4,  # Lower rudiment detection for loose style
    },
    "modern_metal": {
        "timing_bias": 1,
        "velocity_emphasis": 1.5,
        "ghost_multiplier": 0.3,
        "kick_timing_tightness": 1.8,
        "hihat_variation": 0.4,
        "rushing_factor": 0.15,
        "groove_consistency": 0.95,
        "rudiment_sensitivity": 0.5,  # Moderate rudiment detection for metal
    },
}

# Define common drum rudiments for pattern detection.
# Each rudiment contains:
#   - pattern: Hand sticking pattern (R/L).
#   - timing_ratio: Relative timing of each note in the pattern.
#   - velocity_ratio: Relative velocity emphasis for each note.
#   - duration: Total duration of the pattern in beats.
DRUM_RUDIMENTS = {
    "single_paradiddle": {
        "pattern": ["R", "L", "R", "R", "L", "R", "L", "L"],  # RLRR LRLL
        "timing_ratio": [1, 1, 1, 1, 1, 1, 1, 1],  # Even timing
        "velocity_ratio": [1.0, 0.8, 0.9, 0.85, 1.0, 0.8, 0.9, 0.85],
        "duration": 2,  # Duration in beats
    },
    "double_paradiddle": {
        "pattern": ["R", "L", "R", "L", "R", "R", "L", "R", "L", "R", "L", "L"],  # RLRLRR LRLRLL
        "timing_ratio": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        "velocity_ratio": [1.0, 0.85, 0.9, 0.85, 0.9, 0.95, 1.0, 0.85, 0.9, 0.85, 0.9, 0.95],
        "duration": 3,
    },
    "triple_roll": {
        "pattern": ["R", "L", "R", "L", "R", "L"],
        "timing_ratio": [0.666, 0.666, 0.666, 0.666, 0.666, 0.666],
        "velocity_ratio": [1.0, 0.9, 0.95, 0.9, 0.95, 1.0],
        "duration": 1,
    },
    "five_stroke_roll": {
        "pattern": ["R", "L", "R", "L", "R"],
        "timing_ratio": [0.75, 0.75, 0.75, 0.75, 1],
        "velocity_ratio": [0.8, 0.8, 0.85, 0.9, 1.0],
        "duration": 1,
    },
    "flam_tap": {
        "pattern": ["rR", "L", "lL", "R"],  # lowercase represents grace note
        "timing_ratio": [1, 1, 1, 1],
        "velocity_ratio": [1.0, 0.8, 1.0, 0.8],
        "duration": 1,
    },
}

def get_drum_map(library: str = "gm") -> DrumMap:
    """Retrieve the drum map for a specific library.
    
    Args:
        library (str): The name of the drum library to use. Defaults to "gm" (General MIDI).
        
    Returns:
        DrumMap: A DrumMap object containing the note mappings for the specified library.
        
    Raises:
        ValueError: If the specified library is not found.
    """
    if library not in DRUM_MAPS:
        raise ValueError(f"Unknown drum library: {library}. Available libraries: {list(DRUM_MAPS.keys())}")
    return DRUM_MAPS[library]

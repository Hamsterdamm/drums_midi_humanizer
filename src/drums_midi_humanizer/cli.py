"""Command-line interface for the MIDI drum humanizer.

This module provides the entry point for the application, handling command-line
arguments, input validation, and orchestration of the humanization process.
It allows users to specify input/output files and customize various humanization
parameters such as timing, velocity, and drummer style.
"""

import argparse
from pathlib import Path

from .core.humanizer import DrumHumanizer, HumanizerConfig
from .config.drums import DRUMMER_PROFILES


def parse_args() -> argparse.Namespace:
    """Parse and validate command-line arguments.

    Defines the available command-line arguments for the drum humanizer,
    including input/output paths and various humanization parameters.

    Returns:
        argparse.Namespace: An object containing the parsed command-line arguments.
            - input_file (str): Path to the source MIDI file.
            - output (str): Path to the destination MIDI file.
            - timing (int): Max timing variation in ticks.
            - velocity (int): Max velocity variation.
            - ghost (float): Probability of ghost notes (0.0-1.0).
            - accent (float): Probability of accents (0.0-1.0).
            - shuffle (float): Shuffle amount (0.0-0.5).
            - flams (float): Probability of flams (0.0-1.0).
            - style (str): Drummer style profile name.
            - library (str): Drum library mapping name.
            - visualize (bool): Whether to generate a visualization.
    """
    parser = argparse.ArgumentParser(
        description="Humanize MIDI drum tracks with realistic drummer feel",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("input_file", help="Input MIDI file path")
    parser.add_argument(
        "--output", "-o",
        help="Output MIDI file path (default: input_file_humanized.mid)"
    )
    parser.add_argument(
        "--timing", "-t",
        type=int, default=10,
        help="Timing variation in ticks (default: 10)"
    )
    parser.add_argument(
        "--velocity", "-v",
        type=int, default=15,
        help="Velocity variation (default: 15)"
    )
    parser.add_argument(
        "--ghost", "-g",
        type=float, default=0.1,
        help="Ghost note probability (default: 0.1)"
    )
    parser.add_argument(
        "--accent", "-a",
        type=float, default=0.2,
        help="Accent probability (default: 0.2)"
    )
    parser.add_argument(
        "--shuffle", "-s",
        type=float, default=0.0,
        help="Shuffle amount, 0.0-0.5 (default: 0.0)"
    )
    parser.add_argument(
        "--flams", "-f",
        type=float, default=0.0,
        help="Flam probability (default: 0.0)"
    )
    parser.add_argument(
        "--style",
        choices=list(DRUMMER_PROFILES.keys()),
        default="balanced",
        help="Drummer style profile (default: balanced)"
    )
    parser.add_argument(
        "--library",
        choices=["gm", "ad2", "sd3", "ez2", "ssd5", "mtpk2"],
        default="gm",
        help="Drums library mapping (default: gm)"
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Generate visualization comparing original and humanized MIDI"
    )
    return parser.parse_args()


def main() -> None:
    """Main entry point for the humanizer application.

    This function orchestrates the entire process:
    1. Parses command-line arguments.
    2. Validates input file existence and parameter ranges.
    3. Initializes the configuration and humanizer.
    4. Processes the MIDI file and optionally generates a visualization.
    """
    args = parse_args()

    # Validate input file existence
    if not Path(args.input_file).exists():
        print(f"Error: Input file not found: {args.input_file}")
        return

    # Validate probability ranges
    if not 0 <= args.ghost <= 1:
        print("Error: Ghost note probability must be between 0.0 and 1.0")
        return

    if not 0 <= args.accent <= 1:
        print("Error: Accent probability must be between 0.0 and 1.0")
        return

    if not 0 <= args.shuffle <= 0.5:
        print("Error: Shuffle amount must be between 0.0 and 0.5")
        return

    if not 0 <= args.flams <= 1:
        print("Error: Flam probability must be between 0.0 and 1.0")
        return

    # Create humanizer configuration object from arguments
    config = HumanizerConfig(
        timing_variation=args.timing,
        velocity_variation=args.velocity,
        ghost_note_prob=args.ghost,
        accent_prob=args.accent,
        shuffle_amount=args.shuffle,
        flamming_prob=args.flams,
        drummer_style=args.style,
        drum_library=args.library,
        visualize=args.visualize
    )

    # Initialize the humanizer with the config and process the file
    humanizer = DrumHumanizer(config)
    humanizer.process_file(args.input_file, args.output)

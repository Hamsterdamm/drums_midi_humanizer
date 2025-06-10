import mido
import random
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Tuple, Dict
import mido
from humanizer import DrumHumanizer, HumanizerConfig


# Set drummer profile characteristics
drummer_profiles = {
    "balanced": {
        "timing_bias": 0,  # Neutral timing
        "velocity_emphasis": 1.0,  # Normal dynamics
        "ghost_multiplier": 1.0,  # Standard ghost notes
        "kick_timing_tightness": 1.0,  # Standard kick timing
        "hihat_variation": 1.0,  # Standard hi-hat variation
        "rushing_factor": 0,  # No tendency to rush or drag
        "groove_consistency": 0.7,  # Moderately consistent groove
    },
    "jazzy": {
        "timing_bias": -2,  # Slightly behind the beat
        "velocity_emphasis": 1.2,  # More dynamic range
        "ghost_multiplier": 1.5,  # More ghost notes
        "kick_timing_tightness": 0.7,  # Looser kick timing
        "hihat_variation": 1.3,  # More hi-hat variation
        "rushing_factor": -0.3,  # Tendency to lay back
        "groove_consistency": 0.6,  # Less mechanical consistency
    },
    "rock": {
        "timing_bias": 2,  # Slightly ahead of the beat
        "velocity_emphasis": 1.3,  # Stronger dynamics
        "ghost_multiplier": 0.8,  # Fewer ghost notes
        "kick_timing_tightness": 1.2,  # Tighter kick timing
        "hihat_variation": 0.9,  # Less hi-hat variation
        "rushing_factor": 0.2,  # Tendency to push forward
        "groove_consistency": 0.8,  # More consistent groove
    },
    "precise": {
        "timing_bias": 0,  # On the beat
        "velocity_emphasis": 0.8,  # Less dynamic range
        "ghost_multiplier": 0.5,  # Fewer ghost notes
        "kick_timing_tightness": 1.5,  # Very tight kick timing
        "hihat_variation": 0.6,  # Minimal hi-hat variation
        "rushing_factor": 0,  # No tendency to rush or drag
        "groove_consistency": 0.9,  # Very consistent groove
    },
    "loose": {
        "timing_bias": 0,  # Variable timing
        "velocity_emphasis": 1.4,  # Wide dynamic range
        "ghost_multiplier": 1.7,  # Many ghost notes
        "kick_timing_tightness": 0.6,  # Loose kick timing
        "hihat_variation": 1.5,  # Lots of hi-hat variation
        "rushing_factor": 0.1,  # Slight tendency to push
        "groove_consistency": 0.5,  # Inconsistent groove
    },
    "modern_metal": {
        "timing_bias": 1,  # Slightly ahead for aggressive feel
        "velocity_emphasis": 1.5,  # Strong dynamic emphasis
        "ghost_multiplier": 0.3,  # Minimal ghost notes
        "kick_timing_tightness": 1.8,  # Very tight kick timing for blast beats and doubles
        "hihat_variation": 0.4,  # Minimal hi-hat variation for precision
        "rushing_factor": 0.15,  # Slight tendency to push forward
        "groove_consistency": 0.95,  # Very consistent and mechanical
    },
}


# General MIDI Drum Mapping (GM)
GM_DRUM_MAP = {
    # Kicks
    35: "Acoustic Bass Drum",
    36: "Bass Drum 1",
    # Snares
    38: "Acoustic Snare",
    40: "Electric Snare",
    # Hi-hats
    42: "Closed Hi-Hat",
    44: "Pedal Hi-Hat",
    46: "Open Hi-Hat",
    # Toms
    41: "Low Floor Tom",
    43: "High Floor Tom",
    45: "Low Tom",
    47: "Low-Mid Tom",
    48: "Hi-Mid Tom",
    50: "High Tom",
    # Cymbals
    49: "Crash Cymbal 1",
    51: "Ride Cymbal 1",
    52: "Chinese Cymbal",
    53: "Ride Bell",
    55: "Splash Cymbal",
    57: "Crash Cymbal 2",
    59: "Ride Cymbal 2",
}

# Addictive Drums 2 Mapping
AD2_DRUM_MAP = {
    # Kicks
    36: "Kick",
    # Snares
    38: "Snare Center",
    40: "Snare Rimshot",
    37: "Snare Sidestick",
    # Hi-hats
    42: "HH Closed Tip",
    22: "HH Closed Shank",
    44: "HH Pedal",
    46: "HH Open",
    26: "HH Half Open",
    # Toms
    41: "Tom 1",
    43: "Tom 2",
    45: "Tom 3",
    48: "Tom 4",
    # Cymbals
    49: "Crash 1",
    57: "Crash 2",
    52: "China",
    55: "Splash",
    51: "Ride Tip",
    59: "Ride Crash",
    53: "Ride Bell",
}

# Superior Drummer 3 Mapping
SD3_DRUM_MAP = {
    # Kicks
    36: "Kick Right",
    35: "Kick Left",
    # Snares
    38: "Snare Center",
    40: "Snare Rimshot",
    37: "Snare Sidestick",
    39: "Snare Rimclick",
    # Hi-hats
    42: "HH Closed Tip",
    22: "HH Closed Shank",
    44: "HH Pedal",
    46: "HH Open Tip",
    26: "HH Half Open Tip",
    61: "HH Closed Tip",
    64: "HH Closed Shank",
    63: "HH Tight Tip",
    62: "HH Tight Shank",
    64: "HH Open",
    # Toms
    48: "Tom 1",
    45: "Tom 2",
    43: "Tom 3",
    41: "Tom 4",
    # Cymbals
    49: "Crash 1 Tip",
    57: "Crash 2 Tip",
    55: "Splash Tip",
    52: "China Tip",
    51: "Ride Tip",
    53: "Ride Bell",
    59: "Ride Edge",
}

# EZdrummer 2 Mapping
EZ2_DRUM_MAP = {
    # Kicks
    36: "Kick",
    # Snares
    38: "Snare Center",
    40: "Snare Rimshot",
    37: "Snare Sidestick",
    # Hi-hats
    42: "HH Closed Tip",
    22: "HH Closed Shank",
    44: "HH Pedal",
    46: "HH Open",
    26: "HH Half Open",
    61: "HH Closed Tip",
    64: "HH Closed Shank",
    63: "HH Tight Tip",
    62: "HH Tight Shank",
    64: "HH Open",
    # Toms
    48: "Tom 1",
    45: "Tom 2",
    43: "Tom 3",
    41: "Tom 4",
    # Cymbals
    49: "Crash 1",
    57: "Crash 2",
    55: "Splash",
    52: "China",
    51: "Ride Tip",
    53: "Ride Bell",
    59: "Ride Edge",
}

# Steven Slate Drums 5 Mapping
SSD5_DRUM_MAP = {
    # Kicks
    36: "Kick",
    # Snares
    38: "Snare Center",
    40: "Snare Rimshot",
    37: "Snare Sidestick",
    # Hi-hats
    42: "HH Closed",
    44: "HH Pedal",
    46: "HH Open",
    # Toms
    41: "Tom 4 (Floor)",
    43: "Tom 3 (Floor)",
    45: "Tom 2 (Mid)",
    48: "Tom 1 (High)",
    # Cymbals
    49: "Crash 1",
    57: "Crash 2",
    55: "Splash",
    52: "China",
    51: "Ride Bow",
    53: "Ride Bell",
    59: "Ride Edge",
}

# MT Power Drum Kit 2 Mapping
MTPK2_DRUM_MAP = {
    36: "Kick",
    38: "Snare Center",
    40: "Snare Rimshot",
    37: "Snare Sidestick",
    42: "HH Closed",
    44: "HH Pedal",
    46: "HH Open",
    41: "Tom Low",
    43: "Tom Mid",
    45: "Tom High",
    49: "Crash 1",
    57: "Crash 2",
    51: "Ride Bow",
    53: "Ride Bell",
    59: "Ride Edge",
}

# Select the appropriate drum map based on the library
drum_maps = {
    "gm": GM_DRUM_MAP,
    "ad2": AD2_DRUM_MAP,
    "sd3": SD3_DRUM_MAP,
    "ez2": EZ2_DRUM_MAP,
    "ssd5": SSD5_DRUM_MAP,
    "mtpk2": MTPK2_DRUM_MAP,
}


# Define common drum rudiments
drum_rudiments = {
    "single_paradiddle": {
        "pattern": ["R", "L", "R", "R", "L", "R", "L", "L"],  # RLRR LRLL
        "timing_ratio": [1, 1, 1, 1, 1, 1, 1, 1],  # Even timing
        "velocity_ratio": [1.0, 0.8, 0.9, 0.85, 1.0, 0.8, 0.9, 0.85],
        "duration": 2,  # Duration in beats
    },
    "double_paradiddle": {
        "pattern": [
            "R",
            "L",
            "R",
            "L",
            "R",
            "R",
            "L",
            "R",
            "L",
            "R",
            "L",
            "L",
        ],  # RLRLRR LRLRLL
        "timing_ratio": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        "velocity_ratio": [
            1.0,
            0.85,
            0.9,
            0.85,
            0.9,
            0.95,
            1.0,
            0.85,
            0.9,
            0.85,
            0.9,
            0.95,
        ],
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


# Group drum types for specialized handling based on selected mapping
def get_note_groups(drum_map):
    kick_notes = []
    snare_notes = []
    hihat_notes = []
    tom_notes = []
    cymbal_notes = []

    for note, name in drum_map.items():
        name_lower = name.lower()
        if "kick" in name_lower or "bass drum" in name_lower:
            kick_notes.append(note)
        elif "snare" in name_lower:
            snare_notes.append(note)
        elif "hh" in name_lower or "hi-hat" in name_lower or "hihat" in name_lower:
            hihat_notes.append(note)
        elif "tom" in name_lower:
            tom_notes.append(note)
        elif any(x in name_lower for x in ["crash", "ride", "china", "splash"]):
            cymbal_notes.append(note)

    return kick_notes, snare_notes, hihat_notes, tom_notes, cymbal_notes


def humanize_drums(
    input_file,
    output_file,
    timing_variation=10,
    velocity_variation=15,
    ghost_note_prob=0.1,
    accent_prob=0.2,
    shuffle_amount=0.0,
    flamming_prob=0.05,
    drummer_style="balanced",
    drum_library="gm",
    visualize=False,
):
    """Add realistic human feel to a MIDI drum track based on drumming principles."""
    # Validate parameters and create config
    if not 0 <= ghost_note_prob <= 1:
        raise ValueError("Ghost note probability must be between 0 and 1")
    if not 0 <= accent_prob <= 1:
        raise ValueError("Accent probability must be between 0 and 1")
    if not 0 <= shuffle_amount <= 0.5:
        raise ValueError("Shuffle amount must be between 0 and 0.5")
    if not 0 <= flamming_prob <= 1:
        raise ValueError("Flam probability must be between 0 and 1")

    config = HumanizerConfig(
        timing_variation=timing_variation,
        velocity_variation=velocity_variation,
        ghost_note_prob=ghost_note_prob,
        accent_prob=accent_prob,
        shuffle_amount=shuffle_amount,
        flamming_prob=flamming_prob,
        drummer_style=drummer_style,
        drum_library=drum_library,
        visualize=visualize
    )
    humanizer = DrumHumanizer(config)

    print(f"Loading MIDI file: {input_file}")
    try:
        midi_file = mido.MidiFile(input_file)
    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found")
        return
    except mido.MidiFile.IOError as e:
        print(f"Error: Invalid MIDI file - {str(e)}")
        return
    except Exception as e:
        print(f"Unexpected error loading MIDI file: {str(e)}")
        return

    # Create a new MIDI file with the same settings
    new_midi = mido.MidiFile(ticks_per_beat=midi_file.ticks_per_beat)
    humanizer.ticks_per_beat = midi_file.ticks_per_beat

    for track in midi_file.tracks:
        print(f"\nProcessing track {midi_file.tracks.index(track) + 1}/{len(midi_file.tracks)}")
        new_track = mido.MidiTrack()
        new_midi.tracks.append(new_track)

        # Process time signature and tempo metadata
        for msg in track:
            if msg.type == "time_signature":
                humanizer.time_sig_numerator = msg.numerator
                humanizer.time_sig_denominator = msg.denominator
                print(f"Found time signature: {msg.numerator}/{msg.denominator}")

            if msg.type == "set_tempo":
                bpm = 60000000 / msg.tempo
                print(f"Found tempo: {bpm:.1f} BPM")

        # Collect drum notes with absolute timing
        messages = []
        current_time = 0
        for msg in track:
            current_time += msg.time
            messages.append((current_time, msg))

        if not messages:
            continue

        # Analyze beat structure
        ticks_per_measure = (
            humanizer.ticks_per_beat * humanizer.time_sig_numerator * 
            (4 / humanizer.time_sig_denominator)
        )

        # Group notes by time to find patterns
        notes_by_time = defaultdict(list)
        for time, msg in messages:
            if msg.type == "note_on" and msg.velocity > 0:
                notes_by_time[time].append(msg)

        # Process each note
        humanized_notes = []
        last_tempo_update = 0

        for time, msg in messages:
            if msg.type == "note_on" and msg.velocity > 0:
                measure_position = (time % ticks_per_measure) / humanizer.ticks_per_beat
                measure_idx = int(time / ticks_per_measure)
                
                # Determine if in fill
                in_fill = any(start <= time <= end for start, end in humanizer.merged_fills)
                
                # Determine pattern point
                pattern_key = (
                    measure_idx % 2,
                    round(measure_position * 8 / humanizer.time_sig_denominator) 
                    / (8 / humanizer.time_sig_denominator)
                )
                is_pattern_point = False  # Implement pattern detection if needed

                # Update tempo drift
                if time - last_tempo_update > humanizer.ticks_per_beat:
                    drift_factor = humanizer.profile.groove_consistency
                    humanizer.tempo_drift = (
                        humanizer.tempo_drift * drift_factor + 
                        random.uniform(-3, 3) * (1 - drift_factor)
                    )
                    last_tempo_update = time

                # Apply humanization
                total_timing_var = humanizer.humanize_timings(
                    msg, time, notes_by_time, in_fill, 
                    is_pattern_point, pattern_key, measure_position
                )

                new_velocity = humanizer.humanize_velocity(
                    msg, time, in_fill, measure_position, measure_idx
                )

                # Create humanized note
                humanized_time = max(0, time + total_timing_var)
                humanized_note = msg.copy(velocity=new_velocity)
                humanized_notes.append((humanized_time, humanized_note))

                # Add note_off
                humanized_notes.append((
                    humanized_time + humanizer.ticks_per_beat // 8,
                    mido.Message(
                        "note_off",
                        channel=humanized_note.channel,
                        note=humanized_note.note,
                        velocity=0,
                        time=0
                    )
                ))
            elif msg.type != "note_off":
                humanized_notes.append((time, msg.copy()))

        # Sort and convert to relative timing
        humanized_notes.sort(key=lambda x: x[0])
        last_time = 0
        for abs_time, msg in humanized_notes:
            msg_copy = msg.copy(time=int(abs_time - last_time))
            new_track.append(msg_copy)
            last_time = abs_time

    # Save the humanized MIDI file
    print(f"Saving humanized MIDI to: {output_file}")
    new_midi.save(output_file)
    print("Done! Applied drummer style:", drummer_style)

    # Create visualization if requested
    if visualize:
        viz_output = str(Path(output_file).with_suffix(".png"))
        try:
            if not create_drum_visualization(messages, humanized_notes, viz_output):
                print("Make sure matplotlib is installed: pip install matplotlib")
        except ImportError:
            print("Could not create visualization - matplotlib not installed")
            print("Install matplotlib with: pip install matplotlib")


def create_visualization(original_messages, humanized_messages, output_path):
    """Create a simple visualization comparing original and humanized MIDI data.
    This is a legacy function kept for backwards compatibility.
    Use create_drum_visualization for new code.
    """
    return create_drum_visualization(original_messages, humanized_messages, output_path)


def create_drum_visualization(original_messages, humanized_messages, output_png):
    """Create a detailed visualization comparing original and humanized MIDI drum patterns.

    Features:
    - Color-coded notes by drum type (kicks, snares, hi-hats, toms, cymbals)
    - Velocity shown through note size
    - Grid lines aligned with beats
    - Comprehensive drum labels
    - Time axis in both ticks and beats
    """
    try:
        print(f"\nGenerating visualization: {output_png}")

        plt.style.use("dark_background")
        fig = plt.figure(figsize=(15, 12))
        gs = plt.GridSpec(3, 1, height_ratios=[5, 5, 1], hspace=0.3)
        ax1 = plt.subplot(gs[0])
        ax2 = plt.subplot(gs[1])
        ax_legend = plt.subplot(gs[2])

        fig.suptitle(
            "MIDI Drum Pattern Analysis\nOriginal vs. Humanized Pattern",
            fontsize=16,
            y=1.02,
        )

        def plot_notes(ax, messages, alpha=1.0, title=""):
            times = [t for t, m in messages if m.type == "note_on" and m.velocity > 0]
            notes = [
                m.note for t, m in messages if m.type == "note_on" and m.velocity > 0
            ]
            velocities = [
                m.velocity * 3
                for t, m in messages
                if m.type == "note_on" and m.velocity > 0
            ]

            if not times:
                return

            # Color coding and categorization
            drum_categories = {
                "Kicks": {"notes": (35, 36), "color": "#FF4444"},
                "Snares": {"notes": (37, 38, 39, 40), "color": "#44FF44"},
                "Hi-hats": {"notes": (42, 44, 46), "color": "#4444FF"},
                "Toms": {"notes": (41, 43, 45, 47, 48, 50), "color": "#FF44FF"},
                "Cymbals": {"notes": (49, 51, 52, 53, 55, 57, 59), "color": "#FFFF44"},
            }

            # Plot each category separately for legend
            for cat_name, cat_info in drum_categories.items():
                cat_times = []
                cat_notes = []
                cat_vels = []

                for t, n, v in zip(times, notes, velocities):
                    if any(n == note for note in cat_info["notes"]):
                        cat_times.append(t)
                        cat_notes.append(n)
                        cat_vels.append(v)

                if cat_times:
                    ax.scatter(
                        cat_times,
                        cat_notes,
                        alpha=alpha,
                        s=cat_vels,
                        c=cat_info["color"],
                        label=cat_name,
                    )

            # Grid lines for beats
            if times:
                ticks_per_beat = 480  # Standard MIDI resolution
                max_time = max(times)
                beat_lines = np.arange(0, max_time + ticks_per_beat, ticks_per_beat)
                ax.vlines(
                    beat_lines,
                    ax.get_ylim()[0],
                    ax.get_ylim()[1],
                    color="gray",
                    alpha=0.2,
                    linestyle="--",
                )

            ax.set_title(title)
            ax.set_ylabel("Drum Type")
            ax.grid(True, alpha=0.2)

            # Enhanced note labels
            note_names = {
                35: "Acoustic Bass Drum",
                36: "Bass Drum 1",
                38: "Acoustic Snare",
                40: "Electric Snare",
                42: "Closed Hi-Hat",
                44: "Pedal Hi-Hat",
                46: "Open Hi-Hat",
                41: "Low Floor Tom",
                43: "High Floor Tom",
                45: "Low Tom",
                47: "Low-Mid Tom",
                48: "Hi-Mid Tom",
                50: "High Tom",
                49: "Crash Cymbal 1",
                51: "Ride Cymbal 1",
                52: "Chinese Cymbal",
                53: "Ride Bell",
                55: "Splash Cymbal",
                57: "Crash Cymbal 2",
                59: "Ride Cymbal 2",
            }

            unique_notes = sorted(set(notes)) if notes else []
            ax.set_yticks(unique_notes)
            ax.set_yticklabels([note_names.get(n, f"Note {n}") for n in unique_notes])
            # Add beat numbers on x-axis
            if times:
                max_time = max(times)
                ticks_per_beat = 480  # Standard MIDI resolution
                n_beats = int(max_time / ticks_per_beat) + 1
                tick_positions = [i * ticks_per_beat for i in range(n_beats + 1)]
                ax.set_xticks(tick_positions)
                ax.set_xticklabels([str(i) for i in range(n_beats + 1)])
                ax.set_xlabel("Beats")

        # Plot original and humanized patterns
        plot_notes(ax1, original_messages, alpha=0.8, title="Original Pattern")
        plot_notes(ax2, humanized_messages, alpha=0.8, title="Humanized Pattern")

        # Remove legend axes and just use it for shared legend
        ax_legend.axis("off")
        handles = []
        labels = []
        for ax in [ax1, ax2]:
            h, l = ax.get_legend_handles_labels()
            for hi, li in zip(h, l):
                if li not in labels:
                    handles.append(hi)
                    labels.append(li)

        ax_legend.legend(
            handles,
            labels,
            loc="center",
            ncol=5,
            bbox_to_anchor=(0.5, 0.5),
            fontsize=10,
        )

        plt.savefig(output_png, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Visualization saved to: {output_png}")
        return True

    except Exception as e:
        print(f"Error creating visualization: {str(e)}")
        print(f"Error type: {e.__class__.__name__}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Humanize MIDI drum tracks with realistic drummer feel"
    )
    parser.add_argument("input_file", help="Input MIDI file path")
    parser.add_argument(
        "--output",
        "-o",
        help="Output MIDI file path (default: input_file_humanized.mid)",
    )
    parser.add_argument(
        "--timing",
        "-t",
        type=int,
        default=10,
        help="Timing variation in ticks (default: 10)",
    )
    parser.add_argument(
        "--velocity",
        "-v",
        type=int,
        default=15,
        help="Velocity variation (default: 15)",
    )
    parser.add_argument(
        "--ghost",
        "-g",
        type=float,
        default=0.1,
        help="Ghost note probability (default: 0.1)",
    )
    parser.add_argument(
        "--accent",
        "-a",
        type=float,
        default=0.2,
        help="Accent probability (default: 0.2)",
    )
    parser.add_argument(
        "--shuffle",
        "-s",
        type=float,
        default=0.0,
        help="Shuffle amount, 0.0-0.5 (default: 0.0)",
    )
    parser.add_argument(
        "--flams",
        "-f",
        type=float,
        default=0.05,
        help="Flam probability (default: 0.0)",
    )
    parser.add_argument(
        "--style",
        type=str,
        default="balanced",
        choices=["balanced", "jazzy", "rock", "precise", "loose", "modern_metal"],
        help="Drummer style profile (default: balanced)",
    )
    parser.add_argument(
        "--library",
        type=str,
        default="gm",
        choices=["gm", "ad2", "sd3", "ez2", "ssd5", "mtpk2"],
        help="Drums library mapping (default: gm)",
    )
    parser.add_argument(
        "--rudiments",
        action="store_true",
        help="Enable automatic rudiment detection and application",
    )
    parser.add_argument(
        "--rudiment-intensity",
        type=float,
        default=0.5,
        help="Intensity of rudiment application (0.0-1.0)",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Generate visualization comparing original and humanized MIDI",
    )

    args = parser.parse_args()

    # Set default output filename if not provided
    if not args.output:
        input_path = Path(args.input_file)
        args.output = str(input_path.with_stem(f"{input_path.stem}_humanized"))

    humanize_drums(
        args.input_file,
        args.output,
        timing_variation=args.timing,
        velocity_variation=args.velocity,
        ghost_note_prob=args.ghost,
        accent_prob=args.accent,
        shuffle_amount=args.shuffle,
        flamming_prob=args.flams,
        drummer_style=args.style,
        drum_library=args.library,
        visualize=args.visualize,
    )


if __name__ == "__main__":
    main()

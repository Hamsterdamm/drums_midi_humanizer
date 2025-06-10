"""Utility functions for MIDI drum humanization."""

from typing import Dict, List, Set, Tuple
import mido


def calculate_measure_position(time: int, ticks_per_beat: int, time_sig_numerator: int = 4) -> float:
    """Calculate the position within a measure (0.0 to time_sig_numerator).

    Args:
        time: Current MIDI time in ticks
        ticks_per_beat: MIDI file's ticks per quarter note
        time_sig_numerator: Time signature numerator (e.g., 4 for 4/4)

    Returns:
        Position within the measure as a float from 0.0 to time_sig_numerator
    """
    beats = time / ticks_per_beat
    return beats % time_sig_numerator


def detect_rudiment_pattern(notes: List[Tuple[int, int, int]], pattern: List[str], timing_ratios: List[float], 
                          ticks_per_beat: int, tolerance: float = 0.1) -> bool:
    """Check if a sequence of notes matches a rudiment pattern.

    Args:
        notes: List of (time, note, velocity) tuples
        pattern: List of hand patterns ('R', 'L', 'r', 'l' where lowercase indicates grace notes)
        timing_ratios: List of relative timing ratios between notes
        ticks_per_beat: MIDI file's ticks per quarter note
        tolerance: Timing tolerance for pattern matching (0.0-1.0)

    Returns:
        True if the pattern matches, False otherwise
    """
    if len(notes) != len(pattern):
        return False

    # Convert timing ratios to ticks
    expected_times = [0]
    total_ratio = sum(timing_ratios)
    for ratio in timing_ratios[:-1]:
        expected_times.append(expected_times[-1] + (ratio / total_ratio) * ticks_per_beat)

    # Compare actual note timings with expected timings
    first_time = notes[0][0]
    for i in range(1, len(notes)):
        actual_delta = notes[i][0] - first_time
        expected_delta = expected_times[i]
        if abs(actual_delta - expected_delta) > (tolerance * ticks_per_beat):
            return False

    return True


def get_note_groups(drum_map: Dict[int, str]) -> Tuple[Set[int], Set[int], Set[int], Set[int], Set[int]]:
    """Group drum notes by type based on the provided drum map.

    Args:
        drum_map: Dictionary mapping MIDI note numbers to drum names

    Returns:
        Tuple of Sets containing MIDI note numbers for:
        (kick notes, snare notes, hihat notes, tom notes, cymbal notes)
    """
    kick_notes = set()
    snare_notes = set()
    hihat_notes = set()
    tom_notes = set()
    cymbal_notes = set()

    for note, name in drum_map.items():
        name_lower = name.lower()
        if "kick" in name_lower or "bass drum" in name_lower:
            kick_notes.add(note)
        elif "snare" in name_lower:
            snare_notes.add(note)
        elif "hh" in name_lower or "hi-hat" in name_lower or "hihat" in name_lower:
            hihat_notes.add(note)
        elif "tom" in name_lower:
            tom_notes.add(note)
        elif any(x in name_lower for x in ["crash", "ride", "china", "splash"]):
            cymbal_notes.add(note)

    return kick_notes, snare_notes, hihat_notes, tom_notes, cymbal_notes


def get_absolute_times(track: mido.MidiTrack) -> List[Tuple[int, mido.Message]]:
    """Convert relative MIDI times to absolute times.

    Args:
        track: MIDI track with relative timing

    Returns:
        List of tuples containing (absolute time, MIDI message)
    """
    messages = []
    current_time = 0

    for msg in track:
        current_time += msg.time
        messages.append((current_time, msg))

    return messages


def convert_to_relative_times(
    messages: List[Tuple[int, mido.Message]]
) -> List[Tuple[int, mido.Message]]:
    """Convert absolute MIDI times back to relative times.

    Args:
        messages: List of (absolute time, message) tuples

    Returns:
        List of (relative time, message) tuples
    """
    sorted_messages = sorted(messages, key=lambda x: x[0])
    relative_messages = []
    last_time = 0

    for abs_time, msg in sorted_messages:
        relative_time = abs_time - last_time
        relative_messages.append((relative_time, msg))
        last_time = abs_time

    return relative_messages


def detect_fills(
    notes_by_time: Dict[int, List[mido.Message]],
    primary_subdivision: int,
    tom_notes: Set[int]
) -> List[Tuple[int, int]]:
    """Detect fill sections in the MIDI data based on tom patterns.

    Args:
        notes_by_time: Dictionary mapping time to list of MIDI messages
        primary_subdivision: Basic rhythmic subdivision in ticks
        tom_notes: Set of MIDI note numbers representing toms

    Returns:
        List of (start_time, end_time) tuples representing fills
    """
    fills = []
    times = sorted(notes_by_time.keys())

    for i in range(len(times) - 1):
        curr_time = times[i]
        next_time = times[i + 1]
        notes_at_curr = [msg.note for msg in notes_by_time[curr_time]]

        # Check for multiple toms in short succession
        if (
            any(note in tom_notes for note in notes_at_curr)
            and next_time - curr_time < primary_subdivision
        ):
            # Mark region as potential fill
            fill_start = max(0, curr_time - primary_subdivision * 2)
            fill_end = min(
                curr_time + primary_subdivision * 8,
                times[-1] if times else 0
            )
            fills.append((fill_start, fill_end))

    return merge_overlapping_fills(fills)


def merge_overlapping_fills(fills: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """Merge overlapping fill regions.

    Args:
        fills: List of (start_time, end_time) tuples

    Returns:
        List of merged (start_time, end_time) tuples
    """
    if not fills:
        return []

    merged = []
    for fill in sorted(fills):
        if not merged or fill[0] > merged[-1][1]:
            merged.append(fill)
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], fill[1]))

    return merged


def calculate_measure_position(time: int, ticks_per_beat: int, time_sig_numerator: int = 4) -> float:
    """Calculate the position within a measure.

    Args:
        time: Time in ticks.
        ticks_per_beat: Number of ticks per beat.
        time_sig_numerator: Time signature numerator (default: 4).

    Returns:
        Position within the measure as a float between 0 and time_sig_numerator.
    """
    beats_per_measure = float(time_sig_numerator)
    measure_length_ticks = ticks_per_beat * beats_per_measure
    position = (time % measure_length_ticks) / ticks_per_beat
    return position

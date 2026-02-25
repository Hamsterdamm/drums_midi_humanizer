"""Utility functions for MIDI drum humanization.

This module provides helper functions for processing MIDI data, including
time conversion, note grouping, pattern detection, and fill identification.
These utilities support the core humanization logic by abstracting common
MIDI operations.
"""

from typing import Dict, List, Set, Tuple
import mido


def calculate_measure_position(
    time: int, ticks_per_beat: int, time_sig_numerator: int = 4
) -> float:
    """Calculate the beat position within a measure.

    Determines where a specific timestamp falls within a measure, expressed
    as a beat index (e.g., 0.0 for the start of the measure, 1.0 for the
    second beat in 4/4).

    Args:
        time (int): Current MIDI time in ticks.
        ticks_per_beat (int): The MIDI file's resolution (ticks per quarter note).
        time_sig_numerator (int): The numerator of the time signature (default: 4).

    Returns:
        float: The beat position within the measure (0.0 <= position < time_sig_numerator).
    """
    beats = time / ticks_per_beat
    return beats % time_sig_numerator


def detect_rudiment_pattern(
    notes: List[Tuple[int, int, int]],
    pattern: List[str],
    timing_ratios: List[float],
    ticks_per_beat: int,
    tolerance: float = 0.1,
) -> bool:
    """Check if a sequence of notes matches a specific rudiment pattern.

    Analyzes a sequence of notes to see if their relative timing matches
    a predefined rhythmic pattern (e.g., a flam or drag), within a given tolerance.

    Args:
        notes (List[Tuple[int, int, int]]): List of (time, note, velocity) tuples.
        pattern (List[str]): List of hand patterns (unused in logic but kept for signature).
        timing_ratios (List[float]): Expected relative timing ratios between notes.
        ticks_per_beat (int): The MIDI file's resolution.
        tolerance (float): Maximum allowed deviation in beats (default: 0.1).

    Returns:
        bool: True if the notes match the timing pattern, False otherwise.
    """
    if len(notes) != len(pattern):
        return False

    # Convert timing ratios to absolute tick offsets relative to the start
    expected_times = [0.0]
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


def get_note_groups(
    drum_map: Dict[int, str],
) -> Tuple[Set[int], Set[int], Set[int], Set[int], Set[int]]:
    """Group drum notes by instrument category based on their names.

    Parses the drum map names to categorize MIDI note numbers into Kicks,
    Snares, Hi-hats, Toms, and Cymbals. This allows the humanizer to apply
    instrument-specific logic.

    Args:
        drum_map (Dict[int, str]): Dictionary mapping MIDI note numbers to drum names.

    Returns:
        Tuple[Set[int], ...]: A tuple containing five sets of note numbers:
            (kick_notes, snare_notes, hihat_notes, tom_notes, cymbal_notes).
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
    """Convert a MIDI track with relative delta times to absolute timestamps.

    Mido tracks store events with 'time' representing the delta since the
    last event. This function calculates the cumulative absolute time for
    easier processing.

    Args:
        track (mido.MidiTrack): The input MIDI track with relative timing.

    Returns:
        List[Tuple[int, mido.Message]]: A list of tuples where each contains
            the absolute timestamp (in ticks) and the MIDI message.
    """
    messages = []
    current_time = 0

    for msg in track:
        current_time += msg.time
        messages.append((current_time, msg))

    return messages


def convert_to_relative_times(
    messages: List[Tuple[int, mido.Message]],
) -> List[Tuple[int, mido.Message]]:
    """Convert absolute timestamps back to relative delta times for MIDI export.

    Reverses the operation of `get_absolute_times`, preparing the messages
    to be written back to a MIDI track.

    Args:
        messages (List[Tuple[int, mido.Message]]): List of (absolute_time, message) tuples.

    Returns:
        List[Tuple[int, mido.Message]]: List of (delta_time, message) tuples,
            sorted by time.
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
    notes_by_time: Dict[int, List[mido.Message]], primary_subdivision: int, tom_notes: Set[int]
) -> List[Tuple[int, int]]:
    """Detect potential drum fill sections based on tom activity.

    Identifies sections where tom hits occur in rapid succession, which
    often indicates a drum fill.

    Args:
        notes_by_time (Dict[int, List[mido.Message]]): Dictionary mapping absolute
            timestamps to lists of MIDI messages occurring at that time.
        primary_subdivision (int): The tick duration of a primary subdivision.
        tom_notes (Set[int]): Set of MIDI note numbers identified as toms.

    Returns:
        List[Tuple[int, int]]: A list of (start_time, end_time) tuples defining
            the detected fill regions.
    """
    fills = []
    times = sorted(notes_by_time.keys())

    for i in range(len(times) - 1):
        curr_time = times[i]
        next_time = times[i + 1]
        notes_at_curr = [msg.note for msg in notes_by_time[curr_time] if hasattr(msg, "note")]

        # Check for multiple toms in short succession
        if (
            any(note in tom_notes for note in notes_at_curr)
            and next_time - curr_time < primary_subdivision
        ):
            # Mark region as potential fill
            fill_start = max(0, curr_time - primary_subdivision * 2)
            fill_end = min(curr_time + primary_subdivision * 8, times[-1] if times else 0)
            fills.append((fill_start, fill_end))

    return merge_overlapping_fills(fills)


def merge_overlapping_fills(fills: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """Merge overlapping or adjacent fill regions into continuous blocks.

    Args:
        fills (List[Tuple[int, int]]): List of (start_time, end_time) tuples.

    Returns:
        List[Tuple[int, int]]: A sorted list of merged time ranges.
    """
    if not fills:
        return []

    # Sort by start time
    sorted_fills = sorted(fills, key=lambda x: x[0])
    merged = []

    for fill in sorted_fills:
        if not merged or fill[0] > merged[-1][1]:
            merged.append(fill)
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], fill[1]))

    return merged

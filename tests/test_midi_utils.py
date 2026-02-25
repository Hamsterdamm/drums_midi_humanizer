"""Tests for MIDI utility functions."""

import mido

from drums_midi_humanizer.utils.midi import (
    calculate_measure_position,
    convert_to_relative_times,
    detect_fills,
    detect_rudiment_pattern,
    get_absolute_times,
)


def test_calculate_measure_position():
    """Test beat position calculation."""
    # 480 ticks per beat, 4/4 time
    assert calculate_measure_position(0, 480, 4) == 0.0
    assert calculate_measure_position(480, 480, 4) == 1.0
    assert calculate_measure_position(240, 480, 4) == 0.5
    assert calculate_measure_position(1920, 480, 4) == 0.0  # Start of next measure


def test_detect_rudiment_pattern():
    """Test rudiment pattern detection logic."""
    # Simple pattern check (e.g., two even hits)
    notes = [(0, 60, 100), (240, 60, 100)]
    pattern = ["R", "L"]
    timing_ratios = [1, 1]
    ticks_per_beat = 480

    # Should match perfect timing
    assert detect_rudiment_pattern(notes, pattern, timing_ratios, ticks_per_beat) is True

    # Should fail on wrong length
    assert detect_rudiment_pattern(notes[:1], pattern, timing_ratios, ticks_per_beat) is False

    # Should match within tolerance (20 ticks off, tolerance is 48)
    bad_notes = [(0, 60, 100), (260, 60, 100)]
    assert detect_rudiment_pattern(bad_notes, pattern, timing_ratios, ticks_per_beat) is True

    # Should fail outside tolerance (120 ticks off)
    really_bad_notes = [(0, 60, 100), (360, 60, 100)]
    assert (
        detect_rudiment_pattern(really_bad_notes, pattern, timing_ratios, ticks_per_beat) is False
    )


def test_time_conversion():
    """Test absolute/relative time conversion roundtrip."""
    track = mido.MidiTrack()
    track.append(mido.Message("note_on", note=60, time=0))
    track.append(mido.Message("note_off", note=60, time=480))
    track.append(mido.Message("note_on", note=62, time=0))  # Simultaneous with note_off

    abs_times = get_absolute_times(track)
    assert abs_times[0][0] == 0
    assert abs_times[1][0] == 480
    assert abs_times[2][0] == 480

    rel_times = convert_to_relative_times(abs_times)
    assert rel_times[0][1].time == 0
    assert rel_times[1][1].time == 480
    assert rel_times[2][1].time == 0


def test_detect_fills():
    """Test fill detection based on tom activity."""
    # Create messages: 4 tom hits in quick succession
    tom_note = 45
    notes_by_time = {i * 100: [mido.Message("note_on", note=tom_note)] for i in range(4)}
    tom_notes = {tom_note}

    fills = detect_fills(notes_by_time, 480, tom_notes)
    assert len(fills) > 0

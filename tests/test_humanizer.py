"""Tests for the MIDI drum humanizer."""

from pathlib import Path
from unittest.mock import patch

import mido
import pytest

from drums_midi_humanizer.core.humanizer import DrumHumanizer, HumanizerConfig
from drums_midi_humanizer.utils.midi import get_note_groups, merge_overlapping_fills


@pytest.fixture
def basic_config():
    """Create a basic humanizer configuration for testing."""
    return HumanizerConfig(
        timing_variation=10,
        velocity_variation=15,
        ghost_note_prob=0.1,
        accent_prob=0.2,
        shuffle_amount=0.0,
        flamming_prob=0.05,
        drummer_style="balanced",
        drum_library="gm",
        visualize=False,
    )


@pytest.fixture
def humanizer(basic_config):
    """Create a DrumHumanizer instance for testing."""
    return DrumHumanizer(basic_config)


def test_note_groups():
    """Test the note grouping functionality."""
    test_map = {36: "Kick", 38: "Snare", 42: "Hi-hat Closed", 45: "Tom", 49: "Crash"}
    kicks, snares, hihats, toms, cymbals = get_note_groups(test_map)

    assert 36 in kicks
    assert 38 in snares
    assert 42 in hihats
    assert 45 in toms
    assert 49 in cymbals


def test_merge_fills():
    """Test merging of overlapping fill sections."""
    fills = [(0, 100), (50, 150), (200, 300)]
    merged = merge_overlapping_fills(fills)

    assert len(merged) == 2
    assert merged[0] == (0, 150)
    assert merged[1] == (200, 300)


def test_humanizer_initialization(humanizer):
    """Test DrumHumanizer initialization."""
    assert humanizer.config.timing_variation == 10
    assert humanizer.config.velocity_variation == 15
    assert humanizer.time_sig_numerator == 4
    assert humanizer.time_sig_denominator == 4


def test_invalid_config():
    """Test invalid configuration values."""
    with pytest.raises(ValueError):
        HumanizerConfig(
            timing_variation=10,
            velocity_variation=15,
            ghost_note_prob=1.5,  # Invalid: should be <= 1
            accent_prob=0.2,
            shuffle_amount=0.0,
            flamming_prob=0.05,
            drummer_style="balanced",
            drum_library="gm",
            visualize=False,
        )


def test_humanize_timings(humanizer):
    """Test advanced timing variation logic."""
    msg = mido.Message("note_on", note=36, velocity=100)  # Kick
    time = 480
    notes_by_time = {480: [msg]}
    in_fill = False
    is_pattern_point = False
    pattern_key = None
    measure_position = 0.0  # Downbeat

    # Set required attributes usually set in process_file
    humanizer.ticks_per_beat = 480

    # 1. Test with zero variation
    humanizer.config.timing_variation = 0
    humanizer.profile.rushing_factor = 0
    humanizer.tempo_drift = 0

    offset = humanizer.humanize_timings(
        msg, time, notes_by_time, in_fill, is_pattern_point, pattern_key, measure_position
    )
    assert offset == 0

    # 2. Test with variation enabled
    humanizer.config.timing_variation = 10
    offset = humanizer.humanize_timings(
        msg, time, notes_by_time, in_fill, is_pattern_point, pattern_key, measure_position
    )
    assert isinstance(offset, int)
    # Check range (max_var is timing_variation * 2)
    assert -20 <= offset <= 20


def test_humanize_velocity(humanizer):
    """Test advanced velocity variation logic."""
    msg = mido.Message("note_on", note=38, velocity=100)  # Snare
    time = 480
    in_fill = False
    measure_position = 1.0  # Backbeat
    measure_idx = 0

    # Set required attributes
    humanizer.ticks_per_beat = 480

    # 1. Test range
    humanizer.config.velocity_variation = 10
    humanizer.config.accent_prob = 0.0

    new_vel = humanizer.humanize_velocity(msg, time, in_fill, measure_position, measure_idx)
    assert 1 <= new_vel <= 127


def test_humanize_track_integration(humanizer):
    """Test the full track humanization flow."""
    track = mido.MidiTrack()
    # Add a kick drum
    track.append(mido.Message("note_on", note=36, velocity=100, time=0))
    track.append(mido.Message("note_off", note=36, velocity=0, time=480))

    humanizer.ticks_per_beat = 480

    new_track, orig_msgs, human_msgs = humanizer._humanize_track(track)

    # Should have note_on and note_off
    assert len(new_track) >= 2
    assert len(orig_msgs) == 1
    assert len(human_msgs) == 1

"""Tests for the MIDI drum humanizer."""

import pytest
from pathlib import Path
import mido
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
        visualize=False
    )


@pytest.fixture
def humanizer(basic_config):
    """Create a DrumHumanizer instance for testing."""
    return DrumHumanizer(basic_config)


def test_note_groups():
    """Test the note grouping functionality."""
    test_map = {
        36: "Kick",
        38: "Snare",
        42: "Hi-hat Closed",
        45: "Tom",
        49: "Crash"
    }
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
            visualize=False
        )

import pytest

from drums_midi_humanizer.config.drums import (
    DRUMMER_PROFILES,
    DrumMap,
    get_drum_map,
)


def test_drum_map_initialization():
    """Test that DrumMap initializes correctly and groups notes."""
    dm = DrumMap(
        kick_notes={36}, snare_notes={38}, hihat_notes={42}, tom_notes={41}, cymbal_notes={49}, ride_notes={51}
    )

    assert 36 in dm.kick_notes
    assert 38 in dm.snare_notes

    groups = dm.get_note_groups()
    assert len(groups) == 6
    assert groups[0] == {36}  # Kick
    assert groups[1] == {38}  # Snare


def test_get_drum_map_valid():
    """Test retrieving a valid drum map."""
    gm_map = get_drum_map("gm")
    assert isinstance(gm_map, DrumMap)
    # Check a known GM mapping
    assert 36 in gm_map.kick_notes
    assert 38 in gm_map.snare_notes


def test_get_drum_map_invalid():
    """Test that invalid library names raise ValueError."""
    with pytest.raises(ValueError):
        get_drum_map("nonexistent_library")


def test_drummer_profiles_structure():
    """Test that drummer profiles are correctly defined dictionaries."""
    assert "balanced" in DRUMMER_PROFILES
    profile = DRUMMER_PROFILES["balanced"]
    assert "timing_bias" in profile
    assert "velocity_emphasis" in profile
    assert isinstance(profile["timing_bias"], (int, float))

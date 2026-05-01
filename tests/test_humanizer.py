"""Tests for the MIDI drum humanizer."""


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
    test_map = {36: "Kick", 38: "Snare", 42: "Hi-hat Closed", 45: "Tom", 49: "Crash", 51: "Ride"}
    kicks, snares, hihats, toms, cymbals, rides = get_note_groups(test_map)

    assert 36 in kicks
    assert 38 in snares
    assert 42 in hihats
    assert 45 in toms
    assert 49 in cymbals
    assert 51 in rides


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
        msg, time, notes_by_time, in_fill, is_pattern_point, pattern_key, measure_position, 1.0, 4
    )
    assert offset == 0

    # 2. Test with variation enabled
    humanizer.config.timing_variation = 10
    offset = humanizer.humanize_timings(
        msg, time, notes_by_time, in_fill, is_pattern_point, pattern_key, measure_position, 1.0, 4
    )
    assert isinstance(offset, float)
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

    new_vel = humanizer.humanize_velocity(msg, time, in_fill, measure_position, measure_idx, 4)
    assert 1 <= new_vel <= 127


def test_humanize_track_integration(humanizer):
    """Test the full track humanization flow."""
    track = mido.MidiTrack()
    # Add a kick drum
    track.append(mido.Message("note_on", note=36, velocity=100, time=0))
    track.append(mido.Message("note_off", note=36, velocity=0, time=480))

    humanizer.ticks_per_beat = 480
    humanizer.config.ghost_note_prob = 0.0  # Disable ghost notes to avoid unpredictable note counts
    from unittest.mock import MagicMock
    humanizer.timeline = MagicMock()
    humanizer.timeline.get_measure_info.return_value = (0.0, 1920, 0, 4)
    humanizer.timeline.get_tempo_multiplier.return_value = 1.0

    new_track, orig_msgs, human_msgs = humanizer._humanize_track(track)

    # Should have note_on and note_off
    assert len(new_track) >= 2
    assert len(orig_msgs) == 1
    assert len(human_msgs) == 1


def test_rudiment_integration(humanizer):
    """Test that a sequence matching a rudiment triggers rudiment logic."""
    track = mido.MidiTrack()
    # Create a flam tap sequence: 4 notes on snare (38) with uniform timing
    # mido track times are relative (delta)
    # We want notes at absolute times 0, 100, 200, 300
    track.append(mido.Message("note_on", note=38, velocity=100, time=0))
    track.append(mido.Message("note_off", note=38, velocity=0, time=100))
    track.append(mido.Message("note_on", note=38, velocity=100, time=0))
    track.append(mido.Message("note_off", note=38, velocity=0, time=100))
    track.append(mido.Message("note_on", note=38, velocity=100, time=0))
    track.append(mido.Message("note_off", note=38, velocity=0, time=100))
    track.append(mido.Message("note_on", note=38, velocity=100, time=0))
    track.append(mido.Message("note_off", note=38, velocity=0, time=100))

    humanizer.ticks_per_beat = 480
    humanizer.profile.rudiment_sensitivity = 1.0  # Force rudiment detection
    from unittest.mock import MagicMock
    humanizer.timeline = MagicMock()
    humanizer.timeline.get_measure_info.return_value = (0.0, 1920, 0, 4)
    humanizer.timeline.get_tempo_multiplier.return_value = 1.0
    
    # We need a fixed variation to safely check values
    humanizer.config.velocity_variation = 0
    humanizer.profile.velocity_emphasis = 0
    
    # Run humanize track
    _, _, human_msgs = humanizer._humanize_track(track)
    
    # With velocity_variation = 0, new_velocity before rudiment is exactly original (100)
    # For flam tap (4 notes), velocity_ratios are usually [1.0, 0.8, 1.0, 0.8]
    # Blend factor is 0.5. So 100 * (0.5) + (100 * ratio) * 0.5
    # For ratio 1.0: 50 + 50 = 100
    # For ratio 0.8: 50 + 40 = 90
    
    # Check if any humanized message has velocity roughly around 90 (meaning rudiment was mixed)
    velocities = [msg[2] for msg in human_msgs]
    
    # With accent probability there could be some variation, let's force accent_prob to 0
    humanizer.config.accent_prob = 0.0
    # Re-run after zeroing accent probability to ensure perfectly predictable outcome
    _, _, human_msgs = humanizer._humanize_track(track)
    velocities = [msg[2] for msg in human_msgs]
    
    # Note: If no other ghost note or accent variations were applied, we should see 90 
    assert 90 in velocities or 91 in velocities, f"Rudiment velocity blending not applied. Velocities: {velocities}"


def test_crash_kick_alignment(humanizer):
    """Test that a crash and kick on the same tick share the same timing offset."""
    track = mido.MidiTrack()
    # Add a Kick and a Crash at the exact same time (0)
    # Time in Mido tracks is relative.
    track.append(mido.Message("note_on", note=36, velocity=100, time=0))  # Kick
    track.append(mido.Message("note_on", note=49, velocity=100, time=0))  # Crash
    track.append(mido.Message("note_off", note=36, velocity=0, time=480))
    track.append(mido.Message("note_off", note=49, velocity=0, time=0))

    humanizer.ticks_per_beat = 480
    humanizer.config.ghost_note_prob = 0.0  # Disable ghost notes

    from unittest.mock import MagicMock
    humanizer.timeline = MagicMock()
    humanizer.timeline.get_measure_info.return_value = (0.0, 1920, 0, 4)
    humanizer.timeline.get_tempo_multiplier.return_value = 1.0

    # Ensure significant timing variation to verify they don't randomly diverge
    humanizer.config.timing_variation = 50

    _, _, human_msgs = humanizer._humanize_track(track)

    # Find the humanized times for the Kick and the Crash
    kick_times = [msg[0] for msg in human_msgs if msg[1] == 36]
    crash_times = [msg[0] for msg in human_msgs if msg[1] == 49]

    assert len(kick_times) == 1
    assert len(crash_times) == 1

    # The absolute humanized time should be exactly identical
    assert kick_times[0] == crash_times[0], f"Crash time {crash_times[0]} did not align with Kick time {kick_times[0]}"


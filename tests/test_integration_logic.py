import os
import pytest
import mido
from mido import Message, MidiFile, MidiTrack, MetaMessage
from drums_midi_humanizer.core.humanizer import DrumHumanizer, HumanizerConfig
from drums_midi_humanizer.config.drums import DRUMMER_PROFILES

# -----------------------------------------------------------------------------
# Fixtures & Helpers
# -----------------------------------------------------------------------------

@pytest.fixture
def temp_midi_files(tmp_path):
    """Creates paths for input and output MIDI files."""
    input_file = tmp_path / "test_input.mid"
    output_file = tmp_path / "test_output.mid"
    return str(input_file), str(output_file)

def create_quantized_beat(filename, length_bars=1):
    """
    Creates a simple quantized 4/4 beat (Kick on 1/3, Snare on 2/4).
    480 ticks per beat.
    """
    mid = MidiFile()
    track = MidiTrack()
    mid.tracks.append(track)
    
    # Meta messages
    track.append(MetaMessage('set_tempo', tempo=500000, time=0))
    
    # Note: The current implementation of DrumHumanizer assumes 4/4 time (hardcoded).
    # We set the time signature here to match, ensuring the test validates the logic
    # within its current limitations.
    track.append(MetaMessage('time_signature', numerator=4, denominator=4, time=0))
    
    # Simple Rock Beat: Kick (36), Snare (38), HiHat (42)
    # 1 bar = 4 beats
    for _ in range(length_bars):
        for beat in range(4):
            # Beat 1: Kick + HH
            if beat == 0:
                track.append(Message('note_on', note=36, velocity=100, time=0)) # Kick
                track.append(Message('note_on', note=42, velocity=90, time=0))  # HH
                track.append(Message('note_off', note=36, velocity=0, time=100))
                track.append(Message('note_off', note=42, velocity=0, time=0))
                # Remaining time in beat: 480 - 100 = 380
                track.append(MetaMessage('marker', text='beat_end', time=380))

            # Beat 2: Snare + HH
            elif beat == 1:
                track.append(Message('note_on', note=38, velocity=100, time=0)) # Snare
                track.append(Message('note_on', note=42, velocity=90, time=0))  # HH
                track.append(Message('note_off', note=38, velocity=0, time=100))
                track.append(Message('note_off', note=42, velocity=0, time=0))
                track.append(MetaMessage('marker', text='beat_end', time=380))

            # Beat 3: Kick + HH
            elif beat == 2:
                track.append(Message('note_on', note=36, velocity=100, time=0))
                track.append(Message('note_on', note=42, velocity=90, time=0))
                track.append(Message('note_off', note=36, velocity=0, time=100))
                track.append(Message('note_off', note=42, velocity=0, time=0))
                track.append(MetaMessage('marker', text='beat_end', time=380))

            # Beat 4: Snare + HH
            elif beat == 3:
                track.append(Message('note_on', note=38, velocity=100, time=0))
                track.append(Message('note_on', note=42, velocity=90, time=0))
                track.append(Message('note_off', note=38, velocity=0, time=100))
                track.append(Message('note_off', note=42, velocity=0, time=0))
                track.append(MetaMessage('marker', text='beat_end', time=380))
                
    mid.save(filename)
    return mid

# -----------------------------------------------------------------------------
# Integration Tests
# -----------------------------------------------------------------------------

def test_end_to_end_processing(temp_midi_files):
    """
    Verifies that the DrumHumanizer can process a file and produce a valid output.
    """
    input_path, output_path = temp_midi_files
    create_quantized_beat(input_path)
    
    config = HumanizerConfig(
        timing_variation=10,
        velocity_variation=10,
        drummer_style='balanced',
        ghost_note_prob=0.1,
        accent_prob=0.1,
        shuffle_amount=0.0,
        flamming_prob=0.1,
        drum_library='gm',
        visualize=False
    )
    
    humanizer = DrumHumanizer(config)
    humanizer.process_file(input_path, output_path)
    
    assert os.path.exists(output_path), "Output file was not created"
    
    # Verify output is valid MIDI
    try:
        processed = MidiFile(output_path)
        assert len(processed.tracks) > 0
    except Exception as e:
        pytest.fail(f"Output file is not a valid MIDI file: {e}")

def test_velocity_humanization(temp_midi_files):
    """
    Verifies that velocities are altered from the original fixed values.
    """
    input_path, output_path = temp_midi_files
    create_quantized_beat(input_path)
    
    # High velocity variation to ensure we catch it
    config = HumanizerConfig(
        timing_variation=0, # Disable timing to isolate velocity
        velocity_variation=50,
        drummer_style='balanced',
        ghost_note_prob=0.0,
        accent_prob=0.0,
        shuffle_amount=0.0,
        flamming_prob=0.0,
        drum_library='gm',
        visualize=False
    )
    
    humanizer = DrumHumanizer(config)
    humanizer.process_file(input_path, output_path)
    
    processed = MidiFile(output_path)
    
    # Collect all note_on velocities
    velocities = []
    for track in processed.tracks:
        for msg in track:
            if msg.type == 'note_on' and msg.velocity > 0:
                velocities.append(msg.velocity)
                
    # Original velocities were 100 and 90.
    # We expect some deviation.
    exact_matches = sum(1 for v in velocities if v in [100, 90])
    assert exact_matches < len(velocities), "Velocities should have varied from original exact values"

def test_ghost_notes_generation(temp_midi_files):
    """
    Verifies that ghost notes are added when probability is high.
    """
    input_path, output_path = temp_midi_files
    create_quantized_beat(input_path, length_bars=2) # More bars = more chances
    
    config = HumanizerConfig(
        timing_variation=10,
        velocity_variation=10,
        ghost_note_prob=1.0, # Max probability
        drummer_style='balanced',
        accent_prob=0.1,
        shuffle_amount=0.0,
        flamming_prob=0.1,
        drum_library='gm',
        visualize=False
    )
    
    humanizer = DrumHumanizer(config)
    humanizer.process_file(input_path, output_path)
    
    original = MidiFile(input_path)
    processed = MidiFile(output_path)
    
    def count_notes(mid):
        count = 0
        for track in mid.tracks:
            for msg in track:
                if msg.type == 'note_on' and msg.velocity > 0:
                    count += 1
        return count
        
    orig_count = count_notes(original)
    proc_count = count_notes(processed)
    
    # We expect more notes in processed file due to ghost notes
    assert proc_count > orig_count, f"Expected ghost notes to increase note count (Original: {orig_count}, Processed: {proc_count})"

def test_configuration_parameters(temp_midi_files):
    """
    Verifies that different styles and parameters don't crash the application.
    """
    input_path, output_path = temp_midi_files
    create_quantized_beat(input_path)
    
    styles = list(DRUMMER_PROFILES.keys())
    
    for style in styles:
        config = HumanizerConfig(
            timing_variation=10,
            velocity_variation=10,
            drummer_style=style,
            ghost_note_prob=0.1,
            accent_prob=0.1,
            shuffle_amount=0.0,
            flamming_prob=0.1,
            drum_library='gm',
            visualize=False
        )
        humanizer = DrumHumanizer(config)
        # Should not raise exception
        humanizer.process_file(input_path, output_path)
        assert os.path.exists(output_path)

def test_styles_produce_different_timing(temp_midi_files):
    """
    Verifies that different styles produce different timing variations.
    """
    input_path, output_path = temp_midi_files
    output_path_loose = output_path.replace(".mid", "_loose.mid")
    
    # Create a longer beat for better statistical significance
    create_quantized_beat(input_path, length_bars=4)
    
    # 1. Process with 'tight' style
    config_tight = HumanizerConfig(
        timing_variation=20,
        velocity_variation=0,
        drummer_style='tight',
        ghost_note_prob=0.0,
        accent_prob=0.0,
        shuffle_amount=0.0,
        flamming_prob=0.0,
        drum_library='gm',
        visualize=False
    )
    
    humanizer_tight = DrumHumanizer(config_tight)
    humanizer_tight.process_file(input_path, output_path)
    
    # 2. Process with 'loose' style
    config_loose = HumanizerConfig(
        timing_variation=20,
        velocity_variation=0,
        drummer_style='loose',
        ghost_note_prob=0.0,
        accent_prob=0.0,
        shuffle_amount=0.0,
        flamming_prob=0.0,
        drum_library='gm',
        visualize=False
    )
    
    humanizer_loose = DrumHumanizer(config_loose)
    humanizer_loose.process_file(input_path, output_path_loose)
    
    # 3. Calculate deviations
    def get_avg_deviation(midi_path):
        mid = MidiFile(midi_path)
        total_dev = 0
        count = 0
        ticks_per_beat = 480 # Known from create_quantized_beat
        
        for track in mid.tracks:
            abs_time = 0
            for msg in track:
                abs_time += msg.time
                if msg.type == 'note_on' and msg.velocity > 0:
                    # Distance to nearest beat
                    dist = min(abs_time % ticks_per_beat, ticks_per_beat - (abs_time % ticks_per_beat))
                    total_dev += dist
                    count += 1
        return total_dev / count if count > 0 else 0

    dev_tight = get_avg_deviation(output_path)
    dev_loose = get_avg_deviation(output_path_loose)
    
    # Loose style should be "looser" (more deviation) than tight style
    assert dev_loose > dev_tight, f"Loose deviation ({dev_loose}) should be > Tight deviation ({dev_tight})"
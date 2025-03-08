import mido
import random
import argparse
import numpy as np
from pathlib import Path

def humanize_drums(input_file, output_file, 
                   timing_variation=10, 
                   velocity_variation=15, 
                   ghost_note_prob=0.1,
                   accent_prob=0.2,
                   shuffle_amount=0.0):
    """
    Add human feel to a MIDI drum track.
    
    Parameters:
    -----------
    input_file : str
        Path to input MIDI file
    output_file : str
        Path to save humanized MIDI file
    timing_variation : int
        Maximum timing variation in ticks (positive or negative)
    velocity_variation : int
        Maximum velocity variation (positive or negative)
    ghost_note_prob : float
        Probability of adding ghost notes (0.0 to 1.0)
    accent_prob : float
        Probability of accenting certain notes (0.0 to 1.0)
    shuffle_amount : float
        Amount of shuffle/swing feeling to add (0.0 to 0.5)
    """
    print(f"Loading MIDI file: {input_file}")
    midi_file = mido.MidiFile(input_file)
    
    # Create a new MIDI file with the same settings
    new_midi = mido.MidiFile(ticks_per_beat=midi_file.ticks_per_beat)
    
    # Process each track
    for track_idx, track in enumerate(midi_file.tracks):
        # Create a new track
        new_track = mido.MidiTrack()
        new_midi.tracks.append(new_track)
        
        # Copy metadata messages directly
        metadata_end_idx = 0
        for i, msg in enumerate(track):
            if msg.type in ['note_on', 'note_off']:
                metadata_end_idx = i
                break
            new_track.append(msg)
        
        # Collect drum notes for processing
        drum_notes = []
        current_time = 0
        
        for msg in track[metadata_end_idx:]:
            current_time += msg.time
            
            if msg.type in ['note_on', 'note_off']:
                # Store the note with its absolute time for processing
                drum_notes.append((current_time, msg))
            else:
                # Non-note messages pass through unchanged
                new_track.append(msg)
        
        # Skip track if no drum notes
        if not drum_notes:
            continue
        
        # Group notes by their timing to identify patterns
        grouped_notes = {}
        for time, msg in drum_notes:
            if msg.type == 'note_on' and msg.velocity > 0:
                if time not in grouped_notes:
                    grouped_notes[time] = []
                grouped_notes[time].append(msg)
        
        # Sort times to process notes chronologically
        sorted_times = sorted(grouped_notes.keys())
        
        # Identify common patterns (like hi-hat with kick or snare)
        beat_positions = []
        for i in range(len(sorted_times)-1):
            # Calculate distance to next note
            interval = sorted_times[i+1] - sorted_times[i]
            # Store beat position info
            notes = [msg.note for msg in grouped_notes[sorted_times[i]]]
            beat_positions.append((sorted_times[i], interval, notes))
        
        # Identify which notes are probably:
        # - Hi-hats (often highest frequency notes)
        # - Kicks (usually low notes, often on strong beats)
        # - Snares (usually mid notes, often on backbeats)
        all_notes = [msg.note for time, msg in drum_notes if msg.type == 'note_on' and msg.velocity > 0]
        note_counts = {}
        for note in all_notes:
            note_counts[note] = note_counts.get(note, 0) + 1
        
        # Common GM drum mapping as reference
        likely_hihat = max(note_counts.keys()) if note_counts else 42  # Default to hi-hat closed
        likely_kick = min(note_counts.keys()) if note_counts else 36   # Default to kick
        
        # Process and humanize drum notes
        humanized_notes = []
        
        for time, msg in drum_notes:
            if msg.type == 'note_on' and msg.velocity > 0:
                # Apply timing variation depending on note type
                timing_var = random.randint(-timing_variation, timing_variation)
                
                # Apply more shuffle to certain notes (like hi-hats)
                if shuffle_amount > 0 and msg.note == likely_hihat:
                    # Determine if this is an offbeat hi-hat
                    beat_pos = time / midi_file.ticks_per_beat
                    is_offbeat = abs((beat_pos % 1) - 0.5) < 0.1
                    
                    if is_offbeat:
                        # Push offbeats later for shuffle feel
                        shuffle_shift = int(shuffle_amount * midi_file.ticks_per_beat)
                        timing_var += shuffle_shift
                
                # Apply velocity variation based on note type
                new_velocity = msg.velocity
                
                # Accent certain notes
                if random.random() < accent_prob:
                    # Increase velocity for accents
                    new_velocity = min(127, new_velocity + random.randint(10, 25))
                else:
                    # Normal velocity variation
                    new_velocity += random.randint(-velocity_variation, velocity_variation)
                    new_velocity = max(1, min(127, new_velocity))
                
                # Occasionally add ghost notes for snares and toms
                if msg.note not in [likely_hihat, likely_kick] and random.random() < ghost_note_prob:
                    # Create a ghost note with lower velocity
                    ghost_velocity = max(1, msg.velocity - random.randint(40, 60))
                    ghost_time = time - random.randint(midi_file.ticks_per_beat//8, midi_file.ticks_per_beat//4)
                    if ghost_time > 0:
                        ghost_note = msg.copy(velocity=ghost_velocity)
                        humanized_notes.append((ghost_time, ghost_note))
                
                # Add the humanized note
                humanized_note = msg.copy(velocity=new_velocity)
                humanized_notes.append((time + timing_var, humanized_note))
            
            elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                # Also adjust note-off timing proportionally
                timing_var = random.randint(-timing_variation, timing_variation)
                humanized_notes.append((time + timing_var, msg))
        
        # Sort humanized notes by time
        humanized_notes.sort(key=lambda x: x[0])
        
        # Convert back to relative timing
        last_time = 0
        for abs_time, msg in humanized_notes:
            if abs_time < 0:
                abs_time = 0  # Ensure no negative times
            
            # Calculate relative time
            msg_copy = msg.copy(time=abs_time - last_time)
            new_track.append(msg_copy)
            last_time = abs_time
    
    # Save the humanized MIDI file
    print(f"Saving humanized MIDI to: {output_file}")
    new_midi.save(output_file)
    print("Done!")

def main():
    parser = argparse.ArgumentParser(description='Humanize MIDI drum tracks')
    parser.add_argument('input_file', help='Input MIDI file path')
    parser.add_argument('--output', '-o', help='Output MIDI file path (default: input_file_humanized.mid)')
    parser.add_argument('--timing', '-t', type=int, default=10, help='Timing variation in ticks (default: 10)')
    parser.add_argument('--velocity', '-v', type=int, default=15, help='Velocity variation (default: 15)')
    parser.add_argument('--ghost', '-g', type=float, default=0.1, help='Ghost note probability (default: 0.1)')
    parser.add_argument('--accent', '-a', type=float, default=0.2, help='Accent probability (default: 0.2)')
    parser.add_argument('--shuffle', '-s', type=float, default=0.0, help='Shuffle amount, 0.0-0.5 (default: 0.0)')
    
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
        shuffle_amount=args.shuffle
    )

if __name__ == "__main__":
    main()

import mido
import random
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict


# Set drummer profile characteristics
drummer_profiles = {
    "balanced": {
        "timing_bias": 0,           # Neutral timing
        "velocity_emphasis": 1.0,    # Normal dynamics
        "ghost_multiplier": 1.0,     # Standard ghost notes
        "kick_timing_tightness": 1.0, # Standard kick timing
        "hihat_variation": 1.0,      # Standard hi-hat variation
        "rushing_factor": 0,         # No tendency to rush or drag
        "groove_consistency": 0.7,   # Moderately consistent groove
    },
    "jazzy": {
        "timing_bias": -2,          # Slightly behind the beat
        "velocity_emphasis": 1.2,    # More dynamic range
        "ghost_multiplier": 1.5,     # More ghost notes
        "kick_timing_tightness": 0.7, # Looser kick timing
        "hihat_variation": 1.3,      # More hi-hat variation
        "rushing_factor": -0.3,      # Tendency to lay back
        "groove_consistency": 0.6,   # Less mechanical consistency
    },
    "rock": {
        "timing_bias": 2,           # Slightly ahead of the beat
        "velocity_emphasis": 1.3,    # Stronger dynamics
        "ghost_multiplier": 0.8,     # Fewer ghost notes
        "kick_timing_tightness": 1.2, # Tighter kick timing
        "hihat_variation": 0.9,      # Less hi-hat variation
        "rushing_factor": 0.2,       # Tendency to push forward
        "groove_consistency": 0.8,   # More consistent groove
    },
    "precise": {
        "timing_bias": 0,           # On the beat
        "velocity_emphasis": 0.8,    # Less dynamic range
        "ghost_multiplier": 0.5,     # Fewer ghost notes
        "kick_timing_tightness": 1.5, # Very tight kick timing
        "hihat_variation": 0.6,      # Minimal hi-hat variation
        "rushing_factor": 0,         # No tendency to rush or drag
        "groove_consistency": 0.9,   # Very consistent groove
    },
    "loose": {
        "timing_bias": 0,           # Variable timing
        "velocity_emphasis": 1.4,    # Wide dynamic range
        "ghost_multiplier": 1.7,     # Many ghost notes
        "kick_timing_tightness": 0.6, # Loose kick timing
        "hihat_variation": 1.5,      # Lots of hi-hat variation
        "rushing_factor": 0.1,       # Slight tendency to push
        "groove_consistency": 0.5,   # Inconsistent groove
    }
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

# Select the appropriate drum map based on the library
drum_maps = {
    "gm": GM_DRUM_MAP,
    "ad2": AD2_DRUM_MAP,
    "sd3": SD3_DRUM_MAP,
    "ez2": EZ2_DRUM_MAP,
    "ssd5": SSD5_DRUM_MAP
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


def humanize_drums(input_file, output_file, 
                   timing_variation=10, 
                   velocity_variation=15, 
                   ghost_note_prob=0.0,
                   accent_prob=0.2,
                   shuffle_amount=0.0,
                   flamming_prob=0.0,
                   drummer_style="balanced",
                   drum_library="gm"):
    """
    Add realistic human feel to a MIDI drum track based on drumming principles.
    
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
    flamming_prob : float
        Probability of adding flams to snare hits (0.0 to 1.0)
    drummer_style : str
        Style profile to apply ("balanced", "jazzy", "rock", "precise", "loose")
    drum_library : str
        Drum library to use for MIDI mapping ("gm", "ad2", "sd3", "ez2", "ssd5")
    """
    print(f"Loading MIDI file: {input_file}")
    try:
        midi_file = mido.MidiFile(input_file)
    except Exception as e:
        print(f"Error loading MIDI file: {e}")
        return
    
    # Create a new MIDI file with the same settings
    new_midi = mido.MidiFile(ticks_per_beat=midi_file.ticks_per_beat)
    
    # Get selected profile, default to balanced if not found
    profile = drummer_profiles.get(drummer_style.lower(), drummer_profiles["balanced"])
    
    # ----------------------------------
    # Drum Library MIDI Mappings
    # ----------------------------------
    
    selected_map = drum_maps.get(drum_library.lower(), GM_DRUM_MAP)
    print(f"Using drum library mapping: {drum_library.upper() if drum_library.lower() in drum_maps else 'General MIDI'}")
    
    KICK_NOTES, SNARE_NOTES, HIHAT_NOTES, TOM_NOTES, CYMBAL_NOTES = get_note_groups(selected_map)
    
    # Check if we found any drum notes for the selected mapping
    if not any([KICK_NOTES, SNARE_NOTES, HIHAT_NOTES, TOM_NOTES, CYMBAL_NOTES]):
        print(f"Warning: No drum notes identified for the selected library '{drum_library}'. Defaulting to General MIDI.")
        KICK_NOTES, SNARE_NOTES, HIHAT_NOTES, TOM_NOTES, CYMBAL_NOTES = get_note_groups(GM_DRUM_MAP)
    
    # Track time signature and tempo information
    time_sig_numerator = 4  # Default 4/4 time
    time_sig_denominator = 4
    current_tempo = 500000  # Default 120 BPM (microseconds per beat)
    
    # Process each track
    for track_idx, track in enumerate(midi_file.tracks):
        new_track = mido.MidiTrack()
        new_midi.tracks.append(new_track)
        
        # Look for time signature and tempo metadata
        for msg in track:
            if msg.type == 'time_signature':
                time_sig_numerator = msg.numerator
                time_sig_denominator = msg.denominator
                print(f"Found time signature: {time_sig_numerator}/{time_sig_denominator}")
            
            if msg.type == 'set_tempo':
                current_tempo = msg.tempo
                bpm = 60000000 / current_tempo
                print(f"Found tempo: {bpm:.1f} BPM")
        
        # Collect drum notes with absolute timing
        drum_notes = []
        current_time = 0
        
        for msg in track:
            current_time += msg.time
            
            if msg.type in ['note_on', 'note_off']:
                drum_notes.append((current_time, msg))
            else:
                # Copy metadata messages directly
                new_track.append(msg)
        
        # Skip if no drum notes
        if not drum_notes:
            continue
        
        # Analyze the beat structure
        ticks_per_beat = midi_file.ticks_per_beat
        ticks_per_measure = ticks_per_beat * time_sig_numerator * (4 / time_sig_denominator)
        
        # Group notes by their time to find patterns
        notes_by_time = defaultdict(list)
        for time, msg in drum_notes:
            if msg.type == 'note_on' and msg.velocity > 0:
                notes_by_time[time].append(msg)
        
        # Identify beat positions for each note
        beat_positions = []
        times = sorted(notes_by_time.keys())
        
        if not times:
            # No valid note_on events
            continue
            
        for time in times:
            # Calculate position within measure (0.0 to time_sig_numerator for given time signature)
            measure_position = (time % ticks_per_measure) / ticks_per_beat
            
            # Determine if it's a downbeat, backbeat or offbeat
            # Adjust for non-4/4 time signatures
            is_downbeat = measure_position < 0.1 or any(abs(measure_position - i) < 0.1 for i in range(2, time_sig_numerator, 2))
            is_backbeat = any(abs(measure_position - i) < 0.1 for i in range(1, time_sig_numerator, 2))
            is_offbeat = not (is_downbeat or is_backbeat)
            
            # Store beat info with the notes
            for note in notes_by_time[time]:
                beat_positions.append({
                    'time': time,
                    'note': note.note,
                    'velocity': note.velocity,
                    'measure_pos': measure_position,
                    'is_downbeat': is_downbeat,
                    'is_backbeat': is_backbeat,
                    'is_offbeat': is_offbeat
                })
        
        # Find tempo by looking at hi-hat pattern
        hihat_times = sorted([bp['time'] for bp in beat_positions if bp['note'] in HIHAT_NOTES])
        hihat_intervals = []
        for i in range(len(hihat_times) - 1):
            interval = hihat_times[i+1] - hihat_times[i]
            if interval > 0:
                hihat_intervals.append(interval)
        
        # Find most common interval (likely the primary subdivision)
        common_intervals = {}
        for interval in hihat_intervals:
            common_intervals[interval] = common_intervals.get(interval, 0) + 1
        
        primary_subdivision = ticks_per_beat/4  # Default to sixteenth notes
        if common_intervals:
            try:
                primary_subdivision = max(common_intervals.items(), key=lambda x: x[1])[0]
            except ValueError:
                # Handle cases where common_intervals might be empty
                pass
        
        # Detect fills by looking for dense tom patterns
        fills = []
        for i in range(len(times) - 1):
            curr_time = times[i]
            next_time = times[i+1]
            notes_at_curr = [msg.note for msg in notes_by_time[curr_time]]
            
            # Check if multiple toms are hit in short succession
            if any(note in TOM_NOTES for note in notes_at_curr) and next_time - curr_time < primary_subdivision:
                # Mark region as potential fill
                fill_start = max(0, curr_time - primary_subdivision * 2)
                fill_end = min(curr_time + primary_subdivision * 8, times[-1] if times else 0)
                fills.append((fill_start, fill_end))
        
        # Merge overlapping fill regions
        merged_fills = []
        for fill in sorted(fills):
            if not merged_fills or fill[0] > merged_fills[-1][1]:
                merged_fills.append(fill)
            else:
                merged_fills[-1] = (merged_fills[-1][0], max(merged_fills[-1][1], fill[1]))
        
        # Create groove pattern analysis
        groove_patterns = defaultdict(list)
        
        # Look for repeating kick/snare patterns
        for i in range(len(beat_positions)):
            bp = beat_positions[i]
            if bp['note'] in KICK_NOTES + SNARE_NOTES:
                measure_idx = int(bp['time'] / ticks_per_measure)
                pos_in_measure = bp['measure_pos']
                # Quantize to eighth notes by default, adjust based on time signature
                quantize_factor = 8 / time_sig_denominator
                pattern_key = (measure_idx % 2, round(pos_in_measure * quantize_factor) / quantize_factor)
                groove_patterns[pattern_key].append(bp)
        
        # Humanize based on analysis
        humanized_notes = []
        
        # Track tempo modifications for groove consistency
        tempo_drift = 0  # cumulative drift value
        last_tempo_update = 0  # when we last updated the drift
        
        for time, msg in drum_notes:
            if msg.type == 'note_on' and msg.velocity > 0:
                # Get measure position
                measure_position = (time % ticks_per_measure) / ticks_per_beat
                measure_idx = int(time / ticks_per_measure)
                
                # Determine if we're in a fill section
                in_fill = any(start <= time <= end for start, end in merged_fills)
                
                # Determine if this is a common pattern point
                pattern_key = (measure_idx % 2, round(measure_position * 8 / time_sig_denominator) / (8 / time_sig_denominator))
                is_pattern_point = pattern_key in groove_patterns
                
                # Tempo drift for groove consistency
                if time - last_tempo_update > ticks_per_beat:
                    # Update tempo drift periodically to simulate subtle tempo changes
                    drift_factor = profile["groove_consistency"]
                    tempo_drift = tempo_drift * drift_factor + random.uniform(-3, 3) * (1 - drift_factor)
                    last_tempo_update = time
                
                # ======= TIMING HUMANIZATION =======
                # Base timing variation based on note type and beat position
                note_type_timing_var = 0
                
                # Different timing handling based on drum type
                if msg.note in KICK_NOTES:
                    # Kicks should have more conservative timing adjustments
                    kick_tightness = profile["kick_timing_tightness"]
                    # Reduce the timing variation for kicks to prevent losses
                    base_var = timing_variation * 0.4 / kick_tightness  # Reduced from 0.6/0.8
                    
                    note_type_timing_var = random.uniform(-base_var, base_var)
                    
                    # Limit anticipation for downbeats
                    if measure_position < 0.1:
                        # Smaller negative adjustment to avoid pushing too early
                        note_type_timing_var -= min(2, 1 + profile["rushing_factor"])
                    
                elif msg.note in SNARE_NOTES:
                    # Snares on backbeats are fundamental, slightly different handling
                    if any(abs(measure_position - i) < 0.1 for i in range(1, time_sig_numerator, 2)):
                        # Backbeats might be slightly late for a relaxed feel or early for an energetic feel
                        note_type_timing_var = random.uniform(-timing_variation * 0.7, timing_variation * 0.7)
                        note_type_timing_var += profile["timing_bias"]
                    else:
                        # Other snares (often syncopations or fills)
                        note_type_timing_var = random.uniform(-timing_variation, timing_variation)
                
                elif msg.note in HIHAT_NOTES:
                    # Hi-hats have their own feel - experienced drummers often push/pull these
                    hihat_var = timing_variation * profile["hihat_variation"]
                    note_type_timing_var = random.uniform(-hihat_var, hihat_var)
                    
                    # Shuffle feel applied to offbeat hi-hats
                    # Calculate eighth note positions for shuffle
                    eighth_note_pos = round(measure_position * 8) / 8
                    is_offbeat_eighth = eighth_note_pos % 0.5 == 0 and eighth_note_pos % 1.0 != 0
                    
                    if shuffle_amount > 0 and is_offbeat_eighth:
                        # Push offbeats later for shuffle feel
                        shuffle_shift = int(shuffle_amount * ticks_per_beat/2)
                        note_type_timing_var += shuffle_shift
                    
                    # Hi-hat timing often correlates with the kick/snare pattern
                    # Check if there's a kick or snare at this time
                    other_drums = [n.note for n in notes_by_time.get(time, [])]
                    if any(n in KICK_NOTES for n in other_drums):
                        # Hi-hats with kicks are often more precisely aligned
                        note_type_timing_var *= 0.7
                
                elif msg.note in CYMBAL_NOTES:
                    # Cymbals often have more variation and might be slightly ahead for emphasis
                    note_type_timing_var = random.uniform(-timing_variation * 1.2, timing_variation * 1.2)
                    # Crashes often anticipate slightly, especially at phrase beginnings
                    if msg.note in [49, 57] and measure_position < 0.1:
                        note_type_timing_var -= 2 + profile["rushing_factor"] * 3
                
                elif msg.note in TOM_NOTES:
                    # Toms in fills often have deliberate timing for effect
                    if in_fill:
                        # More variation in fills, but with pattern consistency
                        seed = int(time / (ticks_per_beat/4))
                        random.seed(seed)  # Use consistent seed for similar positions
                        note_type_timing_var = random.uniform(-timing_variation * 1.3, timing_variation * 1.3)
                        random.seed(None)  # Reset seed
                    else:
                        # Regular tom hits
                        note_type_timing_var = random.uniform(-timing_variation, timing_variation)
                
                else:
                    # Other percussion
                    note_type_timing_var = random.uniform(-timing_variation, timing_variation)
                
                # Apply rushing/dragging tendency of the drummer
                rushing_component = profile["rushing_factor"] * 5
                if measure_position < 0.1:  # Downbeats often have different timing tendencies
                    rushing_component *= 0.5
                
                # Apply groove consistency by maintaining similar variations at pattern points
                groove_component = 0
                if is_pattern_point and profile["groove_consistency"] > 0.6:
                    # Use pattern-based variation for consistent groove
                    pattern_seed = hash((pattern_key[0], pattern_key[1], msg.note))
                    random.seed(pattern_seed)
                    groove_component = random.uniform(-timing_variation * 0.5, timing_variation * 0.5)
                    random.seed(None)  # Reset seed
                
                # Combine all timing factors
                total_timing_var = int(note_type_timing_var + rushing_component + groove_component + tempo_drift)
                
                # Limit maximum variation
                max_var = timing_variation * 2
                total_timing_var = max(-max_var, min(max_var, total_timing_var))
                
                # ======= VELOCITY HUMANIZATION =======
                # Base velocity adjustment
                new_velocity = msg.velocity
                velocity_var = 0
                
                # Apply different velocity patterns based on drum type and beat position
                if msg.note in KICK_NOTES:
                    # Kicks on downbeats are often stronger
                    if measure_position < 0.1 or any(abs(measure_position - i) < 0.1 for i in range(2, time_sig_numerator, 2)):
                        velocity_var = random.randint(-5, 15) * profile["velocity_emphasis"]
                    else:
                        velocity_var = random.randint(-10, 10) * profile["velocity_emphasis"]
                
                elif msg.note in SNARE_NOTES:
                    # Snares on backbeats are often accented
                    if any(abs(measure_position - i) < 0.1 for i in range(1, time_sig_numerator, 2)):
                        # Backbeat emphasis
                        velocity_var = random.randint(-5, 15) * profile["velocity_emphasis"]
                        # Sometimes really accent these
                        if random.random() < accent_prob * 1.5:
                            velocity_var += random.randint(10, 20)
                    else:
                        velocity_var = random.randint(-10, 10) * profile["velocity_emphasis"]
                
                elif msg.note in HIHAT_NOTES:
                    # Hi-hats often have a specific pattern of accents
                    # Quantize to 16th notes for hi-hat analysis
                    sixteenth_note_pos = round(measure_position * 16) / 16
                    
                    # Quarter notes often stronger than eighth or sixteenth notes
                    if sixteenth_note_pos.is_integer():
                        velocity_var = random.randint(-5, 15) * profile["velocity_emphasis"]
                    elif sixteenth_note_pos * 2 == round(sixteenth_note_pos * 2):  # Eighth notes
                        velocity_var = random.randint(-10, 10) * profile["velocity_emphasis"]
                    else:  # Sixteenth notes
                        velocity_var = random.randint(-15, 5) * profile["velocity_emphasis"]
                
                elif msg.note in CYMBAL_NOTES:
                    # Cymbals at phrase beginnings are stronger
                    if measure_position < 0.1 and measure_idx % 2 == 0:
                        velocity_var = random.randint(0, 20) * profile["velocity_emphasis"]
                    else:
                        velocity_var = random.randint(-10, 15) * profile["velocity_emphasis"]
                
                elif msg.note in TOM_NOTES:
                    # Toms in fills often have dynamic shaping
                    if in_fill:
                        # Find position in the fill
                        fill_idx = next((i for i in merged_fills if i[0] <= time <= i[1]), None)
                        if fill_idx:
                            fill_start, fill_end = fill_idx
                            # Avoid division by zero
                            fill_duration = max(1, fill_end - fill_start)
                            fill_position = (time - fill_start) / fill_duration
                            
                            # Common fill velocity shape: crescendo or decrescendo
                            if random.random() < 0.7:  # crescendo (most common)
                                velocity_var = int((fill_position * 30 - 10) * profile["velocity_emphasis"])
                            else:  # decrescendo
                                velocity_var = int(((1 - fill_position) * 30 - 10) * profile["velocity_emphasis"])
                    else:
                        velocity_var = random.randint(-15, 15) * profile["velocity_emphasis"]
                
                else:
                    # Other percussion
                    velocity_var = random.randint(-velocity_variation, velocity_variation)
                
                # Apply accent probability
                if random.random() < accent_prob:
                    # Increase velocity for accents based on style
                    accent_amount = random.randint(10, 25) * profile["velocity_emphasis"]
                    velocity_var += accent_amount
                
                # Apply final velocity adjustment
                new_velocity = max(1, min(127, msg.velocity + int(velocity_var)))
                
                # ======= GHOST NOTES =======
                # Add ghost notes based on drum type and position
                if msg.note in SNARE_NOTES:
                    # Snares frequently have ghost notes
                    ghost_prob = ghost_note_prob * profile["ghost_multiplier"]
                    
                    # More likely to add ghosts before backbeats
                    if abs(measure_position - 1.0) < 0.2 or abs(measure_position - 3.0) < 0.2:
                        ghost_prob *= 1.5
                    
                    if random.random() < ghost_prob:
                        # Create a ghost note with lower velocity
                        ghost_velocity = max(1, min(msg.velocity - random.randint(40, 60), 30))
                        # Ghost notes usually just before the main beat
                        ghost_offset = random.randint(ticks_per_beat//16, ticks_per_beat//8)
                        ghost_time = time - ghost_offset
                        
                        if ghost_time > 0:
                            ghost_note = msg.copy(velocity=ghost_velocity)
                            humanized_notes.append((ghost_time, ghost_note))
                
                elif msg.note in TOM_NOTES and not in_fill:
                    # Occasional ghost notes on toms outside of fills
                    if random.random() < ghost_note_prob * 0.7 * profile["ghost_multiplier"]:
                        ghost_velocity = max(1, min(msg.velocity - random.randint(40, 60), 35))
                        ghost_time = time - random.randint(ticks_per_beat//16, ticks_per_beat//8)
                        
                        if ghost_time > 0:
                            ghost_note = msg.copy(velocity=ghost_velocity)
                            humanized_notes.append((ghost_time, ghost_note))
                
                # ======= FLAMS =======
                # Add flams (quick double hits) to snares occasionally
                if msg.note in SNARE_NOTES and random.random() < flamming_prob:
                    # Flam is a quieter note right before the main hit
                    flam_velocity = max(1, int(new_velocity * 0.7))
                    flam_offset = random.randint(3, 10)  # Very close to the main hit
                    flam_time = time - flam_offset
                    
                    if flam_time > 0:
                        flam_note = msg.copy(velocity=flam_velocity)
                        humanized_notes.append((flam_time, flam_note))
                
                # Add the main humanized note
                humanized_note = msg.copy(velocity=new_velocity) if new_velocity>0 else msg.copy
                humanized_time = max(0, time + total_timing_var)
                humanized_notes.append((humanized_time, humanized_note))
                
                # Add the corresponding note_off event
                humanized_notes.append((humanized_time+ticks_per_beat//8, mido.Message('note_off', channel=humanized_note.channel, note=humanized_note.note, velocity=humanized_note.velocity, time=0)))
        
        # Sort humanized notes by time
        humanized_notes.sort(key=lambda x: x[0])
        
        # Convert back to relative timing
        last_time = 0
        for abs_time, msg in humanized_notes:
            # Calculate relative time
            msg_copy = msg.copy(time=abs_time - last_time)
            new_track.append(msg_copy)
            last_time = abs_time
    
    # Save the humanized MIDI file
    print(f"Saving humanized MIDI to: {output_file}")
    new_midi.save(output_file)
    print("Done! Applied drummer style:", drummer_style)

def main():
    parser = argparse.ArgumentParser(description='Humanize MIDI drum tracks with realistic drummer feel')
    parser.add_argument('input_file', help='Input MIDI file path')
    parser.add_argument('--output', '-o', help='Output MIDI file path (default: input_file_humanized.mid)')
    parser.add_argument('--timing', '-t', type=int, default=10, help='Timing variation in ticks (default: 10)')
    parser.add_argument('--velocity', '-v', type=int, default=15, help='Velocity variation (default: 15)')
    parser.add_argument('--ghost', '-g', type=float, default=0.0, help='Ghost note probability (default: 0.0)')
    parser.add_argument('--accent', '-a', type=float, default=0.2, help='Accent probability (default: 0.2)')
    parser.add_argument('--shuffle', '-s', type=float, default=0.0, help='Shuffle amount, 0.0-0.5 (default: 0.0)')
    parser.add_argument('--flams', '-f', type=float, default=0.0, help='Flam probability (default: 0.0)')
    parser.add_argument('--style', type=str, default="balanced", 
                        choices=["balanced", "jazzy", "rock", "precise", "loose"],
                        help='Drummer style profile (default: balanced)')
    parser.add_argument('--library', type=str, default="gm", 
                        choices=["gm", "ad2", "sd3", "ez2", "ssd5"],
                        help='Drums library mapping (default: gm)')
    
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
        drum_library=args.library
    )

if __name__ == "__main__":
    main()

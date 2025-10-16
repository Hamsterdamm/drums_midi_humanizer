from dataclasses import dataclass
from typing import Dict, List, Tuple, Set
import random
import mido
from collections import defaultdict
from pathlib import Path

from ..config.drums import get_drum_map, DrummerProfile, DRUM_RUDIMENTS
from ..utils.midi import calculate_measure_position, detect_rudiment_pattern
from ..visualization.visualizer import create_drum_visualization

@dataclass
class HumanizerConfig:
    timing_variation: int
    velocity_variation: int
    ghost_note_prob: float
    accent_prob: float
    shuffle_amount: float
    flamming_prob: float
    drummer_style: str
    drum_library: str
    visualize: bool = False

class DrumHumanizer:
    """Humanize MIDI drum tracks with realistic drummer feel."""

    def __init__(self, config: HumanizerConfig):
        self.config = config
        self.profile = self._get_drummer_profile()
        self.drum_map = get_drum_map(config.drum_library)
        (
            self.KICK_NOTES,
            self.SNARE_NOTES,
            self.HIHAT_NOTES,
            self.TOM_NOTES,
            self.CYMBAL_NOTES
        ) = self.drum_map.get_note_groups()
        self.ticks_per_beat = 0
        self.time_sig_numerator = 4
        self.time_sig_denominator = 4
        self.tempo_drift = 0
        self.merged_fills = []
        self.last_pattern = None
        self.current_timing_variation = self.config.timing_variation

    def _get_drummer_profile(self) -> DrummerProfile:
        """Get the drummer profile from config."""
        from ..config.drums import DRUMMER_PROFILES
        profile_data = DRUMMER_PROFILES[self.config.drummer_style]
        return DrummerProfile(**profile_data)

    def _get_measure_position(self, time: int) -> float:
        """Get position within the measure for a given time.
        
        Args:
            time: Current MIDI time in ticks
            
        Returns:
            Position within measure from 0.0 to time_sig_numerator
        """
        return calculate_measure_position(
            time,
            self.ticks_per_beat,
            self.time_sig_numerator
        )

    def process_file(self, input_file: str, output_file: str | None = None) -> None:
        """Process a MIDI file and apply humanization."""
        if output_file is None:
            input_path = Path(input_file)
            output_file = str(input_path.parent / f"{input_path.stem}_humanized{input_path.suffix}")
        
        print(f"Loading MIDI file: {input_file}")
        try:
            midi_file = mido.MidiFile(input_file)
        except Exception as e:
            print(f"Error loading MIDI file: {e}")
            return

        self.ticks_per_beat = midi_file.ticks_per_beat
        humanized_midi = mido.MidiFile()
        humanized_midi.ticks_per_beat = midi_file.ticks_per_beat

        original_messages = []
        humanized_messages = []

        for track in midi_file.tracks:
            print(f"\nProcessing track {midi_file.tracks.index(track) + 1}/{len(midi_file.tracks)}")
            
            # Group notes by absolute time to find patterns
            notes_by_time = defaultdict(list)
            absolute_time = 0
            for msg in track:
                absolute_time += msg.time
                if msg.type == 'note_on' and msg.velocity > 0:
                    notes_by_time[absolute_time].append(msg.note)

            # Process track with absolute timing
            absolute_time = 0
            new_track = mido.MidiTrack()
            new_track.append(mido.MetaMessage('track_name', name='Humanized Drums', time=0))
            
            humanized_notes = []
            
            # Process MIDI messages, separating notes from other message types
            for msg in track:
                absolute_time += msg.time
                
                if msg.type == 'note_on' and msg.velocity > 0:
                    # Calculate measure position for this note
                    measure_pos = self._get_measure_position(absolute_time)
                    
                    # Store original message for visualization
                    original_messages.append((absolute_time, msg.note, msg.velocity))
                    
                    # Apply humanization
                    new_time = self._apply_timing_variation(absolute_time, msg.note)
                    new_velocity = self._apply_velocity_variation(msg.velocity, msg.note, measure_pos)
                    
                    # Add flams for snare notes
                    if (msg.note in self.SNARE_NOTES and 
                        random.random() < self.config.flamming_prob):
                        flam_time = new_time - int(self.ticks_per_beat / 12)
                        flam_velocity = int(new_velocity * 0.7)
                        humanized_notes.append((flam_time, msg.note, flam_velocity))
                        humanized_messages.append((flam_time, msg.note, flam_velocity))
                    
                    # Add humanized note
                    humanized_notes.append((new_time, msg.note, new_velocity))
                    humanized_messages.append((new_time, msg.note, new_velocity))

                elif msg.type not in ['note_on', 'note_off']:
                    # Pass through non-note messages with original timing
                    new_track.append(msg)

            # Detect and apply drum rudiments to the collected notes
            if self.profile.rudiment_sensitivity > 0.5:
                humanized_notes = self._detect_and_apply_rudiments(humanized_notes)

            # Sort notes by time and create messages with proper delta times
            humanized_notes.sort(key=lambda x: x[0])
            last_time = sum(m.time for m in new_track)
            
            for time, note, velocity in humanized_notes:
                delta_time = time - last_time
                new_track.append(
                    mido.Message('note_on', 
                               note=note,
                               velocity=velocity,
                               time=max(0, delta_time))
                )
                new_track.append(
                    mido.Message('note_off',
                               note=note,
                               velocity=0,
                               time=1) # 1-tick duration for drum hits
                )
                last_time = time + 1

            humanized_midi.tracks.append(new_track)

        # Save the humanized MIDI file
        humanized_midi.save(output_file)
        print(f"\nSaved humanized MIDI to: {output_file}")

        # Generate visualization if requested
        if self.config.visualize:
            visualization_file = output_file.rsplit('.', 1)[0] + '.png'
            print(f"Generating visualization: {visualization_file}")
            create_drum_visualization(original_messages, humanized_messages, visualization_file)

    def humanize_timings(
        self,
        msg: mido.Message,
        time: int,
        notes_by_time: Dict,
        in_fill: bool,
        is_pattern_point: bool,
        pattern_key: Tuple,
        measure_position: float,
    ) -> int:
        """Calculate timing variations for a note."""
        note_type_timing_var = 0

        # Different timing handling based on drum type
        if msg.note in self.KICK_NOTES:
            note_type_timing_var = self._handle_kick_timing(measure_position)
        elif msg.note in self.SNARE_NOTES:
            note_type_timing_var = self._handle_snare_timing(measure_position)
        elif msg.note in self.HIHAT_NOTES:
            note_type_timing_var = self._handle_hihat_timing(notes_by_time, time, measure_position)
        elif msg.note in self.CYMBAL_NOTES:
            note_type_timing_var = self._handle_cymbal_timing(measure_position)
        elif msg.note in self.TOM_NOTES:
            note_type_timing_var = self._handle_tom_timing(in_fill)
        else:
            note_type_timing_var = random.uniform(-self.config.timing_variation, self.config.timing_variation)

        # Apply groove factors
        rushing_component = self._calculate_rushing_component(measure_position)
        groove_component = self._calculate_groove_component(is_pattern_point, pattern_key, msg)

        # Combine all timing factors
        total_timing_var = int(note_type_timing_var + rushing_component + groove_component + self.tempo_drift)
        max_var = self.config.timing_variation * 2
        return max(-max_var, min(max_var, total_timing_var))

    def humanize_velocity(
        self,
        msg: mido.Message,
        time: int,
        in_fill: bool,
        measure_position: float,
        measure_idx: int,
    ) -> int:
        """Calculate velocity variations for a note."""
        velocity_var = 0

        # Apply different velocity patterns based on drum type
        if msg.note in self.KICK_NOTES:
            velocity_var = self._handle_kick_velocity(measure_position)
        elif msg.note in self.SNARE_NOTES:
            velocity_var = self._handle_snare_velocity(measure_position)
        elif msg.note in self.HIHAT_NOTES:
            velocity_var = self._handle_hihat_velocity(measure_position)
        elif msg.note in self.CYMBAL_NOTES:
            velocity_var = self._handle_cymbal_velocity(measure_position, measure_idx)
        elif msg.note in self.TOM_NOTES:
            velocity_var = self._handle_tom_velocity(in_fill, time)
        else:
            velocity_var = random.randint(-self.config.velocity_variation, self.config.velocity_variation)

        # Apply accent probability
        if random.random() < self.config.accent_prob:
            accent_amount = random.randint(0, 5) * self.profile.velocity_emphasis
            velocity_var += accent_amount

        return max(1, min(127, msg.velocity + int(velocity_var)))

    def _handle_kick_timing(self, measure_position: float) -> float:
        base_var = self.config.timing_variation * 0.4 / self.profile.kick_timing_tightness
        var = random.uniform(-base_var, base_var)
        if measure_position < 0.1:
            var -= min(2, 1 + self.profile.rushing_factor)
        return var

    def _handle_snare_timing(self, measure_position: float) -> float:
        if self._is_backbeat(measure_position):
            return random.uniform(
                -self.config.timing_variation * 0.7,
                self.config.timing_variation * 0.7
            ) + self.profile.timing_bias
        return random.uniform(-self.config.timing_variation, self.config.timing_variation)

    def _handle_hihat_timing(self, notes_by_time: Dict, time: int, measure_position: float) -> float:
        hihat_var = self.config.timing_variation * self.profile.hihat_variation
        var = random.uniform(-hihat_var, hihat_var)

        # Add shuffle feel if configured
        if self.config.shuffle_amount > 0 and self._is_offbeat_eighth(measure_position):
            var += int(self.config.shuffle_amount * self.ticks_per_beat / 2)

        # Adjust timing based on kick/snare presence
        if self._has_kick_or_snare_at_time(notes_by_time, time):
            var *= 0.7

        return var

    def _handle_cymbal_timing(self, measure_position: float) -> float:
        var = random.uniform(
            -self.config.timing_variation * 1.2,
            self.config.timing_variation * 1.2
        )
        if measure_position < 0.1:
            var -= 2 + self.profile.rushing_factor * 3
        return var

    def _handle_tom_timing(self, in_fill: bool) -> float:
        if in_fill:
            seed = random.randint(0, 1000)  # Use consistent seed for similar positions
            random.seed(seed)
            var = random.uniform(
                -self.config.timing_variation * 1.3,
                self.config.timing_variation * 1.3
            )
            random.seed(None)
            return var
        return random.uniform(-self.config.timing_variation, self.config.timing_variation)

    def _handle_kick_velocity(self, measure_position: float) -> int:
        if self._is_downbeat(measure_position):
            return random.randint(0, 15) * self.profile.velocity_emphasis
        return random.randint(-10, 0) * self.profile.velocity_emphasis

    def _handle_snare_velocity(self, measure_position: float) -> int:
        if self._is_backbeat(measure_position):
            var = random.randint(0, 15) * self.profile.velocity_emphasis
            if random.random() < self.config.accent_prob * 1.5:
                var += random.randint(0, 5)
            return var
        return random.randint(-10, 0) * self.profile.velocity_emphasis

    def _handle_hihat_velocity(self, measure_position: float) -> int:
        sixteenth_pos = round(measure_position * 16) / 16
        if sixteenth_pos.is_integer():
            return random.randint(0, 15) * self.profile.velocity_emphasis
        elif sixteenth_pos * 2 == round(sixteenth_pos * 2):
            return random.randint(0, 5) * self.profile.velocity_emphasis
        return random.randint(-15, 0) * self.profile.velocity_emphasis

    def _handle_cymbal_velocity(self, measure_position: float, measure_idx: int) -> int:
        if measure_position < 0.1 and measure_idx % 2 == 0:
            return random.randint(0, 20) * self.profile.velocity_emphasis
        return random.randint(-10, 0) * self.profile.velocity_emphasis

    def _handle_tom_velocity(self, in_fill: bool, time: int) -> int:
        if in_fill:
            fill = next((f for f in self.merged_fills if f[0] <= time <= f[1]), None)
            if fill:
                fill_duration = max(1, fill[1] - fill[0])
                fill_position = (time - fill[0]) / fill_duration
                if random.random() < 0.7:  # crescendo
                    return int((fill_position * 30 - 10) * self.profile.velocity_emphasis)
                else:  # decrescendo
                    return int(((1 - fill_position) * 30 - 10) * self.profile.velocity_emphasis)
        return random.randint(-15, 15) * self.profile.velocity_emphasis

    def _calculate_rushing_component(self, measure_position: float) -> float:
        rushing_component = self.profile.rushing_factor * 5
        if measure_position < 0.1:
            rushing_component *= 0.5
        return rushing_component

    def _calculate_groove_component(self, is_pattern_point: bool, pattern_key: Tuple, msg: mido.Message) -> float:
        if is_pattern_point and self.profile.groove_consistency > 0.6:
            pattern_seed = hash((pattern_key[0], pattern_key[1], msg.note))
            random.seed(pattern_seed)
            groove_component = random.uniform(
                -self.config.timing_variation * 0.5,
                self.config.timing_variation * 0.5
            )
            random.seed(None)
            return groove_component
        return 0

    def _is_downbeat(self, measure_position: float) -> bool:
        return measure_position < 0.1 or any(
            abs(measure_position - i) < 0.1
            for i in range(2, self.time_sig_numerator, 2)
        )

    def _is_backbeat(self, measure_position: float) -> bool:
        return any(
            abs(measure_position - i) < 0.1
            for i in range(1, self.time_sig_numerator, 2)
        )

    def _is_offbeat_eighth(self, measure_position: float) -> bool:
        eighth_note_pos = round(measure_position * 8) / 8
        return eighth_note_pos % 0.5 == 0 and eighth_note_pos % 1.0 != 0

    def _has_kick_or_snare_at_time(self, notes_by_time: Dict, time: int) -> bool:
        other_drums = notes_by_time.get(time, [])
        return any(n in self.KICK_NOTES or n in self.SNARE_NOTES for n in other_drums)
    
    def _apply_timing_variation(self, time: int, note: int) -> int:
        """Apply timing variation based on note type and drummer profile."""
        measure_pos = self._get_measure_position(time)
        
        # Base variation based on groove consistency
        base_variation = random.gauss(0, self.current_timing_variation * (2 - self.profile.groove_consistency))
        
        # Add drummer-specific timing bias
        if note in self.KICK_NOTES:
            variation = base_variation * (1 / self.profile.kick_timing_tightness)
        elif note in self.HIHAT_NOTES:
            variation = base_variation * self.profile.hihat_variation
        else:
            variation = base_variation
            
        # Apply rushing/dragging tendency based on measure position
        rush_amount = self.profile.rushing_factor * measure_pos * self.ticks_per_beat / 8
        
        # Apply shuffle feel on relevant subdivisions
        if measure_pos % 0.5 < 0.25:  # Second 8th note of each beat
            shuffle = self.config.shuffle_amount * self.ticks_per_beat / 12
        else:
            shuffle = 0
            
        return int(time + variation + rush_amount + shuffle)

    def _apply_velocity_variation(self, velocity: int, note: int, measure_pos: float) -> int:
        """Apply velocity variation based on note type and drummer profile."""
        if note in self.KICK_NOTES or note in self.SNARE_NOTES:
            # Emphasize strong beats
            emphasis = 1.0 + (0.2 * self.profile.velocity_emphasis * (1 - (measure_pos % 1)))
            if random.random() < self.config.accent_prob:
                emphasis *= 1.2
        elif note in self.HIHAT_NOTES:
            # Lighter variation for hi-hats
            emphasis = 1.0 + random.gauss(0, 0.1) * self.profile.hihat_variation
        else:
            emphasis = 1.0
            
        # Apply ghost notes
        if (note in self.SNARE_NOTES and 
            random.random() < self.config.ghost_note_prob):
            emphasis *= self.profile.ghost_multiplier
            
        new_velocity = int(velocity * emphasis)
        return max(1, min(127, new_velocity))

    def _detect_and_apply_rudiments(self, notes: List[Tuple[int, int, int]]) -> List[Tuple[int, int, int]]:
        """Detect and apply humanization to drum rudiments."""
        if len(notes) < 2:
            return notes
            
        for rudiment_name, rudiment in DRUM_RUDIMENTS.items():
            if detect_rudiment_pattern(
                notes, 
                rudiment["pattern"],
                rudiment["timing_ratio"],
                self.ticks_per_beat
            ):
                # Apply rudiment-specific timing and velocity adjustments
                result = []
                pattern_duration = rudiment["duration"] * self.ticks_per_beat
                start_time = notes[0][0]
                
                for i, (time, note, vel) in enumerate(notes):
                    rel_pos = i / len(notes)
                    adj_time = start_time + int(rel_pos * pattern_duration * rudiment["timing_ratio"][i])
                    adj_vel = int(vel * rudiment["velocity_ratio"][i])
                    result.append((adj_time, note, adj_vel))
                    
                return result
                
        return notes

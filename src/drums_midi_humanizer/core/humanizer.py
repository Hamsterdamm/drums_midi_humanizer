"""Core humanization functionality module.

This module contains the main logic for processing MIDI drum tracks, applying
timing and velocity variations based on drummer profiles and configuration settings.
"""

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
    """Configuration parameters for the drum humanizer.

    Attributes:
        timing_variation (int): Maximum deviation in ticks for timing.
        velocity_variation (int): Maximum deviation for velocity.
        ghost_note_prob (float): Probability (0.0-1.0) of adding ghost notes.
        accent_prob (float): Probability (0.0-1.0) of adding accents.
        shuffle_amount (float): Amount of shuffle/swing to apply (0.0-0.5).
        flamming_prob (float): Probability (0.0-1.0) of simulating flams.
        drummer_style (str): Name of the drummer profile to use.
        drum_library (str): Name of the drum mapping library.
        visualize (bool): Whether to generate a visualization plot.
    """

    timing_variation: int
    velocity_variation: int
    ghost_note_prob: float
    accent_prob: float
    shuffle_amount: float
    flamming_prob: float
    drummer_style: str
    drum_library: str
    visualize: bool = False

    def __post_init__(self):
        """Validate configuration parameters."""
        if not 0 <= self.ghost_note_prob <= 1:
            raise ValueError(
                f"Ghost note probability must be between 0.0 and 1.0, got {self.ghost_note_prob}"
            )
        if not 0 <= self.accent_prob <= 1:
            raise ValueError(
                f"Accent probability must be between 0.0 and 1.0, got {self.accent_prob}"
            )
        if not 0 <= self.flamming_prob <= 1:
            raise ValueError(
                f"Flam probability must be between 0.0 and 1.0, got {self.flamming_prob}"
            )


class DrumHumanizer:
    """Humanize MIDI drum tracks with realistic drummer feel.

    This class orchestrates the humanization process, managing drummer profiles,
    drum mappings, and the application of timing and velocity variations to
    MIDI events.
    """

    def __init__(self, config: HumanizerConfig):
        """Initialize the DrumHumanizer with a configuration.

        Args:
            config (HumanizerConfig): Configuration object containing humanization
                parameters and settings.
        """
        self.config = config
        self.profile = self._get_drummer_profile()
        self.drum_map = get_drum_map(config.drum_library)
        self.KICK_NOTES, self.SNARE_NOTES, self.HIHAT_NOTES, self.TOM_NOTES, self.CYMBAL_NOTES = (
            self.drum_map.get_note_groups()
        )
        self.ticks_per_beat = 0
        self.time_sig_numerator = 4
        self.time_sig_denominator = 4
        self.tempo_drift = 0
        self.merged_fills = []
        self.last_pattern = None
        self.current_timing_variation = self.config.timing_variation

    def _get_drummer_profile(self) -> DrummerProfile:
        """Retrieve the selected drummer profile based on configuration.

        Returns:
            DrummerProfile: The profile object containing style-specific constants.
        """
        from ..config.drums import DRUMMER_PROFILES

        profile_data = DRUMMER_PROFILES[self.config.drummer_style]
        return DrummerProfile(**profile_data)

    def _get_measure_position(self, time: int) -> float:
        """Calculate the beat position within a measure for a given absolute time.

        Args:
            time (int): Current absolute MIDI time in ticks.

        Returns:
            float: Position within measure (e.g., 0.0 to 4.0 for 4/4 time).
        """
        return calculate_measure_position(time, self.ticks_per_beat, self.time_sig_numerator)

    def process_file(self, input_file: str, output_file: str | None = None) -> None:
        """Process a MIDI file and apply humanization.

        Reads the input MIDI file, applies timing and velocity variations to
        drum notes, and saves the result. Optionally generates a visualization.

        Args:
            input_file (str): Path to the source MIDI file.
            output_file (str | None): Path to save the humanized MIDI file. If None,
                defaults to '{input_stem}_humanized.mid'.
        """
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

            absolute_time = 0
            # This list will hold message objects with their absolute time
            events_with_absolute_time = []

            # First pass: Apply humanization and store events with their new absolute times
            for msg in track:
                absolute_time += msg.time

                if msg.type == "note_on" and msg.velocity > 0:
                    measure_pos = self._get_measure_position(absolute_time)
                    original_messages.append((absolute_time, msg.note, msg.velocity))

                    new_time = self._apply_timing_variation(absolute_time, msg.note)
                    new_velocity = self._apply_velocity_variation(
                        msg.velocity, msg.note, measure_pos
                    )

                    # Add humanized note
                    events_with_absolute_time.append(
                        {
                            "time": new_time,
                            "msg": mido.Message("note_on", note=msg.note, velocity=new_velocity),
                        }
                    )
                    # Add corresponding note_off. Using a small duration for drum hits.
                    events_with_absolute_time.append(
                        {
                            "time": new_time + 1,
                            "msg": mido.Message("note_off", note=msg.note, velocity=0),
                        }
                    )

                    humanized_messages.append((new_time, msg.note, new_velocity))

                elif msg.type != "note_off":  # Keep all other messages, discard original note_offs
                    events_with_absolute_time.append({"time": absolute_time, "msg": msg})

            # Sort all events by their absolute time
            events_with_absolute_time.sort(key=lambda e: e["time"])

            # Second pass: Create the new track by calculating correct delta times
            new_track = mido.MidiTrack()
            last_time = 0
            for event in events_with_absolute_time:
                current_time = event["time"]
                delta_time = current_time - last_time
                event["msg"].time = max(0, int(delta_time))
                new_track.append(event["msg"])
                last_time = current_time

            humanized_midi.tracks.append(new_track)

        # Save the humanized MIDI file
        humanized_midi.save(output_file)
        print(f"\nSaved humanized MIDI to: {output_file}")

        # Generate visualization if requested
        if self.config.visualize:
            visualization_file = output_file.rsplit(".", 1)[0] + ".png"
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
        """Calculate complex timing variations for a note (Advanced Logic).

        Args:
            msg (mido.Message): The MIDI message being processed.
            time (int): Absolute time of the event.
            notes_by_time (Dict): Dictionary of all notes indexed by time.
            in_fill (bool): Whether the note is part of a detected fill.
            is_pattern_point (bool): Whether the note is part of a detected pattern.
            pattern_key (Tuple): Key identifying the specific pattern.
            measure_position (float): Position within the measure.

        Returns:
            int: The calculated timing offset in ticks.
        """
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
            note_type_timing_var = random.uniform(
                -self.config.timing_variation, self.config.timing_variation
            )

        # Apply groove factors
        rushing_component = self._calculate_rushing_component(measure_position)
        groove_component = self._calculate_groove_component(is_pattern_point, pattern_key, msg)

        # Combine all timing factors
        total_timing_var = int(
            note_type_timing_var + rushing_component + groove_component + self.tempo_drift
        )
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
        """Calculate complex velocity variations for a note (Advanced Logic).

        Args:
            msg (mido.Message): The MIDI message being processed.
            time (int): Absolute time of the event.
            in_fill (bool): Whether the note is part of a detected fill.
            measure_position (float): Position within the measure.
            measure_idx (int): Index of the current measure.

        Returns:
            int: The new velocity value (clamped between 1 and 127).
        """
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
            velocity_var = random.randint(
                -self.config.velocity_variation, self.config.velocity_variation
            )

        # Apply accent probability
        if random.random() < self.config.accent_prob:
            accent_amount = random.randint(0, 5) * self.profile.velocity_emphasis
            velocity_var += accent_amount

        return max(1, min(127, msg.velocity + int(velocity_var)))

    def _handle_kick_timing(self, measure_position: float) -> float:
        """Calculate timing variation specifically for kick drums."""
        base_var = self.config.timing_variation * 0.4 / self.profile.kick_timing_tightness
        var = random.uniform(-base_var, base_var)
        if measure_position < 0.1:
            var -= min(2, 1 + self.profile.rushing_factor)
        return var

    def _handle_snare_timing(self, measure_position: float) -> float:
        """Calculate timing variation specifically for snare drums."""
        if self._is_backbeat(measure_position):
            return (
                random.uniform(
                    -self.config.timing_variation * 0.7, self.config.timing_variation * 0.7
                )
                + self.profile.timing_bias
            )
        return random.uniform(-self.config.timing_variation, self.config.timing_variation)

    def _handle_hihat_timing(
        self, notes_by_time: Dict, time: int, measure_position: float
    ) -> float:
        """Calculate timing variation specifically for hi-hats."""
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
        """Calculate timing variation specifically for cymbals."""
        var = random.uniform(
            -self.config.timing_variation * 1.2, self.config.timing_variation * 1.2
        )
        if measure_position < 0.1:
            var -= 2 + self.profile.rushing_factor * 3
        return var

    def _handle_tom_timing(self, in_fill: bool) -> float:
        """Calculate timing variation specifically for toms."""
        if in_fill:
            seed = random.randint(0, 1000)  # Use consistent seed for similar positions
            random.seed(seed)
            var = random.uniform(
                -self.config.timing_variation * 1.3, self.config.timing_variation * 1.3
            )
            random.seed(None)
            return var
        return random.uniform(-self.config.timing_variation, self.config.timing_variation)

    def _handle_kick_velocity(self, measure_position: float) -> int:
        """Calculate velocity adjustment for kick drums."""
        if self._is_downbeat(measure_position):
            return random.randint(0, 15) * self.profile.velocity_emphasis
        return random.randint(-10, 0) * self.profile.velocity_emphasis

    def _handle_snare_velocity(self, measure_position: float) -> int:
        """Calculate velocity adjustment for snare drums."""
        if self._is_backbeat(measure_position):
            var = random.randint(0, 15) * self.profile.velocity_emphasis
            if random.random() < self.config.accent_prob * 1.5:
                var += random.randint(0, 5)
            return var
        return random.randint(-10, 0) * self.profile.velocity_emphasis

    def _handle_hihat_velocity(self, measure_position: float) -> int:
        """Calculate velocity adjustment for hi-hats."""
        sixteenth_pos = round(measure_position * 16) / 16
        if sixteenth_pos.is_integer():
            return random.randint(0, 15) * self.profile.velocity_emphasis
        elif sixteenth_pos * 2 == round(sixteenth_pos * 2):
            return random.randint(0, 5) * self.profile.velocity_emphasis
        return random.randint(-15, 0) * self.profile.velocity_emphasis

    def _handle_cymbal_velocity(self, measure_position: float, measure_idx: int) -> int:
        """Calculate velocity adjustment for cymbals."""
        if measure_position < 0.1 and measure_idx % 2 == 0:
            return random.randint(0, 20) * self.profile.velocity_emphasis
        return random.randint(-10, 0) * self.profile.velocity_emphasis

    def _handle_tom_velocity(self, in_fill: bool, time: int) -> int:
        """Calculate velocity adjustment for toms, handling fill dynamics."""
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
        """Calculate the timing offset due to rushing tendencies."""
        rushing_component = self.profile.rushing_factor * 5
        if measure_position < 0.1:
            rushing_component *= 0.5
        return rushing_component

    def _calculate_groove_component(
        self, is_pattern_point: bool, pattern_key: Tuple, msg: mido.Message
    ) -> float:
        """Calculate the timing offset due to groove consistency."""
        if is_pattern_point and self.profile.groove_consistency > 0.6:
            pattern_seed = hash((pattern_key[0], pattern_key[1], msg.note))
            random.seed(pattern_seed)
            groove_component = random.uniform(
                -self.config.timing_variation * 0.5, self.config.timing_variation * 0.5
            )
            random.seed(None)
            return groove_component
        return 0

    def _is_downbeat(self, measure_position: float) -> bool:
        """Check if the position corresponds to a downbeat (1, 3 in 4/4)."""
        return measure_position < 0.1 or any(
            abs(measure_position - i) < 0.1 for i in range(2, self.time_sig_numerator, 2)
        )

    def _is_backbeat(self, measure_position: float) -> bool:
        """Check if the position corresponds to a backbeat (2, 4 in 4/4)."""
        return any(abs(measure_position - i) < 0.1 for i in range(1, self.time_sig_numerator, 2))

    def _is_offbeat_eighth(self, measure_position: float) -> bool:
        """Check if the position is an offbeat eighth note."""
        eighth_note_pos = round(measure_position * 8) / 8
        return eighth_note_pos % 0.5 == 0 and eighth_note_pos % 1.0 != 0

    def _has_kick_or_snare_at_time(self, notes_by_time: Dict, time: int) -> bool:
        """Check if a kick or snare occurs at the specified time."""
        other_drums = notes_by_time.get(time, [])
        return any(n in self.KICK_NOTES or n in self.SNARE_NOTES for n in other_drums)

    def _apply_timing_variation(self, time: int, note: int) -> int:
        """Apply timing variation based on note type and drummer profile.

        This is the primary timing logic used by `process_file`.

        Args:
            time (int): The original absolute timestamp.
            note (int): The MIDI note number.

        Returns:
            int: The new absolute timestamp with humanization applied.
        """
        measure_pos = self._get_measure_position(time)

        # Base variation based on groove consistency
        base_variation = random.gauss(
            0, self.current_timing_variation * (2 - self.profile.groove_consistency)
        )

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
        """Apply velocity variation based on note type and drummer profile.

        This is the primary velocity logic used by `process_file`.

        Args:
            velocity (int): The original velocity.
            note (int): The MIDI note number.
            measure_pos (float): The position within the measure.

        Returns:
            int: The new velocity value.
        """
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
        if note in self.SNARE_NOTES and random.random() < self.config.ghost_note_prob:
            emphasis *= self.profile.ghost_multiplier

        new_velocity = int(velocity * emphasis)
        return max(1, min(127, new_velocity))

    def _detect_and_apply_rudiments(
        self, notes: List[Tuple[int, int, int]]
    ) -> List[Tuple[int, int, int]]:
        """Detect and apply humanization to drum rudiments.

        Scans a sequence of notes for known rudiment patterns and applies specific
        timing and velocity curves if matches are found.

        Args:
            notes (List[Tuple[int, int, int]]): List of (time, note, velocity) tuples.

        Returns:
            List[Tuple[int, int, int]]: The processed list of notes.
        """
        if len(notes) < 2:
            return notes

        for rudiment_name, rudiment in DRUM_RUDIMENTS.items():
            if detect_rudiment_pattern(
                notes, rudiment["pattern"], rudiment["timing_ratio"], self.ticks_per_beat
            ):
                # Apply rudiment-specific timing and velocity adjustments
                result = []
                pattern_duration = rudiment["duration"] * self.ticks_per_beat
                start_time = notes[0][0]

                for i, (time, note, vel) in enumerate(notes):
                    rel_pos = i / len(notes)
                    adj_time = start_time + int(
                        rel_pos * pattern_duration * rudiment["timing_ratio"][i]
                    )
                    adj_vel = int(vel * rudiment["velocity_ratio"][i])
                    result.append((adj_time, note, adj_vel))

                return result

        return notes

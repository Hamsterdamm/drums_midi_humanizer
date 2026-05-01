"""Core humanization functionality module.

This module contains the main logic for processing MIDI drum tracks, applying
timing and velocity variations based on drummer profiles and configuration settings.
"""

import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import mido

from ..config.drums import DrummerProfile, get_drum_map, DRUM_RUDIMENTS
from ..utils.midi import (
    MidiTimeline,
    convert_to_relative_times,
    detect_fills,
    get_absolute_times,
    detect_rudiment_pattern,
)
from ..visualization.visualizer import create_drum_visualization

logger = logging.getLogger(__name__)


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
        logger.debug(f"Loaded drummer profile '{config.drummer_style}': {self.profile}")
        self.drum_map = get_drum_map(config.drum_library)
        logger.debug(f"Loaded drum map for library '{config.drum_library}'")
        (
            self.KICK_NOTES,
            self.SNARE_NOTES,
            self.HIHAT_NOTES,
            self.TOM_NOTES,
            self.CYMBAL_NOTES,
            self.RIDE_NOTES,
        ) = self.drum_map.get_note_groups()
        self.ticks_per_beat = 0
        self.tempo_drift = 0
        self.merged_fills: List[Tuple[int, int]] = []

    def _get_drummer_profile(self) -> DrummerProfile:
        """Retrieve the selected drummer profile based on configuration.

        Returns:
            DrummerProfile: The profile object containing style-specific constants.
        """
        from ..config.drums import DRUMMER_PROFILES

        profile_data = DRUMMER_PROFILES[self.config.drummer_style]
        return DrummerProfile(**profile_data)



    def process_file(self, input_file: str, output_file: str | None = None) -> Tuple[List[Tuple], List[Tuple]]:
        """Process a MIDI file and apply humanization.

        Reads the input MIDI file, applies timing and velocity variations to
        drum notes, and saves the result. Optionally generates a visualization.

        Args:
            input_file (str): Path to the source MIDI file.
            output_file (str | None): Path to save the humanized MIDI file. If None,
                defaults to '{input_stem}_humanized.mid'.
                
        Returns:
            Tuple[List[Tuple], List[Tuple]]: A tuple containing the original and humanized
            message lists for visualization formatting.
        """
        if output_file is None:
            input_path = Path(input_file)
            output_file = str(input_path.parent / f"{input_path.stem}_humanized{input_path.suffix}")

        logger.info(f"Loading MIDI file: {input_file}")
        try:
            midi_file = mido.MidiFile(input_file)
        except Exception as e:
            logger.error(f"Error loading MIDI file: {e}")
            return [], []

        self.ticks_per_beat = midi_file.ticks_per_beat
        self.timeline = MidiTimeline(midi_file)
        logger.info(f"MIDI file loaded. Ticks per beat: {self.ticks_per_beat}, Tracks: {len(midi_file.tracks)}")
        humanized_midi = mido.MidiFile()
        humanized_midi.ticks_per_beat = midi_file.ticks_per_beat

        original_messages_for_viz = []
        humanized_messages_for_viz = []

        for i, track in enumerate(midi_file.tracks):
            logger.info(f"Processing track {i + 1}/{len(midi_file.tracks)}")
            humanized_track, orig_msgs, human_msgs = self._humanize_track(track)
            humanized_midi.tracks.append(humanized_track)
            original_messages_for_viz.extend(orig_msgs)
            humanized_messages_for_viz.extend(human_msgs)

        # Save the humanized MIDI file
        humanized_midi.save(output_file)
        logger.info(f"Saved humanized MIDI to: {output_file}")

        # Generate visualization if requested
        if self.config.visualize:
            visualization_file = output_file.rsplit(".", 1)[0] + ".png"
            logger.info(f"Generating visualization: {visualization_file}")
            create_drum_visualization(
                original_messages_for_viz, humanized_messages_for_viz, visualization_file
            )
            
        return original_messages_for_viz, humanized_messages_for_viz

    def _generate_ghost_notes(self, events: List[Tuple[int, mido.Message]]) -> List[Tuple[int, int, int]]:
        """Generate ghost notes based on probability and empty spaces."""
        if not events:
            return []
            
        ghosts = []
        last_time = events[-1][0]
        sixteenth_ticks = self.ticks_per_beat // 4
        
        # Map occupied 16th slots
        occupied_slots = set()
        for time, msg in events:
            if msg.type == 'note_on' and msg.velocity > 0:
                slot = round(time / sixteenth_ticks)
                occupied_slots.add(slot)
        
        # Iterate slots
        num_slots = int(last_time / sixteenth_ticks)
        snare_note = list(self.SNARE_NOTES)[0] if self.SNARE_NOTES else 38
        
        for i in range(num_slots):
            if i not in occupied_slots:
                # Probability check.
                # We scale down the global probability (0.4 factor) because we are iterating
                # over every single 16th note slot. Without scaling, a 10% probability
                # would result in an unrealistically busy ghost note pattern.
                if random.random() < self.config.ghost_note_prob * 0.4:
                    time = int(i * sixteenth_ticks)
                    # Add some timing variation
                    time += random.randint(-self.config.timing_variation, self.config.timing_variation)
                    if time < 0: time = 0
                    velocity = random.randint(15, 40)
                    ghosts.append((time, snare_note, velocity))
                    
        return ghosts

    def _humanize_track(
        self, track: mido.MidiTrack
    ) -> Tuple[mido.MidiTrack, List[Tuple], List[Tuple]]:
        """Humanize a single MIDI track.

        Args:
            track (mido.MidiTrack): The MIDI track to process.

        Returns:
            Tuple[mido.MidiTrack, List[Tuple], List[Tuple]]: A tuple containing the
            humanized track, a list of original note messages for visualization,
            and a list of humanized note messages for visualization.
        """
        events_with_absolute_time = get_absolute_times(track)

        # Build context: notes by time
        # We need random access to notes occurring at specific times to make context-aware decisions
        # (e.g., adjusting hi-hat timing if a snare is played at the same time).
        notes_by_time: Dict[int, List[mido.Message]] = {}
        for time, msg in events_with_absolute_time:
            if msg.type == "note_on" and msg.velocity > 0:
                if time not in notes_by_time:
                    notes_by_time[time] = []
                notes_by_time[time].append(msg)

        # Detect fills
        # Fills usually happen on faster subdivisions (16th notes or faster), so we use that as a heuristic.
        primary_subdivision = self.ticks_per_beat // 4
        self.merged_fills = detect_fills(notes_by_time, primary_subdivision, self.TOM_NOTES)

        processed_events = []
        original_messages = []
        humanized_messages = []
        notes_processed_count = 0

        # Pair note_on and note_off events to preserve duration
        parsed_notes = []
        active_notes: Dict[int, List[Tuple[int, int, int, mido.Message]]] = {}  # note -> list of (start_time, velocity, channel, msg)
        non_note_events = []

        for time, msg in events_with_absolute_time:
            if msg.type == "note_on" and msg.velocity > 0:
                if msg.note not in active_notes:
                    active_notes[msg.note] = []
                active_notes[msg.note].append((time, msg.velocity, msg.channel, msg))
            elif (msg.type == "note_off") or (msg.type == "note_on" and msg.velocity == 0):
                if msg.note in active_notes and active_notes[msg.note]:
                    start_time, velocity, channel, orig_msg = active_notes[msg.note].pop(0)
                    duration = time - start_time
                    parsed_notes.append({
                        "start": start_time,
                        "duration": duration,
                        "note": msg.note,
                        "velocity": velocity,
                        "channel": channel,
                        "orig_msg": orig_msg
                    })
            else:
                non_note_events.append((time, msg))

        # Handle notes without a corresponding note_off
        for note, pending in active_notes.items():
            for start_time, velocity, channel, orig_msg in pending:
                # Default to a 16th note duration if cut off
                duration = self.ticks_per_beat // 4
                parsed_notes.append({
                    "start": start_time,
                    "duration": duration,
                    "note": note,
                    "velocity": velocity,
                    "channel": channel,
                    "orig_msg": orig_msg
                })

        # Sort notes by start time to process in order
        parsed_notes.sort(key=lambda x: x["start"])

        # Rudiment Detection Algorithm
        hand_notes = []
        for note_data in parsed_notes:
            if note_data["note"] not in self.KICK_NOTES:
                hand_notes.append(note_data)

        matched_note_ids = set()
        sorted_rudiments = sorted(DRUM_RUDIMENTS.items(), key=lambda item: len(item[1]["pattern"]), reverse=True)

        for rudiment_name, r_data in sorted_rudiments:
            pattern_len = len(r_data["pattern"])
            timing_ratios = r_data["timing_ratio"]
            velocity_ratios = r_data["velocity_ratio"]

            if pattern_len > len(hand_notes):
                continue

            for i in range(len(hand_notes) - pattern_len + 1):
                window = hand_notes[i:i + pattern_len]
                
                # Check if any note in the window is already matched
                if any(id(n) in matched_note_ids for n in window):
                    continue

                notes_for_detection = [(n["start"], n["note"], n["velocity"]) for n in window]

                if detect_rudiment_pattern(
                    notes=notes_for_detection,
                    pattern=r_data["pattern"],
                    timing_ratios=timing_ratios,
                    ticks_per_beat=self.ticks_per_beat,
                    tolerance=0.15
                ) and random.random() < self.profile.rudiment_sensitivity:
                    # Match found! Tag the notes
                    start_time = window[0]["start"]
                    pattern_key = (rudiment_name, start_time)
                    for idx, n_data in enumerate(window):
                        n_data["rudiment_metadata"] = {
                            "name": rudiment_name,
                            "pattern_key": pattern_key,
                            "pattern_idx": idx,
                            "velocity_ratio": velocity_ratios[idx]
                        }
                        matched_note_ids.add(id(n_data))

        # Add non-note events to the processed list
        processed_events.extend(non_note_events)

        for note_data in parsed_notes:
            time = note_data["start"]
            msg = note_data["orig_msg"]

            notes_processed_count += 1
            original_messages.append((time, msg.note, msg.velocity))

            measure_pos, measure_duration, measure_idx, numerator = self.timeline.get_measure_info(time)
            tempo_multiplier = self.timeline.get_tempo_multiplier(time)
            in_fill = any(start <= time <= end for start, end in self.merged_fills)

            rudiment_metadata = note_data.get("rudiment_metadata")
            is_pattern_point = rudiment_metadata is not None
            pattern_key = rudiment_metadata["pattern_key"] if is_pattern_point else None

            # Apply advanced humanization logic
            timing_offset = self.humanize_timings(
                msg, time, notes_by_time, in_fill, is_pattern_point, pattern_key, measure_pos, tempo_multiplier, numerator
            )
            new_time = max(0, time + int(timing_offset))
            new_velocity = self.humanize_velocity(
                msg, time, in_fill, measure_pos, measure_idx, numerator
            )

            if is_pattern_point:
                target_ratio = rudiment_metadata["velocity_ratio"]
                blend_factor = 0.5
                mixed_velocity = new_velocity * (1 - blend_factor) + (new_velocity * target_ratio) * blend_factor
                new_velocity = int(max(1, min(127, mixed_velocity)))

            # Preserve original duration
            new_duration = note_data["duration"]
            new_end = new_time + new_duration

            # Ensure end time is after start time
            if new_end <= new_time:
                new_end = new_time + 1

            processed_events.append(
                (new_time, mido.Message("note_on", note=msg.note, velocity=new_velocity, channel=note_data["channel"]))
            )
            processed_events.append(
                (new_end, mido.Message("note_off", note=msg.note, velocity=0, channel=note_data["channel"]))
            )

            humanized_messages.append((new_time, msg.note, new_velocity))

        # Add ghost notes
        if self.config.ghost_note_prob > 0:
            ghost_notes = self._generate_ghost_notes(events_with_absolute_time)
            for t, n, v in ghost_notes:
                processed_events.append((t, mido.Message("note_on", note=n, velocity=v)))
                processed_events.append((t + self.ticks_per_beat // 4, mido.Message("note_off", note=n, velocity=0)))
                humanized_messages.append((t, n, v))
        
        processed_events.sort(key=lambda x: x[0])
        logger.info(f"Processed {notes_processed_count} notes in track.")

        relative_time_events = convert_to_relative_times(processed_events)

        new_track = mido.MidiTrack()
        for delta_time, msg in relative_time_events:
            msg.time = delta_time
            new_track.append(msg)

        return new_track, original_messages, humanized_messages

    def humanize_timings(
        self,
        msg: mido.Message,
        time: int,
        notes_by_time: Dict,
        in_fill: bool,
        is_pattern_point: bool,
        pattern_key: Tuple,
        measure_position: float,
        tempo_multiplier: float,
        numerator: int,
    ) -> float:
        """Calculate complex timing variations for a note (Advanced Logic).

        Args:
            msg (mido.Message): The MIDI message being processed.
            time (int): Absolute time of the event.
            notes_by_time (Dict): Dictionary of all notes indexed by time.
            in_fill (bool): Whether the note is part of a detected fill.
            is_pattern_point (bool): Whether the note is part of a detected pattern.
            pattern_key (Tuple): Key identifying the specific pattern.
            measure_position (float): Position within the measure.
            tempo_multiplier (float): Scaling factor based on tempo.
            numerator (int): Current time signature numerator.

        Returns:
            float: The calculated timing offset in ticks.
        """
        note_type_timing_var: float = 0.0

        # Different timing handling based on drum type
        if msg.note in self.KICK_NOTES:
            note_type_timing_var = self._handle_kick_timing(measure_position)
        elif msg.note in self.SNARE_NOTES:
            note_type_timing_var = self._handle_snare_timing(measure_position, numerator)
        elif msg.note in self.HIHAT_NOTES:
            note_type_timing_var = self._handle_hihat_timing(notes_by_time, time, measure_position)
        elif msg.note in self.CYMBAL_NOTES:
            note_type_timing_var = self._handle_cymbal_timing(measure_position)
        elif msg.note in self.RIDE_NOTES:
            note_type_timing_var = self._handle_ride_timing(notes_by_time, time, measure_position)
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
        total_timing_var = (
            note_type_timing_var + rushing_component + groove_component + self.tempo_drift
        ) * tempo_multiplier
        
        max_var = self.config.timing_variation * 2 * tempo_multiplier
        # Clamp the variation to prevent extreme outliers that would break the rhythm entirely.
        return max(-max_var, min(max_var, total_timing_var))

    def humanize_velocity(
        self,
        msg: mido.Message,
        time: int,
        in_fill: bool,
        measure_position: float,
        measure_idx: int,
        numerator: int,
    ) -> int:
        """Calculate complex velocity variations for a note (Advanced Logic).

        Args:
            msg (mido.Message): The MIDI message being processed.
            time (int): Absolute time of the event.
            in_fill (bool): Whether the note is part of a detected fill.
            measure_position (float): Position within the measure.
            measure_idx (int): Index of the current measure.
            numerator (int): Current time signature numerator.

        Returns:
            int: The new velocity value (clamped between 1 and 127).
        """
        velocity_var: int = 0

        # Apply different velocity patterns based on drum type
        if msg.note in self.KICK_NOTES:
            velocity_var = self._handle_kick_velocity(measure_position, numerator)
        elif msg.note in self.SNARE_NOTES:
            velocity_var = self._handle_snare_velocity(measure_position, numerator)
        elif msg.note in self.HIHAT_NOTES:
            velocity_var = self._handle_hihat_velocity(measure_position)
        elif msg.note in self.CYMBAL_NOTES:
            velocity_var = self._handle_cymbal_velocity(measure_position, measure_idx)
        elif msg.note in self.RIDE_NOTES:
            velocity_var = self._handle_ride_velocity(measure_position)
        elif msg.note in self.TOM_NOTES:
            velocity_var = self._handle_tom_velocity(in_fill, time)
        else:
            velocity_var = random.randint(
                -self.config.velocity_variation, self.config.velocity_variation
            )

        # Apply accent probability
        if random.random() < self.config.accent_prob:
            accent_amount = int(random.randint(0, 5) * self.profile.velocity_emphasis)
            velocity_var += accent_amount

        return max(1, min(127, msg.velocity + velocity_var))

    def _handle_kick_timing(self, measure_position: float) -> float:
        """Calculate timing variation specifically for kick drums."""
        base_var = self.config.timing_variation * 0.4 / self.profile.kick_timing_tightness
        var = random.uniform(-base_var, base_var)
        if measure_position < 0.1:
            # Kick drums on the downbeat (beat 1) define the start of the measure.
            # Drummers tend to be slightly ahead or behind consistently, but usually tighter here.
            var -= min(2, 1 + self.profile.rushing_factor)
        return var

    def _handle_snare_timing(self, measure_position: float, numerator: int) -> float:
        """Calculate timing variation specifically for snare drums."""
        if self._is_backbeat(measure_position, numerator):
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
        # When hitting multiple drums simultaneously, limbs interact.
        # This tightens the timing when a hi-hat coincides with a kick or snare.
        if self._has_kick_or_snare_at_time(notes_by_time, time):
            var *= 0.7

        return var

    def _handle_ride_timing(
        self, notes_by_time: Dict, time: int, measure_position: float
    ) -> float:
        """Calculate timing variation specifically for ride cymbals."""
        # We apply the same hi-hat variation logic since the ride often acts as the primary timekeeper
        ride_var = self.config.timing_variation * self.profile.hihat_variation
        var = random.uniform(-ride_var, ride_var)

        # Add shuffle feel if configured
        if self.config.shuffle_amount > 0 and self._is_offbeat_eighth(measure_position):
            var += int(self.config.shuffle_amount * self.ticks_per_beat / 2)

        # Adjust timing based on kick/snare presence (tightens timing when limb interaction occurs)
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

    def _handle_kick_velocity(self, measure_position: float, numerator: int) -> int:
        """Calculate velocity adjustment for kick drums."""
        if self._is_downbeat(measure_position, numerator):
            return int(random.randint(0, 15) * self.profile.velocity_emphasis)
        return int(random.randint(-10, 0) * self.profile.velocity_emphasis)

    def _handle_snare_velocity(self, measure_position: float, numerator: int) -> int:
        """Calculate velocity adjustment for snare drums."""
        if self._is_backbeat(measure_position, numerator):
            var = int(random.randint(0, 15) * self.profile.velocity_emphasis)
            if random.random() < self.config.accent_prob * 1.5:
                var += random.randint(0, 5)
            return var
        return int(random.randint(-10, 0) * self.profile.velocity_emphasis)

    def _handle_hihat_velocity(self, measure_position: float) -> int:
        """Calculate velocity adjustment for hi-hats."""
        sixteenth_pos = round(measure_position * 16) / 16
        if sixteenth_pos.is_integer():
            return int(random.randint(0, 15) * self.profile.velocity_emphasis)
        elif sixteenth_pos * 2 == round(sixteenth_pos * 2):
            return int(random.randint(0, 5) * self.profile.velocity_emphasis)
        return int(random.randint(-15, 0) * self.profile.velocity_emphasis)

    def _handle_ride_velocity(self, measure_position: float) -> int:
        """Calculate velocity adjustment for ride cymbals."""
        # Applying the same 16th-note subdivision emphasis as hi-hats
        sixteenth_pos = round(measure_position * 16) / 16
        if sixteenth_pos.is_integer():
            return int(random.randint(0, 15) * self.profile.velocity_emphasis)
        elif sixteenth_pos * 2 == round(sixteenth_pos * 2):
            return int(random.randint(0, 5) * self.profile.velocity_emphasis)
        return int(random.randint(-15, 0) * self.profile.velocity_emphasis)

    def _handle_cymbal_velocity(self, measure_position: float, measure_idx: int) -> int:
        """Calculate velocity adjustment for cymbals."""
        if measure_position < 0.1 and measure_idx % 2 == 0:
            return int(random.randint(0, 20) * self.profile.velocity_emphasis)
        return int(random.randint(-10, 0) * self.profile.velocity_emphasis)

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
        return int(random.randint(-15, 15) * self.profile.velocity_emphasis)

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
            # Use a deterministic seed based on the pattern key.
            # This ensures that every time this specific pattern occurs, the timing deviation
            # is identical, simulating a drummer's consistent "pocket" or "feel" for that groove.
            pattern_seed = hash((pattern_key[0], pattern_key[1], msg.note))
            random.seed(pattern_seed)
            groove_component = random.uniform(
                -self.config.timing_variation * 0.5, self.config.timing_variation * 0.5
            )
            random.seed(None)
            return groove_component
        return 0

    def _is_downbeat(self, measure_position: float, numerator: int) -> bool:
        """Check if the position corresponds to a downbeat (1, 3 in 4/4)."""
        return measure_position < 0.1 or any(
            abs(measure_position - i) < 0.1 for i in range(2, numerator, 2)
        )

    def _is_backbeat(self, measure_position: float, numerator: int) -> bool:
        """Check if the position corresponds to a backbeat (2, 4 in 4/4)."""
        return any(abs(measure_position - i) < 0.1 for i in range(1, numerator, 2))

    def _is_offbeat_eighth(self, measure_position: float) -> bool:
        """Check if the position is an offbeat eighth note."""
        eighth_note_pos = round(measure_position * 8) / 8
        return eighth_note_pos % 0.5 == 0 and eighth_note_pos % 1.0 != 0

    def _has_kick_or_snare_at_time(self, notes_by_time: Dict, time: int) -> bool:
        """Check if a kick or snare occurs at the specified time."""
        other_drums = notes_by_time.get(time, [])
        return any(
            msg.note in self.KICK_NOTES or msg.note in self.SNARE_NOTES for msg in other_drums
        )

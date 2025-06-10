from dataclasses import dataclass
from typing import Dict, List, Tuple, Set
import random
import mido
from collections import defaultdict
from pathlib import Path

@dataclass
class DrummerProfile:
    timing_bias: float
    velocity_emphasis: float
    ghost_multiplier: float
    kick_timing_tightness: float
    hihat_variation: float
    rushing_factor: float
    groove_consistency: float

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
    def __init__(self, config: HumanizerConfig):
        self.config = config
        self.profile = self._get_drummer_profile()
        self.drum_map = self._get_drum_map()
        self._init_note_groups()
        self.ticks_per_beat = 0
        self.time_sig_numerator = 4
        self.time_sig_denominator = 4
        self.tempo_drift = 0
        self.merged_fills = []

    def _init_note_groups(self):
        self.KICK_NOTES: Set[int] = set()
        self.SNARE_NOTES: Set[int] = set()
        self.HIHAT_NOTES: Set[int] = set()
        self.TOM_NOTES: Set[int] = set()
        self.CYMBAL_NOTES: Set[int] = set()
        self._set_note_groups()

    def _get_drummer_profile(self) -> DrummerProfile:
        # Initialize with default profile
        return DrummerProfile(
            timing_bias=0,
            velocity_emphasis=1.0,
            ghost_multiplier=1.0,
            kick_timing_tightness=1.0,
            hihat_variation=1.0,
            rushing_factor=0,
            groove_consistency=0.8,
        )

    def _get_drum_map(self) -> Dict:
        # Return appropriate drum map based on library
        return {}  # Implement based on your existing drum maps

    def _set_note_groups(self):
        # Initialize note groups based on drum map
        pass  # Implement based on your existing get_note_groups function

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
        other_drums = [n.note for n in notes_by_time.get(time, [])]
        return any(n in self.KICK_NOTES for n in other_drums)

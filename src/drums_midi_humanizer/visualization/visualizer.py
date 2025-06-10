"""Visualization tools for MIDI drum patterns."""

from typing import List, Tuple, Dict
import matplotlib.pyplot as plt
import numpy as np
import mido

class DrumVisualizer:
    """Class for creating visualizations of MIDI drum patterns."""

    def __init__(self):
        """Initialize the visualizer with default settings."""
        self.drum_categories = {
            "Kicks": {"notes": (35, 36), "color": "#FF4444"},
            "Snares": {"notes": (37, 38, 39, 40), "color": "#44FF44"},
            "Hi-hats": {"notes": (42, 44, 46, 54, 56), "color": "#4444FF"},  # Added closed/open variations
            "Toms": {"notes": (41, 43, 45, 47, 48, 50), "color": "#FF44FF"},
            "Cymbals": {"notes": (49, 51, 52, 53, 55, 57, 59, 56, 58), "color": "#FFFF44"},  # Added all cymbal types
        }
        self.analysis_window = 16  # Analysis window in beats for timing variations
        
    def create_comparison_plot(
        self,
        original_messages: List[Tuple[int, mido.Message]],
        humanized_messages: List[Tuple[int, mido.Message]],
        output_path: str,
        ticks_per_beat: int = 480
    ) -> bool:
        """Create a detailed visualization comparing original and humanized MIDI drum patterns.

        Args:
            original_messages: List of (time, message) tuples for original MIDI
            humanized_messages: List of (time, message) tuples for humanized MIDI
            output_path: Path to save the visualization
            ticks_per_beat: MIDI ticks per quarter note

        Returns:
            bool: True if visualization was created successfully
        """
        try:
            print(f"\nGenerating visualization: {output_path}")

            plt.style.use("dark_background")
            fig = plt.figure(figsize=(15, 15))
            gs = plt.GridSpec(5, 1, height_ratios=[5, 5, 3, 3, 1], hspace=0.4)
            
            # Create subplots
            ax_orig = plt.subplot(gs[0])
            ax_human = plt.subplot(gs[1])
            ax_timing = plt.subplot(gs[2])
            ax_velocity = plt.subplot(gs[3])
            ax_legend = plt.subplot(gs[4])

            fig.suptitle(
                "MIDI Drum Pattern Analysis\nOriginal vs. Humanized Pattern",
                fontsize=16,
                y=1.02,
            )

            # Plot patterns
            self._plot_notes(ax_orig, original_messages, alpha=0.8, title="Original Pattern")
            self._plot_notes(ax_human, humanized_messages, alpha=0.8, title="Humanized Pattern")
            
            # Plot timing analysis
            self._plot_timing_analysis(
                ax_timing, 
                original_messages, 
                humanized_messages, 
                ticks_per_beat
            )
            
            # Plot velocity analysis
            self._plot_velocity_analysis(
                ax_velocity,
                original_messages,
                humanized_messages
            )

            # Handle legend
            self._setup_legend(ax_legend, [ax_orig, ax_human])

            # Save and clean up
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            plt.close()
            print(f"Visualization saved to: {output_path}")
            return True

        except Exception as e:
            print(f"Error creating visualization: {str(e)}")
            print(f"Error type: {e.__class__.__name__}")
            return False

    def _plot_notes(
        self,
        ax: plt.Axes,
        messages: List[Tuple[int, mido.Message]],
        alpha: float = 1.0,
        title: str = ""
    ) -> None:
        """Plot MIDI notes on the given axes with improved visualization."""
        times = [t for t, m in messages if m.type == "note_on" and m.velocity > 0]
        notes = [m.note for t, m in messages if m.type == "note_on" and m.velocity > 0]
        velocities = [m.velocity * 3 for t, m in messages if m.type == "note_on" and m.velocity > 0]

        if not times:
            return

        # Plot grid for timing reference
        if times:
            grid_times = np.arange(min(times), max(times) + 1, 480)  # Grid every quarter note
            ax.vlines(grid_times, 35, 60, color='gray', alpha=0.2, linestyle=':')

        # Plot each category with enhanced styling
        for cat_name, cat_info in self.drum_categories.items():
            cat_times, cat_notes, cat_vels = self._filter_category(
                times, notes, velocities, cat_info["notes"]
            )
            if cat_times:
                # Main notes
                ax.scatter(
                    cat_times,
                    cat_notes,
                    alpha=alpha,
                    s=cat_vels,
                    c=cat_info["color"],
                    label=cat_name,
                    edgecolors='white',
                    linewidths=0.5
                )
                
                # Add velocity lines
                for t, n, v in zip(cat_times, cat_notes, cat_vels):
                    ax.vlines(t, n - 0.3, n + 0.3, color=cat_info["color"], 
                            alpha=min(1.0, v/200), linewidth=1)

        ax.set_ylabel("MIDI Note")
        ax.set_title(title)
        ax.grid(True, alpha=0.2)
        ax.set_ylim(34, 60)

    def _filter_category(
        self,
        times: List[int],
        notes: List[int],
        velocities: List[int],
        category_notes: Tuple[int, ...]
    ) -> Tuple[List[int], List[int], List[int]]:
        """Filter notes by category."""
        cat_times = []
        cat_notes = []
        cat_vels = []

        for t, n, v in zip(times, notes, velocities):
            if any(n == note for note in category_notes):
                cat_times.append(t)
                cat_notes.append(n)
                cat_vels.append(v)

        return cat_times, cat_notes, cat_vels

    def _add_grid_lines(self, ax: plt.Axes, times: List[int]) -> None:
        """Add beat grid lines to the plot."""
        if times:
            ticks_per_beat = 480  # Standard MIDI resolution
            max_time = max(times)
            beat_lines = np.arange(0, max_time + ticks_per_beat, ticks_per_beat)
            ax.vlines(
                beat_lines,
                ax.get_ylim()[0],
                ax.get_ylim()[1],
                color="gray",
                alpha=0.2,
                linestyle="--",
            )

    def _setup_axes(self, ax: plt.Axes, title: str, notes: List[int]) -> None:
        """Configure axes labels and appearance."""
        ax.set_title(title)
        ax.set_ylabel("Drum Type")
        ax.grid(True, alpha=0.2)

        # Note labels
        note_names = self._get_note_names()
        unique_notes = sorted(set(notes)) if notes else []
        ax.set_yticks(unique_notes)
        ax.set_yticklabels([note_names.get(n, f"Note {n}") for n in unique_notes])

        # Beat numbers on x-axis
        if notes:
            max_time = max(n for n in notes if isinstance(n, (int, float)))
            ticks_per_beat = 480
            n_beats = int(max_time / ticks_per_beat) + 1
            tick_positions = [i * ticks_per_beat for i in range(n_beats + 1)]
            ax.set_xticks(tick_positions)
            ax.set_xticklabels([str(i) for i in range(n_beats + 1)])
            ax.set_xlabel("Beats")

    def _setup_legend(self, ax_legend: plt.Axes, plot_axes: List[plt.Axes]) -> None:
        """Set up the shared legend."""
        ax_legend.axis("off")
        handles = []
        labels = []
        
        for ax in plot_axes:
            h, l = ax.get_legend_handles_labels()
            for hi, li in zip(h, l):
                if li not in labels:
                    handles.append(hi)
                    labels.append(li)

        ax_legend.legend(
            handles,
            labels,
            loc="center",
            ncol=5,
            bbox_to_anchor=(0.5, 0.5),
            fontsize=10,
        )

    def _get_note_names(self) -> Dict[int, str]:
        """Get dictionary of MIDI note numbers to drum names."""
        return {
            35: "Acoustic Bass Drum",
            36: "Bass Drum 1",
            38: "Acoustic Snare",
            40: "Electric Snare",
            42: "Closed Hi-Hat",
            44: "Pedal Hi-Hat",
            46: "Open Hi-Hat",
            41: "Low Floor Tom",
            43: "High Floor Tom",
            45: "Low Tom",
            47: "Low-Mid Tom",
            48: "Hi-Mid Tom",
            50: "High Tom",
            49: "Crash Cymbal 1",
            51: "Ride Cymbal 1",
            52: "Chinese Cymbal",
            53: "Ride Bell",
            55: "Splash Cymbal",
            57: "Crash Cymbal 2",
            59: "Ride Cymbal 2",
        }

    def _plot_timing_analysis(
        self,
        ax: plt.Axes,
        original_messages: List[Tuple[int, mido.Message]],
        humanized_messages: List[Tuple[int, mido.Message]],
        ticks_per_beat: int
    ) -> None:
        """Plot timing variation analysis."""
        def get_timing_variations(messages):
            times = np.array([t for t, m in messages if m.type == "note_on" and m.velocity > 0])
            if len(times) < 2:
                return []
            intervals = np.diff(times)
            return intervals - np.round(intervals / (ticks_per_beat/4)) * (ticks_per_beat/4)
        
        orig_vars = get_timing_variations(original_messages)
        human_vars = get_timing_variations(humanized_messages)
        
        if orig_vars.size > 0 and human_vars.size > 0:
            bins = np.linspace(-50, 50, 40)
            ax.hist(orig_vars, bins=bins, alpha=0.5, label="Original", color="#888888")
            ax.hist(human_vars, bins=bins, alpha=0.7, label="Humanized", color="#44FF44")
            
            ax.set_xlabel("Timing Variation (ticks)")
            ax.set_ylabel("Count")
            ax.set_title("Timing Variations from Grid")
            ax.legend()
            ax.grid(True, alpha=0.2)

    def _plot_velocity_analysis(
        self,
        ax: plt.Axes,
        original_messages: List[Tuple[int, mido.Message]],
        humanized_messages: List[Tuple[int, mido.Message]]
    ) -> None:
        """Plot velocity distribution analysis."""
        def get_velocities(messages):
            return [m.velocity for _, m in messages if m.type == "note_on" and m.velocity > 0]
            
        orig_vels = get_velocities(original_messages)
        human_vels = get_velocities(humanized_messages)
        
        if orig_vels and human_vels:
            bins = np.linspace(0, 127, 40)
            ax.hist(orig_vels, bins=bins, alpha=0.5, label="Original", color="#888888")
            ax.hist(human_vels, bins=bins, alpha=0.7, label="Humanized", color="#44FF44")
            
            ax.set_xlabel("Velocity")
            ax.set_ylabel("Count")
            ax.set_title("Velocity Distribution")
            ax.legend()
            ax.grid(True, alpha=0.2)

def create_drum_visualization(original_messages: List[Tuple[int, int, int]],
                            humanized_messages: List[Tuple[int, int, int]],
                            output_png: str) -> None:
    """Create a detailed visualization comparing original and humanized MIDI drum patterns.

    Args:
        original_messages: List of (time, note, velocity) tuples from original MIDI.
        humanized_messages: List of (time, note, velocity) tuples from humanized MIDI.
        output_png: Path to save the output PNG visualization.
    """
    # Set dark style for better visibility
    plt.style.use("dark_background")
    fig = plt.figure(figsize=(15, 12))
    gs = plt.GridSpec(3, 1, height_ratios=[5, 5, 1], hspace=0.3)

    # Categories for grouping drums
    categories = {
        "Kicks": list(range(35, 37)),
        "Snares": list(range(37, 41)),
        "Hi-hats": list(range(42, 47)),
        "Toms": list(range(41, 51)),
        "Cymbals": list(range(51, 60))
    }

    # Plot original notes
    ax1 = plt.subplot(gs[0])
    _plot_drum_grid(original_messages, categories, ax1, "Original MIDI")

    # Plot humanized notes
    ax2 = plt.subplot(gs[1])
    _plot_drum_grid(humanized_messages, categories, ax2, "Humanized MIDI")

    # Plot velocity differences
    ax3 = plt.subplot(gs[2])
    _plot_velocity_differences(original_messages, humanized_messages, ax3)

    plt.tight_layout()
    plt.savefig(output_png, dpi=300, bbox_inches='tight')
    plt.close()


def _plot_drum_grid(messages: List[Tuple[int, int, int]], categories: dict,
                   ax: plt.Axes, title: str) -> None:
    """Plot a grid of drum hits colored by category and sized by velocity."""
    colors = plt.cm.tab10(np.linspace(0, 1, len(categories)))
    
    for i, (cat, notes) in enumerate(categories.items()):
        times, vels = [], []
        for t, n, v in messages:
            if n in notes:
                times.append(t)
                vels.append(v)
        if times:  # Only plot if we have notes in this category
            ax.scatter(times, [i] * len(times), s=np.array(vels) * 2,
                      c=[colors[i]], alpha=0.6, label=cat)

    ax.set_title(title)
    ax.set_yticks(range(len(categories)))
    ax.set_yticklabels(categories.keys())
    ax.grid(True, alpha=0.2)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')


def _plot_velocity_differences(original: List[Tuple[int, int, int]],
                             humanized: List[Tuple[int, int, int]],
                             ax: plt.Axes) -> None:
    """Plot the velocity differences between original and humanized notes."""
    # Create lookup for humanized velocities
    humanized_dict = {(t, n): v for t, n, v in humanized}
    
    # Calculate differences
    times, diffs = [], []
    for t, n, v in original:
        if (t, n) in humanized_dict:
            times.append(t)
            diffs.append(humanized_dict[(t, n)] - v)

    # Plot differences
    if times:
        ax.bar(times, diffs, alpha=0.6, width=50)
        ax.axhline(y=0, color='w', linestyle='-', alpha=0.2)
        ax.set_title("Velocity Differences")
        ax.set_ylabel("Δ Velocity")
        ax.grid(True, alpha=0.2)

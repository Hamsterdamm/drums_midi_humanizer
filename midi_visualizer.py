import matplotlib.pyplot as plt
import numpy as np


def create_comparison_plot(original_messages, humanized_messages, output_path=None):
    """
    Create a visualization comparing original and humanized MIDI data.

    Args:
        original_messages: List of (time, msg) tuples from original MIDI
        humanized_messages: List of (time, msg) tuples from humanized MIDI
        output_path: Optional path to save the plot
    """
    plt.style.use("dark_background")
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 10), height_ratios=[2, 1, 1])
    fig.suptitle("MIDI Drum Pattern Comparison\nOriginal vs. Humanized", fontsize=16)

    # Colors for different drum types
    colors = {
        "kick": "#FF4444",
        "snare": "#44FF44",
        "hihat": "#4444FF",
        "tom": "#FF44FF",
        "cymbal": "#FFFF44",
        "other": "#FFFFFF",
    }

    def get_drum_type(note):
        # Categorize notes by type
        if note in range(35, 37):  # Kick drums
            return "kick"
        elif note in range(37, 41):  # Snares
            return "snare"
        elif note in (42, 44, 46):  # Hi-hats
            return "hihat"
        elif note in (41, 43, 45, 47, 48, 50):  # Toms
            return "tom"
        elif note in (49, 51, 52, 53, 55, 57, 59):  # Cymbals
            return "cymbal"
        return "other"

    # Plot note events
    for i, (messages, alpha) in enumerate(
        [(original_messages, 0.5), (humanized_messages, 1.0)]
    ):
        times = []
        notes = []
        velocities = []
        colors_list = []

        # Collect note data
        for time, msg in messages:
            if msg.type == "note_on" and msg.velocity > 0:
                times.append(time)
                notes.append(msg.note)
                velocities.append(msg.velocity)
                drum_type = get_drum_type(msg.note)
                colors_list.append(colors[drum_type])

        # Plot notes
        if times:  # Only plot if we have data
            ax1.scatter(
                times, notes, c=colors_list, alpha=alpha, s=velocities, marker="o"
            )

    # Customize the note plot
    ax1.set_ylabel("MIDI Note")
    ax1.set_title("Note Events")
    ax1.grid(True, alpha=0.3)

    # Plot velocity distribution
    def plot_velocities(ax, messages, label, alpha=1.0):
        velocities = [
            msg.velocity
            for _, msg in messages
            if msg.type == "note_on" and msg.velocity > 0
        ]
        if velocities:
            ax.hist(velocities, bins=30, alpha=alpha, label=label, color="#44FF44")
            ax.legend()
            ax.grid(True, alpha=0.3)

    plot_velocities(ax2, original_messages, "Original", alpha=0.5)
    plot_velocities(ax2, humanized_messages, "Humanized", alpha=0.7)
    ax2.set_ylabel("Count")
    ax2.set_title("Velocity Distribution")

    # Plot timing variations
    def plot_timing_variations(ax, messages, label, alpha=1.0):
        times = [
            time for time, msg in messages if msg.type == "note_on" and msg.velocity > 0
        ]
        if len(times) > 1:
            intervals = np.diff(times)
            ax.hist(intervals, bins=30, alpha=alpha, label=label, color="#4444FF")
            ax.legend()
            ax.grid(True, alpha=0.3)

    plot_timing_variations(ax3, original_messages, "Original", alpha=0.5)
    plot_timing_variations(ax3, humanized_messages, "Humanized", alpha=0.7)
    ax3.set_xlabel("Time (ticks)")
    ax3.set_ylabel("Count")
    ax3.set_title("Timing Intervals")

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()

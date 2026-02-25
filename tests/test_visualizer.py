"""Tests for the DrumVisualizer."""

from unittest.mock import patch

import mido
import pytest

from drums_midi_humanizer.visualization.visualizer import DrumVisualizer


@pytest.fixture
def visualizer():
    return DrumVisualizer()


def test_visualizer_init(visualizer):
    """Test initialization of visualizer categories."""
    assert "Kicks" in visualizer.drum_categories
    assert visualizer.analysis_window == 16


def test_filter_category(visualizer):
    """Test filtering notes by category."""
    times = [0, 100, 200]
    notes = [36, 38, 36]  # Kick, Snare, Kick
    velocities = [100, 100, 100]
    category_notes = (36,)

    cat_times, cat_notes, cat_vels = visualizer._filter_category(
        times, notes, velocities, category_notes
    )

    assert len(cat_times) == 2
    assert cat_notes == [36, 36]
    assert cat_vels == [100, 100]


def test_get_note_names(visualizer):
    """Test note name mapping."""
    names = visualizer._get_note_names()
    assert names[36] == "Bass Drum 1"
    assert names[38] == "Acoustic Snare"


@patch("matplotlib.pyplot.subplot")
@patch("matplotlib.pyplot.figure")
def test_create_comparison_plot(mock_figure, mock_subplot, visualizer, tmp_path):
    """Test plot creation flow (mocking matplotlib)."""
    orig_msgs = [(0, mido.Message("note_on", note=36, velocity=100))]
    human_msgs = [(0, mido.Message("note_on", note=36, velocity=105))]

    output_path = str(tmp_path / "test_plot.png")

    # Configure mock axes to return valid values for unpacking
    mock_ax = mock_subplot.return_value
    mock_ax.get_legend_handles_labels.return_value = ([], [])

    # Mock plt to avoid actual rendering
    with patch("matplotlib.pyplot.savefig"), patch("matplotlib.pyplot.close"):
        result = visualizer.create_comparison_plot(orig_msgs, human_msgs, output_path)
        assert result is True

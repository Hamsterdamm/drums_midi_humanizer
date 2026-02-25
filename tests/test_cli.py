"""Tests for the CLI argument parsing."""

import sys
from unittest.mock import patch
from drums_midi_humanizer.cli import parse_args

def test_parse_args_defaults():
    """Test default argument values."""
    test_args = ["prog", "input.mid"]
    with patch.object(sys, 'argv', test_args):
        args = parse_args()
        assert args.input_file == "input.mid"
        assert args.output is None
        assert args.style == "balanced"
        assert args.library == "gm"
        assert args.visualize is False
        assert args.timing == 10
        assert args.velocity == 15
        assert args.ghost == 0.1
        assert args.accent == 0.2
        assert args.shuffle == 0.0
        assert args.flams == 0.0

def test_parse_args_custom():
    """Test custom argument parsing."""
    test_args = [
        "prog", "input.mid",
        "--output", "out.mid",
        "--style", "rock",
        "--library", "sd3",
        "--visualize",
        "--timing", "20",
        "--velocity", "25",
        "--ghost", "0.3",
        "--accent", "0.4",
        "--shuffle", "0.1",
        "--flams", "0.2",
    ]
    with patch.object(sys, 'argv', test_args):
        args = parse_args()
        assert args.input_file == "input.mid"
        assert args.output == "out.mid"
        assert args.style == "rock"
        assert args.library == "sd3"
        assert args.visualize is True
        assert args.timing == 20
        assert args.velocity == 25
        assert args.ghost == 0.3
        assert args.accent == 0.4
        assert args.shuffle == 0.1
        assert args.flams == 0.2
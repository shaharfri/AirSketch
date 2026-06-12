"""Tests for the voice command parser (no audio/model needed)."""
import pytest

from airsketch.classroom.voice_commands import IntentType, parse_command


class TestDraw:
    def test_draw_a_triangle(self):
        i = parse_command("draw a triangle")
        assert i.type == IntentType.DRAW
        assert i.target == "triangle"

    def test_draw_house(self):
        i = parse_command("Draw a house!")
        assert i.type == IntentType.DRAW
        assert i.target == "house"

    def test_lets_draw_a_cat(self):
        i = parse_command("ok everyone, let's draw a cat")
        assert i.type == IntentType.DRAW
        assert i.target == "cat"

    def test_bare_target_word(self):
        i = parse_command("triangle")
        assert i.type == IntentType.DRAW
        assert i.target == "triangle"

    def test_synonym_plane(self):
        i = parse_command("draw a plane")
        assert i.type == IntentType.DRAW
        assert i.target == "airplane"

    def test_synonym_box(self):
        i = parse_command("draw a box")
        assert i.type == IntentType.DRAW
        assert i.target == "square"


class TestControl:
    def test_next(self):
        assert parse_command("next").type == IntentType.NEXT
        assert parse_command("next challenge please").type == IntentType.NEXT
        assert parse_command("another one").type == IntentType.NEXT

    def test_submit(self):
        assert parse_command("submit").type == IntentType.SUBMIT
        assert parse_command("I'm done").type == IntentType.SUBMIT
        assert parse_command("check it").type == IntentType.SUBMIT

    def test_clear(self):
        assert parse_command("clear").type == IntentType.CLEAR
        assert parse_command("erase that").type == IntentType.CLEAR
        assert parse_command("start over").type == IntentType.CLEAR

    def test_retry(self):
        assert parse_command("again").type == IntentType.RETRY
        assert parse_command("try again").type == IntentType.RETRY
        assert parse_command("retry").type == IntentType.RETRY
        assert parse_command("once more").type == IntentType.RETRY

    def test_next_still_distinct_from_retry(self):
        assert parse_command("next").type == IntentType.NEXT
        assert parse_command("another one").type == IntentType.NEXT

    def test_capture_board(self):
        assert parse_command("read the board").type == IntentType.CAPTURE_BOARD
        assert parse_command("capture the board").type == IntentType.CAPTURE_BOARD
        assert parse_command("whiteboard").type == IntentType.CAPTURE_BOARD
        assert parse_command("board").type == IntentType.CAPTURE_BOARD
        assert parse_command("take a picture of the board").type == IntentType.CAPTURE_BOARD

    def test_board_mention_without_verb_is_dictation(self):
        # A long sentence merely mentioning the board should NOT trigger capture
        i = parse_command("everyone please look at the board over there now")
        assert i.type == IntentType.DICTATION


class TestTheme:
    def test_switch_geometry(self):
        i = parse_command("switch to geometry")
        assert i.type == IntentType.SET_THEME
        assert i.target == "geometry"

    def test_objects_mode(self):
        i = parse_command("objects mode")
        assert i.type == IntentType.SET_THEME
        assert i.target == "objects"

    def test_bare_mixed(self):
        i = parse_command("mixed")
        assert i.type == IntentType.SET_THEME
        assert i.target == "mixed"


class TestFuzzy:
    """Mis-heard speech (Whisper-base is imperfect) should still map to DRAW."""

    def test_misheard_circle(self):
        i = parse_command("draw a sirkle")
        assert i.type == IntentType.DRAW and i.target == "circle"

    def test_misheard_triangle(self):
        i = parse_command("drew a tryangle")
        assert i.type == IntentType.DRAW and i.target == "triangle"

    def test_misheard_house(self):
        i = parse_command("draw a haus")
        assert i.type == IntentType.DRAW and i.target == "house"

    def test_long_dictation_not_drawn(self):
        # A shape word in a long sentence without a draw-verb stays dictation
        i = parse_command("today we will learn about the angles inside a triangle shape")
        assert i.type == IntentType.DICTATION


class TestFallback:
    def test_dictation(self):
        i = parse_command("today we will learn about the area of a triangle formula")
        # Long sentence with a shape word but no draw verb -> dictation
        assert i.type == IntentType.DICTATION

    def test_empty(self):
        assert parse_command("").type == IntentType.NONE
        assert parse_command("   ").type == IntentType.NONE

    def test_text_preserved(self):
        i = parse_command("Draw a Triangle")
        assert i.text == "Draw a Triangle"

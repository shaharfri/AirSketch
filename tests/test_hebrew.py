"""Tests for Hebrew language support across the OpenVINO-native paths:
voice-command parsing, the Whisper language token, and the board-LLM prompt.
No models are loaded (pure logic / static helpers)."""
import importlib.util
import os
import sys

import pytest

from airsketch.classroom.voice_commands import IntentType, parse_command


# ----------------------------------------------------------------- voice commands
class TestHebrewCommands:
    def test_draw_shapes(self):
        cases = {
            "צייר משולש": "triangle",
            "ציירי עיגול": "circle",
            "צייר ריבוע": "square",
            "צייר מלבן": "rectangle",
            "צייר כוכב": "star",
            "צייר בית": "house",
            "תצייר חתול": "cat",
            "צייר מטוס": "airplane",
        }
        for text, target in cases.items():
            intent = parse_command(text)
            assert intent.type == IntentType.DRAW, text
            assert intent.target == target, text

    def test_bare_target_word(self):
        assert parse_command("משולש").type == IntentType.DRAW
        assert parse_command("משולש").target == "triangle"

    def test_next(self):
        assert parse_command("הבא").type == IntentType.NEXT
        assert parse_command("קדימה").type == IntentType.NEXT

    def test_retry(self):
        assert parse_command("שוב").type == IntentType.RETRY
        assert parse_command("עוד פעם").type == IntentType.RETRY

    def test_submit(self):
        assert parse_command("שלח").type == IntentType.SUBMIT
        assert parse_command("סיימתי").type == IntentType.SUBMIT

    def test_clear(self):
        assert parse_command("נקה").type == IntentType.CLEAR
        assert parse_command("מחק").type == IntentType.CLEAR

    def test_theme(self):
        assert parse_command("צורות").type == IntentType.SET_THEME
        assert parse_command("צורות").target == "geometry"
        assert parse_command("חפצים").target == "objects"
        assert parse_command("מעורב").target == "mixed"

    def test_board_capture(self):
        assert parse_command("קרא את הלוח").type == IntentType.CAPTURE_BOARD
        assert parse_command("לוח").type == IntentType.CAPTURE_BOARD

    def test_dictation_fallback(self):
        # A normal Hebrew sentence with no command words → dictation.
        intent = parse_command("היום נלמד על השברים בכיתה")
        assert intent.type == IntentType.DICTATION

    def test_english_still_works(self):
        # Regression: Hebrew additions must not break English parsing.
        assert parse_command("draw a triangle").type == IntentType.DRAW
        assert parse_command("draw a triangle").target == "triangle"
        assert parse_command("next").type == IntentType.NEXT
        assert parse_command("clear").type == IntentType.CLEAR


# ----------------------------------------------------------------- whisper token
class TestWhisperLanguageToken:
    def test_token_mapping(self):
        from airsketch.voice import VoiceRecorder
        assert VoiceRecorder.language_token("he") == "<|he|>"
        assert VoiceRecorder.language_token("hebrew") == "<|he|>"
        assert VoiceRecorder.language_token("en") == "<|en|>"
        assert VoiceRecorder.language_token("anything-else") == "<|en|>"


# ----------------------------------------------------------------- LLM prompt
class TestLessonLLMPrompt:
    def test_hebrew_prompt_uses_hebrew_instructions(self):
        from airsketch.lesson_llm import LessonUnderstander
        he = LessonUnderstander._build_prompt("2+2=4", language="he")
        en = LessonUnderstander._build_prompt("2+2=4", language="en")
        assert "ענה בעברית" in he          # "answer in Hebrew"
        assert "עוזר הוראה" in he           # "teaching assistant"
        assert "2+2=4" in he                # board text embedded
        assert "ענה בעברית" not in en       # English prompt unaffected
        assert "teaching assistant" in en


# ----------------------------------------------------------------- CLI / launcher
class TestHebrewRenderer:
    def test_has_hebrew(self):
        from airsketch import hebrew_text as ht
        assert ht.has_hebrew("שלום")
        assert ht.has_hebrew("Round 3 שלב")
        assert not ht.has_hebrew("Hello 123")

    def test_to_visual_reverses_hebrew(self):
        from airsketch import hebrew_text as ht
        # pure Hebrew -> reversed to visual order
        assert ht.to_visual("אבג") == "גבא"

    def test_to_visual_keeps_digit_runs_ltr(self):
        from airsketch import hebrew_text as ht
        # numbers must stay readable (not reversed to "21")
        assert "12" in ht.to_visual("אב 12")

    def test_measure_positive(self):
        from airsketch import hebrew_text as ht
        w, h = ht.measure("שלום", 1.0)
        assert w > 0 and h > 0

    def test_draw_text_marks_frame(self):
        import numpy as np
        from airsketch import hebrew_text as ht
        frame = np.zeros((60, 300, 3), dtype=np.uint8)
        ok = ht.draw_text(frame, "שלום", (10, 40), 1.0, (255, 255, 255))
        assert ok is True                      # PIL + font available
        assert frame.sum() > 0                 # something was drawn

    def test_utils_text_size_hebrew_vs_ascii(self):
        from airsketch.utils import text_size
        assert text_size("שלום", 1.0)[0] > 0
        assert text_size("Hello", 1.0)[0] > 0


def test_lang_flag_parses_in_main():
    from airsketch.main import parse_args
    old = sys.argv
    try:
        sys.argv = ["airsketch", "--classroom", "--lang", "he"]
        cfg = parse_args()
    finally:
        sys.argv = old
    assert cfg.language == "he"


def test_launcher_emits_lang_flag():
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "launcher.py")
    spec = importlib.util.spec_from_file_location("airsketch_launcher_he", path)
    launcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launcher)
    # default 'en' -> no flag; 'he' -> --lang he
    assert "--lang" not in launcher.build_args({"mode": "classroom", "language": "en"})
    args = launcher.build_args({"mode": "classroom", "language": "he"})
    assert args[args.index("--lang") + 1] == "he"

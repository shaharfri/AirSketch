"""Tests for the lesson-LLM response parser (no model / inference needed)."""
from airsketch.lesson_llm import Understanding, parse_understanding


def test_well_formed_json():
    text = (
        '{"summary": "A lesson on adding fractions.", "topic": "Fractions", '
        '"key_points": ["1/2 + 1/4 = 3/4", "denominators must match"], '
        '"corrected": "1/2 + 1/4 = 3/4"}'
    )
    u = parse_understanding(text)
    assert u.topic == "Fractions"
    assert u.summary.startswith("A lesson")
    assert u.key_points == ["1/2 + 1/4 = 3/4", "denominators must match"]


def test_strips_think_block_and_im_end():
    text = (
        "<think>let me reason</think>"
        '{"summary": "s", "topic": "t", "key_points": [], "corrected": "c"}'
        "<|im_end|>trailing"
    )
    u = parse_understanding(text)
    assert u.summary == "s" and u.topic == "t"


def test_key_points_non_list_coerced():
    u = parse_understanding('{"summary":"s","topic":"t","key_points":"only one"}')
    assert u.key_points == ["only one"]


def test_non_json_fallback_uses_text_as_summary():
    u = parse_understanding("This board covers photosynthesis basics.",
                            fallback_text="raw ocr")
    assert "photosynthesis" in u.summary
    assert u.corrected == "raw ocr"
    assert u.key_points == []


def test_empty():
    u = parse_understanding("")
    assert isinstance(u, Understanding)
    assert u.summary == ""

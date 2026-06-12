"""Tests for the OCR line-assembly logic (no model / inference needed)."""
import numpy as np

from airsketch.ocr_reader import TextLine, assemble_text


def _line(text, cx, cy, h=20.0, conf=0.95):
    box = np.array([[cx - 30, cy - h / 2], [cx + 30, cy - h / 2],
                    [cx + 30, cy + h / 2], [cx - 30, cy + h / 2]], np.float32)
    return TextLine(text=text, confidence=conf, box=box, cx=cx, cy=cy, height=h)


def test_empty():
    assert assemble_text([]) == ""


def test_orders_top_to_bottom():
    lines = [_line("third", 100, 300), _line("first", 100, 100), _line("second", 100, 200)]
    assert assemble_text(lines) == "first\nsecond\nthird"


def test_same_row_joined_left_to_right():
    # Two boxes at nearly the same y -> one row, ordered by x
    lines = [_line("world", 300, 100), _line("hello", 100, 100)]
    assert assemble_text(lines) == "hello world"


def test_rows_separated_by_height():
    lines = [
        _line("a", 100, 100, h=20),
        _line("b", 100, 160, h=20),   # ~60px below -> separate row
    ]
    assert assemble_text(lines) == "a\nb"

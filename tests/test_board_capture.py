"""Tests for board capture data model (no OCR model / camera needed)."""
from airsketch.board_capture import BoardNote


def test_board_note_defaults():
    note = BoardNote(timestamp="20260601_120000",
                     transcription="hello", summary="greeting")
    assert note.items == []
    assert note.image_path == ""
    assert note.round_index == 0
    assert note.raw_response == ""


def test_board_note_full():
    note = BoardNote(
        timestamp="20260601_120000",
        transcription="line one\nline two",
        summary="line one",
        items=["line one", "line two"],
        image_path="outputs/board_x.png",
        round_index=3,
    )
    assert note.items == ["line one", "line two"]
    assert note.round_index == 3

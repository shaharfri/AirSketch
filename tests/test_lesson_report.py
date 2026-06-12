"""Tests for the Phase 4 lesson report HTML export (no model / camera needed)."""
from airsketch.board_capture import BoardNote
from airsketch.classroom.challenge_engine import Challenge, ChallengeResult
from airsketch.exporter import export_lesson_report


def _result(target, detected, score, stars, passed, rnd):
    ch = Challenge(target=target, theme="geometry", round_index=rnd)
    return ChallengeResult(challenge=ch, detected=detected, score=score,
                           stars=stars, passed=passed)


def test_report_written_with_notes_and_scoreboard(tmp_path):
    notes = [BoardNote(timestamp="20260601_120000",
                       transcription="1/2 + 1/4 = 3/4\nKey points:",
                       summary="Adding fractions lesson",
                       items=["numerator on top", "denominators must match"],
                       topic="Fractions", round_index=2)]
    results = [_result("circle", "circle", 96, 3, True, 1),
               _result("triangle", "rectangle", 0, 0, False, 2)]
    out = tmp_path / "lesson.html"
    path = export_lesson_report(notes, results, str(out), session_label="Theme: geometry")

    html = out.read_text(encoding="utf-8")
    assert path == str(out)
    assert "Lesson Report" in html
    assert "Adding fractions lesson" in html
    assert "Fractions" in html
    assert "denominators must match" in html
    assert "1/2 + 1/4 = 3/4" in html          # transcription embedded
    assert "circle" in html and "triangle" in html  # scoreboard rows
    assert "Theme: geometry" in html


def test_report_includes_narration(tmp_path):
    out = tmp_path / "narr.html"
    export_lesson_report([], [], str(out),
                         narration=["Today we cover fractions.", "Remember the denominator."])
    html = out.read_text(encoding="utf-8")
    assert "Lesson narration" in html
    assert "Today we cover fractions." in html
    assert "Remember the denominator." in html


def test_report_handles_empty(tmp_path):
    out = tmp_path / "empty.html"
    export_lesson_report([], [], str(out))
    html = out.read_text(encoding="utf-8")
    assert "Lesson Report" in html
    assert "No challenges played" in html
    assert "No board captures" in html


def test_report_escapes_html(tmp_path):
    notes = [BoardNote(timestamp="t", transcription="<script>x</script>",
                       summary="a & b <c>", topic="t")]
    out = tmp_path / "esc.html"
    export_lesson_report(notes, [], str(out))
    html = out.read_text(encoding="utf-8")
    assert "<script>x</script>" not in html       # raw transcription not injected
    assert "&lt;script&gt;" in html

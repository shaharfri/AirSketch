"""Notebook session export — single-file HTML + JSON."""
import base64
import html as html_module
import json
import os
import time
from typing import List

import cv2
import numpy as np

from airsketch.stroke import Diagram, DiagramStatus


def _img_to_b64_png(img: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", img)
    return base64.b64encode(buf.tobytes()).decode("ascii") if ok else ""


def _file_to_b64_png(path: str) -> str:
    """Read an image file from disk and return base64 PNG (re-encoded)."""
    if not path or not os.path.exists(path):
        return ""
    img = cv2.imread(path)
    return _img_to_b64_png(img) if img is not None else ""


_HTML_STYLE = """
:root {
  --bg: #0f1116;
  --card: #181c24;
  --card-2: #1f242f;
  --text: #e8ecf2;
  --muted: #8a93a3;
  --accent: #00e5b8;
  --accent-2: #6c8cff;
  --tag-bg: #20283a;
  --tag-fg: #aecbff;
  --shadow: 0 8px 30px rgba(0,0,0,0.35);
}
* { box-sizing: border-box; }
body {
  font-family: -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
  background: linear-gradient(180deg, #0b0d12 0%, #131720 100%);
  color: var(--text);
  margin: 0;
  padding: 32px 16px 64px;
  min-height: 100vh;
}
.container { max-width: 960px; margin: 0 auto; }
header {
  display: flex; flex-direction: column; gap: 6px;
  padding: 0 4px 20px;
  border-bottom: 1px solid #22283540;
  margin-bottom: 28px;
}
header h1 { margin: 0; font-size: 28px; letter-spacing: 0.2px; }
header .sub {
  color: var(--muted); font-size: 14px;
  display: flex; gap: 14px; flex-wrap: wrap;
}
.diagram {
  background: var(--card);
  border-radius: 14px;
  padding: 22px 24px;
  margin: 18px 0;
  box-shadow: var(--shadow);
  display: grid;
  grid-template-columns: 1fr 280px;
  gap: 24px;
  align-items: start;
}
@media (max-width: 720px) {
  .diagram { grid-template-columns: 1fr; }
}
.info .topic {
  display: inline-block;
  color: var(--accent);
  font-size: 11px;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.12em;
  margin-bottom: 6px;
}
.info h2 {
  margin: 0 0 10px;
  font-size: 22px;
  line-height: 1.2;
}
.info .num { color: var(--muted); font-weight: 500; }
.info .desc { font-size: 15px; line-height: 1.55; color: #d4d8e0; }
.info .meta {
  margin-top: 14px; font-size: 12px; color: var(--muted);
  display: flex; gap: 14px; flex-wrap: wrap;
}
.tags { margin-top: 12px; display: flex; flex-wrap: wrap; gap: 6px; }
.tag {
  background: var(--tag-bg);
  color: var(--tag-fg);
  padding: 3px 10px;
  border-radius: 999px;
  font-size: 12px;
  border: 1px solid #2a3450;
}
.thumbs { display: flex; flex-direction: column; gap: 10px; }
.thumb {
  background: #fff;
  border-radius: 10px;
  overflow: hidden;
  border: 1px solid #2a3041;
  padding: 8px;
  display: flex; align-items: center; justify-content: center;
  position: relative;
}
.thumb img { display: block; width: 100%; height: auto; image-rendering: -webkit-optimize-contrast; }
.thumb .lbl {
  position: absolute; top: 6px; left: 8px;
  background: rgba(0,0,0,0.55); color: #fff;
  font-size: 10px; letter-spacing: 0.08em; text-transform: uppercase;
  padding: 2px 7px; border-radius: 4px;
}
.thumb.clean { border: 1px solid #4d6;}
.thumb.clean .lbl { background: rgba(40,150,80,0.85); }
.thumb.raw .lbl { background: rgba(150,90,40,0.85); }
.empty { color: var(--muted); padding: 40px 0; text-align: center; }
.failed { color: #ff7a7a; }
.pending { color: #ffd980; }
.footer { margin-top: 40px; color: var(--muted); font-size: 12px; text-align: center; }
"""


def _render_diagram_html(idx: int, d: Diagram) -> str:
    title = "Untitled"
    desc = ""
    topic = "sketch"
    tags: list = []
    status_class = ""

    if d.analysis:
        title = d.analysis.title
        desc = d.analysis.description
        topic = d.analysis.topic
        tags = d.analysis.tags
    elif d.status == DiagramStatus.FAILED:
        title = "Analysis Failed"
        desc = d.error or "Unknown error."
        topic = "error"
        status_class = "failed"
    else:
        title = "Pending Analysis"
        topic = "pending"
        status_class = "pending"

    tags_html = "".join(
        f"<span class='tag'>{html_module.escape(str(t))}</span>" for t in tags
    )

    img_b64 = _img_to_b64_png(d.canvas) if d.canvas is not None else ""
    thumbs_html = (
        f"<div class='thumb'>"
        f"<img src='data:image/png;base64,{img_b64}' alt='diagram'>"
        f"</div>" if img_b64 else ""
    )

    analyzer = d.analysis.analyzer_name if d.analysis else "n/a"
    prim_summary = ""
    if d.primitives:
        from collections import Counter as _C
        c = _C(p.kind for p in d.primitives)
        prim_summary = ", ".join(f"{n}× {lbl}" for lbl, n in c.most_common())
    meta = (
        f"<span>{len(d.strokes)} strokes</span>"
        f"<span>{d.total_points} points</span>"
        + (f"<span>primitives: {html_module.escape(prim_summary)}</span>" if prim_summary else "")
        + f"<span>analyzer: {html_module.escape(analyzer)}</span>"
    )

    return (
        f"<article class='diagram'>"
        f"  <div class='info'>"
        f"    <div class='topic {status_class}'>{html_module.escape(topic)}</div>"
        f"    <h2><span class='num'>{idx}.</span> {html_module.escape(title)}</h2>"
        f"    <div class='desc'>{html_module.escape(desc)}</div>"
        f"    <div class='tags'>{tags_html}</div>"
        f"    <div class='meta'>{meta}</div>"
        f"  </div>"
        f"  <div class='thumbs'>{thumbs_html}</div>"
        f"</article>"
    )


def export_html(
    diagrams: List[Diagram],
    output_path: str,
    session_label: str = "",
) -> str:
    """Write a self-contained HTML notebook to `output_path`."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)

    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    parts = [
        "<!DOCTYPE html>",
        "<html lang='en'><head>",
        "<meta charset='utf-8'>",
        "<meta name='viewport' content='width=device-width,initial-scale=1'>",
        "<title>AirNotes — Session</title>",
        f"<style>{_HTML_STYLE}</style>",
        "</head><body>",
        "<div class='container'>",
        "<header>",
        "<h1>AirNotes — Session</h1>",
        "<div class='sub'>",
        f"<span>{html_module.escape(session_label)}</span>" if session_label else "",
        f"<span>generated {timestamp}</span>",
        f"<span>{len(diagrams)} diagram{'s' if len(diagrams) != 1 else ''}</span>",
        "</div>",
        "</header>",
    ]

    if not diagrams:
        parts.append("<div class='empty'>No diagrams in this session.</div>")
    else:
        for i, d in enumerate(diagrams, 1):
            parts.append(_render_diagram_html(i, d))

    parts.append(
        "<div class='footer'>Created with AirDraw AR · AirNotes mode</div>"
    )
    parts.append("</div></body></html>")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("".join(parts))
    return output_path


_LESSON_STYLE = """
.report-grid { display: grid; grid-template-columns: 280px 1fr; gap: 22px; align-items: start; }
@media (max-width: 720px) { .report-grid { grid-template-columns: 1fr; } }
.board-shot { background:#fff; border-radius:10px; border:1px solid #2a3041; padding:6px; }
.board-shot img { display:block; width:100%; height:auto; border-radius:6px; }
.kpts { margin:10px 0 0; padding-left:18px; }
.kpts li { margin:4px 0; color:#d4d8e0; }
.transcript {
  white-space: pre-wrap; font-family: ui-monospace, "Cascadia Code", Consolas, monospace;
  background: var(--card-2); border:1px solid #2a3041; border-radius:8px;
  padding:12px 14px; font-size:13px; color:#c9d2e0; margin-top:12px;
}
details > summary { cursor:pointer; color: var(--muted); font-size:13px; margin-top:8px; }
table.score { width:100%; border-collapse:collapse; margin-top:8px; font-size:14px; }
table.score th, table.score td { text-align:left; padding:8px 10px; border-bottom:1px solid #232a38; }
table.score th { color: var(--muted); font-weight:600; font-size:12px; text-transform:uppercase; letter-spacing:0.08em; }
.pass { color: var(--accent); } .fail { color:#ff7a7a; }
.stars { color:#ffd166; letter-spacing:2px; }
.totals { display:flex; gap:22px; flex-wrap:wrap; margin:10px 2px 0; color:#d4d8e0; font-size:15px; }
.totals b { color: var(--accent); }
.section-title { margin:34px 0 6px; font-size:20px; }
"""


def _render_board_note_html(idx: int, note) -> str:
    img_b64 = _file_to_b64_png(getattr(note, "image_path", ""))
    shot = (f"<div class='board-shot'><img src='data:image/png;base64,{img_b64}'"
            f" alt='board'></div>" if img_b64 else "")
    topic = getattr(note, "topic", "") or "board"
    summary = getattr(note, "summary", "") or ""
    items = getattr(note, "items", []) or []
    transcription = getattr(note, "transcription", "") or ""
    rnd = getattr(note, "round_index", 0)
    ts = getattr(note, "timestamp", "")

    kpts = "".join(f"<li>{html_module.escape(str(k))}</li>" for k in items)
    kpts_html = f"<ul class='kpts'>{kpts}</ul>" if kpts else ""
    meta = []
    if ts:
        meta.append(f"captured {html_module.escape(ts)}")
    if rnd:
        meta.append(f"during round {rnd}")
    meta_html = (f"<div class='meta'><span>" + "</span><span>".join(meta)
                 + "</span></div>") if meta else ""
    transcript_html = (
        "<details><summary>Raw OCR transcription</summary>"
        f"<div class='transcript'>{html_module.escape(transcription)}</div></details>"
        if transcription else ""
    )

    return (
        "<article class='diagram'>"
        "  <div class='report-grid'>"
        f"    <div class='thumbs'>{shot}</div>"
        "    <div class='info'>"
        f"      <div class='topic'>{html_module.escape(topic)}</div>"
        f"      <h2><span class='num'>{idx}.</span> {html_module.escape(summary) or 'Board capture'}</h2>"
        f"      {kpts_html}"
        f"      {meta_html}"
        f"      {transcript_html}"
        "    </div>"
        "  </div>"
        "</article>"
    )


def _render_scoreboard_html(results: list) -> str:
    if not results:
        return "<div class='empty'>No challenges played this session.</div>"
    rows = []
    total_score = total_stars = passed = 0
    for r in results:
        ch = r.challenge
        ok = bool(r.passed)
        passed += int(ok)
        total_score += int(r.score)
        total_stars += int(r.stars)
        cls = "pass" if ok else "fail"
        mark = "PASS" if ok else "miss"
        rows.append(
            "<tr>"
            f"<td>{ch.round_index}</td>"
            f"<td>{html_module.escape(str(ch.target))}</td>"
            f"<td>{html_module.escape(str(r.detected))}</td>"
            f"<td>{int(r.score)}</td>"
            f"<td class='stars'>{'★' * int(r.stars)}{'☆' * (3 - int(r.stars))}</td>"
            f"<td class='{cls}'>{mark}</td>"
            "</tr>"
        )
    totals = (
        f"<div class='totals'><span>Played <b>{len(results)}</b></span>"
        f"<span>Passed <b>{passed}</b></span>"
        f"<span>Score <b>{total_score}</b></span>"
        f"<span>Stars <b>{total_stars}</b></span></div>"
    )
    return (
        "<table class='score'><thead><tr>"
        "<th>#</th><th>Challenge</th><th>Drew</th><th>Score</th><th>Stars</th><th>Result</th>"
        "</tr></thead><tbody>" + "".join(rows) + "</tbody></table>" + totals
    )


def export_lesson_report(
    notes: list,
    results: list,
    output_path: str,
    session_label: str = "",
    narration: "list | None" = None,
) -> str:
    """Write a self-contained HTML lesson report.

    `notes`     — list of BoardNote (board captures: image + OCR transcription + LLM summary).
    `results`   — list of ChallengeResult (the classroom scoreboard).
    `narration` — list of str: the teacher's transcribed spoken narration (gated to teacher).
    All duck-typed so this stays decoupled from the classroom package.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    notes = notes or []
    results = results or []
    narration = narration or []

    parts = [
        "<!DOCTYPE html>",
        "<html lang='en'><head><meta charset='utf-8'>",
        "<meta name='viewport' content='width=device-width,initial-scale=1'>",
        "<title>AirSketch — Lesson Report</title>",
        f"<style>{_HTML_STYLE}{_LESSON_STYLE}</style>",
        "</head><body><div class='container'>",
        "<header><h1>AirSketch — Lesson Report</h1><div class='sub'>",
        (f"<span>{html_module.escape(session_label)}</span>" if session_label else ""),
        f"<span>generated {timestamp}</span>",
        f"<span>{len(notes)} board capture{'s' if len(notes) != 1 else ''}</span>",
        f"<span>{len(results)} challenge{'s' if len(results) != 1 else ''}</span>",
        "</div></header>",
        "<h2 class='section-title'>Challenge scoreboard</h2>",
        _render_scoreboard_html(results),
        "<h2 class='section-title'>Board notes</h2>",
    ]
    if notes:
        for i, n in enumerate(notes, 1):
            parts.append(_render_board_note_html(i, n))
    else:
        parts.append("<div class='empty'>No board captures this session "
                     "(press B during the lesson to add one).</div>")

    if narration:
        parts.append("<h2 class='section-title'>Lesson narration (teacher)</h2>")
        lines = "".join(f"<li>{html_module.escape(str(t))}</li>" for t in narration)
        parts.append(f"<ul class='kpts'>{lines}</ul>")

    parts.append("<div class='footer'>Created with AirSketch · Classroom mode</div>")
    parts.append("</div></body></html>")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("".join(parts))
    return output_path


def export_json(diagrams: List[Diagram], output_path: str) -> str:
    """Write a structured JSON representation of the notebook."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)

    data = []
    for i, d in enumerate(diagrams, 1):
        entry = {
            "index": i,
            "status": d.status,
            "created_at": d.created_at,
            "finalized_at": d.finalized_at,
            "stroke_count": len(d.strokes),
            "point_count": d.total_points,
        }
        if d.analysis:
            entry["analysis"] = {
                "title": d.analysis.title,
                "description": d.analysis.description,
                "topic": d.analysis.topic,
                "tags": d.analysis.tags,
                "confidence": d.analysis.confidence,
                "analyzer": d.analysis.analyzer_name,
            }
        if d.error:
            entry["error"] = d.error
        entry["shapes_detected"] = [
            {"label": s.label, "confidence": s.confidence} for s in d.shapes_detected
        ]
        entry["primitives"] = [
            {"kind": p.kind, "confidence": p.confidence} for p in d.primitives
        ]
        data.append(entry)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            {"generated_at": time.time(), "diagrams": data},
            f,
            indent=2,
        )
    return output_path

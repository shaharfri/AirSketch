"""Phase B — turn raw OCR board text into a structured lesson understanding.

A small instruct LLM (Qwen2.5-3B-Instruct, INT4, on the OpenVINO GenAI runtime)
reads the OCR transcription of a whiteboard and produces a concise summary, a
topic label, cleaned key points, and an OCR-typo-corrected version.

Lazy-loaded on first use. Inference is a few seconds on CPU; callers run
`understand()` on a worker thread.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Understanding:
    summary: str = ""
    topic: str = ""
    key_points: List[str] = field(default_factory=list)
    corrected: str = ""
    raw_response: str = ""


_SYSTEM = (
    "You are a teaching assistant. You receive the raw OCR transcription of a "
    "classroom whiteboard, which may contain small OCR errors. Summarize and "
    "structure the lesson content. Be faithful to the board; do not invent facts."
)

_USER_TEMPLATE = (
    "Whiteboard text:\n\"\"\"\n{text}\n\"\"\"\n\n"
    "Reply with ONLY a single JSON object (no markdown, no commentary):\n"
    '{{"summary": "one or two sentences describing the lesson", '
    '"topic": "short subject label, e.g. Fractions", '
    '"key_points": ["concise cleaned point", "..."], '
    '"corrected": "the board text with obvious OCR typos fixed"}}'
)

# --- Hebrew variants (used when cfg.language == "he") ---
_SYSTEM_HE = (
    "אתה עוזר הוראה. אתה מקבל תמלול OCR גולמי של לוח כיתה, שעשוי להכיל שגיאות OCR "
    "קטנות. סכם ובנה את תוכן השיעור. היה נאמן ללוח; אל תמציא עובדות. ענה בעברית."
)

_USER_TEMPLATE_HE = (
    "טקסט מהלוח:\n\"\"\"\n{text}\n\"\"\"\n\n"
    "החזר אך ורק אובייקט JSON יחיד (ללא markdown, ללא הערות), עם ערכים בעברית:\n"
    '{{"summary": "משפט או שניים המתארים את השיעור", '
    '"topic": "תווית נושא קצרה, למשל שברים", '
    '"key_points": ["נקודה מנוקה ותמציתית", "..."], '
    '"corrected": "טקסט הלוח עם תיקון שגיאות OCR ברורות"}}'
)


def _clean_output(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"<think>.*", "", text, flags=re.DOTALL)
    text = re.sub(r"<\|im_end\|>.*", "", text, flags=re.DOTALL)
    return text.strip()


def parse_understanding(text: str, fallback_text: str = "") -> Understanding:
    """Extract an Understanding from an LLM response (tolerant of non-JSON)."""
    cleaned = _clean_output(text)
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(0))
            kp = data.get("key_points", [])
            if not isinstance(kp, list):
                kp = [str(kp)]
            return Understanding(
                summary=str(data.get("summary", "")).strip()[:400],
                topic=str(data.get("topic", "")).strip()[:60],
                key_points=[str(k).strip() for k in kp if str(k).strip()][:8],
                corrected=str(data.get("corrected", "")).strip()[:1000],
                raw_response=text,
            )
        except json.JSONDecodeError:
            pass
    # Fallback: no parseable JSON -> use the cleaned text as the summary.
    return Understanding(
        summary=cleaned[:400],
        topic="",
        key_points=[],
        corrected=fallback_text,
        raw_response=text,
    )


class LessonUnderstander:
    """Lazy-loading Qwen2.5-3B (OpenVINO GenAI) summarizer for board text."""

    name = "lesson_llm"

    def __init__(self, config):
        self._cfg = config
        self._pipe = None
        self._gen = None
        self._device = "CPU"
        self._loaded = False
        self._load_error = ""

    @property
    def load_error(self) -> str:
        return self._load_error

    def _ensure_pipe(self) -> bool:
        if self._loaded:
            return self._pipe is not None
        self._loaded = True
        try:
            import openvino_genai as ov_genai
            from airsketch.diagram_analyzer import (
                pick_device, download_vlm_model, ensure_ov_tokenizer,
            )

            import os
            self._device = pick_device(getattr(self._cfg, "llm_device", "CPU"))
            cache_dir = getattr(self._cfg, "vlm_model_cache_dir", "models")
            # Prefer a bundled, ready-to-use local model dir (offline, no download).
            local = os.path.join(cache_dir, "qwen2.5-3b-instruct-ov")
            if os.path.isdir(local) and any(f.endswith(".xml") for f in os.listdir(local)):
                model_dir = local
                print(f"[board-llm] Using bundled model: {local}")
            else:
                model_dir = download_vlm_model(
                    self._cfg.board_llm_model_id,
                    cache_dir=cache_dir,
                    offline_only=getattr(self._cfg, "board_llm_offline_only", False),
                )
            ensure_ov_tokenizer(model_dir)   # no-op if OV tokenizer already shipped
            print(f"[board-llm] Loading {self._cfg.board_llm_model_id} on "
                  f"{self._device}...", flush=True)
            self._pipe = ov_genai.LLMPipeline(model_dir, self._device)
            self._gen = ov_genai.GenerationConfig()
            self._gen.max_new_tokens = 320
            self._gen.do_sample = False
            print("[board-llm] Understanding model ready.", flush=True)
        except Exception as e:
            self._pipe = None
            self._load_error = f"{type(e).__name__}: {e}"
            print(f"[board-llm] LLM load failed: {self._load_error}")
        return self._pipe is not None

    @staticmethod
    def _build_prompt(text: str, language: str = "en") -> str:
        is_he = str(language).lower().startswith("he")
        system = _SYSTEM_HE if is_he else _SYSTEM
        template = _USER_TEMPLATE_HE if is_he else _USER_TEMPLATE
        return (
            "<|im_start|>system\n" + system + "<|im_end|>\n"
            "<|im_start|>user\n" + template.format(text=text) + "<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

    def understand(self, transcription: str) -> Optional[Understanding]:
        """Summarize/structure the OCR transcription. None if the model won't load."""
        if not transcription.strip():
            return Understanding()
        if not self._ensure_pipe():
            return None
        prompt = self._build_prompt(transcription, getattr(self._cfg, "language", "en"))
        try:
            raw = self._pipe.generate(prompt, self._gen)
        except Exception as e:
            print(f"[board-llm] generate failed: {type(e).__name__}: {e}")
            return Understanding(summary="", corrected=transcription)
        return parse_understanding(str(raw), fallback_text=transcription)

    def release(self) -> None:
        self._pipe = None

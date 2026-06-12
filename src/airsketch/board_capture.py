"""Phase 3 — capture a photo of the physical whiteboard and transcribe it.

The classroom app grabs the live camera frame (a photo of the teacher's
whiteboard / paper) and runs PaddleOCR's PP-OCR models on the OpenVINO runtime
(see `ocr_reader.PPOCROpenVINOReader`) to read the text on the board. The result
is a structured `BoardNote` that accumulates into a lesson record the Phase 4
report can consume.

OCR (not a VLM) is used for transcription: a small, fast model that actually
reads text reliably. A text LLM can be layered on later to summarize / interpret
the transcribed content (Phase B).

The OCR reader is LAZY-loaded on first capture. Inference takes ~1-3 s on CPU;
the caller runs `transcribe()` on a worker thread so the UI never blocks.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import List, Optional

import cv2
import numpy as np


@dataclass
class BoardNote:
    """One captured whiteboard snapshot, transcribed by OCR (+ optional LLM)."""
    timestamp: str
    transcription: str
    summary: str
    items: List[str] = field(default_factory=list)
    topic: str = ""
    image_path: str = ""
    round_index: int = 0
    raw_response: str = ""


class BoardCapturer:
    """Lazy-loading PP-OCR (OpenVINO) wrapper for whiteboard transcription."""

    name = "board_capturer"

    def __init__(self, config):
        self._cfg = config
        self._reader = None
        self._device = str(getattr(config, "ocr_device", "CPU"))
        self._loaded = False
        self._load_error = ""

        # Optional Phase B understanding LLM (lazy-loads on first capture).
        self._understander = None
        if getattr(config, "board_llm_enabled", False):
            from airsketch.lesson_llm import LessonUnderstander
            self._understander = LessonUnderstander(config)

    @property
    def load_error(self) -> str:
        return self._load_error

    def _ensure_reader(self) -> bool:
        """Load the OCR reader once (idempotent). Returns True if usable."""
        if self._loaded:
            return self._reader is not None
        self._loaded = True
        try:
            from airsketch.ocr_reader import PPOCROpenVINOReader
            print(f"[board] Loading PP-OCR (OpenVINO) on {self._device}...", flush=True)
            self._reader = PPOCROpenVINOReader(device=self._device)
            print("[board] Board reader ready.", flush=True)
        except Exception as e:
            self._reader = None
            self._load_error = f"{type(e).__name__}: {e}"
            print(f"[board] OCR load failed: {self._load_error}")
        return self._reader is not None

    def transcribe(
        self,
        frame: np.ndarray,
        round_index: int = 0,
        output_dir: str = "outputs",
    ) -> Optional[BoardNote]:
        """Save the frame, OCR it, and return a BoardNote.

        Returns None only if the OCR reader could not be loaded at all.
        Designed to be called on a worker thread.
        """
        if not self._ensure_reader():
            return None

        ts = time.strftime("%Y%m%d_%H%M%S")
        image_path = ""
        try:
            os.makedirs(output_dir, exist_ok=True)
            image_path = os.path.join(output_dir, f"board_{ts}.png")
            cv2.imwrite(image_path, frame)
        except Exception as e:
            print(f"[board] could not save snapshot: {type(e).__name__}: {e}")

        lines = self._reader.read(frame)
        from airsketch.ocr_reader import assemble_text
        transcription = assemble_text(lines)
        ocr_items = [l.text for l in sorted(lines, key=lambda x: (x.cy, x.cx))]

        # Defaults from OCR alone (used when the LLM is off or fails).
        summary = ocr_items[0] if ocr_items else ""
        items = ocr_items
        topic = ""
        raw = ""

        if self._understander is not None and transcription.strip():
            u = self._understander.understand(transcription)
            if u is not None:
                summary = u.summary or summary
                topic = u.topic
                items = u.key_points or ocr_items
                raw = u.raw_response

        return BoardNote(
            timestamp=ts,
            transcription=transcription,
            summary=summary,
            items=items,
            topic=topic,
            image_path=image_path,
            round_index=round_index,
            raw_response=raw,
        )

    def release(self) -> None:
        self._reader = None
        if self._understander is not None:
            self._understander.release()


def _main() -> None:
    """CLI: transcribe a still image with PP-OCR — no camera, no game.

        python -m airsketch.board_capture path/to/whiteboard.jpg

    Lets you validate the OCR transcription against any image file, which is far
    easier to debug than the live loop.
    """
    import argparse

    from airsketch.config import AppConfig

    p = argparse.ArgumentParser(
        prog="airsketch.board_capture",
        description="Transcribe a whiteboard/paper image with PP-OCR on OpenVINO (no camera).",
    )
    p.add_argument("image", help="Path to an image file (whiteboard / paper / printout)")
    p.add_argument("--ocr-device", default="CPU", help="OpenVINO device: CPU | GPU | NPU | AUTO")
    p.add_argument("--understand", action="store_true",
                   help="Also run the Phase B LLM to summarize/structure the text (~1.8 GB)")
    p.add_argument("--llm-device", default="CPU", help="OpenVINO device for the LLM")
    p.add_argument("--output-dir", default="outputs")
    args = p.parse_args()

    img = cv2.imread(args.image)
    if img is None:
        raise SystemExit(f"ERROR: could not read image: {args.image}")
    print(f"[board] read image {args.image}  shape={img.shape}")

    cfg = AppConfig()
    cfg.ocr_device = args.ocr_device.upper()
    if args.understand:
        cfg.board_llm_enabled = True
        cfg.llm_device = args.llm_device.upper()

    cap = BoardCapturer(cfg)
    note = cap.transcribe(img, output_dir=args.output_dir)
    if note is None:
        raise SystemExit(f"OCR unavailable: {cap.load_error}")

    print("\n===== BOARD NOTE =====")
    print(f"topic   : {note.topic}")
    print(f"summary : {note.summary}")
    print(f"saved   : {note.image_path}")
    print(f"items   : {len(note.items)}")
    for it in note.items:
        print(f"   - {it}")
    print("transcription:")
    print(note.transcription)


if __name__ == "__main__":
    _main()

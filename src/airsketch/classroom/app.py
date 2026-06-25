"""Classroom challenge game — state machine + main loop.

Flow:
    READY  --SPACE-->  ANNOUNCE (countdown)  -->  DRAWING
    DRAWING  --ENTER / timer / submit-->  judge  -->  RESULT (celebration)
    RESULT  --done-->  READY

Reuses the tested Notebook for stroke capture + live snap-to-shape.
"""
from __future__ import annotations

import math
import threading
import time
from enum import Enum

import cv2
import numpy as np

from airsketch.camera import Camera
from airsketch.classroom.celebration import Celebration
from airsketch.classroom.challenge_engine import ChallengeEngine
from airsketch.classroom.judge import Judge
from airsketch.config import AppConfig
from airsketch.diagram_analyzer import LocalAnalyzer
from airsketch.gesture_detector import IndexPointingDetector
from airsketch.hand_tracker import create_hand_tracker
from airsketch.notebook import Notebook
from airsketch.shape_recognizer import ShapeRecognizer
from airsketch.utils import (
    draw_neon_line, draw_panel, draw_text_with_shadow, text_size,
    fill_rounded_rect, stroke_rounded_rect,
)


def _std_flush() -> None:
    """Flush stdout/stderr if present. A PyInstaller ``--windowed`` (no-console)
    build has ``sys.stdout``/``sys.stderr`` set to None, so a bare
    ``sys.stdout.flush()`` raises AttributeError on shutdown — guard it."""
    import sys
    for stream in (sys.stdout, sys.stderr):
        try:
            if stream is not None:
                stream.flush()
        except Exception:
            pass


class GameState(Enum):
    READY = "ready"
    ANNOUNCE = "announce"
    DRAWING = "drawing"
    RESULT = "result"


ANNOUNCE_SECONDS = 3.0
DRAW_SECONDS = 25.0

# --- UI palette (BGR) — cohesive, lightweight theme ---
UI_ACCENT = (200, 232, 0)      # aqua/teal accent
UI_TEXT = (238, 240, 246)      # near-white
UI_MUTED = (150, 156, 172)     # muted grey-blue
UI_GOOD = (90, 230, 140)       # green
UI_WARN = (60, 175, 255)       # amber/orange
UI_BAD = (80, 90, 255)         # red
UI_PANEL = (26, 28, 36)        # panel fill
UI_BORDER = (66, 72, 88)       # panel border

# --- Hebrew display names + UI strings (used when cfg.language == "he") ---
HE_TARGETS = {
    "circle": "עיגול", "triangle": "משולש", "rectangle": "מלבן", "square": "ריבוע",
    "star": "כוכב", "arrow": "חץ", "line": "קו",
    "house": "בית", "cat": "חתול", "tree": "עץ", "sun": "שמש",
    "flower": "פרח", "fish": "דג", "car": "מכונית", "airplane": "מטוס",
}
HE_THEMES = {"geometry": "צורות", "objects": "חפצים", "mixed": "מעורב"}
HE_UI = {
    # legend labels
    "play": "שחק", "theme": "נושא", "submit": "שלח", "clear": "נקה",
    "retry": "שוב", "board": "לוח", "talk": "דבר", "dictate": "הכתבה",
    "new voice": "קול חדש", "quit": "יציאה", "next": "הבא",
    # center banners
    "Ready to play!": "מוכנים לשחק!", "pick a challenge": "בחרו אתגר",
    "Get ready...": "להתכונן...",
    "Learning your voice": "לומד את הקול שלך",
    "say a sentence, then press D or L": "אמרו משפט, ואז D או L",
    "Dictating": "מכתיב", "speak, then press D": "דברו, ואז D",
    "Listening": "מקשיב", "speak, then press V": "דברו, ואז V",
    "Reading the board...": "קורא את הלוח...",
    "transcribing — please wait": "מתמלל — אנא המתינו",
    "voice off — no microphone": "הקול כבוי — אין מיקרופון",
    "voice commands disabled": "פקודות קוליות מושבתות",
}


class ClassroomApp:
    def __init__(self, config: AppConfig, theme: str = "geometry",
                 voice: bool = False):
        self.cfg = config
        self.engine = ChallengeEngine(theme=theme)
        self.theme = theme
        self._lang = getattr(config, "language", "en")

        # CNN classifier (optional — needed for object challenges)
        self._cnn = None
        try:
            from airsketch.sketch_classifier import SketchClassifier
            self._cnn = SketchClassifier(device=config.cnn_device)
            print(f"[classroom] CNN loaded: {self._cnn.labels}")
        except Exception as e:
            print(f"[classroom] CNN not available ({type(e).__name__}); "
                  f"object challenges will be skipped. Geometry challenges still work.")

        self.judge = Judge(self._cnn)
        self.celebration = Celebration()

        # Board capture (Phase 3) — optional, needs --vlm (Qwen2-VL).
        # The VLM model is lazy-loaded on the first capture, so enabling it here
        # costs nothing at startup.
        self._board = None
        self._board_busy = False
        self._board_msg = ""
        self._board_msg_until = 0.0
        self._last_camera_frame = None
        self._report_path = None
        self.lesson_notes = []   # list[BoardNote] — consumed by the Phase 4 report
        if getattr(config, "board_enabled", False):
            from airsketch.board_capture import BoardCapturer
            self._board = BoardCapturer(config)
            print("[classroom] Board capture enabled (PP-OCR/OpenVINO). Press B "
                  "(or say 'read the board') to transcribe the whiteboard.")

        # Voice control (optional — needs mic + Whisper model)
        self._voice = None
        self._voice_requested = bool(voice)   # user asked for voice (V/D/L)
        self._voice_unavailable_reason = ""
        if voice:
            from airsketch.classroom.voice_controller import VoiceController
            self._voice = VoiceController(
                device=getattr(config, "voice_device", "CPU"),
                language=getattr(config, "language", "en"),
                probe_timeout=getattr(config, "mic_probe_timeout", 6.0),
                whisper_model=getattr(config, "whisper_model", "base"))
            if self._voice.available:
                print("[classroom] Voice control ready. Press V to talk.")
            else:
                self._voice_unavailable_reason = self._voice.error or "no microphone"
                print(f"[classroom] Voice unavailable: {self._voice_unavailable_reason}")
                self._voice = None

        # Teacher voice — the Dictation button (D). First press learns the
        # teacher's voice; afterwards it records dictation, gated to that voice.
        self._speaker = None
        self._teacher_profile = None
        self._voice_mode = None     # None | "learn" | "dictate" — set while the D recording runs
        self._voice_msg = ""
        self._voice_msg_until = 0.0
        self.dictation_lines = []   # teacher narration → lesson report
        self._profile_path = getattr(config, "speaker_profile_path", "models/teacher_voice.json")
        if getattr(config, "teacher_voice_enabled", False) and self._voice is not None:
            try:
                from airsketch.speaker_id import SpeakerEmbedder, SpeakerProfile
                self._speaker = SpeakerEmbedder(
                    device=getattr(config, "speaker_device", "CPU"))
                self._teacher_profile = SpeakerProfile.load(self._profile_path)
                if self._teacher_profile:
                    print(f"[classroom] Teacher voice loaded ({self._teacher_profile.count} "
                          f"sample(s)). Press D to dictate, L to switch to a new voice.")
                else:
                    print("[classroom] Press D to LEARN your voice (say a sentence), "
                          "then press D to DICTATE. (L re-learns / switches voice anytime.)")
            except Exception as e:
                print(f"[classroom] Speaker recognition unavailable ({type(e).__name__}: {e})")
                self._speaker = None
        elif getattr(config, "teacher_voice_enabled", False):
            print("[classroom] --teacher-voice needs --voice (mic + Whisper).")

    # -------------------- localization helpers --------------------------

    def _t(self, text: str) -> str:
        """Translate a fixed UI string to Hebrew when language == 'he'."""
        return HE_UI.get(text, text) if self._lang == "he" else text

    def _theme_name(self, theme: str) -> str:
        return HE_THEMES.get(theme, theme) if self._lang == "he" else theme

    def _prompt_text(self, challenge) -> str:
        """The on-screen 'draw a X' prompt, localized."""
        if challenge is None:
            return ""
        if self._lang == "he":
            return f"צייר {HE_TARGETS.get(challenge.target, challenge.target)}!"
        return challenge.prompt

    # -------------------------------------------------------------------

    def run(self) -> None:
        cfg = self.cfg
        source = cfg.video_path if cfg.video_path else cfg.camera_index
        with Camera(source, cfg.frame_width, cfg.frame_height,
                    rotate_180=cfg.rotate_180, mirror=cfg.mirror) as cam:
            tracker = create_hand_tracker(cfg)
            self._gesture = IndexPointingDetector(confirm_frames=cfg.point_confirm_frames)
            self._notebook = Notebook(
                analyzer=LocalAnalyzer(),
                recognizer=ShapeRecognizer(),
                pause_seconds=999,   # never auto-finalize; the game controls timing
                canvas_render_size=cfg.canvas_render_size,
                thumbnail_size=cfg.thumbnail_size,
                tail_trim=cfg.stroke_tail_trim,
                live_snap_enabled=cfg.live_snap_enabled,
            )

            cv2.namedWindow("AirSketch Classroom", cv2.WINDOW_NORMAL)
            print("[classroom] Ready. Click the window, then press SPACE (or say a shape) to start.")

            self._state = GameState.READY
            self._current = None
            self._phase_start = time.time()
            self._last_result = None
            self._retry_available = False
            # One report file per session; rewritten incrementally after every
            # board capture / dictation / challenge so a messy exit never loses it.
            import os as _os
            _out = getattr(self.cfg, "output_dir", "outputs")
            self._report_path = _os.path.join(
                _out, f"lesson_{time.strftime('%Y%m%d_%H%M%S')}.html")
            prev_t = time.perf_counter()
            fps = 0.0

            last_frame = None
            try:
                while True:
                    now = time.perf_counter()
                    dt = now - prev_t
                    fps = 1.0 / dt if dt > 0 else 0.0
                    prev_t = now

                    # CRITICAL: never read the camera while the mic is recording.
                    # On integrated camera+mic subsystems (and on VDI's shared
                    # redirection channel) simultaneous access stalls the camera
                    # pipeline and blocks cam.read(), freezing the whole UI.
                    # During recording we reuse the last frame and skip hand
                    # tracking; nobody is drawing during a spoken command anyway.
                    recording_now = self._voice is not None and self._voice.is_recording

                    if recording_now:
                        frame = (last_frame.copy() if last_frame is not None
                                 else np.zeros((cfg.frame_height, cfg.frame_width, 3), np.uint8))
                        hand = None
                    else:
                        frame = cam.read()
                        last_frame = frame
                        # Snapshot the clean frame (before HUD/strokes are drawn
                        # onto it in-place) for board capture. Only pay the copy
                        # cost when board capture is actually enabled.
                        if self._board is not None:
                            self._last_camera_frame = frame.copy()
                        hand = tracker.process(frame)

                    elapsed = time.time() - self._phase_start

                    # ---------------- VOICE ----------------
                    if self._voice is not None:
                        res = self._voice.take_result()
                        if res is not None:
                            text, audio = res
                            mode, self._voice_mode = self._voice_mode, None
                            if mode == "learn":
                                self._handle_learn_sample(audio)
                            elif mode == "dictate":
                                self._handle_dictation(text, audio)
                            elif text:
                                from airsketch.classroom.voice_commands import parse_command
                                self._apply_intent(parse_command(text), audio)

                    # ---------------- STATE MACHINE (skipped while recording) ----------------
                    if recording_now:
                        pass
                    elif self._state == GameState.ANNOUNCE:
                        if elapsed >= ANNOUNCE_SECONDS:
                            self._notebook.clear_current()
                            self._gesture.reset()
                            self._state = GameState.DRAWING
                            self._phase_start = time.time()

                    elif self._state == GameState.DRAWING:
                        pen = self._gesture.update(hand.landmarks) if hand.visible else False
                        if hand.visible and pen and hand.fingertip is not None:
                            if not self._notebook.is_drawing:
                                self._notebook.begin_stroke(hand.fingertip)
                            else:
                                self._notebook.append_to_stroke(hand.fingertip)
                            tx, ty = int(hand.fingertip[0]), int(hand.fingertip[1])
                            cv2.circle(frame, (tx, ty), 13, (0, 0, 0), -1, cv2.LINE_AA)
                            cv2.circle(frame, (tx, ty), 12, (0, 255, 100), 3, cv2.LINE_AA)
                        else:
                            if self._notebook.is_drawing:
                                self._notebook.end_stroke()
                        if elapsed >= DRAW_SECONDS:
                            self._submit_now()

                    elif self._state == GameState.RESULT:
                        if not self.celebration.is_active:
                            self._state = GameState.READY

                    # ---------------- RENDER ----------------
                    if (not recording_now) and self._state == GameState.DRAWING:
                        self._draw_strokes(frame, self._notebook)

                    self._draw_hud(frame, self._state, self._current, self._last_result, fps, elapsed)
                    self._draw_voice_indicator(frame)
                    self._draw_board_status(frame)
                    self._draw_voice_msg(frame)
                    self._draw_key_legend(frame, self._state)

                    # RESULT celebration is its own full overlay; otherwise a single
                    # priority-picked center banner (never stacked — see _draw_center_overlay).
                    if (not recording_now) and self._state == GameState.RESULT:
                        self.celebration.render(frame)
                    else:
                        self._draw_center_overlay(frame, self._state, self._current, elapsed)

                    cv2.imshow("AirSketch Classroom", frame)

                    # Window closed via the title-bar X → exit cleanly so the
                    # finally block still runs (and the final report is written).
                    try:
                        if cv2.getWindowProperty("AirSketch Classroom",
                                                 cv2.WND_PROP_VISIBLE) < 1:
                            break
                    except cv2.error:
                        break

                    # ---------------- KEYS ----------------
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord("q"), ord("Q"), 27):
                        break
                    elif key == ord(" ") and self._state == GameState.READY:
                        self._start_challenge(self.engine.next_challenge())
                    elif key in (13, 10) and self._state == GameState.DRAWING:
                        self._submit_now()
                    elif key in (ord("c"), ord("C")) and self._state == GameState.DRAWING:
                        self._notebook.clear_current()
                    elif key in (ord("r"), ord("R")) and self._retry_available \
                            and self._state in (GameState.RESULT, GameState.READY):
                        self._retry_challenge()
                    elif key in (ord("t"), ord("T")) and self._state == GameState.READY:
                        self._cycle_theme()
                    elif key in (ord("b"), ord("B")) and self._board is not None:
                        self._capture_board()
                    elif key in (ord("d"), ord("D")) and self._speaker is not None:
                        self._toggle_dictation()
                    elif key in (ord("l"), ord("L")) and self._speaker is not None:
                        self._toggle_relearn()
                    elif key in (ord("v"), ord("V")) and self._voice is not None \
                            and self._voice_mode is None:
                        self._voice.toggle()

            except Exception:
                # Surface the real error — os._exit() in __main__ would otherwise
                # kill the process before the traceback is ever printed.
                import traceback, sys as _sys
                print("\n===== AirSketch crashed — traceback =====")
                traceback.print_exc()
                _std_flush()
            finally:
                # 1) Release the camera ON A SIDE THREAD right away. cv2's
                #    VideoCapture.release() is what actually turns the LED off, but
                #    it's slow on the Windows MSMF backend — so we run it in parallel
                #    with the report write below and only wait a bounded time for it.
                import os as _os, sys as _sys, threading as _th

                def _release_camera():
                    try:
                        cam.release()
                    except Exception:
                        pass
                    try:
                        tracker.release()
                    except Exception:
                        pass

                rel = _th.Thread(target=_release_camera, daemon=True)
                rel.start()

                # 2) The must-not-lose work (overlaps the camera release).
                self._print_scoreboard()
                self._write_report(final=True)
                try:
                    self._notebook.shutdown(wait=False)
                except Exception:
                    pass
                if self._voice is not None:
                    try:
                        self._voice.close()
                    except Exception:
                        pass
                try:
                    cv2.destroyAllWindows()
                    cv2.waitKey(1)
                except Exception:
                    pass

                # 3) Give the camera release up to a few seconds to finish (LED
                #    off); then hard-exit so MediaPipe/OpenVINO threads can't hang us.
                rel.join(4.0)
                _std_flush()
                _os._exit(0)

    # -------------------- action helpers (shared by keys + voice) --------

    def _start_challenge(self, challenge) -> None:
        self._current = challenge
        self._state = GameState.ANNOUNCE
        self._phase_start = time.time()
        self._retry_available = False
        print(f"[challenge {challenge.round_index}] {challenge.prompt} (theme={challenge.theme})")

    def _retry_challenge(self) -> None:
        """Re-attempt the last (missed) challenge — same target, same round."""
        challenge = self.engine.retry_last()
        if challenge is None:
            return
        print(f"[retry] re-attempting {challenge.target}")
        self._start_challenge(challenge)

    def _submit_now(self) -> None:
        self._notebook.end_stroke()
        diagram = self._notebook.finalize_current_diagram()
        if diagram is None:
            from airsketch.classroom.challenge_engine import ChallengeResult
            result = ChallengeResult(
                challenge=self._current, detected="nothing", score=0,
                stars=0, passed=False, explanation="no drawing",
            )
        else:
            result = self.judge.judge(diagram, self._current)
        self.engine.record_result(result)
        self._last_result = result
        if result.passed:
            self.celebration.start_success(stars=result.stars, score=result.score)
        else:
            self.celebration.start_failure(target=self._current.target, score=result.score)
        self._retry_available = not result.passed
        self._state = GameState.RESULT
        self._phase_start = time.time()
        print(f"[result] {result.detected} | score={result.score} "
              f"stars={result.stars} passed={result.passed} ({result.explanation})")
        self._write_report()

    def _capture_board(self) -> None:
        """Photograph the whiteboard and transcribe it with Qwen-VL.

        Runs the (slow) VLM inference on a daemon thread so the game loop never
        blocks; the result lands in self.lesson_notes when it's done.
        """
        if self._board is None or self._board_busy:
            return
        if self._last_camera_frame is None:
            self._board_msg = "No camera frame yet"
            self._board_msg_until = time.time() + 3.0
            return
        frame = self._last_camera_frame.copy()
        # When NOT in mirror (selfie) mode, text held to the camera comes out
        # horizontally reversed, so OCR reads mirror-writing. Flip it back for the
        # OCR/LLM (and the saved report image) so the text is readable. In mirror
        # mode the frame is already the right way round, so leave it.
        if not getattr(self.cfg, "mirror", False):
            frame = cv2.flip(frame, 1)
        round_index = self._current.round_index if self._current else 0
        self._board_busy = True
        self._board_msg = "Reading the board..."
        self._board_msg_until = time.time() + 3.0

        def worker():
            try:
                note = self._board.transcribe(
                    frame, round_index=round_index, output_dir=self.cfg.output_dir)
                if note is None:
                    self._board_msg = "Board reader unavailable (VLM load failed)"
                else:
                    self.lesson_notes.append(note)
                    self._board_msg = (f"Captured: {note.summary[:48]}"
                                       if note.summary else "Board captured")
                    print(f"[board] captured (round {round_index}) -> {note.summary}")
                    if note.transcription:
                        print(f"[board] {note.transcription[:400]}")
                    self._write_report()
            except Exception as e:
                self._board_msg = f"Board capture failed ({type(e).__name__})"
                print(f"[board] capture failed: {type(e).__name__}: {e}")
            finally:
                self._board_msg_until = time.time() + 5.0
                self._board_busy = False

        threading.Thread(target=worker, daemon=True).start()

    # ---------------- teacher voice: the Dictation button (D) ----------------

    def _set_voice_msg(self, msg: str, secs: float = 4.0) -> None:
        self._voice_msg = msg
        self._voice_msg_until = time.time() + secs
        print(f"[teacher-voice] {msg}")

    def _toggle_dictation(self) -> None:
        """D key: one button for the whole dictation feature.

        First use (no voice learned yet) records a sample to LEARN the teacher's
        voice; afterwards it records DICTATION (gated to that voice). Press once
        to start the recording, press again to stop.
        """
        if self._speaker is None or self._voice is None:
            return
        if self._voice.is_recording:
            self._voice.toggle()              # stop -> processed on result
            return
        if self._teacher_profile is None:
            self._voice_mode = "learn"
            self._set_voice_msg("Learning your voice: say a sentence, then press D")
        else:
            self._voice_mode = "dictate"
            self._set_voice_msg("Dictating: speak, then press D")
        self._voice.toggle(purpose="dictation")   # start recording (free speech)

    def _toggle_relearn(self) -> None:
        """L key: (re)learn / switch the active voice — overwrites the saved one.

        Lets a different teacher take over at any time WITHOUT being prompted on
        every dictation. D keeps dictating with the current voice; press L only
        when you actually want to change whose voice is recognized.
        """
        if self._speaker is None or self._voice is None:
            return
        if self._voice.is_recording:
            self._voice.toggle()              # stop -> processed on result
            return
        self._voice_mode = "learn"            # _handle_learn_sample overwrites the profile
        had = self._teacher_profile is not None
        self._voice.toggle(purpose="dictation")
        self._set_voice_msg(
            ("Switching voice: say a sentence, then press L" if had
             else "Learning your voice: say a sentence, then press L"))

    def _handle_learn_sample(self, audio) -> None:
        """Build (or replace) the teacher's voice profile from one spoken sentence."""
        emb = self._speaker.embed(audio) if (self._speaker and audio is not None) else None
        if emb is None:
            self._set_voice_msg("Too short — try again and say a full sentence")
            return
        from airsketch.speaker_id import build_profile
        replaced = self._teacher_profile is not None
        self._teacher_profile = build_profile(
            [emb], threshold=getattr(self.cfg, "speaker_threshold", 0.5))
        try:
            import os
            os.makedirs(os.path.dirname(os.path.abspath(self._profile_path)) or ".", exist_ok=True)
            self._teacher_profile.save(self._profile_path)
        except Exception as e:
            print(f"[teacher-voice] could not save profile: {type(e).__name__}: {e}")
        self._set_voice_msg("New voice set! Press D to dictate." if replaced
                            else "Voice learned! Press D to dictate.")

    def _handle_dictation(self, text: str, audio) -> None:
        """Capture free speech as lesson narration, gated to the enrolled teacher."""
        if not text.strip():
            self._set_voice_msg("heard nothing — press D and speak")
            return
        if self._teacher_profile is not None:
            emb = self._speaker.embed(audio) if audio is not None else None
            if emb is None or not self._teacher_profile.matches(emb):
                score = self._teacher_profile.score(emb) if emb is not None else 0.0
                self._set_voice_msg(f"ignored — voice didn't match teacher (match {score:.2f})")
                return
        self.dictation_lines.append(text.strip())
        self._set_voice_msg(f"noted: {text.strip()[:40]}")
        self._write_report()

    def _cycle_theme(self) -> None:
        self.theme = {"geometry": "objects", "objects": "mixed",
                      "mixed": "geometry"}[self.theme]
        self.engine.theme = self.theme
        print(f"[classroom] theme -> {self.theme}")

    def _apply_intent(self, intent, audio=None) -> None:
        """Dispatch a parsed voice intent into game actions."""
        from airsketch.classroom.voice_commands import IntentType
        print(f"[voice] heard: '{intent.text}' -> {intent.type.value}"
              + (f" ({intent.target})" if intent.target else ""))
        if intent.type == IntentType.DICTATION:
            # Free speech on the command button (V) is not a command. Dictation
            # has its own button (D), so just nudge the user there.
            if self._speaker is not None:
                self._set_voice_msg("Press D to dictate (V is for commands)")
            return
        if intent.type == IntentType.DRAW and self._state == GameState.READY:
            self._start_challenge(self.engine.challenge_for(intent.target))
        elif intent.type == IntentType.RETRY and self._retry_available \
                and self._state in (GameState.RESULT, GameState.READY):
            self._retry_challenge()
        elif intent.type == IntentType.NEXT and self._state == GameState.READY:
            self._start_challenge(self.engine.next_challenge())
        elif intent.type == IntentType.SUBMIT and self._state == GameState.DRAWING:
            self._submit_now()
        elif intent.type == IntentType.CLEAR and self._state == GameState.DRAWING:
            self._notebook.clear_current()
        elif intent.type == IntentType.SET_THEME and self._state == GameState.READY:
            self.theme = intent.target
            self.engine.theme = self.theme
            print(f"[classroom] theme -> {self.theme}")
        elif intent.type == IntentType.CAPTURE_BOARD and self._board is not None:
            self._capture_board()

    # ---------------- drawing helpers ----------------

    def _draw_strokes(self, frame, notebook) -> None:
        for stroke in notebook.current.strokes:
            pts = [(int(x), int(y)) for x, y in stroke.points]
            if len(pts) >= 2:
                draw_neon_line(frame, pts, self.cfg.trail_color, base_thickness=2)
        if notebook.current_stroke is not None:
            pts = [(int(x), int(y)) for x, y in notebook.current_stroke.points]
            if len(pts) >= 2:
                draw_neon_line(frame, pts, (0, 255, 255), base_thickness=2)

    def _draw_hud(self, frame, state, challenge, last_result, fps, elapsed) -> None:
        h, w = frame.shape[:2]

        # Top status bar — rounded glass panel with an accent bar + logo dot
        draw_panel(frame, 12, 12, 336, 92, alpha=0.62, bg=UI_PANEL,
                   border=UI_BORDER, radius=16, accent=UI_ACCENT)
        cv2.circle(frame, (34, 34), 8, UI_ACCENT, -1, cv2.LINE_AA)
        cv2.circle(frame, (34, 34), 8, (255, 255, 255), 1, cv2.LINE_AA)
        draw_text_with_shadow(frame, "AirSketch", (52, 40), 0.7, UI_TEXT, thickness=2)
        if self._lang == "he":
            stats = (f"סבב {self.engine.round_count}    ניקוד {self.engine.total_score}"
                     f"    כוכבים {self.engine.total_stars}")
            theme_line = f"נושא {self._theme_name(self.theme)}   |   {fps:.0f} FPS"
        else:
            stats = (f"Round {self.engine.round_count}    Score {self.engine.total_score}"
                     f"    Stars {self.engine.total_stars}")
            theme_line = f"Theme {self.theme}   |   {fps:.0f} FPS"
        draw_text_with_shadow(frame, stats, (24, 66), 0.52, (212, 216, 228))
        draw_text_with_shadow(frame, theme_line, (24, 88), 0.46, UI_MUTED)

        if state == GameState.DRAWING and challenge:
            # Challenge reminder chip (top-center) + a rounded timer bar (above the
            # bottom key legend). Controls live in the legend, not here.
            txt = self._prompt_text(challenge)
            ts = text_size(txt, 1.0, 2)
            draw_panel(frame, (w - ts[0]) // 2 - 20, 12, ts[0] + 40, 46,
                       alpha=0.58, bg=UI_PANEL, border=UI_BORDER, radius=14,
                       accent=UI_GOOD)
            draw_text_with_shadow(frame, txt, ((w - ts[0]) // 2, 43), 1.0, UI_GOOD, thickness=2)
            frac = max(0.0, 1.0 - elapsed / DRAW_SECONDS)
            tx, ty, tw, th = 20, h - 52, w - 40, 12
            stroke_rounded_rect(frame, tx, ty, tw, th, UI_BORDER, 1, 6)
            fw = int(tw * frac)
            if fw > 6:
                col = UI_GOOD if frac > 0.5 else UI_WARN if frac > 0.25 else UI_BAD
                fill_rounded_rect(frame, tx, ty, fw, th, col, 6)

        # Retry prompt after a miss (visible in RESULT and the following READY).
        # The 'R' key itself lives in the legend; this is just status.
        if self._retry_available and state in (GameState.RESULT, GameState.READY):
            tgt = self._current.target if self._current else "shape"
            if self._lang == "he":
                msg = f"כמעט! נסו את ה{HE_TARGETS.get(tgt, tgt)} שוב"
            else:
                msg = f"Missed it — try the {tgt} again"
            ms = text_size(msg, 0.6, 2)
            bx, by = (w - ms[0]) // 2 - 18, int(h * 0.70) - 22
            draw_panel(frame, bx, by, ms[0] + 36, 36, alpha=0.55, bg=UI_PANEL,
                       border=UI_BORDER, radius=18, accent=UI_WARN)
            draw_text_with_shadow(frame, msg, ((w - ms[0]) // 2, int(h * 0.70)),
                                  0.6, UI_WARN, thickness=2)

    def _draw_voice_indicator(self, frame) -> None:
        if self._voice is None:
            # Voice was requested but couldn't start (e.g. no mic on the VDI).
            # Say so explicitly — don't just silently drop the V/D/L controls.
            if self._voice_requested:
                h, w = frame.shape[:2]
                x, y = w - 260, 16
                cv2.circle(frame, (x, y + 6), 7, (90, 90, 110), 1, cv2.LINE_AA)
                draw_text_with_shadow(frame, self._t("voice off — no microphone"),
                                      (x + 18, y + 12), 0.45, (120, 130, 200))
                draw_text_with_shadow(frame, self._t("voice commands disabled"),
                                      (x + 18, y + 34), 0.42, (110, 110, 130))
            return
        h, w = frame.shape[:2]
        x, y = w - 230, 16
        if not self._voice.available:
            # Mic failed at runtime — show it's off, don't pretend it's ready
            cv2.circle(frame, (x, y + 6), 7, (90, 90, 110), 1, cv2.LINE_AA)
            draw_text_with_shadow(frame, "voice off (no mic)", (x + 18, y + 12),
                                  0.45, (150, 150, 160))
            return
        if self._voice.is_recording:
            # Pulsing red REC dot
            import time as _t
            r = 8 + int(3 * abs(math.sin(_t.time() * 6)))
            cv2.circle(frame, (x, y + 6), r, (60, 60, 255), -1, cv2.LINE_AA)
            draw_text_with_shadow(frame, "REC", (x + 18, y + 12), 0.5, (120, 120, 255))
        elif self._voice.is_transcribing:
            draw_text_with_shadow(frame, "transcribing...", (x, y + 12), 0.5, (0, 200, 255))
        else:
            cv2.circle(frame, (x, y + 6), 7, (90, 160, 90), 1, cv2.LINE_AA)
            draw_text_with_shadow(frame, "voice ready", (x + 18, y + 12), 0.5, (150, 180, 150))
        if self._voice.last_text:
            txt = f"\"{self._voice.last_text[:40]}\""
            draw_text_with_shadow(frame, txt, (x - 40, y + 34), 0.45, (180, 200, 220))

    def _draw_voice_msg(self, frame) -> None:
        """Dictation status line + toast (the learn/dictate banner is the modal overlay)."""
        if self._speaker is None:
            return
        h, w = frame.shape[:2]
        if self._voice_msg and time.time() < self._voice_msg_until:
            draw_text_with_shadow(frame, self._voice_msg, (20, h - 108),
                                  0.55, (210, 190, 130), thickness=2)
        state = "ready" if self._teacher_profile else "not learned"
        draw_text_with_shadow(frame, f"Dictation: {state}   notes: {len(self.dictation_lines)}",
                              (20, h - 130), 0.45, (170, 160, 200))

    def _draw_board_status(self, frame) -> None:
        """Board last-result toast + note count (the 'Reading...' banner is the modal overlay)."""
        if self._board is None:
            return
        h, w = frame.shape[:2]
        if (not self._board_busy) and self._board_msg and time.time() < self._board_msg_until:
            draw_text_with_shadow(frame, self._board_msg, (20, h - 64),
                                  0.55, (120, 220, 160), thickness=2)
        draw_text_with_shadow(frame, f"Board notes: {len(self.lesson_notes)}",
                              (20, h - 86), 0.45, (150, 180, 210))

    def _draw_center_overlay(self, frame, state, challenge, elapsed) -> None:
        """Draw exactly ONE center banner, picked by priority — never stacked.

        Priority: an active recording (learn / dictate / command) > board reading >
        the READY/ANNOUNCE state banner. DRAWING/RESULT show no center banner here
        (RESULT uses the celebration overlay).
        """
        T = self._t
        recording = self._voice is not None and self._voice.is_recording
        if recording:
            if self._voice_mode == "learn":
                self._center_banner(frame, T("Learning your voice"),
                                    T("say a sentence, then press D or L"), (200, 160, 255))
            elif self._voice_mode == "dictate":
                self._center_banner(frame, T("Dictating"),
                                    T("speak, then press D"), (140, 220, 255))
            else:
                self._center_banner(frame, T("Listening"), T("speak, then press V"),
                                    (60, 200, 255))
            return
        if self._board_busy:
            self._center_banner(frame, T("Reading the board..."),
                                T("transcribing — please wait"), UI_WARN)
            return
        if state == GameState.READY:
            self._center_banner(frame, T("Ready to play!"), T("pick a challenge"), UI_ACCENT)
        elif state == GameState.ANNOUNCE and challenge:
            remaining = max(0, ANNOUNCE_SECONDS - elapsed)
            self._center_banner(frame, self._prompt_text(challenge),
                                f"{T('Get ready...')} {remaining:.0f}", UI_GOOD)

    def _draw_key_legend(self, frame, state) -> None:
        """All keyboard shortcuts, in ONE consistent place: a bottom-center bar."""
        items = []
        if state == GameState.READY:
            items += [("SPACE", "play"), ("T", "theme")]
        elif state == GameState.DRAWING:
            items += [("ENTER", "submit"), ("C", "clear")]
        elif state == GameState.RESULT:
            items += [("SPACE", "next")]
        if self._retry_available and state in (GameState.RESULT, GameState.READY):
            items.append(("R", "retry"))
        if self._board is not None:
            items.append(("B", "board"))
        if self._voice is not None:
            items.append(("V", "talk"))
        if self._speaker is not None:
            items.append(("D", "dictate"))
            items.append(("L", "new voice"))
        items.append(("Q", "quit"))

        F = cv2.FONT_HERSHEY_SIMPLEX
        gap = 22
        segs, total = [], 0
        for key, label in items:
            label = self._t(label)            # localize the label (keys stay Latin)
            kw = cv2.getTextSize(key, F, 0.5, 2)[0][0]
            lw = text_size(label, 0.48)[0]    # Hebrew-aware width
            sw = kw + 7 + lw
            segs.append((key, label, kw, sw))
            total += sw + gap
        total = max(0, total - gap)
        if total <= 0:
            return
        h, w = frame.shape[:2]
        px = (w - total) // 2
        draw_panel(frame, px - 18, h - 34, total + 36, 28, alpha=0.5,
                   bg=UI_PANEL, border=UI_BORDER, radius=14)
        x, y = px, h - 14
        for key, label, kw, sw in segs:
            draw_text_with_shadow(frame, key, (x, y), 0.5, UI_ACCENT, thickness=2)
            draw_text_with_shadow(frame, label, (x + kw + 7, y), 0.48, UI_MUTED, thickness=1)
            x += sw + gap

    @staticmethod
    def _center_banner(frame, title, subtitle, color):
        h, w = frame.shape[:2]
        ts = text_size(title, 1.3, 2)
        pw = max(ts[0] + 88, 380)
        px, py = (w - pw) // 2, h // 2 - 58
        draw_panel(frame, px, py, pw, 108, alpha=0.66, bg=UI_PANEL,
                   border=UI_BORDER, radius=20, accent=color)
        draw_text_with_shadow(frame, title, ((w - ts[0]) // 2, h // 2 - 2),
                              1.3, color, thickness=2)
        # accent underline under the title
        uw = min(ts[0], pw - 56)
        cv2.line(frame, ((w - uw) // 2, h // 2 + 12), ((w + uw) // 2, h // 2 + 12),
                 color, 2, cv2.LINE_AA)
        ss = text_size(subtitle, 0.6, 1)
        draw_text_with_shadow(frame, subtitle, ((w - ss[0]) // 2, h // 2 + 38),
                              0.6, UI_MUTED)

    def _write_report(self, final: bool = False) -> None:
        """(Re)write the session lesson report. Called incrementally after every
        board capture / dictation / challenge AND on exit, so a forced/messy exit
        (e.g. a stalled camera) never loses the report."""
        if self._report_path is None:
            return
        if not self.engine.history and not self.lesson_notes and not self.dictation_lines:
            return
        try:
            import os
            from airsketch.exporter import export_lesson_report
            os.makedirs(os.path.dirname(os.path.abspath(self._report_path)) or ".", exist_ok=True)
            export_lesson_report(
                self.lesson_notes, self.engine.history, self._report_path,
                session_label=f"Theme: {self.theme}",
                narration=self.dictation_lines)
            if final:
                print(f"[report] Lesson report: {self._report_path}")
        except Exception as e:
            print(f"[report] export failed: {type(e).__name__}: {e}")

    def _print_scoreboard(self) -> None:
        print("\n===== Classroom session scoreboard =====")
        print(f"  Rounds played : {self.engine.round_count}")
        print(f"  Passed        : {self.engine.passed_count}")
        print(f"  Total score   : {self.engine.total_score}")
        print(f"  Total stars   : {self.engine.total_stars}")
        for r in self.engine.history:
            mark = "OK " if r.passed else "x  "
            print(f"   {mark} round {r.challenge.round_index}: "
                  f"{r.challenge.target} -> {r.detected} "
                  f"({r.score}, {'*' * r.stars})")
        if self.lesson_notes:
            print(f"  Board notes   : {len(self.lesson_notes)}")
            for n in self.lesson_notes:
                print(f"     - [{n.timestamp}] {n.summary}")
        print("========================================\n")


def main():
    import argparse
    p = argparse.ArgumentParser(prog="airsketch.classroom",
                                description="AirSketch Classroom — challenge game")
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--video", type=str, default=None)
    p.add_argument("--no-rotate", action="store_true")
    p.add_argument("--mirror", action="store_true")
    p.add_argument("--theme", choices=("geometry", "objects", "mixed"), default="geometry")
    p.add_argument("--lang", choices=("en", "he"), default=None,
                   help="Language for voice + board LLM: en (default) or he (Hebrew)")
    p.add_argument("--hand-backend", choices=("mediapipe", "openvino"), default=None,
                   help="Hand tracking backend: 'openvino' (default) or 'mediapipe' (fallback)")
    p.add_argument("--hand-device", default=None,
                   help="OpenVINO device for hand tracking: AUTO | CPU | GPU | NPU")
    p.add_argument("--hand-debug", action="store_true",
                   help="Overlay live OpenVINO hand-tracking diagnostics for troubleshooting")
    p.add_argument("--cnn-device", default=None)
    p.add_argument("--snap", action="store_true",
                   help="Enable live snap-to-shape (off by default — freehand)")
    p.add_argument("--voice", action="store_true",
                   help="Enable voice control (needs mic + Whisper model)")
    p.add_argument("--board", action="store_true",
                   help="Enable whiteboard capture (PP-OCR on OpenVINO)")
    p.add_argument("--ocr-device", default=None,
                   help="OpenVINO device for board OCR: CPU | GPU | NPU | AUTO")
    p.add_argument("--understand", action="store_true",
                   help="Add the LLM that summarizes board text (implies --board; ~1.8 GB)")
    p.add_argument("--llm-device", default=None,
                   help="OpenVINO device for the understanding LLM: CPU | GPU | NPU | AUTO")
    p.add_argument("--voice-device", default=None,
                   help="OpenVINO device for voice STT (Whisper): CPU | GPU | NPU | AUTO")
    p.add_argument("--speaker-device", default=None,
                   help="OpenVINO device for speaker ID (WeSpeaker): CPU | GPU | NPU | AUTO")
    p.add_argument("--teacher-voice", action="store_true",
                   help="Speaker recognition: enroll (E) the teacher's voice, gate dictation (implies --voice)")
    p.add_argument("--speaker-threshold", type=float, default=None,
                   help="Cosine acceptance threshold for the teacher's voice (default 0.5)")
    p.add_argument("--mic-timeout", type=float, default=None,
                   help="Seconds to wait for the mic to open at startup (default 6)")
    p.add_argument("--whisper-model", choices=("base", "small"), default=None,
                   help="Whisper model (OpenVINO): base (default) or small (more accurate)")
    args = p.parse_args()

    cfg = AppConfig()
    if args.video:
        cfg.video_path = args.video
    else:
        cfg.camera_index = args.camera
    if args.no_rotate:
        cfg.rotate_180 = False
    if args.mirror:
        cfg.mirror = True
    if args.lang:
        cfg.language = args.lang
    if args.hand_backend:
        cfg.hand_tracker_backend = args.hand_backend
    if args.hand_device:
        cfg.hand_device = args.hand_device.upper()
    if args.hand_debug:
        cfg.hand_debug = True
    if args.cnn_device:
        cfg.cnn_device = args.cnn_device.upper()
    if args.snap:
        cfg.live_snap_enabled = True
    if args.board:
        cfg.board_enabled = True
    if args.ocr_device:
        cfg.ocr_device = args.ocr_device.upper()
    if args.understand:
        cfg.board_enabled = True
        cfg.board_llm_enabled = True
    if args.llm_device:
        cfg.llm_device = args.llm_device.upper()
    if args.voice_device:
        cfg.voice_device = args.voice_device.upper()
    if args.speaker_device:
        cfg.speaker_device = args.speaker_device.upper()
    if args.teacher_voice:
        cfg.teacher_voice_enabled = True
    if args.speaker_threshold is not None:
        cfg.speaker_threshold = args.speaker_threshold
    if args.mic_timeout is not None:
        cfg.mic_probe_timeout = args.mic_timeout
    if args.whisper_model:
        cfg.whisper_model = args.whisper_model

    ClassroomApp(cfg, theme=args.theme, voice=args.voice or args.teacher_voice).run()


if __name__ == "__main__":
    import os
    try:
        main()
    finally:
        os._exit(0)

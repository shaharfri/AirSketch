"""Parse transcribed teacher speech into classroom intents.

Pure logic — no audio / model dependencies — so it is fully unit-testable.
"""
from __future__ import annotations

import difflib
import re
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from airsketch.classroom.challenge_engine import GEOMETRY_TARGETS, OBJECT_TARGETS

ALL_TARGETS = GEOMETRY_TARGETS + OBJECT_TARGETS

# Spoken synonyms → canonical target label (English + Hebrew)
_TARGET_ALIASES = {
    "square": "square", "box": "square", "rectangle": "rectangle",
    "circle": "circle", "round": "circle", "ring": "circle",
    "triangle": "triangle",
    "star": "star",
    "arrow": "arrow",
    "line": "line",
    "house": "house", "home": "house",
    "cat": "cat", "kitty": "cat",
    "tree": "tree",
    "sun": "sun",
    "flower": "flower",
    "fish": "fish",
    "car": "car", "vehicle": "car",
    "airplane": "airplane", "plane": "airplane", "aeroplane": "airplane",
    # --- Hebrew ---
    "עיגול": "circle", "עגול": "circle", "מעגל": "circle",
    "ריבוע": "square",
    "מלבן": "rectangle",
    "משולש": "triangle",
    "כוכב": "star",
    "חץ": "arrow",
    "קו": "line",
    "בית": "house",
    "חתול": "cat", "חתולה": "cat",
    "עץ": "tree",
    "שמש": "sun",
    "פרח": "flower",
    "דג": "fish",
    "מכונית": "car", "אוטו": "car", "רכב": "car",
    "מטוס": "airplane",
}


class IntentType(Enum):
    DRAW = "draw"            # start a challenge with a specific target
    NEXT = "next"            # start a random challenge
    RETRY = "retry"          # re-attempt the same challenge after a miss
    CAPTURE_BOARD = "capture_board"  # photograph + transcribe the whiteboard
    SUBMIT = "submit"        # submit the current drawing
    CLEAR = "clear"          # clear current drawing
    SET_THEME = "set_theme"  # switch challenge theme
    DICTATION = "dictation"  # general speech → lesson notes
    NONE = "none"            # nothing actionable


@dataclass
class Intent:
    type: IntentType
    target: Optional[str] = None   # for DRAW: the shape; for SET_THEME: the theme
    text: str = ""                 # original transcription


_THEME_WORDS = {
    "geometry": "geometry", "shapes": "geometry", "geometric": "geometry",
    "objects": "objects", "object": "objects", "things": "objects",
    "mixed": "mixed", "everything": "mixed", "all": "mixed",
    # --- Hebrew ---
    "גאומטריה": "geometry", "צורות": "geometry", "גיאומטריה": "geometry",
    "חפצים": "objects", "אובייקטים": "objects", "עצמים": "objects",
    "מעורב": "mixed", "הכל": "mixed", "מעורבב": "mixed",
}


def _normalize(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", " ", text)        # strip punctuation
    text = re.sub(r"\s+", " ", text).strip()
    return text


# Hebrew one-letter prefixes that attach to a noun: ה(the) ו(and) ב(in) ל(to)
# כ(as) מ(from) ש(that). Stripping one lets "המשולש" match "משולש".
_HE_PREFIXES = "הובלכמש"


def _he_variants(word: str) -> list[str]:
    """The word plus, for Hebrew words, the form with a leading prefix removed."""
    out = [word]
    if len(word) > 2 and word[0] in _HE_PREFIXES and any("֐" <= c <= "ת" for c in word):
        out.append(word[1:])
    return out


def _find_target(words: list[str]) -> Optional[str]:
    for w in words:
        for cand in _he_variants(w):
            if cand in _TARGET_ALIASES:
                return _TARGET_ALIASES[cand]
    return None


# All spoken forms we accept for a target, for fuzzy matching.
_ALIAS_KEYS = list(_TARGET_ALIASES.keys())
# Draw-verb variants Whisper commonly produces (English + Hebrew).
_DRAW_WORDS = {"draw", "draws", "drew", "drawe", "drawa", "drawing", "drow", "draua",
               "צייר", "ציירי", "תצייר", "תציירי", "לצייר", "ציור", "נצייר"}


def _fuzzy_target(words: list[str], cutoff: float = 0.6) -> Optional[str]:
    """Find the best phonetically/spelling-close target among the words.

    Handles Whisper errors like 'drosirken' / 'syrical' -> circle by also
    scanning sub-tokens. Returns the canonical target or None.
    """
    best: tuple[float, Optional[str]] = (0.0, None)
    for w in words:
        if len(w) < 3:
            continue
        m = difflib.get_close_matches(w, _ALIAS_KEYS, n=1, cutoff=cutoff)
        if m:
            score = difflib.SequenceMatcher(None, w, m[0]).ratio()
            if score > best[0]:
                best = (score, _TARGET_ALIASES[m[0]])
    return best[1]


def _looks_like_draw(words: list[str], cutoff: float = 0.7) -> bool:
    for w in words:
        if w in _DRAW_WORDS:
            return True
        if difflib.get_close_matches(w, list(_DRAW_WORDS), n=1, cutoff=cutoff):
            return True
    return False


def parse_command(text: str) -> Intent:
    """Map a transcription to a classroom intent.

    Recognized patterns (case/punctuation insensitive):
        "draw a triangle" / "draw triangle" / "let's draw a house"  -> DRAW(target)
        "next" / "next challenge" / "another one"                   -> NEXT
        "submit" / "done" / "finished" / "check"                    -> SUBMIT
        "clear" / "erase" / "start over"                            -> CLEAR
        "switch to geometry" / "objects mode" / "mixed"             -> SET_THEME(theme)
        anything else                                               -> DICTATION
    """
    raw = text or ""
    norm = _normalize(raw)
    if not norm:
        return Intent(IntentType.NONE, text=raw)
    words = norm.split()
    wordset = set(words)

    # Theme switch (check before DRAW, since "objects" alone is a theme word)
    if any(w in wordset for w in ("switch", "mode", "theme", "change",
                                  "החלף", "שנה", "מצב", "נושא")) or \
       (len(words) <= 2 and words[0] in _THEME_WORDS):
        for w in words:
            if w in _THEME_WORDS:
                return Intent(IntentType.SET_THEME, target=_THEME_WORDS[w], text=raw)

    # Draw command — explicit draw verb (English or Hebrew) + a target word
    target = _find_target(words)
    if (wordset & _DRAW_WORDS) and target:
        return Intent(IntentType.DRAW, target=target, text=raw)
    # Retry the same challenge after a miss ("again", "try again", "retry", ...;
    # Hebrew "שוב", "שנית", "עוד פעם")
    if wordset & {"retry", "redo", "redraw", "again", "שוב", "שנית"} or \
       ("try" in wordset and "again" in wordset) or \
       ("once" in wordset and "more" in wordset) or \
       ("עוד" in wordset and "פעם" in wordset):
        return Intent(IntentType.RETRY, text=raw)
    # "next"/"another" → new random challenge (Hebrew "הבא", "אחר", "קדימה")
    if wordset & {"next", "another", "go", "הבא", "אחר", "קדימה"}:
        return Intent(IntentType.NEXT, text=raw)
    # Submit / check (Hebrew "שלח", "סיימתי", "בדוק", "מוכן", "סיום")
    if wordset & {"submit", "done", "finished", "finish", "check", "ready",
                  "שלח", "סיימתי", "בדוק", "מוכן", "סיום"}:
        return Intent(IntentType.SUBMIT, text=raw)
    # Clear (Hebrew "נקה", "מחק", "איפוס")
    if wordset & {"clear", "erase", "reset", "נקה", "מחק", "איפוס", "נקי"} or \
       ("start" in wordset and "over" in wordset):
        return Intent(IntentType.CLEAR, text=raw)
    # Capture / read the physical whiteboard (Phase 3). Require an action verb
    # near "board" (or a bare "board"/"whiteboard") so it doesn't hijack every
    # dictation sentence that merely mentions the board.
    _board_verbs = {"capture", "read", "scan", "save", "snapshot",
                    "grab", "take", "photo", "picture", "record",
                    "קרא", "סרוק", "צלם", "שמור", "תצלם"}
    has_he_board = any("לוח" in w for w in words)   # matches הלוח / בלוח / ...
    if "whiteboard" in wordset or \
       ("board" in wordset and (len(words) <= 3 or (wordset & _board_verbs))) or \
       (has_he_board and (len(words) <= 3 or (wordset & _board_verbs))):
        return Intent(IntentType.CAPTURE_BOARD, text=raw)
    # Bare target word (e.g. teacher just says "triangle")
    if target and len(words) <= 3:
        return Intent(IntentType.DRAW, target=target, text=raw)

    # --- Fuzzy fallback for mis-heard speech (Whisper-base is imperfect) ---
    # If it sounds like a draw command, or a shape word is phonetically close,
    # treat it as DRAW. This rescues "drow syrical" -> circle, etc.
    fuzzy = _fuzzy_target(words)
    if fuzzy and (_looks_like_draw(words) or len(words) <= 3):
        return Intent(IntentType.DRAW, target=fuzzy, text=raw)

    # Fallback: dictation
    return Intent(IntentType.DICTATION, text=raw)

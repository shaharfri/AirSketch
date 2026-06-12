"""Challenge curriculum + round selection for the classroom game."""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import List, Optional


# Geometry targets are judged by the primitive classifier.
GEOMETRY_TARGETS = ["circle", "triangle", "rectangle", "square", "star", "arrow", "line"]

# Object targets are judged by the Quick-Draw CNN.
OBJECT_TARGETS = ["house", "cat", "tree", "sun", "flower", "fish", "car", "airplane"]

# Friendly article + emoji-free prompt phrasing
_ARTICLE = {
    "circle": "a", "triangle": "a", "rectangle": "a", "square": "a",
    "star": "a", "arrow": "an", "line": "a",
    "house": "a", "cat": "a", "tree": "a", "sun": "a", "flower": "a",
    "fish": "a", "car": "a", "airplane": "an",
}


@dataclass
class Challenge:
    target: str          # e.g. "triangle"
    theme: str           # "geometry" | "objects"
    round_index: int     # 1-based
    prompt: str = ""     # "Draw a TRIANGLE!"

    def __post_init__(self):
        if not self.prompt:
            art = _ARTICLE.get(self.target, "a")
            self.prompt = f"Draw {art} {self.target.upper()}!"


@dataclass
class ChallengeResult:
    challenge: Challenge
    detected: str
    score: int           # 0-100
    stars: int           # 0-3
    passed: bool
    explanation: str = ""


@dataclass
class ChallengeEngine:
    """Picks challenges from a curriculum and tracks the running score."""

    theme: str = "geometry"          # "geometry" | "objects" | "mixed"
    seed: Optional[int] = None
    avoid_repeats: bool = True

    _rng: random.Random = field(init=False)
    _round: int = field(init=False, default=0)
    _history: List[ChallengeResult] = field(init=False, default_factory=list)
    _last_target: Optional[str] = field(init=False, default=None)

    def __post_init__(self):
        self._rng = random.Random(self.seed)

    def _pool(self) -> List[str]:
        if self.theme == "geometry":
            return list(GEOMETRY_TARGETS)
        if self.theme == "objects":
            return list(OBJECT_TARGETS)
        return GEOMETRY_TARGETS + OBJECT_TARGETS  # mixed

    def _theme_of(self, target: str) -> str:
        return "objects" if target in OBJECT_TARGETS else "geometry"

    def next_challenge(self) -> Challenge:
        pool = self._pool()
        if self.avoid_repeats and self._last_target and len(pool) > 1:
            pool = [t for t in pool if t != self._last_target]
        target = self._rng.choice(pool)
        self._last_target = target
        self._round += 1
        return Challenge(
            target=target,
            theme=self._theme_of(target),
            round_index=self._round,
        )

    def challenge_for(self, target: str) -> Challenge:
        """Create a challenge for a specific target (e.g. from a voice command)."""
        self._last_target = target
        self._round += 1
        return Challenge(
            target=target,
            theme=self._theme_of(target),
            round_index=self._round,
        )

    def retry_last(self) -> Optional[Challenge]:
        """Re-issue the most recent challenge for another attempt.

        Drops the recorded result being retried (so a retry doesn't count
        against the student) and reuses the same round_index — the round count
        is unchanged. Returns None if there is nothing to retry.
        """
        if not self._history:
            return None
        last = self._history.pop()
        target = last.challenge.target
        self._last_target = target
        return Challenge(
            target=target,
            theme=self._theme_of(target),
            round_index=last.challenge.round_index,
        )

    def record_result(self, result: ChallengeResult) -> None:
        self._history.append(result)

    @property
    def round_count(self) -> int:
        return self._round

    @property
    def total_score(self) -> int:
        return sum(r.score for r in self._history)

    @property
    def total_stars(self) -> int:
        return sum(r.stars for r in self._history)

    @property
    def passed_count(self) -> int:
        return sum(1 for r in self._history if r.passed)

    @property
    def history(self) -> List[ChallengeResult]:
        return list(self._history)

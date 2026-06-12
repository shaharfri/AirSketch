"""AirSketch classroom mode — teacher/student lesson activities.

Phase 1: the challenge game (draw-on-demand with judging + celebration).
Later phases add voice transcription, board capture, and lesson reports.
"""
from airsketch.classroom.challenge_engine import (
    Challenge,
    ChallengeEngine,
    ChallengeResult,
    GEOMETRY_TARGETS,
    OBJECT_TARGETS,
)
from airsketch.classroom.celebration import Celebration
from airsketch.classroom.judge import Judge
from airsketch.classroom.voice_commands import Intent, IntentType, parse_command

__all__ = [
    "Challenge", "ChallengeEngine", "ChallengeResult",
    "GEOMETRY_TARGETS", "OBJECT_TARGETS",
    "Celebration", "Judge",
    "Intent", "IntentType", "parse_command",
]

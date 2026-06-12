"""Gesture detection from MediaPipe hand landmarks.

Two strategies:
 - IndexPointingDetector (default, recommended):
     Pen-down when only the index finger is extended (pointing pose).
     Uses 3D world landmarks (orientation-invariant) and fuses TWO signals
     per finger — joint angle AND compactness ratio — for robustness.
     Falls back to 2D pixel landmarks if world landmarks aren't available.
 - PinchDetector (alternative): pen-down when thumb tip and index tip touch.
"""
import math
from typing import List, Sequence, Tuple

# MediaPipe landmark indices
WRIST = 0
THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 1, 2, 3, 4
INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 9, 10, 11, 12
RING_MCP, RING_PIP, RING_DIP, RING_TIP = 13, 14, 15, 16
PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP = 17, 18, 19, 20

# Point can be 2D or 3D — algorithms handle both via Sequence indexing
Point = Tuple[float, ...]


# ----------------------------------------------------------------------------
# Geometry helpers (dimension-agnostic — works for both 2D and 3D)
# ----------------------------------------------------------------------------

def _dist(a: Sequence[float], b: Sequence[float]) -> float:
    return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(min(len(a), len(b)))))


def _angle_deg(a: Sequence[float], b: Sequence[float], c: Sequence[float]) -> float:
    """Angle in degrees at vertex b formed by rays b->a and b->c."""
    dim = min(len(a), len(b), len(c))
    v1 = [a[i] - b[i] for i in range(dim)]
    v2 = [c[i] - b[i] for i in range(dim)]
    n1 = math.sqrt(sum(x * x for x in v1))
    n2 = math.sqrt(sum(x * x for x in v2))
    if n1 < 1e-6 or n2 < 1e-6:
        return 180.0
    dot = sum(v1[i] * v2[i] for i in range(dim))
    cos_t = max(-1.0, min(1.0, dot / (n1 * n2)))
    return math.degrees(math.acos(cos_t))


def finger_joint_angle(
    landmarks: Sequence[Sequence[float]],
    mcp: int, pip: int, dip: int, tip: int,
) -> float:
    """The smaller of the two joint angles along the finger (PIP and DIP).

    Fully extended finger → both angles ≈ 180°.
    Curled finger → at least one angle drops far below.
    Taking the min sharpens the signal at the curl transition.
    """
    a1 = _angle_deg(landmarks[mcp], landmarks[pip], landmarks[dip])
    a2 = _angle_deg(landmarks[pip], landmarks[dip], landmarks[tip])
    return min(a1, a2)


def finger_compactness(
    landmarks: Sequence[Sequence[float]],
    mcp: int, pip: int, dip: int, tip: int,
) -> float:
    """Compactness = distance(MCP, TIP) / sum of chain segments.

    Independent of finger length / camera distance.
      - Extended finger:  ratio ≈ 1.0  (straight chain → direct ≈ chain sum)
      - Curled finger:    ratio < 0.5  (tip folds back toward MCP)
    """
    direct = _dist(landmarks[mcp], landmarks[tip])
    chain = (
        _dist(landmarks[mcp], landmarks[pip])
        + _dist(landmarks[pip], landmarks[dip])
        + _dist(landmarks[dip], landmarks[tip])
    )
    if chain < 1e-6:
        return 1.0
    return direct / chain


# Kept for backwards compatibility (used by tests / older code)
def finger_angle(
    landmarks: Sequence[Sequence[float]],
    mcp: int, pip: int, dip: int, tip: int,
) -> float:
    return finger_joint_angle(landmarks, mcp, pip, dip, tip)


# ----------------------------------------------------------------------------
# Per-finger evidence
# ----------------------------------------------------------------------------

class FingerEvidence:
    """Holds the angle + compactness signals for one finger."""

    __slots__ = ("angle", "compact")

    def __init__(self, angle: float, compact: float):
        self.angle = angle
        self.compact = compact

    @classmethod
    def from_landmarks(
        cls,
        landmarks: Sequence[Sequence[float]],
        mcp: int, pip: int, dip: int, tip: int,
    ) -> "FingerEvidence":
        return cls(
            finger_joint_angle(landmarks, mcp, pip, dip, tip),
            finger_compactness(landmarks, mcp, pip, dip, tip),
        )

    # --- Fused decision rules ---

    def is_strongly_extended(self) -> bool:
        return self.angle >= 150.0 and self.compact >= 0.88

    def is_weakly_extended(self) -> bool:
        """Used in hysteresis: easier to stay extended than to engage."""
        return self.angle >= 125.0 and self.compact >= 0.72

    def is_curled(self) -> bool:
        return self.angle <= 130.0 or self.compact <= 0.75

    def as_dict(self) -> dict:
        return {"angle": round(self.angle, 1), "compact": round(self.compact, 3)}


# ----------------------------------------------------------------------------
# Main gesture detector
# ----------------------------------------------------------------------------

class IndexPointingDetector:
    """Pen-down when the index finger is extended and the hand is not flat-open.

    Prefers 3D world landmarks (rotation-invariant). Falls back to 2D pixel
    landmarks if world landmarks aren't provided.

    Decision logic (with hysteresis):
      ENGAGE  → index is_strongly_extended AND not open_palm
      RELEASE → index NOT is_weakly_extended OR open_palm
      open_palm → ALL FOUR fingers strongly extended

    Confirmation: state flips require `confirm_frames` consecutive agreements.
    """

    def __init__(self, confirm_frames: int = 3):
        self._confirm_frames = confirm_frames
        self._currently_active = False
        self._candidate_state = False
        self._candidate_count = 0
        self._last_diagnostics: dict = {}

    def update(
        self,
        landmarks: Sequence[Sequence[float]] | None,
        world_landmarks: Sequence[Sequence[float]] | None = None,
    ) -> bool:
        # Prefer 3D world landmarks; fall back to 2D
        lms = world_landmarks if world_landmarks else landmarks
        coord_space = "3d" if world_landmarks else "2d"

        if lms is None or len(lms) < 21:
            self._candidate_state = False
            self._candidate_count = 0
            self._currently_active = False
            self._last_diagnostics = {"reason": "no_landmarks", "coords": coord_space}
            return False

        idx = FingerEvidence.from_landmarks(lms, INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP)
        mid = FingerEvidence.from_landmarks(lms, MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP)
        rng = FingerEvidence.from_landmarks(lms, RING_MCP, RING_PIP, RING_DIP, RING_TIP)
        pky = FingerEvidence.from_landmarks(lms, PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP)

        # Open-palm override: all four fingers clearly extended → pause drawing
        open_palm = (
            idx.is_strongly_extended()
            and mid.is_strongly_extended()
            and rng.is_strongly_extended()
            and pky.is_strongly_extended()
        )

        # Fused decision with hysteresis
        if self._currently_active:
            target = idx.is_weakly_extended() and not open_palm
        else:
            target = idx.is_strongly_extended() and not open_palm

        self._last_diagnostics = {
            "coords": coord_space,
            "index": idx.as_dict(),
            "middle": mid.as_dict(),
            "ring": rng.as_dict(),
            "pinky": pky.as_dict(),
            "open_palm": open_palm,
            "target": target,
        }

        if target == self._candidate_state:
            self._candidate_count += 1
        else:
            self._candidate_state = target
            self._candidate_count = 1
        if self._candidate_count >= self._confirm_frames:
            self._currently_active = target

        return self._currently_active

    @property
    def is_active(self) -> bool:
        return self._currently_active

    @property
    def diagnostics(self) -> dict:
        return dict(self._last_diagnostics)

    def reset(self) -> None:
        self._currently_active = False
        self._candidate_state = False
        self._candidate_count = 0


# ----------------------------------------------------------------------------
# PinchDetector (alternative gesture) — unchanged
# ----------------------------------------------------------------------------

class PinchDetector:
    """Detects pinch (thumb tip touching index tip).

    Uses thumb-index distance normalized by a hand-size reference, with
    hysteresis (different on/off thresholds) and a confirmation window.
    """

    def __init__(
        self,
        pinch_threshold: float = 0.30,
        release_threshold: float = 0.45,
        confirm_frames: int = 3,
    ):
        self._pinch_threshold = pinch_threshold
        self._release_threshold = release_threshold
        self._confirm_frames = confirm_frames
        self._currently_pinching = False
        self._candidate_state = False
        self._candidate_count = 0

    @staticmethod
    def _ratio(thumb_tip, index_tip, hand_size: float) -> float:
        return _dist(thumb_tip, index_tip) / max(hand_size, 1.0)

    def update(
        self,
        thumb_tip: Point | None,
        index_tip: Point | None,
        hand_size: float,
    ) -> bool:
        if thumb_tip is None or index_tip is None:
            self._candidate_state = False
            self._candidate_count = 0
            self._currently_pinching = False
            return False

        rel = self._ratio(thumb_tip, index_tip, hand_size)
        target = rel <= (
            self._release_threshold if self._currently_pinching else self._pinch_threshold
        )

        if target == self._candidate_state:
            self._candidate_count += 1
        else:
            self._candidate_state = target
            self._candidate_count = 1
        if self._candidate_count >= self._confirm_frames:
            self._currently_pinching = target
        return self._currently_pinching

    @property
    def is_pinching(self) -> bool:
        return self._currently_pinching

    def reset(self) -> None:
        self._currently_pinching = False
        self._candidate_state = False
        self._candidate_count = 0

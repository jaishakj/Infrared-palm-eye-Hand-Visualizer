"""tracker.py — MediaPipe hand detection wrapper.

Wraps mediapipe.solutions.hands and exposes a clean per-frame API that
returns pixel-space landmarks plus a persistence timer used for the
eye-awakening sequence.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import mediapipe as mp
import numpy as np

mp_hands = mp.solutions.hands

# Landmark indices (MediaPipe hand model)
WRIST = 0
THUMB_MCP, THUMB_PIP, THUMB_TIP = 2, 3, 4
INDEX_MCP, INDEX_PIP, INDEX_TIP = 5, 6, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_TIP = 9, 10, 12
RING_MCP, RING_PIP, RING_TIP = 13, 14, 16
PINKY_MCP, PINKY_PIP, PINKY_TIP = 17, 18, 20

PALM_LANDMARKS = [WRIST, INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP]

# Skeleton connections (subset of mp_hands.HAND_CONNECTIONS as plain tuples)
CONNECTIONS = list(mp_hands.HAND_CONNECTIONS)

FINGERTIPS = [4, 8, 12, 16, 20]

EYE_APPEAR_SECONDS = 2.5
EYE_FADE_SECONDS = 0.6  # fade-in duration once timer passes threshold


@dataclass
class HandFrame:
    """Result of processing a single frame."""

    present: bool = False
    landmarks_px: np.ndarray | None = None  # (21, 2) int32 pixel coords
    landmarks_norm: np.ndarray | None = None  # (21, 3) normalized
    handedness: str = ""
    hand_timer: float = 0.0
    eye_alpha: float = 0.0
    extras: dict = field(default_factory=dict)


class HandTracker:
    """Stateful hand tracker. Keeps a presence timer across frames so the
    renderer knows when to awaken the palm eye."""

    def __init__(
        self,
        max_hands: int = 1,
        detection_conf: float = 0.6,
        tracking_conf: float = 0.55,
    ) -> None:
        self._hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_hands,
            model_complexity=1,
            min_detection_confidence=detection_conf,
            min_tracking_confidence=tracking_conf,
        )
        self._first_seen: float | None = None
        self._last_seen: float = 0.0
        # small grace period so a single dropped detection frame doesn't
        # reset the awakening timer
        self._grace = 0.35

    def close(self) -> None:
        self._hands.close()

    def process(self, frame_bgr: np.ndarray) -> HandFrame:
        h, w = frame_bgr.shape[:2]
        rgb = frame_bgr[:, :, ::-1]  # BGR -> RGB view, no copy needed by mp
        rgb = np.ascontiguousarray(rgb)
        results = self._hands.process(rgb)

        now = time.monotonic()
        out = HandFrame()

        if results.multi_hand_landmarks:
            lm = results.multi_hand_landmarks[0].landmark
            norm = np.array([[p.x, p.y, p.z] for p in lm], dtype=np.float32)
            px = np.empty((21, 2), dtype=np.int32)
            px[:, 0] = np.clip(norm[:, 0] * w, 0, w - 1).astype(np.int32)
            px[:, 1] = np.clip(norm[:, 1] * h, 0, h - 1).astype(np.int32)

            if self._first_seen is None:
                self._first_seen = now
            self._last_seen = now

            out.present = True
            out.landmarks_px = px
            out.landmarks_norm = norm
            if results.multi_handedness:
                out.handedness = (
                    results.multi_handedness[0].classification[0].label
                )
        else:
            # allow a short grace gap before resetting the timer
            if self._first_seen is not None and (now - self._last_seen) > self._grace:
                self._first_seen = None

        if self._first_seen is not None:
            out.hand_timer = now - self._first_seen
        else:
            out.hand_timer = 0.0

        # Eye alpha: 0 until threshold, then eased fade over EYE_FADE_SECONDS
        if out.present and out.hand_timer >= EYE_APPEAR_SECONDS:
            t = min(1.0, (out.hand_timer - EYE_APPEAR_SECONDS) / EYE_FADE_SECONDS)
            out.eye_alpha = t * t * (3 - 2 * t)  # smoothstep
        else:
            out.eye_alpha = 0.0

        return out

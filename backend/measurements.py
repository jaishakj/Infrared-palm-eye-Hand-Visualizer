"""measurements.py — real-world hand measurements from pixel landmarks.

Calibration: the wrist→index-MCP pixel distance multiplied by 1.8 is
assumed to equal HAND_REAL_WIDTH_CM (8.5 cm), giving a px→cm scale that
adapts to how far the hand is from the camera.

Joint angles are the interior angle at each finger's PIP joint computed
via the dot product of the two bone vectors meeting there (MCP→PIP and
TIP→PIP). A straight finger reads near 180°, a curled one near 60–90°.
"""

from __future__ import annotations

import math

import numpy as np

from tracker import (
    WRIST,
    INDEX_MCP,
    THUMB_MCP, THUMB_PIP, THUMB_TIP,
    INDEX_PIP, INDEX_TIP,
    MIDDLE_MCP, MIDDLE_PIP, MIDDLE_TIP,
    RING_MCP, RING_PIP, RING_TIP,
    PINKY_MCP, PINKY_PIP, PINKY_TIP,
)

HAND_REAL_WIDTH_CM = 8.5
WIDTH_FACTOR = 1.8

# finger -> (MCP, PIP, TIP)
FINGER_JOINTS: dict[str, tuple[int, int, int]] = {
    "THUMB": (THUMB_MCP, THUMB_PIP, THUMB_TIP),
    "INDEX": (INDEX_MCP, INDEX_PIP, INDEX_TIP),
    "MIDDLE": (MIDDLE_MCP, MIDDLE_PIP, MIDDLE_TIP),
    "RING": (RING_MCP, RING_PIP, RING_TIP),
    "PINKY": (PINKY_MCP, PINKY_PIP, PINKY_TIP),
}


def px_to_cm_scale(landmarks_px: np.ndarray) -> float:
    """cm-per-pixel scale from the wrist→index-MCP distance."""
    d = float(np.linalg.norm(landmarks_px[INDEX_MCP] - landmarks_px[WRIST]))
    if d < 1e-6:
        return 0.0
    return HAND_REAL_WIDTH_CM / (d * WIDTH_FACTOR)


def palm_area_cm2(landmarks_px: np.ndarray, hull_px: np.ndarray | None) -> float:
    """Palm/hand area in cm² using the convex hull polygon (shoelace)."""
    scale = px_to_cm_scale(landmarks_px)
    if scale == 0.0 or hull_px is None or len(hull_px) < 3:
        return 0.0
    pts = hull_px.reshape(-1, 2).astype(np.float64)
    x, y = pts[:, 0], pts[:, 1]
    area_px = 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    return round(area_px * scale * scale, 1)


def joint_angle_deg(a: np.ndarray, pivot: np.ndarray, c: np.ndarray) -> float:
    """Interior angle at `pivot` between rays pivot→a and pivot→c."""
    v1 = a.astype(np.float64) - pivot
    v2 = c.astype(np.float64) - pivot
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return 0.0
    cosang = float(np.dot(v1, v2) / (n1 * n2))
    cosang = max(-1.0, min(1.0, cosang))
    return math.degrees(math.acos(cosang))


def all_joint_angles(landmarks_px: np.ndarray) -> dict[str, int]:
    out: dict[str, int] = {}
    for name, (mcp, pip_, tip) in FINGER_JOINTS.items():
        ang = joint_angle_deg(
            landmarks_px[mcp], landmarks_px[pip_], landmarks_px[tip]
        )
        out[name] = int(round(ang))
    return out


def measure(landmarks_px: np.ndarray, hull_px: np.ndarray | None) -> dict:
    return {
        "palm_area_cm2": palm_area_cm2(landmarks_px, hull_px),
        "joint_angles": all_joint_angles(landmarks_px),
    }

"""renderer.py — OpenCV drawing pipeline.

Pipeline order (per frame):
  1. COLORMAP_HOT applied to grayscale, blended 75/25 with the raw frame
  2. Convex hull contour echoes at offsets [4, 8, 14, 20] px
  3. MediaPipe skeleton, double pass (thick glow + thin bright line),
     orange fingertip joints
  4. Mystical eye composited at eye_alpha over the palm centre —
     2.5 Hz iris pulse, 18 radial spokes, upper/lower eyelid arcs
"""

from __future__ import annotations

import time

import cv2
import numpy as np

from tracker import (
    HandFrame,
    CONNECTIONS,
    FINGERTIPS,
    PALM_LANDMARKS,
    WRIST,
    MIDDLE_MCP,
)

# BGR palette (matches frontend tokens)
CLR_GLOW = (26, 107, 255)      # #ff6b1a orange (BGR)
CLR_LINE = (140, 200, 255)     # pale hot line
CLR_TIP = (26, 107, 255)
CLR_HULL = (60, 160, 255)
CLR_EYE_RING = (40, 130, 255)
CLR_EYE_SPOKE = (90, 170, 255)
CLR_IRIS = (255, 220, 160)     # icy core against the hot palette
CLR_PUPIL = (16, 10, 8)

HULL_OFFSETS = [4, 8, 14, 20]
IRIS_PULSE_HZ = 2.5
EYE_SPOKES = 18


def thermal_base(frame_bgr: np.ndarray) -> np.ndarray:
    """Step 1 — infrared look: COLORMAP_HOT blended 75/25 with raw."""
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    hot = cv2.applyColorMap(gray, cv2.COLORMAP_HOT)
    return cv2.addWeighted(hot, 0.75, frame_bgr, 0.25, 0)


def compute_hull(landmarks_px: np.ndarray) -> np.ndarray:
    return cv2.convexHull(landmarks_px.reshape(-1, 1, 2))


def _offset_polygon(hull: np.ndarray, offset_px: int) -> np.ndarray:
    """Expand hull outward from its centroid by ~offset_px."""
    pts = hull.reshape(-1, 2).astype(np.float64)
    centroid = pts.mean(axis=0)
    vecs = pts - centroid
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms < 1e-6] = 1.0
    expanded = pts + vecs / norms * offset_px
    return expanded.astype(np.int32).reshape(-1, 1, 2)


def draw_hull_echoes(img: np.ndarray, hull: np.ndarray) -> None:
    """Step 2 — concentric hull contours at fixed pixel offsets."""
    overlay = img.copy()
    for i, off in enumerate(HULL_OFFSETS):
        poly = _offset_polygon(hull, off)
        alpha_w = max(1, 2 - i // 2)
        cv2.polylines(overlay, [poly], True, CLR_HULL, alpha_w, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.55, img, 0.45, 0, dst=img)


def draw_skeleton(img: np.ndarray, landmarks_px: np.ndarray) -> None:
    """Step 3 — double-pass skeleton: soft glow then bright core line."""
    # pass 1: glow
    glow = img.copy()
    for a, b in CONNECTIONS:
        cv2.line(glow, tuple(landmarks_px[a]), tuple(landmarks_px[b]),
                 CLR_GLOW, 7, cv2.LINE_AA)
    cv2.addWeighted(glow, 0.35, img, 0.65, 0, dst=img)

    # pass 2: bright thin line
    for a, b in CONNECTIONS:
        cv2.line(img, tuple(landmarks_px[a]), tuple(landmarks_px[b]),
                 CLR_LINE, 2, cv2.LINE_AA)

    # joints
    for i, (x, y) in enumerate(landmarks_px):
        if i in FINGERTIPS:
            cv2.circle(img, (x, y), 6, CLR_TIP, -1, cv2.LINE_AA)
            cv2.circle(img, (x, y), 8, CLR_LINE, 1, cv2.LINE_AA)
        else:
            cv2.circle(img, (x, y), 3, CLR_LINE, -1, cv2.LINE_AA)


def eye_geometry(landmarks_px: np.ndarray) -> tuple[tuple[int, int], int]:
    """Palm centre = mean of landmarks [0,5,9,13,17]; radius from
    wrist→middle-MCP distance × 0.35, clamped to [20, 80] px."""
    pts = landmarks_px[PALM_LANDMARKS].astype(np.float64)
    cx, cy = pts.mean(axis=0)
    span = float(np.linalg.norm(landmarks_px[MIDDLE_MCP] - landmarks_px[WRIST]))
    radius = int(np.clip(span * 0.35, 20, 80))
    return (int(cx), int(cy)), radius


def draw_eye(img: np.ndarray, center: tuple[int, int], radius: int,
             alpha: float) -> None:
    """Step 4 — mystical eye composited at `alpha`."""
    if alpha <= 0.01:
        return

    t = time.monotonic()
    pulse = 0.5 + 0.5 * np.sin(2 * np.pi * IRIS_PULSE_HZ * t)
    iris_r = int(radius * (0.42 + 0.10 * pulse))
    pupil_r = max(3, int(iris_r * (0.45 + 0.12 * (1 - pulse))))

    layer = img.copy()
    cx, cy = center

    # outer ring + halo
    cv2.circle(layer, center, radius, CLR_EYE_RING, 2, cv2.LINE_AA)
    cv2.circle(layer, center, int(radius * 1.12), CLR_EYE_RING, 1, cv2.LINE_AA)

    # 18 radial spokes between iris and outer ring
    for k in range(EYE_SPOKES):
        ang = 2 * np.pi * k / EYE_SPOKES + t * 0.6  # slow rotation
        r0, r1 = iris_r + 3, radius - 3
        x0 = int(cx + r0 * np.cos(ang)); y0 = int(cy + r0 * np.sin(ang))
        x1 = int(cx + r1 * np.cos(ang)); y1 = int(cy + r1 * np.sin(ang))
        cv2.line(layer, (x0, y0), (x1, y1), CLR_EYE_SPOKE, 1, cv2.LINE_AA)

    # iris (pulsing) + pupil
    cv2.circle(layer, center, iris_r, CLR_IRIS, 2, cv2.LINE_AA)
    cv2.circle(layer, center, pupil_r, CLR_PUPIL, -1, cv2.LINE_AA)
    cv2.circle(layer, center, pupil_r, CLR_IRIS, 1, cv2.LINE_AA)

    # eyelid arcs (upper + lower)
    axes = (int(radius * 1.35), int(radius * 0.95))
    cv2.ellipse(layer, center, axes, 0, 200, 340, CLR_EYE_RING, 2, cv2.LINE_AA)
    cv2.ellipse(layer, center, axes, 0, 20, 160, CLR_EYE_RING, 2, cv2.LINE_AA)

    cv2.addWeighted(layer, alpha, img, 1 - alpha, 0, dst=img)


def render(frame_bgr: np.ndarray, hand: HandFrame) -> tuple[np.ndarray, np.ndarray | None]:
    """Full pipeline. Returns (rendered_frame, hull or None)."""
    out = thermal_base(frame_bgr)
    hull = None

    if hand.present and hand.landmarks_px is not None:
        hull = compute_hull(hand.landmarks_px)
        draw_hull_echoes(out, hull)
        draw_skeleton(out, hand.landmarks_px)
        center, radius = eye_geometry(hand.landmarks_px)
        draw_eye(out, center, radius, hand.eye_alpha)

    return out, hull


def encode_jpeg_b64(img: np.ndarray, quality: int = 82) -> str:
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        return ""
    import base64
    return base64.b64encode(buf.tobytes()).decode("ascii")

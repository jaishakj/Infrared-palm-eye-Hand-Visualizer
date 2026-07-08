"""main.py — FastAPI app: serves the frontend and streams frames over /ws.

Run from the backend/ directory:

    uvicorn main:app --port 8000

Then open http://localhost:8000
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

import cv2
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles

from tracker import HandTracker
from renderer import render, encode_jpeg_b64
from measurements import measure

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"

CAM_INDEX = 0
CAM_WIDTH = 960
CAM_HEIGHT = 540
JPEG_QUALITY = 82
TARGET_FPS = 30

app = FastAPI(title="Infrared Palm Eye")


@app.websocket("/ws")
async def ws_stream(ws: WebSocket) -> None:
    """Per-connection capture → track → render → measure → send loop."""
    await ws.accept()
    loop = asyncio.get_event_loop()

    cap = cv2.VideoCapture(CAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
    tracker = HandTracker()

    fps = 0.0
    last_t = time.perf_counter()
    frame_budget = 1.0 / TARGET_FPS

    try:
        if not cap.isOpened():
            await ws.send_text(json.dumps({
                "error": "camera_unavailable",
                "message": "Could not open the webcam. Close other apps using it and reconnect.",
            }))
            await ws.close()
            return

        while True:
            tick = time.perf_counter()

            ok, frame = await loop.run_in_executor(None, cap.read)
            if not ok or frame is None:
                await asyncio.sleep(0.05)
                continue

            frame = cv2.flip(frame, 1)  # mirror for a natural feel

            # Heavy work off the event loop
            hand = await loop.run_in_executor(None, tracker.process, frame)
            rendered, hull = await loop.run_in_executor(None, render, frame, hand)

            measurements = None
            if hand.present and hand.landmarks_px is not None:
                measurements = measure(hand.landmarks_px, hull)

            b64 = await loop.run_in_executor(
                None, encode_jpeg_b64, rendered, JPEG_QUALITY
            )
            if not b64:
                continue

            now = time.perf_counter()
            dt = now - last_t
            last_t = now
            inst = 1.0 / dt if dt > 0 else 0.0
            fps = inst if fps == 0.0 else fps * 0.9 + inst * 0.1

            await ws.send_text(json.dumps({
                "frame": b64,
                "fps": round(fps, 1),
                "hand_present": hand.present,
                "hand_timer": round(hand.hand_timer, 2),
                "eye_alpha": round(hand.eye_alpha, 2),
                "measurements": measurements,
            }))

            # Pace the loop toward TARGET_FPS
            elapsed = time.perf_counter() - tick
            if elapsed < frame_budget:
                await asyncio.sleep(frame_budget - elapsed)

    except WebSocketDisconnect:
        pass
    finally:
        cap.release()
        tracker.close()


# Mounted last so /ws keeps precedence; html=True serves index.html at /
app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")

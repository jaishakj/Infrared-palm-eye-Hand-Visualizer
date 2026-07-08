# ◉ Infrared Palm Eye

Real-time hand visualiser. Your webcam feed is re-rendered in an infrared/thermal
aesthetic, a mystical eye awakens in your palm centre after 2.5 seconds of
steady presence, and live measurements stream to a sidebar. Runs entirely in
the browser against a local FastAPI + WebSocket backend.

## Stack

| Layer    | Tech |
|----------|------|
| Backend  | FastAPI · WebSocket · MediaPipe Hands · OpenCV · NumPy |
| Frontend | Vanilla HTML/CSS/JS · Material 3 dark · Rajdhani + Share Tech Mono |

## Structure

```
infrared-palm-eye/
├── backend/
│   ├── main.py              # FastAPI app + WS stream + serves frontend
│   ├── tracker.py           # MediaPipe hand detection + presence timer
│   ├── renderer.py          # OpenCV thermal/hull/skeleton/eye pipeline
│   ├── measurements.py      # Palm area cm² + PIP joint angles °
│   └── requirements.txt
├── frontend/
│   ├── index.html
│   ├── style.css            # M3 dark tokens + grid layout + scanlines
│   └── app.js               # WS manager, canvas renderer, UI updater
└── README.md
```

## Run

```bash
cd backend
pip install -r requirements.txt   # or: pip install fastapi uvicorn mediapipe opencv-python numpy websockets
uvicorn main:app --port 8000
# open http://localhost:8000
```

Click **CONNECT** (the URL auto-fills from the current host), show your palm,
hold it steady for 2.5 s, and the eye opens.

## How it works

### WebSocket payload (`/ws`, per frame)

```json
{
  "frame": "<base64 JPEG q=82>",
  "fps": 29.4,
  "hand_present": true,
  "hand_timer": 1.84,
  "eye_alpha": 0.72,
  "measurements": {
    "palm_area_cm2": 42.3,
    "joint_angles": { "THUMB": 162, "INDEX": 171, "MIDDLE": 168, "RING": 155, "PINKY": 143 }
  }
}
```

### Render pipeline (backend, per frame)

1. **Thermal base** — `COLORMAP_HOT` on the equalised grayscale, blended
   75/25 with the raw frame.
2. **Hull echoes** — convex hull of the 21 landmarks, redrawn at outward
   offsets of 4 / 8 / 14 / 20 px.
3. **Skeleton** — MediaPipe hand connections drawn twice: a thick soft glow
   pass, then a thin bright line pass. Fingertip joints in orange.
4. **The eye** — composited at `eye_alpha`. Iris pulses at 2.5 Hz, 18 radial
   spokes rotate slowly, upper/lower eyelid arcs frame the ring.

**Eye placement** — centre is the mean of landmarks `[0, 5, 9, 13, 17]`
(wrist + four MCP knuckles). Radius = `dist(wrist, middle_MCP) × 0.35`,
clamped to 20–80 px. The eye only appears after `hand_timer ≥ 2.5 s`, then
fades in with a smoothstep over ~0.6 s.

### Measurements

- **px→cm calibration** — `dist(wrist, index_MCP) × 1.8` is assumed equal to
  a real hand width of **8.5 cm**, so scale adapts with distance from camera.
- **Palm area** — shoelace area of the convex hull polygon, converted to cm².
- **Joint angles** — interior angle at each finger's PIP joint via the dot
  product of the MCP→PIP and TIP→PIP vectors. Straight finger ≈ 180°.

### Frontend

- CSS grid: top bar / camera canvas + sidebar / footer.
- Sidebar cards: **Connection** (URL input + connect button, status dot) →
  **Eye status** (dormant / awakening progress arc / active glow) →
  **Palm area** (big mono number) → **Joint angles** (5 rows with animated
  fill bars, colour-coded ≥150° green · 90–149° amber · <90° red).
- Frames arrive as base64 JPEG and are painted to `<canvas>` via `drawImage`,
  throttled to one draw per animation frame. Scanline overlay via `::after`.
- WS auto-reconnect with exponential backoff: 1 → 2 → 4 → 8 → 30 s max.

## Notes & tuning

- `HAND_REAL_WIDTH_CM` in `measurements.py` — set to your actual palm width
  for more accurate cm² readings.
- `CAM_INDEX` / resolution in `main.py` if you have multiple cameras.
- The presence timer has a 0.35 s grace window so a single dropped detection
  frame doesn't reset the awakening sequence.
- The frame is mirrored (`cv2.flip(frame, 1)`) for natural interaction.

/* ═══════════════════════════════════════════════════════════
   INFRARED PALM EYE — frontend controller
   WS manager (exp. backoff) · canvas renderer · UI updater
   ═══════════════════════════════════════════════════════════ */

"use strict";

/* ─── DOM refs ──────────────────────────────────────────── */

const $ = (id) => document.getElementById(id);

const canvas = $("feed");
const ctx = canvas.getContext("2d");

const overlay = $("stage-overlay");
const overlayTitle = overlay.querySelector(".overlay-title");
const overlayHint = overlay.querySelector(".overlay-hint");

const wsUrlInput = $("ws-url");
const btnConnect = $("btn-connect");
const connDot = $("conn-dot");
const connLabel = $("conn-label");

const statFps = $("stat-fps");
const statHand = $("stat-hand");
const statTimer = $("stat-timer");

const eyeWidget = document.querySelector(".eye-widget");
const eyeArc = $("eye-arc");
const eyePhase = $("eye-phase");
const eyeSub = $("eye-sub");

const areaValue = $("area-value");
const jointList = $("joint-list");
const footerClock = $("footer-clock");

const EYE_APPEAR_SECONDS = 2.5;
const ARC_CIRCUMFERENCE = 314.16; // 2π × 50

const FINGERS = ["THUMB", "INDEX", "MIDDLE", "RING", "PINKY"];

/* ─── Joint rows (built once) ───────────────────────────── */

const jointEls = {};
for (const name of FINGERS) {
  const row = document.createElement("div");
  row.className = "joint-row";
  row.innerHTML = `
    <span class="joint-name">${name}</span>
    <div class="joint-bar"><div class="joint-fill"></div></div>
    <span class="joint-deg">--°</span>
  `;
  jointList.appendChild(row);
  jointEls[name] = {
    fill: row.querySelector(".joint-fill"),
    deg: row.querySelector(".joint-deg"),
  };
}

/* ─── WebSocket manager with exponential backoff ────────── */

const BACKOFF_STEPS = [1000, 2000, 4000, 8000, 30000]; // 1→2→4→8→30s max

const wsManager = {
  ws: null,
  url: "",
  wantConnected: false,
  attempt: 0,
  reconnectTimer: null,

  connect(url) {
    this.url = url;
    this.wantConnected = true;
    this.attempt = 0;
    this._open();
  },

  disconnect() {
    this.wantConnected = false;
    clearTimeout(this.reconnectTimer);
    if (this.ws) this.ws.close();
    this.ws = null;
    setConnState("offline");
  },

  _open() {
    clearTimeout(this.reconnectTimer);
    setConnState("connecting");

    let ws;
    try {
      ws = new WebSocket(this.url);
    } catch (err) {
      this._scheduleReconnect();
      return;
    }
    this.ws = ws;

    ws.onopen = () => {
      this.attempt = 0;
      setConnState("online");
    };

    ws.onmessage = (ev) => {
      let data;
      try {
        data = JSON.parse(ev.data);
      } catch {
        return;
      }
      if (data.error) {
        showOverlay("FEED ERROR", data.message || data.error);
        return;
      }
      handleFrame(data);
    };

    ws.onclose = () => {
      if (this.ws !== ws) return; // stale socket
      this.ws = null;
      if (this.wantConnected) {
        this._scheduleReconnect();
      } else {
        setConnState("offline");
      }
      showOverlay("NO SIGNAL", "Connection closed. Waiting to retry…");
      resetReadouts();
    };

    ws.onerror = () => {
      ws.close();
    };
  },

  _scheduleReconnect() {
    const delay =
      BACKOFF_STEPS[Math.min(this.attempt, BACKOFF_STEPS.length - 1)];
    this.attempt += 1;
    setConnState("connecting", `RETRY ${Math.round(delay / 1000)}s`);
    showOverlay("RECONNECTING", `Next attempt in ${delay / 1000}s`);
    this.reconnectTimer = setTimeout(() => {
      if (this.wantConnected) this._open();
    }, delay);
  },
};

/* ─── Connection UI ─────────────────────────────────────── */

function setConnState(state, labelOverride) {
  connDot.className = "dot";
  if (state === "online") {
    connDot.classList.add("online");
    connLabel.textContent = "ONLINE";
    btnConnect.textContent = "DISCONNECT";
    btnConnect.classList.add("disconnect");
  } else if (state === "connecting") {
    connDot.classList.add("connecting");
    connLabel.textContent = labelOverride || "CONNECTING";
    btnConnect.textContent = "CANCEL";
    btnConnect.classList.add("disconnect");
  } else {
    connLabel.textContent = "OFFLINE";
    btnConnect.textContent = "CONNECT";
    btnConnect.classList.remove("disconnect");
  }
}

btnConnect.addEventListener("click", () => {
  if (wsManager.wantConnected) {
    wsManager.disconnect();
  } else {
    wsManager.connect(wsUrlInput.value.trim());
  }
});

wsUrlInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !wsManager.wantConnected) {
    wsManager.connect(wsUrlInput.value.trim());
  }
});

/* ─── Stage overlay ─────────────────────────────────────── */

function showOverlay(title, hint) {
  overlayTitle.textContent = title;
  overlayHint.textContent = hint;
  overlay.classList.remove("hidden");
}

function hideOverlay() {
  overlay.classList.add("hidden");
}

/* ─── Frame handling ────────────────────────────────────── */

const frameImg = new Image();
let pendingFrame = null;
let drawScheduled = false;

frameImg.onload = () => {
  if (canvas.width !== frameImg.naturalWidth) {
    canvas.width = frameImg.naturalWidth;
    canvas.height = frameImg.naturalHeight;
  }
  ctx.drawImage(frameImg, 0, 0);
};

function handleFrame(data) {
  hideOverlay();

  // draw at most once per rAF to avoid backlog
  pendingFrame = data.frame;
  if (!drawScheduled) {
    drawScheduled = true;
    requestAnimationFrame(() => {
      drawScheduled = false;
      if (pendingFrame) {
        frameImg.src = "data:image/jpeg;base64," + pendingFrame;
        pendingFrame = null;
      }
    });
  }

  updateStats(data);
  updateEye(data);
  updateMeasurements(data.measurements);
}

/* ─── Top-bar stats ─────────────────────────────────────── */

function updateStats(d) {
  statFps.textContent = (d.fps ?? 0).toFixed(1);
  statHand.textContent = d.hand_present ? "YES" : "NO";
  statHand.style.color = d.hand_present
    ? "var(--success)"
    : "var(--on-surface-dim)";
  statTimer.textContent = (d.hand_timer ?? 0).toFixed(2) + "s";
}

/* ─── Eye status widget ─────────────────────────────────── */

function updateEye(d) {
  const timer = d.hand_timer ?? 0;
  const alpha = d.eye_alpha ?? 0;

  eyeWidget.classList.remove("awakening", "active");

  if (!d.hand_present) {
    eyeArc.style.strokeDashoffset = ARC_CIRCUMFERENCE;
    eyePhase.textContent = "DORMANT";
    eyeSub.textContent = "Show a palm to the lens";
    return;
  }

  if (alpha >= 0.99) {
    eyeWidget.classList.add("active");
    eyeArc.style.strokeDashoffset = 0;
    eyePhase.textContent = "ACTIVE";
    eyeSub.textContent = "The eye is watching";
  } else if (timer > 0) {
    eyeWidget.classList.add("awakening");
    const progress = Math.min(1, timer / EYE_APPEAR_SECONDS);
    eyeArc.style.strokeDashoffset = ARC_CIRCUMFERENCE * (1 - progress);
    eyePhase.textContent = alpha > 0 ? "OPENING" : "AWAKENING";
    eyeSub.textContent = `Hold steady · ${Math.max(
      0,
      EYE_APPEAR_SECONDS - timer
    ).toFixed(1)}s`;
  }
}

/* ─── Measurements ──────────────────────────────────────── */

function updateMeasurements(m) {
  if (!m) return;

  const area = m.palm_area_cm2 ?? 0;
  areaValue.textContent = area > 0 ? area.toFixed(1) : "--.-";

  const angles = m.joint_angles || {};
  for (const name of FINGERS) {
    const el = jointEls[name];
    const deg = angles[name] ?? 0;

    if (deg <= 0) {
      el.fill.style.width = "0%";
      el.deg.textContent = "--°";
      continue;
    }

    // map 0–180° onto the bar
    el.fill.style.width = Math.min(100, (deg / 180) * 100) + "%";
    el.deg.textContent = deg + "°";

    el.fill.classList.remove("amber", "red");
    if (deg < 90) el.fill.classList.add("red");
    else if (deg < 150) el.fill.classList.add("amber");
    // ≥150° stays green (default)
  }
}

function resetReadouts() {
  statFps.textContent = "--.-";
  statHand.textContent = "NO";
  statHand.style.color = "var(--on-surface-dim)";
  statTimer.textContent = "0.00s";
  areaValue.textContent = "--.-";
  eyeWidget.classList.remove("awakening", "active");
  eyeArc.style.strokeDashoffset = ARC_CIRCUMFERENCE;
  eyePhase.textContent = "DORMANT";
  eyeSub.textContent = "Show a palm to the lens";
  for (const name of FINGERS) {
    jointEls[name].fill.style.width = "0%";
    jointEls[name].deg.textContent = "--°";
  }
}

/* ─── Footer clock ──────────────────────────────────────── */

setInterval(() => {
  footerClock.textContent = new Date().toLocaleTimeString("en-GB");
}, 1000);

/* ─── Boot: default URL from current host ───────────────── */

(function initUrl() {
  if (location.protocol.startsWith("http")) {
    const proto = location.protocol === "https:" ? "wss" : "ws";
    wsUrlInput.value = `${proto}://${location.host}/ws`;
  }
})();

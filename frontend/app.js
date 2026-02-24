/**
 * Vocalize — Frontend Application
 *
 * Captures webcam frames, streams them via WebSocket to the backend
 * inference endpoint, and displays real-time predictions + transcript log.
 */

// ──────────────────────────────────────────────
// Configuration
// ──────────────────────────────────────────────
const WS_PROTOCOL = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
const WS_BASE = `${WS_PROTOCOL}//${window.location.host}`;
const TARGET_FPS = 15;
const FRAME_QUALITY = 0.5;   // JPEG quality (lower = smaller payload)
const MAX_CAPTURE_W = 640;   // max downscale width, height derived from aspect ratio

// ──────────────────────────────────────────────
// State
// ──────────────────────────────────────────────
let ws = null;
let mediaStream = null;
let captureTimer = null;
let sessionId = null;
let frameCount = 0;
let fpsTimestamp = 0;
let transcriptLog = [];

// ──────────────────────────────────────────────
// DOM references
// ──────────────────────────────────────────────
const video = document.getElementById('webcam');
const canvas = document.getElementById('captureCanvas');
const ctx = canvas.getContext('2d');
const predLabel = document.getElementById('predictionLabel');
const predMeta = document.getElementById('predMeta');
const confLabel = document.getElementById('confLabel');
const confFill = document.getElementById('confFill');
const bufferText = document.getElementById('bufferText');
const bufferDots = document.getElementById('bufferDots');
const statusBadge = document.getElementById('statusBadge');
const statusText = document.getElementById('statusText');
const sessionIdEl = document.getElementById('sessionId');
const detectionCount = document.getElementById('detectionCount');
const uniqueSigns = document.getElementById('uniqueSigns');
const logCount = document.getElementById('logCount');
const transcriptList = document.getElementById('transcriptList');
const fpsCounter = document.getElementById('fpsCounter');
const btnStart = document.getElementById('btnStart');
const btnStop = document.getElementById('btnStop');

// Canvas dimensions set dynamically once webcam starts (preserves aspect ratio)

// Generate buffer dots
for (let i = 0; i < 30; i++) {
    const dot = document.createElement('span');
    bufferDots.appendChild(dot);
}
const dots = bufferDots.querySelectorAll('span');

// ──────────────────────────────────────────────
// Helpers
// ──────────────────────────────────────────────

function generateSessionId() {
    return 'sess_' + Date.now().toString(36) + '_' + Math.random().toString(36).slice(2, 8);
}

function formatTime(date) {
    return date.toLocaleTimeString('en-US', { hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit' });
}

function setStatus(state, text) {
    statusBadge.className = `status-badge ${state}`;
    statusText.textContent = text;
}

// ──────────────────────────────────────────────
// Webcam
// ──────────────────────────────────────────────

async function startWebcam() {
    try {
        mediaStream = await navigator.mediaDevices.getUserMedia({
            video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: 'user' },
            audio: false,
        });
        video.srcObject = mediaStream;

        // Wait for video dimensions to be available before sizing canvas
        await new Promise(resolve => {
            video.onloadedmetadata = () => resolve();
        });

        // Size canvas to match webcam aspect ratio (max MAX_CAPTURE_W wide)
        const vw = video.videoWidth;
        const vh = video.videoHeight;
        const scale = Math.min(MAX_CAPTURE_W / vw, 1);
        canvas.width = Math.round(vw * scale);
        canvas.height = Math.round(vh * scale);
        console.log(`[CAM] Webcam: ${vw}x${vh} → Canvas: ${canvas.width}x${canvas.height}`);

        return true;
    } catch (err) {
        console.error('Webcam error:', err);
        setStatus('disconnected', 'Camera denied');
        return false;
    }
}

function stopWebcam() {
    if (mediaStream) {
        mediaStream.getTracks().forEach(t => t.stop());
        mediaStream = null;
        video.srcObject = null;
    }
}

// ──────────────────────────────────────────────
// WebSocket
// ──────────────────────────────────────────────

function connectWebSocket() {
    sessionId = generateSessionId();
    sessionIdEl.textContent = sessionId;

    const url = `${WS_BASE}/ws/inference/${sessionId}`;
    console.log('[WS] Connecting to', url);
    setStatus('disconnected', 'Connecting...');

    ws = new WebSocket(url);

    ws.onopen = () => {
        console.log('[WS] Connected');
        setStatus('connected', 'Live');
        startCapture();
    };

    ws.onmessage = (event) => {
        const msg = JSON.parse(event.data);
        handleServerMessage(msg);
    };

    ws.onerror = (err) => {
        console.error('[WS] Error:', err);
        setStatus('disconnected', 'Error');
    };

    ws.onclose = () => {
        console.log('[WS] Disconnected');
        setStatus('disconnected', 'Disconnected');
        stopCapture();
    };
}

// ──────────────────────────────────────────────
// Frame capture & send
// ──────────────────────────────────────────────

function captureAndSend() {
    if (!ws || ws.readyState !== WebSocket.OPEN) return;
    if (!video.videoWidth || !canvas.width) return;

    // Draw aspect-ratio-preserving downscaled frame to hidden canvas
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    // Convert to JPEG Base64
    const dataUrl = canvas.toDataURL('image/jpeg', FRAME_QUALITY);

    // Send to backend
    ws.send(JSON.stringify({
        type: 'frame',
        data: dataUrl,
    }));

    // FPS tracking
    frameCount++;
    const now = performance.now();
    if (now - fpsTimestamp >= 1000) {
        fpsCounter.textContent = `${frameCount} FPS`;
        frameCount = 0;
        fpsTimestamp = now;
    }
}

function startCapture() {
    if (captureTimer) return;
    frameCount = 0;
    fpsTimestamp = performance.now();
    captureTimer = setInterval(captureAndSend, 1000 / TARGET_FPS);
}

function stopCapture() {
    if (captureTimer) {
        clearInterval(captureTimer);
        captureTimer = null;
    }
    fpsCounter.textContent = '— FPS';
}

// ──────────────────────────────────────────────
// Handle server messages
// ──────────────────────────────────────────────

function handleServerMessage(msg) {
    if (msg.type === 'prediction') {
        updatePrediction(msg.label, msg.confidence, msg.buffer);
    } else if (msg.type === 'status') {
        updateBuffer(msg.buffer || 0);
        predMeta.textContent = msg.message;
    }
}

function updatePrediction(label, confidence, buffer) {
    // Update prediction display
    predLabel.textContent = label.toUpperCase();

    // Confidence bar
    const pct = Math.round(confidence * 100);
    confLabel.textContent = `Confidence: ${pct}%`;
    confFill.style.width = `${pct}%`;

    if (confidence >= 0.60) {
        confFill.style.background = 'linear-gradient(90deg, #22c55e, #4ade80)';
        predMeta.textContent = `Detected with ${pct}% confidence`;

        // Add to transcript log
        addTranscriptEntry(label, confidence);
    } else {
        confFill.style.background = 'linear-gradient(90deg, #f59e0b, #fbbf24)';
        predMeta.textContent = `Low confidence (${pct}%) — keep signing`;
    }

    // Update buffer dots
    updateBuffer(buffer);
}

function updateBuffer(count) {
    bufferText.textContent = `${count}/30`;
    dots.forEach((d, i) => {
        d.classList.toggle('filled', i < count);
    });
}

// ──────────────────────────────────────────────
// Transcript log (with client-side dedup)
// ──────────────────────────────────────────────

let lastLoggedLabel = '';
let lastLoggedTime = 0;
const LOG_COOLDOWN = 1500; // ms

function addTranscriptEntry(label, confidence) {
    const now = Date.now();
    if (label === lastLoggedLabel && (now - lastLoggedTime) < LOG_COOLDOWN) return;

    lastLoggedLabel = label;
    lastLoggedTime = now;

    const entry = { label, confidence, time: new Date() };
    transcriptLog.push(entry);

    // Update stats
    detectionCount.textContent = transcriptLog.length;
    const unique = new Set(transcriptLog.map(e => e.label));
    uniqueSigns.textContent = unique.size;
    logCount.textContent = transcriptLog.length;

    // Clear empty state
    const empty = transcriptList.querySelector('.empty-state');
    if (empty) empty.remove();

    // Create entry element
    const item = document.createElement('div');
    item.className = 'transcript-item';
    item.innerHTML = `
        <div class="transcript-index">${transcriptLog.length}</div>
        <div class="transcript-body">
            <div class="transcript-word">${label}</div>
            <div class="transcript-time">${formatTime(entry.time)}</div>
        </div>
        <div class="transcript-conf">${Math.round(confidence * 100)}%</div>
    `;

    // Prepend (newest first)
    transcriptList.insertBefore(item, transcriptList.firstChild);

    // Keep max 100 entries in DOM
    while (transcriptList.children.length > 100) {
        transcriptList.removeChild(transcriptList.lastChild);
    }
}

// ──────────────────────────────────────────────
// Start / Stop controls
// ──────────────────────────────────────────────

async function startStreaming() {
    btnStart.disabled = true;

    const camOK = await startWebcam();
    if (!camOK) {
        btnStart.disabled = false;
        return;
    }

    connectWebSocket();

    btnStop.disabled = false;
}

function stopStreaming() {
    btnStop.disabled = true;

    stopCapture();

    if (ws) {
        ws.close();
        ws = null;
    }

    stopWebcam();
    setStatus('disconnected', 'Stopped');

    predLabel.textContent = 'Stopped';
    predMeta.textContent = 'Press Start to resume';
    confLabel.textContent = 'Confidence: —';
    confFill.style.width = '0%';

    btnStart.disabled = false;
}

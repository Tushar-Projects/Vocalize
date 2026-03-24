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
const fpsCounter     = document.getElementById('fpsCounter');
const btnStart       = document.getElementById('btnStart');
const btnStop        = document.getElementById('btnStop');

// NLP card references
const nlpTranslated  = document.getElementById('nlpTranslated');
const nlpEnglish     = document.getElementById('nlpEnglish');
const nlpGloss       = document.getElementById('nlpGloss');
const nlpPills       = document.getElementById('nlpPills');
const nlpStatus      = document.getElementById('nlpStatus');
const langSelect     = document.getElementById('langSelect');

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
        // Send current language selection to backend
        ws.send(JSON.stringify({
            type: 'config',
            target_language: langSelect.value,
        }));
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
    } else if (msg.type === 'nlp') {
        renderNlpResult(msg);
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

        // Fetch the updated transcript from the DB
        fetchTranscriptHistory();
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
// DB-backed Transcript History
// ──────────────────────────────────────────────

const API_BASE = `${window.location.protocol}//${window.location.host}`;
const transcriptContainer = document.getElementById('transcriptContainer');
const transcriptEmpty = document.getElementById('transcriptEmpty');

let lastFetchedCount = 0;   // avoid redundant re-renders
let isFetching = false;

/**
 * Fetch the full transcript for the current session from the
 * backend REST API and render it into the sidebar.
 *
 * Called from handleServerMessage whenever a high-confidence
 * prediction arrives — no polling needed.
 */
async function fetchTranscriptHistory() {
    if (!sessionId || isFetching) return;
    isFetching = true;

    try {
        const res = await fetch(`${API_BASE}/transcripts/${sessionId}`);

        if (res.status === 404) {
            // No transcripts saved yet — nothing to render
            isFetching = false;
            return;
        }

        if (!res.ok) {
            console.warn('[FETCH] Transcript fetch failed:', res.status);
            isFetching = false;
            return;
        }

        const data = await res.json(); // array of TranscriptRead objects

        // Skip re-render if count hasn't changed
        if (data.length === lastFetchedCount) {
            isFetching = false;
            return;
        }
        lastFetchedCount = data.length;

        renderTranscriptList(data);
    } catch (err) {
        console.error('[FETCH] Transcript fetch error:', err);
    }

    isFetching = false;
}

/**
 * Render the full transcript array into the sidebar container.
 * Items are ordered by sequence_index ascending (chronological).
 */
function renderTranscriptList(items) {
    // Hide empty state
    if (transcriptEmpty) transcriptEmpty.style.display = 'none';

    // Clear existing entries
    transcriptContainer.querySelectorAll('.transcript-item').forEach(el => el.remove());

    // Sort by sequence_index ascending (server should already do this)
    items.sort((a, b) => a.sequence_index - b.sequence_index);

    // Update sidebar stats
    detectionCount.textContent = items.length;
    const uniqueLabels = new Set(items.map(i => i.sign_label));
    uniqueSigns.textContent = uniqueLabels.size;
    logCount.textContent = items.length;

    // Render each entry
    items.forEach((item) => {
        const el = document.createElement('div');
        el.className = 'transcript-item';

        const time = new Date(item.created_at);
        const pct = Math.round(item.confidence * 100);

        el.innerHTML = `
            <div class="transcript-index">${item.sequence_index}</div>
            <div class="transcript-body">
                <div class="transcript-word">${item.sign_label}</div>
                <div class="transcript-time">${formatTime(time)}</div>
            </div>
            <div class="transcript-conf">${pct}%</div>
        `;

        transcriptContainer.appendChild(el);
    });

    // Auto-scroll to the newest entry (bottom)
    transcriptContainer.scrollTop = transcriptContainer.scrollHeight;
}

// ──────────────────────────────────────────────
// NLP Result Rendering
// ──────────────────────────────────────────────

function renderNlpResult(msg) {
    const english     = msg.corrected_english || '';
    const translated  = msg.translated_text || english;
    const suggestions = msg.auto_complete || [];
    const rawGloss    = msg.raw_gloss || [];
    const lang        = msg.target_language || 'English';

    // Update status badge
    nlpStatus.textContent = 'updated';
    setTimeout(() => { nlpStatus.textContent = 'idle'; }, 2000);

    // Translated text (prominent)
    if (translated) {
        nlpTranslated.textContent = translated;
        nlpTranslated.classList.remove('empty');
    } else {
        nlpTranslated.textContent = 'Waiting for signs...';
        nlpTranslated.classList.add('empty');
    }

    // English subtitle (shown when target != English)
    if (lang !== 'English' && english) {
        nlpEnglish.textContent = `EN: ${english}`;
    } else {
        nlpEnglish.textContent = '';
    }

    // Raw gloss display
    if (rawGloss.length > 0) {
        nlpGloss.innerHTML = `Raw gloss: <span>${rawGloss.join(' \u2192 ')}</span>`;
    } else {
        nlpGloss.innerHTML = '';
    }

    // Auto-complete pills
    nlpPills.innerHTML = '';
    if (suggestions.length === 0) {
        nlpPills.innerHTML = '<span style="font-size:0.75rem; color:var(--text-dim);">\u2014</span>';
    } else {
        suggestions.forEach(word => {
            const pill = document.createElement('button');
            pill.className = 'nlp-pill';
            pill.textContent = word;
            pill.title = `Predicted next word: ${word}`;
            pill.addEventListener('click', () => {
                pill.style.background = 'rgba(0, 212, 170, 0.2)';
                pill.style.borderColor = 'var(--accent-alt)';
                setTimeout(() => {
                    pill.style.background = '';
                    pill.style.borderColor = '';
                }, 400);
            });
            nlpPills.appendChild(pill);
        });
    }

    console.log(`[NLP] ${lang}:`, translated, '| EN:', english, '| suggestions:', suggestions);
}

function resetNlpCard() {
    nlpTranslated.textContent = 'Waiting for signs...';
    nlpTranslated.classList.add('empty');
    nlpEnglish.textContent = '';
    nlpGloss.innerHTML = '';
    nlpPills.innerHTML = '<span style="font-size:0.75rem; color:var(--text-dim);">\u2014</span>';
    nlpStatus.textContent = 'idle';
}

// Send language change to backend immediately
langSelect.addEventListener('change', () => {
    if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({
            type: 'config',
            target_language: langSelect.value,
        }));
        console.log('[CONFIG] Language changed to:', langSelect.value);
    }
});

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

    // Reset transcript state for the new session
    lastFetchedCount = 0;
    if (transcriptEmpty) transcriptEmpty.style.display = '';
    transcriptContainer.querySelectorAll('.transcript-item').forEach(el => el.remove());
    detectionCount.textContent = '0';
    uniqueSigns.textContent    = '0';
    logCount.textContent       = '0';

    // Reset NLP card
    resetNlpCard();

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


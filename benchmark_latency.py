#!/usr/bin/env python3
"""End-to-end latency benchmark for the Vocalize project.

Run from the Vocalize project root (or place this file in that root):
    python benchmark_latency.py --source 0 --duration 45
    python benchmark_latency.py --source path/to/sign_video.mp4 --duration 45

The script starts a local Uvicorn instance, connects to the real inference
WebSocket, streams frames using the same 640px / JPEG-quality-50 / Base64
protocol as frontend/app.js, and instruments the project's real backend
functions. It reports cold-start, per-component, WebSocket, NLP and peer-relay
latencies. Temporary transcript rows created by this benchmark are deleted.

Requirements beyond requirements.txt:
    pip install websockets

The model files must be at:
    backend/action.h5
    backend/action_labels.json

A running PostgreSQL database is required. GROQ_API_KEY is required to measure
actual Groq NLP latency. The script does not measure browser Web Speech API
latency because that recognizer runs inside the browser, outside this Python
process.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import csv
import json
import math
import os
import platform
import statistics
import sys
import threading
import time
import uuid
from datetime import datetime
from collections import defaultdict
from functools import wraps
from pathlib import Path
from typing import Any, Callable
from urllib.request import urlopen

_IMPORT_STARTED = time.perf_counter()
PROJECT_ROOT = Path(__file__).resolve().parent
os.chdir(PROJECT_ROOT)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import cv2
    import numpy as np
    import uvicorn
    import websockets
except ImportError as exc:
    raise SystemExit(
        f"Missing dependency: {exc}. Install project requirements and run: "
        "pip install websockets"
    ) from exc

from backend.main import app
from backend.database import SessionLocal
from backend.models import Transcript
import backend.routers.inference as inference

_IMPORT_MS = (time.perf_counter() - _IMPORT_STARTED) * 1000.0


class Metrics:
    """Thread-safe collection of duration samples and counters."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.samples: dict[str, list[float]] = defaultdict(list)
        self.counters: dict[str, int] = defaultdict(int)
        self.notes: dict[str, str] = {}

    def add(self, name: str, milliseconds: float) -> None:
        with self._lock:
            self.samples[name].append(float(milliseconds))

    def increment(self, name: str, amount: int = 1) -> None:
        with self._lock:
            self.counters[name] += amount

    def note(self, name: str, value: str) -> None:
        with self._lock:
            self.notes[name] = value

    def get_samples(self, name: str) -> list[float]:
        with self._lock:
            return list(self.samples.get(name, []))

    def get_counters(self) -> dict[str, int]:
        with self._lock:
            return dict(self.counters)

    def get_notes(self) -> dict[str, str]:
        with self._lock:
            return dict(self.notes)


METRICS = Metrics()
_MODEL_METRIC_NAMES: dict[int, str] = {}
_PATCHED_MODEL_CLASSES: dict[type, Callable[..., Any]] = {}
_LAST_FRAME_RECEIVED_AT: dict[int, float] = {}


def _timed_sync(metric_name: str, function: Callable[..., Any]) -> Callable[..., Any]:
    @wraps(function)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        started = time.perf_counter()
        try:
            return function(*args, **kwargs)
        finally:
            METRICS.add(metric_name, (time.perf_counter() - started) * 1000.0)
    return wrapper


def _timed_async(metric_name: str, function: Callable[..., Any]) -> Callable[..., Any]:
    @wraps(function)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        started = time.perf_counter()
        try:
            return await function(*args, **kwargs)
        finally:
            METRICS.add(metric_name, (time.perf_counter() - started) * 1000.0)
    return wrapper


def _install_model_call_timer(model: Any, metric_name: str) -> None:
    """Time actual Keras forward calls without changing model predictions."""
    if model is None:
        return

    model_class = type(model)
    _MODEL_METRIC_NAMES[id(model)] = metric_name

    if model_class in _PATCHED_MODEL_CLASSES:
        return

    original_call = model_class.__call__
    _PATCHED_MODEL_CLASSES[model_class] = original_call

    def profiled_call(self: Any, *args: Any, **kwargs: Any) -> Any:
        metric = _MODEL_METRIC_NAMES.get(id(self))
        if metric is None:
            return original_call(self, *args, **kwargs)

        started = time.perf_counter()
        try:
            return original_call(self, *args, **kwargs)
        finally:
            METRICS.add(metric, (time.perf_counter() - started) * 1000.0)

    model_class.__call__ = profiled_call


def install_backend_instrumentation() -> None:
    """Wrap the exact backend functions called by /ws/inference/{session_id}."""
    # Frame path: Base64/JPEG decode -> MediaPipe -> 1,662 keypoint vector.
    for attr, metric in (
        ("decode_frame", "server_jpeg_decode_ms"),
        ("mediapipe_detect", "server_mediapipe_holistic_ms"),
        ("extract_keypoints", "server_keypoint_extraction_ms"),
        ("save_transcript", "server_transcript_db_commit_ms"),
        ("_load_gesture_cache", "server_custom_gesture_db_load_ms"),
        ("_ensure_gesture_cache", "server_custom_gesture_cache_lookup_ms"),
        ("nearest_custom_gesture", "server_custom_gesture_knn_ms"),
    ):
        setattr(inference, attr, _timed_sync(metric, getattr(inference, attr)))

    # Model loading and warm-up happen lazily when the WebSocket connects.
    original_get_model = inference.get_model

    @wraps(original_get_model)
    def timed_get_model() -> Any:
        is_cold = inference._model is None
        started = time.perf_counter()
        result = original_get_model()
        elapsed = (time.perf_counter() - started) * 1000.0
        if is_cold:
            METRICS.add("server_classifier_load_and_warmup_ms", elapsed)
        _install_model_call_timer(result[0], "server_classifier_forward_ms")
        return result

    inference.get_model = timed_get_model

    original_get_embed_model = inference.get_embed_model

    @wraps(original_get_embed_model)
    def timed_get_embed_model() -> Any:
        is_cold = inference._embed_model is None
        started = time.perf_counter()
        result = original_get_embed_model()
        elapsed = (time.perf_counter() - started) * 1000.0
        if is_cold:
            METRICS.add("server_embedding_model_build_and_warmup_ms", elapsed)
        _install_model_call_timer(result, "server_embedding_forward_ms")
        return result

    inference.get_embed_model = timed_get_embed_model

    original_get_holistic = inference.get_holistic

    @wraps(original_get_holistic)
    def timed_get_holistic() -> Any:
        is_cold = inference._holistic is None
        started = time.perf_counter()
        result = original_get_holistic()
        if is_cold:
            METRICS.add(
                "server_mediapipe_holistic_initialization_ms",
                (time.perf_counter() - started) * 1000.0,
            )
        return result

    inference.get_holistic = timed_get_holistic

    # NLP includes the Groq HTTP request, response parsing and result shaping.
    inference.correct_and_predict = _timed_async(
        "server_groq_nlp_round_trip_ms", inference.correct_and_predict
    )

    # Time ASGI WebSocket sends by message type.
    from starlette.websockets import WebSocket
    original_send_text = WebSocket.send_text
    original_receive_text = WebSocket.receive_text

    async def timed_receive_text(self: Any) -> str:
        data = await original_receive_text(self)
        try:
            parsed = json.loads(data)
            if parsed.get("type") == "frame":
                _LAST_FRAME_RECEIVED_AT[id(self)] = time.perf_counter()
        except Exception:
            pass
        return data

    async def timed_send_text(self: Any, data: str) -> Any:
        message_type = "other"
        parsed: dict[str, Any] = {}
        try:
            parsed = json.loads(data)
            message_type = str(parsed.get("type", "other"))
        except Exception:
            pass

        frame_received_at = _LAST_FRAME_RECEIVED_AT.get(id(self))
        if frame_received_at is not None:
            elapsed_ms = (time.perf_counter() - frame_received_at) * 1000.0
            if message_type in {"prediction", "status"} and "buffer" in parsed:
                METRICS.add("server_frame_receive_to_response_ms", elapsed_ms)
            elif message_type == "nlp":
                METRICS.add("server_frame_receive_to_nlp_send_ms", elapsed_ms)

        started = time.perf_counter()
        try:
            return await original_send_text(self, data)
        finally:
            METRICS.add(
                f"server_websocket_send_{message_type}_ms",
                (time.perf_counter() - started) * 1000.0,
            )

    WebSocket.receive_text = timed_receive_text
    WebSocket.send_text = timed_send_text


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    position = max(0, min(len(ordered) - 1, math.ceil((pct / 100.0) * len(ordered)) - 1))
    return ordered[position]


def print_metrics_table() -> None:
    labels = {
        "client_camera_read_ms": "Client camera/video frame read",
        "client_resize_ms": "Client resize to max 640 px width",
        "client_jpeg_encode_ms": "Client JPEG encode (quality 0.50)",
        "client_base64_json_encode_ms": "Client Base64 + JSON serialization",
        "client_websocket_send_ms": "Client WebSocket send call",
        "server_jpeg_decode_ms": "Server Base64/JPEG decode",
        "server_mediapipe_holistic_ms": "Server MediaPipe Holistic per frame",
        "server_keypoint_extraction_ms": "Server 1,662-feature extraction",
        "server_classifier_forward_ms": "Server Bi-LSTM + Attention forward pass",
        "server_embedding_forward_ms": "Server custom-gesture embedding forward pass",
        "server_custom_gesture_knn_ms": "Server custom-gesture nearest-prototype match",
        "server_transcript_db_commit_ms": "Server PostgreSQL transcript commit",
        "server_groq_nlp_round_trip_ms": "Server Groq NLP request + parse",
        "server_custom_gesture_db_load_ms": "Server custom-gesture DB load",
        "server_custom_gesture_cache_lookup_ms": "Server custom-gesture cache lookup",
        "server_websocket_send_prediction_ms": "Server WebSocket prediction send",
        "server_websocket_send_nlp_ms": "Server WebSocket NLP result send",
    }

    print("\nCOMPONENT LATENCY (milliseconds)")
    print("-" * 106)
    print(f"{'Component':48} {'n':>6} {'mean':>10} {'p50':>10} {'p95':>10} {'max':>10}")
    print("-" * 106)

    for key, label in labels.items():
        values = METRICS.get_samples(key)
        if not values:
            continue
        print(
            f"{label[:48]:48} {len(values):6d} "
            f"{statistics.fmean(values):10.2f} {percentile(values, 50):10.2f} "
            f"{percentile(values, 95):10.2f} {max(values):10.2f}"
        )

    startup_labels = {
        "python_import_ms": "Python + project dependency imports",
        "server_classifier_load_and_warmup_ms": "Classifier load + initial warm-up",
        "server_embedding_model_build_and_warmup_ms": "Embedding model build + warm-up",
        "server_mediapipe_holistic_initialization_ms": "MediaPipe Holistic initialization",
        "server_boot_and_database_startup_ms": "Uvicorn startup + DB table initialization",
        "websocket_connect_to_ready_ms": "WebSocket connect to ready status",
    }
    startup_items = [
        (label, METRICS.get_samples(key))
        for key, label in startup_labels.items()
        if METRICS.get_samples(key)
    ]
    if startup_items:
        print("\nCOLD-START / INITIALIZATION")
        print("-" * 68)
        for label, values in startup_items:
            print(f"{label:48} {values[-1]:10.2f} ms")

    counters = METRICS.get_counters()
    if counters:
        print("\nCOUNTS")
        print("-" * 68)
        for key, value in sorted(counters.items()):
            print(f"{key:48} {value}")

    notes = METRICS.get_notes()
    if notes:
        print("\nNOTES")
        print("-" * 68)
        for key, value in notes.items():
            print(f"{key}: {value}")


def start_local_server(host: str, port: int) -> tuple[uvicorn.Server, threading.Thread]:
    config = uvicorn.Config(
        app,
        host=host,
        port=port,
        log_level="warning",
        access_log=False,
        lifespan="on",
    )
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, name="vocalize-benchmark-server", daemon=True)

    started = time.perf_counter()
    thread.start()
    deadline = time.perf_counter() + 90.0

    while time.perf_counter() < deadline:
        if server.started:
            METRICS.add(
                "server_boot_and_database_startup_ms",
                (time.perf_counter() - started) * 1000.0,
            )
            return server, thread
        if not thread.is_alive():
            raise RuntimeError(
                "Uvicorn exited during startup. Check PostgreSQL, DATABASE_URL, "
                "and whether the selected port is available."
            )
        time.sleep(0.05)

    server.should_exit = True
    raise TimeoutError("Timed out waiting for the local Vocalize server to start.")


def run_database_probe(session_id: str) -> bool:
    """Exercise the same save_transcript function used by the inference route."""
    inference.save_transcript(session_id, "__latency_probe__", 1.0, 0)
    db = SessionLocal()
    try:
        found = (
            db.query(Transcript)
            .filter(
                Transcript.session_id == session_id,
                Transcript.sign_label == "__latency_probe__",
            )
            .first()
            is not None
        )
        return found
    finally:
        db.close()


def timed_http_get(metric_name: str, url: str) -> tuple[int, bytes]:
    """Measure a real HTTP GET to one of the application's REST endpoints."""
    started = time.perf_counter()
    try:
        with urlopen(url, timeout=15.0) as response:
            body = response.read()
            status = int(response.status)
        METRICS.add(metric_name, (time.perf_counter() - started) * 1000.0)
        METRICS.increment(f"{metric_name}_http_{status}")
        return status, body
    except Exception:
        METRICS.add(metric_name, (time.perf_counter() - started) * 1000.0)
        raise


def cleanup_probe_rows(session_id: str) -> None:
    db = SessionLocal()
    try:
        db.query(Transcript).filter(Transcript.session_id == session_id).delete(
            synchronize_session=False
        )
        db.commit()
    except Exception as exc:
        db.rollback()
        print(f"[WARN] Could not remove benchmark transcript rows: {exc}")
    finally:
        db.close()


def parse_source(source: str) -> int | str:
    return int(source) if source.isdigit() else source


async def stream_camera_frames(
    websocket: Any,
    capture: cv2.VideoCapture,
    args: argparse.Namespace,
    events: dict[str, Any],
) -> None:
    """Stream frames at the frontend's nominal 15 FPS and JPEG settings."""
    interval = 1.0 / args.fps
    total_seconds = args.duration + args.tail
    stream_started = time.perf_counter()
    next_send_due = stream_started
    last_frame: np.ndarray | None = None
    frame_count = 0

    while (time.perf_counter() - stream_started) < total_seconds:
        # Camera or video capture.
        started = time.perf_counter()
        ok, frame = capture.read()
        METRICS.add("client_camera_read_ms", (time.perf_counter() - started) * 1000.0)

        if ok and frame is not None:
            last_frame = frame
        elif last_frame is not None:
            # For a finite video, keep sending the final frame during the tail
            # so the backend's 2-second silence debounce can be evaluated.
            frame = last_frame
        else:
            raise RuntimeError("No frame could be read from the selected camera/video.")

        # Match frontend/app.js: preserve aspect ratio and cap width at 640 px.
        started = time.perf_counter()
        height, width = frame.shape[:2]
        scale = min(640.0 / float(width), 1.0)
        out_width = max(1, int(round(width * scale)))
        out_height = max(1, int(round(height * scale)))
        if out_width != width or out_height != height:
            resized = cv2.resize(frame, (out_width, out_height), interpolation=cv2.INTER_AREA)
        else:
            resized = frame
        METRICS.add("client_resize_ms", (time.perf_counter() - started) * 1000.0)

        # Match canvas.toDataURL('image/jpeg', 0.5) as closely as OpenCV allows.
        started = time.perf_counter()
        encoded_ok, encoded = cv2.imencode(
            ".jpg", resized, [int(cv2.IMWRITE_JPEG_QUALITY), 50]
        )
        METRICS.add("client_jpeg_encode_ms", (time.perf_counter() - started) * 1000.0)
        if not encoded_ok:
            raise RuntimeError("OpenCV failed to JPEG-encode a captured frame.")

        started = time.perf_counter()
        data_url = "data:image/jpeg;base64," + base64.b64encode(encoded).decode("ascii")
        payload = json.dumps({"type": "frame", "data": data_url}, separators=(",", ":"))
        METRICS.add("client_base64_json_encode_ms", (time.perf_counter() - started) * 1000.0)

        started = time.perf_counter()
        await websocket.send(payload)
        METRICS.add("client_websocket_send_ms", (time.perf_counter() - started) * 1000.0)
        frame_count += 1
        METRICS.increment("client_frames_sent")

        sent_at = time.perf_counter()
        if events.get("first_frame_sent_at") is None:
            events["first_frame_sent_at"] = sent_at

        next_send_due += interval
        delay = next_send_due - time.perf_counter()
        if delay > 0:
            await asyncio.sleep(delay)

    METRICS.note("streamed_frames", str(frame_count))


async def read_server_messages(websocket: Any, events: dict[str, Any]) -> None:
    """Read status, prediction and NLP messages without blocking frame sending."""
    try:
        async for raw in websocket:
            received_at = time.perf_counter()
            try:
                message = json.loads(raw)
            except json.JSONDecodeError:
                METRICS.increment("unparseable_server_messages")
                continue

            message_type = message.get("type")
            if message_type == "status":
                text = str(message.get("message", ""))
                if text.startswith("Connected.") and not events.get("ready_at"):
                    events["ready_at"] = received_at
                    if events.get("ws_connect_started_at") is not None:
                        METRICS.add(
                            "websocket_connect_to_ready_ms",
                            (received_at - events["ws_connect_started_at"]) * 1000.0,
                        )
                    events["ready_event"].set()
                METRICS.increment("server_status_messages")

            elif message_type == "prediction":
                METRICS.increment("server_predictions_received")
                label = str(message.get("label", ""))
                confidence = float(message.get("confidence", 0.0))
                if confidence >= inference.CONFIDENCE_THRESH and label:
                    if label != events.get("last_confident_label"):
                        events["last_confident_label"] = label
                        events["last_confident_new_sign_at"] = received_at
                        if events.get("first_confident_new_sign_at") is None:
                            events["first_confident_new_sign_at"] = received_at
                            events["first_confident_label"] = label
                if events.get("first_prediction_at") is None:
                    events["first_prediction_at"] = received_at
                    first_sent = events.get("first_frame_sent_at")
                    if first_sent is not None:
                        METRICS.add(
                            "client_first_frame_to_first_prediction_ms",
                            (received_at - first_sent) * 1000.0,
                        )

            elif message_type == "nlp":
                METRICS.increment("server_nlp_results_received")
                if events.get("first_nlp_at") is None:
                    events["first_nlp_at"] = received_at
                    first_sent = events.get("first_frame_sent_at")
                    first_sign = events.get("first_confident_new_sign_at")
                    if first_sent is not None:
                        METRICS.add(
                            "client_first_frame_to_first_nlp_ms",
                            (received_at - first_sent) * 1000.0,
                        )
                    if first_sign is not None:
                        METRICS.add(
                            "client_first_confident_sign_to_nlp_ms",
                            (received_at - first_sign) * 1000.0,
                        )
                    events["first_nlp_payload"] = message
                    events["nlp_event"].set()

    except Exception as exc:
        if not events.get("stopping"):
            events["reader_error"] = str(exc)


async def run_inference_websocket(args: argparse.Namespace, session_id: str, client_id: str) -> dict[str, Any]:
    host = args.host
    uri = f"ws://{host}:{args.port}/ws/inference/{session_id}?client_id={client_id}"

    events: dict[str, Any] = {
        "ws_connect_started_at": time.perf_counter(),
        "first_frame_sent_at": None,
        "first_prediction_at": None,
        "first_confident_new_sign_at": None,
        "last_confident_new_sign_at": None,
        "first_nlp_at": None,
        "last_confident_label": None,
        "first_confident_label": None,
        "ready_event": asyncio.Event(),
        "nlp_event": asyncio.Event(),
        "stopping": False,
    }

    async with websockets.connect(
        uri,
        max_size=None,
        ping_interval=None,
        close_timeout=5,
    ) as websocket:
        reader_task = asyncio.create_task(read_server_messages(websocket, events))
        try:
            await asyncio.wait_for(events["ready_event"].wait(), timeout=180.0)

            config_payload = json.dumps(
                {"type": "config", "target_language": args.target_language},
                separators=(",", ":"),
            )
            await websocket.send(config_payload)

            print(
                f"\nStreaming for {args.duration:.1f}s at {args.fps:.1f} FPS "
                f"(+ {args.tail:.1f}s tail). Perform signs from the model vocabulary."
            )
            print("Keep sending frames during the tail; the backend checks its 2-second pause timer on incoming frames.")

            source = parse_source(args.source)
            capture = cv2.VideoCapture(source)
            if not capture.isOpened():
                raise RuntimeError(f"Could not open video source: {args.source}")
            try:
                if isinstance(source, int):
                    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                await stream_camera_frames(websocket, capture, args, events)
            finally:
                capture.release()

            if os.getenv("GROQ_API_KEY"):
                try:
                    await asyncio.wait_for(events["nlp_event"].wait(), timeout=args.nlp_wait)
                    print("Received an NLP result from the live inference pipeline.")
                except asyncio.TimeoutError:
                    print(
                        "[WARN] No NLP result arrived before the timeout. Check that a sign "
                        "was classified with confidence >= 0.60 and that GROQ_API_KEY is valid."
                    )
            else:
                print("[WARN] GROQ_API_KEY is not set; an actual Groq latency cannot be measured.")

            await asyncio.sleep(0.25)
        finally:
            events["stopping"] = True
            reader_task.cancel()
            try:
                await reader_task
            except asyncio.CancelledError:
                pass

    return events


async def benchmark_peer_relay(host: str, port: int) -> None:
    """Measure the project's room WebSocket relay using two local peers."""
    room_id = f"bench-room-{uuid.uuid4().hex[:12]}"
    uri = f"ws://{host}:{port}/ws/signaling/{room_id}"
    try:
        async with websockets.connect(uri, max_size=None, ping_interval=None) as peer_a:
            await asyncio.wait_for(peer_a.recv(), timeout=5.0)  # room_info
            async with websockets.connect(uri, max_size=None, ping_interval=None) as peer_b:
                await asyncio.wait_for(peer_b.recv(), timeout=5.0)  # room_info
                await asyncio.sleep(0.05)
                for index in range(5):
                    message = json.dumps(
                        {
                            "type": "nlp_relay",
                            "from": "benchmark-peer-a",
                            "corrected_english": f"latency relay test {index}",
                            "translated_text": f"relay test {index}",
                            "raw_gloss": ["latency", "relay"],
                            "target_language": "English",
                        },
                        separators=(",", ":"),
                    )
                    started = time.perf_counter()
                    await peer_a.send(message)
                    received = await asyncio.wait_for(peer_b.recv(), timeout=5.0)
                    elapsed = (time.perf_counter() - started) * 1000.0
                    if received != message:
                        raise RuntimeError("Peer relay returned a different payload.")
                    METRICS.add("peer_room_websocket_relay_ms", elapsed)
    except Exception as exc:
        METRICS.note("peer_relay_benchmark", f"Not measured: {exc}")


def delete_benchmark_rows(session_id: str) -> None:
    try:
        cleanup_probe_rows(session_id)
    except Exception as exc:
        print(f"[WARN] Transcript cleanup failed: {exc}")



def export_metrics(args: argparse.Namespace, session_id: str) -> tuple[Path, Path]:
    """Write raw timing samples and a summary so results can be audited later."""
    output_dir = PROJECT_ROOT / "latency_results"
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    json_path = output_dir / f"vocalize_latency_{stamp}.json"
    csv_path = output_dir / f"vocalize_latency_{stamp}.csv"

    with METRICS._lock:
        raw_samples = {key: list(values) for key, values in METRICS.samples.items()}
        counters = dict(METRICS.counters)
        notes = dict(METRICS.notes)

    summary = {}
    for name, values in raw_samples.items():
        if values:
            summary[name] = {
                "count": len(values),
                "mean_ms": statistics.fmean(values),
                "p50_ms": percentile(values, 50),
                "p95_ms": percentile(values, 95),
                "max_ms": max(values),
                "min_ms": min(values),
            }

    try:
        gpu_devices = [device.name for device in inference.tf.config.list_physical_devices("GPU")]
    except Exception:
        gpu_devices = []
    report = {
        "project": "Vocalize",
        "timestamp_local": datetime.now().astimezone().isoformat(),
        "session_id": session_id,
        "python_version": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "versions": {
            "tensorflow": getattr(inference.tf, "__version__", "unknown"),
            "mediapipe": getattr(inference.mp, "__version__", "unknown"),
            "opencv": getattr(cv2, "__version__", "unknown"),
        },
        "tensorflow_detected_gpu_devices": gpu_devices,
        "project_root": str(PROJECT_ROOT),
        "model_path": str(PROJECT_ROOT / "backend" / "action.h5"),
        "labels_path": str(PROJECT_ROOT / "backend" / "action_labels.json"),
        "parameters": vars(args),
        "summary": summary,
        "raw_samples_ms": raw_samples,
        "counters": counters,
        "notes": notes,
    }
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "sample_index", "latency_ms"])
        for metric_name, values in sorted(raw_samples.items()):
            for index, value in enumerate(values, start=1):
                writer.writerow([metric_name, index, f"{value:.6f}"])

    return json_path, csv_path

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure Vocalize end-to-end and component-level latency."
    )
    parser.add_argument("--source", default="0", help="Webcam index (default 0) or video file path.")
    parser.add_argument("--duration", type=float, default=45.0, help="Active streaming duration in seconds (default 45).")
    parser.add_argument("--tail", type=float, default=4.0, help="Extra frame-streaming time for the 2-second NLP debounce (default 4).")
    parser.add_argument("--fps", type=float, default=15.0, help="Frame send rate; frontend target is 15 FPS (default 15).")
    parser.add_argument("--target-language", default="English", help="NLP target language.")
    parser.add_argument("--nlp-wait", type=float, default=30.0, help="Maximum wait for the first NLP result after streaming ends.")
    parser.add_argument("--host", default="127.0.0.1", help="Local benchmark server host.")
    parser.add_argument("--port", type=int, default=8765, help="Local benchmark server port.")
    parser.add_argument("--client-id", default=None, help="Existing Vocalize browser client ID; enables custom-gesture matching if gestures are registered.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.fps <= 0 or args.duration <= 0 or args.tail < 0:
        print("[ERROR] FPS and duration must be positive; tail cannot be negative.")
        return 2

    model_path = PROJECT_ROOT / "backend" / "action.h5"
    labels_path = PROJECT_ROOT / "backend" / "action_labels.json"
    missing = [str(path) for path in (model_path, labels_path) if not path.is_file()]
    if missing:
        print("[ERROR] Required model files are missing:")
        for path in missing:
            print(f"  - {path}")
        print("The inference router expects both files inside the backend/ directory.")
        return 2

    if not os.getenv("DATABASE_URL"):
        print("[INFO] DATABASE_URL not set; backend/config.py will use its default PostgreSQL URL.")
    if not os.getenv("GROQ_API_KEY"):
        print("[WARN] GROQ_API_KEY not detected. Recognition and component timings will run, but NLP latency will not be valid.")

    session_id = f"bench-{uuid.uuid4().hex[:24]}"
    client_id = args.client_id or f"bench-{uuid.uuid4().hex[:24]}"

    print("Vocalize latency benchmark")
    print(f"Project root : {PROJECT_ROOT}")
    print(f"Model        : {model_path}")
    print(f"Labels       : {labels_path}")
    print(f"Source       : {args.source}")
    print(f"Client ID    : {client_id}")
    print(f"Imports      : {_IMPORT_MS:.2f} ms")
    try:
        gpu_devices = [device.name for device in inference.tf.config.list_physical_devices("GPU")]
    except Exception:
        gpu_devices = []
    print(f"TensorFlow   : {getattr(inference.tf, '__version__', 'unknown')}")
    print(f"GPU devices  : {gpu_devices if gpu_devices else 'none detected'}")
    print("Starting instrumented local backend...")

    install_backend_instrumentation()
    server: uvicorn.Server | None = None
    server_thread: threading.Thread | None = None
    try:
        server, server_thread = start_local_server(args.host, args.port)
        print(f"Backend ready: http://{args.host}:{args.port}")

        db_ok = run_database_probe(session_id)
        METRICS.note("database_probe", "successful" if db_ok else "FAILED — check DATABASE_URL and PostgreSQL")
        if not db_ok:
            raise RuntimeError("PostgreSQL transcript write probe failed; latency results would be incomplete.")

        base_url = f"http://{args.host}:{args.port}"
        gesture_status, _ = timed_http_get(
            "client_get_gestures_http_ms", f"{base_url}/gestures/{client_id}"
        )
        METRICS.note("gesture_list_http_status", str(gesture_status))

        events = asyncio.run(run_inference_websocket(args, session_id, client_id))
        transcript_status, _ = timed_http_get(
            "client_get_transcript_http_ms", f"{base_url}/transcripts/{session_id}"
        )
        METRICS.note("transcript_http_status", str(transcript_status))
        asyncio.run(benchmark_peer_relay(args.host, args.port))

        counters = METRICS.get_counters()
        if counters.get("server_predictions_received", 0) == 0:
            print("[WARN] No predictions were returned. Check camera framing, sign vocabulary, model threshold and lighting.")
        if args.client_id is None:
            print("[INFO] No existing --client-id supplied; custom gesture matching will have no stored prototypes to compare.")

        first_sent = events.get("first_frame_sent_at")
        first_prediction = events.get("first_prediction_at")
        first_sign = events.get("first_confident_new_sign_at")
        first_nlp = events.get("first_nlp_at")
        if first_sent and first_prediction:
            print(f"\nFirst frame -> first prediction: {(first_prediction - first_sent) * 1000.0:.2f} ms")
        if first_sent and first_nlp:
            print(f"First frame -> first NLP result: {(first_nlp - first_sent) * 1000.0:.2f} ms")
        if first_sign and first_nlp:
            print(f"First confident new sign -> first NLP result: {(first_nlp - first_sign) * 1000.0:.2f} ms")
        if events.get("first_confident_label"):
            print(f"First confident label: {events['first_confident_label']}")
        if events.get("reader_error"):
            print(f"[WARN] WebSocket reader: {events['reader_error']}")

        # Record local script import time as a startup-related measurement.
        METRICS.add("python_import_ms", _IMPORT_MS)

    except KeyboardInterrupt:
        print("\n[INFO] Benchmark interrupted.")
    except Exception as exc:
        print(f"\n[ERROR] Benchmark failed: {exc}")
        return 1
    finally:
        delete_benchmark_rows(session_id)
        if server is not None:
            server.should_exit = True
        if server_thread is not None:
            server_thread.join(timeout=10.0)

    report_json, report_csv = export_metrics(args, session_id)
    print_metrics_table()
    print(f"\nSaved detailed JSON report: {report_json}")
    print(f"Saved raw CSV samples:      {report_csv}")
    print("\nInterpretation notes")
    print("- Cold-start values are initialization costs; do not mix them with warm per-frame inference values.")
    print("- MediaPipe and keypoint extraction are per processed server frame. Model inference is measured only after the 30-frame buffer fills.")
    print("- PostgreSQL commit time is measured through the same save_transcript() function used by the live route; a separate probe guarantees at least one DB sample.")
    print("- NLP latency includes the Groq request and response parsing. The live first-sign-to-NLP time also includes the configured 2-second debounce and streaming/network delays.")
    print("- The client uses OpenCV for webcam/video capture and JPEG encoding. Browser getUserMedia/canvas timing can differ slightly.")
    print("- The peer-relay metric is a loopback two-client WebSocket relay test, not an Internet/network-wide measurement.")
    print("- Web Speech API timing is excluded because it executes in the browser and is independent of the backend sign-inference route.")
    print("- Use p50 and p95 over multiple runs in the paper. Report the hardware, CPU/GPU mode, frame size, FPS, vocabulary, signer and whether warm-up was excluded.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

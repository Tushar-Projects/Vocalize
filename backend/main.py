"""
Vocalize — FastAPI Application Entry Point

Boots the app, creates database tables on startup via the lifespan
context manager, and mounts all routers.

Usage
─────
    uvicorn backend.main:app --reload --port 8000
"""

import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.config import settings
from backend.database import engine, Base

# Import models so Base.metadata knows about all tables
from backend import models  # noqa: F401

# Ensure project root is on sys.path so inference.py can import
# landmark_extractor and realtime_inference from the top level
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from backend.routers import transcripts, signaling, inference, gestures

# ──────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)-24s  %(levelname)-7s  %(message)s",
)
logger = logging.getLogger("vocalize")


# ──────────────────────────────────────────────
# Lifespan (startup / shutdown)
# ──────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Runs on startup:
      - Creates all tables in PostgreSQL via Base.metadata.create_all().
        Tables that already exist are left untouched (checkfirst=True
        is the default behaviour).

    Runs on shutdown:
      - Disposes the engine connection pool.
    """
    logger.info("Creating database tables (if they don't exist)...")
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables ready.")

    yield  # ← app is running

    engine.dispose()
    logger.info("Database engine disposed. Goodbye!")


# ──────────────────────────────────────────────
# FastAPI app
# ──────────────────────────────────────────────

app = FastAPI(
    title=settings.APP_TITLE,
    version=settings.APP_VERSION,
    description=(
        "Vocalize backend — real-time ASL/ISL sign-language translation. "
        "Provides transcript logging and WebRTC signaling."
    ),
    lifespan=lifespan,
)

# ── CORS ──
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # permissive for local dev
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Mount routers ──
app.include_router(transcripts.router)
app.include_router(signaling.router)
app.include_router(inference.router)
app.include_router(gestures.router)

# ── Serve frontend static files ──
_FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
if _FRONTEND_DIR.is_dir():
    app.mount("/app", StaticFiles(directory=str(_FRONTEND_DIR), html=True), name="frontend")


# ── Health check ──
@app.get("/", tags=["health"])
async def root():
    return {
        "app": settings.APP_TITLE,
        "version": settings.APP_VERSION,
        "status": "running",
    }


@app.get("/health", tags=["health"])
async def health():
    return {"status": "ok"}

"""
Vocalize — Transcript CRUD Router

POST /   → save a new detection
GET /{session_id}  → fetch ordered conversation history
GET /{session_id}/summary → quick session stats
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from sqlalchemy import func

from backend.database import get_db
from backend.models import Transcript
from backend.schemas import TranscriptCreate, TranscriptRead, SessionSummary

router = APIRouter(
    prefix="/transcripts",
    tags=["transcripts"],
)


# ──────────────────────────────────────────────
# POST — log a new detection
# ──────────────────────────────────────────────

@router.post(
    "/",
    response_model=TranscriptRead,
    status_code=status.HTTP_201_CREATED,
    summary="Log a new sign detection",
)
def create_transcript(
    payload: TranscriptCreate,
    db: Session = Depends(get_db),
):
    """
    Save a single sign-language detection to the database.

    Called by the inference pipeline whenever a prediction is made
    with sufficient confidence.
    """
    entry = Transcript(
        session_id=payload.session_id,
        sign_label=payload.sign_label,
        confidence=payload.confidence,
        sequence_index=payload.sequence_index,
        notes=payload.notes,
    )
    db.add(entry)
    db.commit()
    db.refresh(entry)
    return entry


# ──────────────────────────────────────────────
# GET — fetch session history (ordered)
# ──────────────────────────────────────────────

@router.get(
    "/{session_id}",
    response_model=list[TranscriptRead],
    summary="Get conversation transcript",
)
def get_session_transcripts(
    session_id: str,
    db: Session = Depends(get_db),
):
    """
    Retrieve all detections for a given session, ordered by
    `sequence_index` ascending for correct replay order.
    """
    results = (
        db.query(Transcript)
        .filter(Transcript.session_id == session_id)
        .order_by(Transcript.sequence_index.asc())
        .all()
    )
    if not results:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No transcripts found for session: {session_id}",
        )
    return results


# ──────────────────────────────────────────────
# GET — session summary
# ──────────────────────────────────────────────

@router.get(
    "/{session_id}/summary",
    response_model=SessionSummary,
    summary="Get session summary stats",
)
def get_session_summary(
    session_id: str,
    db: Session = Depends(get_db),
):
    """Quick overview of a session: total count, unique labels, time range."""
    rows = (
        db.query(Transcript)
        .filter(Transcript.session_id == session_id)
        .order_by(Transcript.sequence_index.asc())
        .all()
    )
    if not rows:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No transcripts found for session: {session_id}",
        )

    labels = list(dict.fromkeys(r.sign_label for r in rows))  # ordered unique
    return SessionSummary(
        session_id=session_id,
        total_detections=len(rows),
        labels=labels,
        first_detection=rows[0].created_at,
        last_detection=rows[-1].created_at,
    )

"""
Vocalize — Pydantic Schemas

Request / response models for the REST API.
"""

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


# ──────────────────────────────────────────────
# Transcript
# ──────────────────────────────────────────────

class TranscriptCreate(BaseModel):
    """Incoming payload when logging a new detection."""
    session_id: str = Field(..., max_length=64, description="Conversation session ID")
    sign_label: str = Field(..., max_length=100, description="Predicted sign label")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Model confidence")
    sequence_index: int = Field(0, ge=0, description="Order within session")
    notes: str | None = Field(None, description="Optional notes or raw output")


class TranscriptRead(BaseModel):
    """Outgoing representation of a stored transcript entry."""
    id: UUID
    session_id: str
    sign_label: str
    confidence: float
    sequence_index: int
    created_at: datetime
    notes: str | None = None

    model_config = ConfigDict(from_attributes=True)


class SessionSummary(BaseModel):
    """Quick summary of a session's transcript history."""
    session_id: str
    total_detections: int
    labels: list[str]
    first_detection: datetime | None = None
    last_detection: datetime | None = None


# ──────────────────────────────────────────────
# Custom Gesture (Phase 11)
# ──────────────────────────────────────────────

class CustomGestureRead(BaseModel):
    """Outgoing representation of a stored custom gesture (no raw embedding)."""
    id: UUID
    client_id: str
    label: str
    rep_count: int
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


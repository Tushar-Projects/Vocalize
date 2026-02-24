"""
Vocalize — ORM Models

All SQLAlchemy table definitions live here.  Import `Base` from
database.py so that `Base.metadata.create_all()` picks up every model.
"""

import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    Column,
    String,
    Float,
    Integer,
    DateTime,
    Index,
    Text,
)
from sqlalchemy.dialects.postgresql import UUID

from backend.database import Base


class Transcript(Base):
    """
    Stores individual sign-language detections from inference sessions.

    Each row represents one predicted sign within a session.
    The `session_id` column isolates conversations so that queries
    for a specific session are fast and secure.

    Indexes
    -------
    - ``ix_transcripts_session_id``  → fast session-scoped lookups
    - ``ix_transcripts_created_at``  → fast time-range queries
    - composite (session_id, sequence_index) → ordered replay
    """

    __tablename__ = "transcripts"

    # ── Primary key ──
    id = Column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        nullable=False,
        comment="Unique transcript entry ID",
    )

    # ── Session isolation ──
    session_id = Column(
        String(64),
        nullable=False,
        index=True,
        comment="Groups detections by conversation session",
    )

    # ── Detection data ──
    sign_label = Column(
        String(100),
        nullable=False,
        comment="Predicted ASL sign label",
    )

    confidence = Column(
        Float,
        nullable=False,
        comment="Model prediction confidence (0.0 – 1.0)",
    )

    sequence_index = Column(
        Integer,
        nullable=False,
        default=0,
        comment="Order of this detection within the session",
    )

    # ── Metadata ──
    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
        index=True,
        comment="Timestamp of detection",
    )

    # ── Optional extended data ──
    notes = Column(
        Text,
        nullable=True,
        comment="Optional notes or raw model output",
    )

    # ── Composite index for ordered session replay ──
    __table_args__ = (
        Index("ix_transcripts_session_seq", "session_id", "sequence_index"),
    )

    def __repr__(self) -> str:
        return (
            f"<Transcript(id={self.id!s:.8}, session={self.session_id}, "
            f"sign={self.sign_label}, conf={self.confidence:.2f})>"
        )

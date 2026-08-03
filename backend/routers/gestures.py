"""
Vocalize — Custom Gesture CRUD Router

GET    /gestures/{client_id}  → list custom gestures for a client (200 + [] if none)
DELETE /gestures/{id}         → remove one custom gesture and invalidate cache
"""

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from backend.database import get_db
from backend.models import CustomGesture
from backend.schemas import CustomGestureRead

router = APIRouter(
    prefix="/gestures",
    tags=["gestures"],
)


# ──────────────────────────────────────────────
# GET — list custom gestures for a client
# ──────────────────────────────────────────────

@router.get(
    "/{client_id}",
    response_model=list[CustomGestureRead],
    summary="List custom gestures for a client",
)
def list_gestures(
    client_id: str,
    db: Session = Depends(get_db),
):
    """
    Retrieve all custom gestures for a given client, ordered by
    `created_at` ascending.

    Returns 200 with an empty list when no gestures exist —
    deliberately diverges from the transcripts router's 404-on-empty
    convention, because this endpoint is fetched on every page load
    and a brand-new client with zero gestures is the default state,
    not an error.
    """
    results = (
        db.query(CustomGesture)
        .filter(CustomGesture.client_id == client_id)
        .order_by(CustomGesture.created_at.asc())
        .all()
    )
    return results


# ──────────────────────────────────────────────
# DELETE — remove a custom gesture
# ──────────────────────────────────────────────

@router.delete(
    "/{gesture_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a custom gesture",
)
def delete_gesture(
    gesture_id: UUID,
    db: Session = Depends(get_db),
):
    """
    Remove a single custom gesture by its UUID.

    Also invalidates the in-memory embedding cache for the gesture's
    client_id so KNN matching stops using the deleted prototype
    immediately, without requiring a server restart.

    Returns 404 if the gesture ID doesn't exist.
    """
    gesture = (
        db.query(CustomGesture)
        .filter(CustomGesture.id == gesture_id)
        .first()
    )
    if not gesture:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Custom gesture not found: {gesture_id}",
        )

    client_id = gesture.client_id

    db.delete(gesture)
    db.commit()

    # Invalidate the in-memory cache — atomic swap, never in-place mutation.
    # Import here to avoid circular import (gestures → inference → gestures).
    from backend.routers.inference import invalidate_gesture_cache
    invalidate_gesture_cache(client_id)

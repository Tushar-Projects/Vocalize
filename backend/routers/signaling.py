"""
Vocalize — WebRTC Signaling Router

WebSocket endpoint at /ws/signaling/{session_id} that relays SDP
offers/answers and ICE candidates between peers in the same room.
"""

import json
import logging
from typing import Dict, List

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

logger = logging.getLogger("vocalize.signaling")

router = APIRouter(tags=["signaling"])


# ──────────────────────────────────────────────
# Connection Manager
# ──────────────────────────────────────────────

class ConnectionManager:
    """
    Manages active WebSocket connections grouped by session_id.

    Each session_id maps to a list of connected WebSocket peers.
    Messages received from one peer are broadcast to all *other*
    peers in the same session room.
    """

    def __init__(self):
        self.active_rooms: Dict[str, List[WebSocket]] = {}

    async def connect(self, session_id: str, websocket: WebSocket):
        """Accept a WebSocket and add it to the session room."""
        await websocket.accept()

        if session_id not in self.active_rooms:
            self.active_rooms[session_id] = []
        self.active_rooms[session_id].append(websocket)

        peer_count = len(self.active_rooms[session_id])
        logger.info(f"[+] Peer joined room '{session_id}' (total: {peer_count})")

    def disconnect(self, session_id: str, websocket: WebSocket):
        """Remove a WebSocket from its session room."""
        if session_id in self.active_rooms:
            self.active_rooms[session_id].remove(websocket)
            remaining = len(self.active_rooms[session_id])
            logger.info(f"[-] Peer left room '{session_id}' (remaining: {remaining})")

            # Clean up empty rooms
            if not self.active_rooms[session_id]:
                del self.active_rooms[session_id]
                logger.info(f"[x] Room '{session_id}' closed (empty)")

    async def broadcast_to_others(
        self,
        session_id: str,
        sender: WebSocket,
        message: str,
    ):
        """
        Send a message to all peers in the room *except* the sender.

        This is the core signaling relay: when Peer A sends an SDP
        offer, it should reach Peer B (and vice versa), but not
        echo back to the sender.
        """
        if session_id not in self.active_rooms:
            return

        for peer in self.active_rooms[session_id]:
            if peer is not sender:
                try:
                    await peer.send_text(message)
                except Exception as e:
                    logger.warning(f"Failed to send to peer in '{session_id}': {e}")

    def get_room_count(self, session_id: str) -> int:
        """Get the number of peers in a room."""
        return len(self.active_rooms.get(session_id, []))

    def get_all_rooms(self) -> Dict[str, int]:
        """Get all active rooms and their peer counts."""
        return {sid: len(peers) for sid, peers in self.active_rooms.items()}


# Singleton manager instance
manager = ConnectionManager()


# ──────────────────────────────────────────────
# WebSocket endpoint
# ──────────────────────────────────────────────

@router.websocket("/ws/signaling/{session_id}")
async def signaling_endpoint(websocket: WebSocket, session_id: str):
    """
    WebRTC signaling relay via WebSocket.

    Protocol
    --------
    Each message should be a JSON string with a "type" field:

        {"type": "offer",     "sdp": "..."}
        {"type": "answer",    "sdp": "..."}
        {"type": "candidate", "candidate": {...}}
        {"type": "chat",      "text": "..."}

    The server does NOT interpret the payload — it simply relays
    the raw JSON text to all other peers in the same session room.
    """
    await manager.connect(session_id, websocket)

    # Notify the new peer how many others are in the room
    try:
        peer_count = manager.get_room_count(session_id)
        await websocket.send_text(json.dumps({
            "type": "room_info",
            "session_id": session_id,
            "peer_count": peer_count,
        }))
    except Exception:
        pass

    try:
        while True:
            # Receive raw text from this peer
            data = await websocket.receive_text()

            # Relay to all other peers in the same room
            await manager.broadcast_to_others(session_id, websocket, data)

    except WebSocketDisconnect:
        manager.disconnect(session_id, websocket)

        # Notify remaining peers that someone left
        leave_msg = json.dumps({
            "type": "peer_left",
            "session_id": session_id,
            "remaining": manager.get_room_count(session_id),
        })
        for peer in manager.active_rooms.get(session_id, []):
            try:
                await peer.send_text(leave_msg)
            except Exception:
                pass


# ──────────────────────────────────────────────
# REST helper (optional — check room status)
# ──────────────────────────────────────────────

@router.get(
    "/signaling/rooms",
    summary="List active signaling rooms",
    tags=["signaling"],
)
async def list_rooms():
    """Return all active session rooms and their peer counts."""
    return {"rooms": manager.get_all_rooms()}

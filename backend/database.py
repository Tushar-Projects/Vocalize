"""
Vocalize — SQLAlchemy Database Setup (Synchronous)

Provides the engine, session factory, declarative Base, and a
FastAPI-compatible dependency for injecting DB sessions into routes.
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base, Session
from typing import Generator

from backend.config import settings

# ──────────────────────────────────────────────
# Engine
# ──────────────────────────────────────────────
engine = create_engine(
    settings.DATABASE_URL,
    # Connection pool tuning for a moderate-traffic app
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True,   # verify connections before checkout
    echo=settings.DEBUG,  # log SQL statements when DEBUG=True
)

# ──────────────────────────────────────────────
# Session factory
# ──────────────────────────────────────────────
SessionLocal = sessionmaker(
    bind=engine,
    autocommit=False,
    autoflush=False,
)

# ──────────────────────────────────────────────
# Declarative Base
# ──────────────────────────────────────────────
Base = declarative_base()

# ──────────────────────────────────────────────
# FastAPI dependency
# ──────────────────────────────────────────────

def get_db() -> Generator[Session, None, None]:
    """
    Yield a SQLAlchemy session and ensure it is closed after the
    request completes, even if an exception occurs.

    Usage in routes:
        @router.get("/")
        def read_items(db: Session = Depends(get_db)):
            ...
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

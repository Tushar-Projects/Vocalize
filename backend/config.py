"""
Vocalize — Backend Configuration

Loads settings from .env file, falling back to defaults.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables / .env file.

    Priority: environment variable → .env file → default value.
    """

    # ── Database ──
    DATABASE_URL: str = "postgresql://postgres:postgres@localhost:5432/vocalize_db"

    # ── App ──
    APP_TITLE: str = "Vocalize API"
    APP_VERSION: str = "0.5.0"
    DEBUG: bool = True

    # ── CORS (comma-separated origins) ──
    CORS_ORIGINS: str = "http://localhost:3000,http://localhost:5173,http://127.0.0.1:8000"

    # ── Groq NLP ──
    GROQ_API_KEY: str = ""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )


# Singleton instance — import this everywhere
settings = Settings()

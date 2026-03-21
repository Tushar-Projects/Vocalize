"""
Vocalize — NLP Engine (Groq-powered Grammar Correction & Auto-Complete)

Takes raw sign language gloss predictions and converts them into
natural English using the Groq API (llama3-8b-8192).

Returns:
    corrected_sentence  — grammatically correct English string
    auto_complete       — list of 3 logical next-word predictions
"""

import json
import logging
import os

from dotenv import load_dotenv
from groq import AsyncGroq

# Ensure .env is loaded before any os.getenv calls
load_dotenv()

logger = logging.getLogger("vocalize.nlp")

# ──────────────────────────────────────────────
# Groq client (lazy singleton)
# ──────────────────────────────────────────────
_client: AsyncGroq | None = None


def _get_client() -> AsyncGroq:
    global _client
    if _client is None:
        api_key = os.getenv("GROQ_API_KEY", "")
        if not api_key:
            raise RuntimeError(
                "GROQ_API_KEY is not set. Add it to your .env file."
            )
        _client = AsyncGroq(api_key=api_key)
        logger.info("Groq AsyncClient initialized.")
    return _client


# ──────────────────────────────────────────────
# System prompt — strict data-formatter mode
# ──────────────────────────────────────────────
SYSTEM_PROMPT = """\
You are a sign-language gloss-to-English converter.

RULES — follow ALL of them strictly:
1. You receive a JSON array of raw sign-language glosses (individual words detected from ASL/ISL).
2. Reconstruct them into ONE grammatically correct, natural English sentence.
3. Fix grammar, add articles/prepositions/conjunctions as needed, correct tense, but preserve the speaker's intent.
4. Predict the 3 most likely NEXT words the speaker might sign, based on context.
5. Return ONLY a JSON object — no markdown, no explanation, no filler text.
6. The JSON object MUST have exactly two keys:
   - "corrected_sentence": string (the corrected English sentence)
   - "auto_complete": array of exactly 3 strings (next-word predictions)
7. If the input is too short or ambiguous, still produce your best guess.
8. Never add conversational filler like "Sure!" or "Here you go".
"""

MODEL = "llama-3.1-8b-instant"


# ──────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────

async def correct_and_predict(words: list[str]) -> dict:
    """
    Send raw gloss words to Groq and get back corrected sentence
    plus auto-complete suggestions.

    Args:
        words: List of raw sign labels, e.g. ["I", "going", "shop"]

    Returns:
        {
            "corrected_sentence": "I am going to the shop.",
            "auto_complete": ["tomorrow", "today", "now"]
        }
        On error, returns a fallback dict with the raw words joined.
    """
    if not words:
        return {
            "corrected_sentence": "",
            "auto_complete": [],
        }

    client = _get_client()
    user_payload = json.dumps(words)

    try:
        response = await client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_payload},
            ],
            response_format={"type": "json_object"},
            temperature=0.3,
            max_tokens=200,
        )

        raw_text = response.choices[0].message.content or ""
        result = json.loads(raw_text)

        # Validate expected keys
        sentence = result.get("corrected_sentence", " ".join(words))
        suggestions = result.get("auto_complete", [])

        # Enforce exactly 3 suggestions
        if not isinstance(suggestions, list):
            suggestions = []
        suggestions = [str(s) for s in suggestions[:3]]

        logger.info(
            f"[NLP] {words} → \"{sentence}\"  "
            f"next: {suggestions}"
        )

        return {
            "corrected_sentence": sentence,
            "auto_complete": suggestions,
        }

    except Exception as e:
        logger.error(f"[NLP] Groq API error: {e}")
        return {
            "corrected_sentence": " ".join(words),
            "auto_complete": [],
        }

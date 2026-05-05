"""
Vocalize — NLP Engine (Groq-powered Grammar Correction, Translation & Auto-Complete)

Takes raw sign language gloss predictions and converts them into
natural English, optionally translates to a target language, using
the Groq API (llama-3.1-8b-instant).

Returns:
    corrected_english  — grammatically correct English string
    translated_text    — sentence translated to the target language
    auto_complete      — list of 3 logical next-word predictions
"""

import json
import logging
import os
import re

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
# System prompt — single-pass correction + translation
# ──────────────────────────────────────────────
SYSTEM_PROMPT = """\
You are a sign-language gloss-to-English converter and translator.

RULES — follow ALL of them strictly:
1. You receive a JSON object with two keys:
   - "words": array of raw sign-language glosses (individual words detected from ASL/ISL).
   - "target_language": the language to translate into (e.g. "English", "Hindi", "Kannada", "Tamil", "Spanish").
2. First, reconstruct the glosses into ONE grammatically correct, natural English sentence.
3. Fix grammar, add articles/prepositions/conjunctions as needed, correct tense, but preserve the speaker's intent.
4. Then translate that English sentence into the target_language.
   - If target_language is "English", set translated_text to the same value as corrected_english.
5. Predict the 3 most likely NEXT words the speaker might sign, based on context. These should always be in English.
6. Return ONLY a JSON object — no markdown, no explanation, no filler text.
7. The JSON object MUST have exactly three keys:
   - "corrected_english": string (the grammatically correct English sentence)
   - "translated_text": string (the sentence in the target language)
   - "auto_complete": array of exactly 3 strings (next-word predictions in English)
8. If the input is too short or ambiguous, still produce your best guess.
9. Never add conversational filler like "Sure!" or "Here you go".
"""

MODEL = "llama-3.1-8b-instant"


# ──────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────

async def correct_and_predict(
    words: list[str],
    target_language: str = "English",
) -> dict:
    """
    Send raw gloss words to Groq and get back corrected sentence,
    translation, and auto-complete suggestions.

    Args:
        words:           List of raw sign labels, e.g. ["I", "going", "shop"]
        target_language: Target language for translation (default: "English")

    Returns:
        {
            "corrected_english": "I am going to the shop.",
            "translated_text": "मैं दुकान जा रहा हूँ।",
            "auto_complete": ["tomorrow", "today", "now"]
        }
        On error, returns a fallback dict with the raw words joined.
    """
    if not words:
        return {
            "corrected_english": "",
            "translated_text": "",
            "auto_complete": [],
        }

    client = _get_client()
    user_payload = json.dumps({
        "words": words,
        "target_language": target_language,
    })

    try:
        response = await client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_payload},
            ],
            response_format={"type": "json_object"},
            temperature=0.3,
            max_tokens=300,
        )

        raw_text = response.choices[0].message.content or ""

        # ── Strip markdown fences the LLM sometimes wraps around JSON ──
        raw_text = re.sub(r"^```(?:json)?\s*", "", raw_text.strip())
        raw_text = re.sub(r"\s*```$", "", raw_text.strip())

        # ── Parse JSON with safe fallback ──
        try:
            result = json.loads(raw_text)
        except json.JSONDecodeError as je:
            logger.error(f"[NLP] JSON parse failed: {je}  raw={raw_text!r}")
            print(f"[NLP] JSON parse failed: {je}\n  raw_text={raw_text!r}")
            return {
                "corrected_english": " ".join(words),
                "translated_text": " ".join(words),
                "auto_complete": [],
            }

        # Validate expected keys
        english = result.get("corrected_english", " ".join(words))
        translated = result.get("translated_text", english)
        suggestions = result.get("auto_complete", [])

        # Enforce exactly 3 suggestions
        if not isinstance(suggestions, list):
            suggestions = []
        suggestions = [str(s) for s in suggestions[:3]]

        logger.info(
            f"[NLP] {words} → EN:\"{english}\"  "
            f"{target_language}:\"{translated}\"  next: {suggestions}"
        )

        return {
            "corrected_english": english,
            "translated_text": translated,
            "auto_complete": suggestions,
        }

    except Exception as e:
        logger.error(f"[NLP] Groq API error: {e}")
        print(f"[NLP] Groq API error: {e}")
        fallback = " ".join(words)
        return {
            "corrected_english": fallback,
            "translated_text": fallback,
            "auto_complete": [],
        }


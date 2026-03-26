"""Language detection and translation helpers for one product string at a time."""

import logging
import re
import time
from collections.abc import Callable

import translators as ts
from lingua import Language, LanguageDetectorBuilder

logger = logging.getLogger(__name__)

TranslatorFn = Callable[[str, str], str]
SUPPORTED_LANGUAGES = (
    Language.ENGLISH,
    Language.SPANISH,
    Language.PORTUGUESE,
    Language.FRENCH,
    Language.GERMAN,
)
DETECTOR = LanguageDetectorBuilder.from_languages(*SUPPORTED_LANGUAGES).build()
TRANSLATION_MAX_RETRIES = 3
TRANSLATION_BACKOFF_SECONDS = 1.0


def _normalize_text(text: str | None) -> str:
    if text is None:
        return ""
    return re.sub(r"\s+", " ", str(text)).strip()


def detect_language(text: str) -> str:
    """Detect the ISO 639-1 language code for a single input string."""
    normalized = _normalize_text(text)
    if not normalized:
        return "unknown"

    language = DETECTOR.detect_language_of(normalized)
    if language is None or language.iso_code_639_1 is None:
        return "unknown"

    return language.iso_code_639_1.name.lower()


def _google_translate(text: str, from_lang: str) -> str:
    return ts.translate_text(
        text,
        translator="google",
        from_language=from_lang,
        to_language="en",
    )


def _translate_with_retry(
    text: str,
    from_lang: str,
    translator: TranslatorFn,
    max_retries: int = TRANSLATION_MAX_RETRIES,
    backoff_seconds: float = TRANSLATION_BACKOFF_SECONDS,
) -> str:
    last_error: Exception | None = None

    for attempt in range(1, max_retries + 1):
        try:
            return translator(text, from_lang)
        except Exception as exc:
            last_error = exc
            if attempt == max_retries:
                break

            sleep_seconds = backoff_seconds * attempt
            logger.warning(
                "Translation failed for language '%s' on attempt %s/%s; retrying in %.1fs",
                from_lang,
                attempt,
                max_retries,
                sleep_seconds,
            )
            time.sleep(sleep_seconds)

    logger.warning(
        "Translation failed for language '%s' after %s attempts; using original text. Error: %s",
        from_lang,
        max_retries,
        last_error,
    )
    return text


def translate_eng(
    text: str,
    from_lang: str | None = None,
    translator: TranslatorFn | None = None,
) -> str:
    """Translate one string to English via Google unless a backend override is passed."""
    normalized = _normalize_text(text)
    source_lang = from_lang or detect_language(normalized)

    if not normalized or source_lang in {"en", "unknown"}:
        return normalized

    backend = translator or _google_translate
    translated = _normalize_text(_translate_with_retry(normalized, source_lang, backend))
    return translated or normalized


def translate_to_english(
    text: str,
    translator: TranslatorFn | None = None,
) -> tuple[str, str]:
    """Compatibility wrapper returning translated text and detected language."""
    detected_lang = detect_language(text)
    translated = translate_eng(text, from_lang=detected_lang, translator=translator)
    return translated, detected_lang

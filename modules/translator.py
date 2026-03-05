"""Detect language and translate to English if needed."""

from langdetect import detect


def translate_to_english(text: str) -> tuple[str, str]:
    """Detect language and translate to English if not already English.

    Args:
        text: Input product description string.

    Returns:
        Tuple of (translated_text, detected_language_code).
        If already English, returns (text, "en") unchanged.
    """
    lang = detect(text)

    if lang == "en":
        return text, lang

    # TODO: call translation API here (e.g. deep-translator, DeepL, Claude)
    translated = text  # placeholder
    return translated, lang

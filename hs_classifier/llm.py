"""Shared OpenAI LLM configuration and Instructor client helpers."""

import os

import instructor
from openai import OpenAI

DEFAULT_OPENAI_MODEL = "gpt-5-nano"


def get_openai_model() -> str:
    """Return the configured OpenAI model, falling back to the supported default."""
    return os.environ.get("OPENAI_MODEL", DEFAULT_OPENAI_MODEL).strip() or DEFAULT_OPENAI_MODEL


def get_openai_client():
    """Create the shared Instructor client for OpenAI-backed structured outputs."""
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "Missing required environment variable 'OPENAI_API_KEY'. "
            "Set it in your environment or .env before using HS classification."
        )

    return instructor.from_openai(OpenAI(api_key=api_key))

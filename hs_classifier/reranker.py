"""Rerank a shortlist of HS code candidates using an LLM.

Uses Instructor with a Pydantic model for structured output.
Provider-agnostic: works with OpenAI, Anthropic, Gemini, etc.
"""

import instructor
import polars as pl
from pydantic import BaseModel, Field


class RerankResult(BaseModel):
    """Top N HS code picks with reasoning."""

    codes: list[str] = Field(
        description="HS codes ordered best to worst (e.g. ['0306', '1605'])"
    )
    reason: str = Field(
        description="A 50 word reason for your choices. If none match, "
        "write 'None of the provided codes'."
    )


def format_shortlist(shortlist: pl.DataFrame) -> list[str]:
    """Format HS code DataFrame rows as 'Code XXXX: Description' strings for the prompt."""
    return [f"Code {r['code']}: {r['description']}" for r in shortlist.iter_rows(named=True)]


def rerank_codes(
    shortlist: pl.DataFrame,
    query: str,
    context: str,
    model: str,
    temperature: float = 0.1,
    top_n: int = 2,
) -> dict:
    """Use an LLM to select the top N HS codes from the retrieved shortlist.

    Args:
        shortlist: DataFrame of ~25 candidate HS codes from retrieval.
        query: Product name or description (in English).
        context: Shipping/packaging context.
        model: Provider/model string, e.g. "google/gemini-2.5-flash-lite".
        top_n: How many top codes to return (default 2).

    Returns:
        Dict with codes (list), descriptions (list), and reason (str).
    """
    client = instructor.from_provider(model)
    formatted_codes = format_shortlist(shortlist)

    context_line = f"Shipping context: '{context}'\n" if context else ""

    prompt = (
        f"You are an expert trade classifier. Given a product description, pick the "
        f"top {top_n} most relevant HS codes from the shortlist below.\n\n"
        f"Product: '{query}'\n"
        f"{context_line}\n"
        f"Work through your reasoning step by step, then return your top {top_n} "
        f"HS code picks (best first) and a 50-word justification. If none match, "
        f"write 'None of the provided codes'.\n\n"
        f"Candidate codes:\n" + "\n".join(formatted_codes)
    )

    result = client.create(
        response_model=RerankResult,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
    )

    # look up descriptions from the shortlist
    codes = result.codes[:top_n]
    descriptions = []
    for code in codes:
        match = shortlist.filter(pl.col("code") == code)
        descriptions.append(match["description"][0] if len(match) else "(not found)")

    return {
        "codes": codes,
        "descriptions": descriptions,
        "reason": result.reason,
    }

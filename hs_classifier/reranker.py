"""Rerank a shortlist of HS code candidates using an LLM.

Uses Instructor with a Pydantic model for structured output.
Provider-agnostic: works with OpenAI, Anthropic, Gemini, etc.
"""

import instructor
import polars as pl
from pydantic import BaseModel, Field


class RerankResult(BaseModel):
    """Top 2 HS code picks with reasoning."""

    code_first: str = Field(description="The HS code (e.g. '0101') that is your first choice")
    code_second: str = Field(description="The HS code (e.g. '0101') that is your second choice")
    reason: str = Field(
        description="A 50 word reason for your choices. If none match, "
        "write 'None of the provided codes' for each label."
    )


def format_shortlist(shortlist: pl.DataFrame) -> list[str]:
    """Format HS code DataFrame rows as 'Code XXXX: Description' strings for the prompt."""
    return [f"Code {r['code']}: {r['description']}" for r in shortlist.iter_rows(named=True)]


def rerank_codes(
    shortlist: pl.DataFrame,
    query: str,
    context: str,
    model: str,
) -> dict:
    """Use an LLM to select the top 2 HS codes from the retrieved shortlist.

    Args:
        shortlist: DataFrame of ~25 candidate HS codes from retrieval.
        query: Product name or description (in English).
        context: Shipping/packaging context.
        model: Provider/model string, e.g. "google/gemini-2.5-flash-lite".

    Returns:
        Dict with code_first, desc_first, code_second, desc_second, reason.
    """
    client = instructor.from_provider(model)
    formatted_codes = format_shortlist(shortlist)

    context_line = f"Shipping context: '{context}'\n" if context else ""

    prompt = (
        f"You are an expert trade classifier. Given a product description, pick the "
        f"two most relevant HS codes from the shortlist below.\n\n"
        f"Product: '{query}'\n"
        f"{context_line}\n"
        f"Work through your reasoning step by step, then return your top 2 HS codes "
        f"picks and a 50-word justification. If none match, write 'None of the "
        f"provided codes' for each label.\n\n"
        f"Candidate codes:\n" + "\n".join(formatted_codes)
    )

    result = client.create(
        response_model=RerankResult,
        messages=[{"role": "user", "content": prompt}],
    )

    # look up descriptions from the shortlist
    first = shortlist.filter(pl.col("code") == result.code_first)
    second = shortlist.filter(pl.col("code") == result.code_second)

    return {
        "code_first": result.code_first,
        "desc_first": first["description"][0] if len(first) else "(not found)",
        "code_second": result.code_second,
        "desc_second": second["description"][0] if len(second) else "(not found)",
        "reason": result.reason,
    }

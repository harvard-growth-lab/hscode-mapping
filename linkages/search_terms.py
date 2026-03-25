"""Generate HS-vocabulary search terms for a product description.

Uses Instructor with a Pydantic model for structured output.
Provider-agnostic: works with OpenAI, Anthropic, Gemini, etc.
"""

from pathlib import Path

import instructor
import polars as pl
from pydantic import BaseModel, Field


class SearchTerms(BaseModel):
    """5-8 HS-vocabulary search terms for vector embedding search."""

    term1: str = Field(description="First term")
    term2: str = Field(description="Second term")
    term3: str = Field(description="Third term")
    term4: str = Field(description="Fourth term")
    term5: str = Field(description="Fifth term")
    term6: str | None = Field(default=None, description="Sixth term (optional)")
    term7: str | None = Field(default=None, description="Seventh term (optional)")
    term8: str | None = Field(default=None, description="Eighth term (optional)")

    def to_list(self) -> list[str]:
        """Return non-None terms as a list."""
        return [v for v in self.model_dump().values() if v is not None]


def load_hs_descriptions(index_path: Path) -> list[str]:
    """Load HS code descriptions from the saved parquet index."""
    return pl.read_parquet(index_path, columns=["description"])["description"].to_list()


def generate_search_terms(
    query: str,
    context: str,
    hs_descriptions: list[str],
    model: str,
) -> list[str]:
    """Generate 5-8 HS-vocabulary search terms for a product description.

    Args:
        query: Product name or description (in English).
        context: Optional context about the product (container info, etc.).
        hs_descriptions: Full list of HS code description strings to reference.
        model: Provider/model string, e.g. "google/gemini-2.5-flash".

    Returns:
        List of search terms.
    """
    client = instructor.from_provider(model)

    prompt = (
        f"TASK: I have a description of a very specific product, namely: '{query}', "
        f"that I would like you to generate 5-8 product description variations, or "
        f"closely related search terms that would maximize our chance of finding this "
        f"product in official product lists.\n\n"
        f"This is some coverage of the product, which should help you decide what it is.\n\n"
        f"'{context}'\n\n"
        f"Return only words that are generic product classes that specifically tell me "
        f"what the product is. DO NOT include company names, adjectives, process terms, "
        f"or non-product phrases. Return 5 to 8 unique (non repeated) terms from the "
        f"list you feel are the most relevant. If there are multiple, just share the one "
        f"common term from the group.\n\n"
        f"Product list to reference: {hs_descriptions}"
    )

    terms = client.create(
        response_model=SearchTerms,
        messages=[{"role": "user", "content": prompt}],
    )

    return terms.to_list()

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


def load_hs_chapters(chapters_path: Path) -> list[str]:
    """Load HS2 chapter descriptions from the chapters parquet."""
    return pl.read_parquet(chapters_path, columns=["description"])["description"].to_list()


def generate_search_terms(
    query: str,
    context: str,
    hs_chapters: list[str],
    model: str,
) -> list[str]:
    """Generate 5-8 HS-vocabulary search terms for a product description.

    Args:
        query: Product name or description (in English).
        context: Optional context about the product (container info, etc.).
        hs_chapters: List of HS2 chapter descriptions (97 entries) for guidance.
        model: Provider/model string, e.g. "google/gemini-2.5-flash-lite".

    Returns:
        List of search terms.
    """
    client = instructor.from_provider(model)

    hs_list = "\n".join(f"- {ch}" for ch in hs_chapters)

    prompt = (
        f"I have a trade product described as: '{query}'\n"
        f"Shipping context: '{context}'\n\n"
        f"What general class of products does this belong to? Below are the 97 chapters "
        f"of the Harmonized System (HS) trade classification.\n\n"
        f"Return 5-8 terms from this list that this product could fall under. "
        f"Pick the most relevant matches — include both specific and broader categories "
        f"if applicable. If no exact match exists, return the closest description.\n\n"
        f"HS chapters:\n{hs_list}"
    )

    print("--- PROMPT ---")
    print(prompt)
    print("--- END PROMPT ---")

    terms = client.create(
        response_model=SearchTerms,
        messages=[{"role": "user", "content": prompt}],
    )

    return terms.to_list()

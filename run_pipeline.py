"""Single-row classifier entry point.

Pipeline flow for one row:
  1. Load a row from the CSV
  2. Build a query string from the row's product description
  3. Detect the language and translate to English
  4. Generate HS-vocabulary search terms via LLM
"""

import warnings

warnings.filterwarnings("ignore", category=SyntaxWarning, module="translators")

from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

import fire
import polars as pl

from linkages.build_query import build_query
from linkages.search_terms import generate_search_terms, load_hs_descriptions
from linkages.translator import detect_language, translate_eng

INDEX_PATH = Path("data/intermediate/hs12_4_index.parquet")
DEFAULT_MODEL = "google/gemini-2.5-flash-lite"


def load_sample_row(
    csv_path: str = "data/raw/ecuador_sample.csv",
    row_index: int = 0,
) -> dict:
    """Load one row from the sample CSV as a dict."""
    data = pl.read_csv(csv_path)
    if row_index < 0 or row_index >= data.height:
        raise IndexError(f"Row {row_index} is out of range for {csv_path}")
    return data.row(row_index, named=True)


def translate_text(text: str) -> dict[str, str]:
    """Detect language of the input text and translate it to English."""
    detected_lang = detect_language(text)
    english_text = translate_eng(text, from_lang=detected_lang)
    return {
        "input_text": text,
        "detected_language": detected_lang,
        "english_text": english_text,
    }


def process_row(row: dict, model: str = DEFAULT_MODEL) -> dict:
    """Full pipeline for one CSV row: build query, translate, generate search terms."""
    # extract product description and context fields from the row
    query_input = build_query(row)
    # detect language and translate the product description
    result = translate_text(query_input.query)
    result["context"] = query_input.context
    # generate search terms from the translated text
    hs_descriptions = load_hs_descriptions(INDEX_PATH)
    terms = generate_search_terms(
        query=result["english_text"],
        context=result["context"],
        hs_descriptions=hs_descriptions,
        model=model,
    )
    result["search_terms"] = terms
    return result


def classify(
    csv_path: str = "data/raw/ecuador_sample.csv",
    row_index: int = 1,
    model: str = DEFAULT_MODEL,
) -> None:
    """Classify a single row from the sample CSV."""
    row = load_sample_row(csv_path=csv_path, row_index=row_index)
    print(process_row(row, model=model))


if __name__ == "__main__":
    fire.Fire(classify)

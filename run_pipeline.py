"""Single-row classifier entry point.

Pipeline flow for one row:
  1. Load a row from the CSV
  2. Build a query string from the row's product description
  3. Detect the language and translate to English
  4. Return the result
"""

import warnings

warnings.filterwarnings("ignore", category=SyntaxWarning, module="translators")

import fire
import polars as pl

from linkages.build_query import build_query
from linkages.translator import detect_language, translate_eng


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


def process_row(row: dict) -> dict[str, str]:
    """Full pipeline for one CSV row: build query, translate, return result."""
    # extract product description and context fields from the row
    query_input = build_query(row)
    # detect language and translate the product description
    result = translate_text(query_input.query)
    # attach the extra context (container description, item unit, etc.)
    result["context"] = query_input.context
    return result


def classify(csv_path: str = "data/raw/ecuador_sample.csv", row_index: int = 1) -> None:
    """Classify a single row from the sample CSV."""
    row = load_sample_row(csv_path=csv_path, row_index=row_index)
    print(process_row(row))


if __name__ == "__main__":
    fire.Fire(classify)

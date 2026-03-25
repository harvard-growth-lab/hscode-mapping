"""CLI entry point — classify a single CSV row.

Usage:
  uv run run_pipeline.py                          # default: row 1 from ecuador_sample
  uv run run_pipeline.py --row_index 5            # different row
  uv run run_pipeline.py --csv_path data/raw/other.csv --row_index 0
"""

import logging

logging.basicConfig(level=logging.INFO, format="%(name)s | %(message)s")

import fire
import polars as pl

from hs_classifier import init_classifier, classify_row


def load_sample_row(
    csv_path: str = "data/raw/ecuador_sample.csv",
    row_index: int = 0,
) -> dict:
    """Load one row from the sample CSV as a dict."""
    data = pl.read_csv(csv_path)
    if row_index < 0 or row_index >= data.height:
        raise IndexError(f"Row {row_index} is out of range for {csv_path}")
    return data.row(row_index, named=True)


def classify(
    csv_path: str = "data/raw/ecuador_sample.csv",
    row_index: int = 1,
) -> None:
    """Classify a single row from the sample CSV."""
    classifier = init_classifier()
    row = load_sample_row(csv_path=csv_path, row_index=row_index)
    result = classify_row(row, classifier)
    print(result)


if __name__ == "__main__":
    fire.Fire(classify)

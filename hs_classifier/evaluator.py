"""Evaluate pipeline predictions against ground truth HS codes."""

from typing import Any

import pandas as pd
import polars as pl


def _to_polars(df: Any) -> tuple[pl.DataFrame, bool]:
    if isinstance(df, pl.DataFrame):
        return df, False
    if isinstance(df, pd.DataFrame):
        return pl.from_pandas(df), True
    raise TypeError("Expected a pandas or polars DataFrame")


def evaluation_report(
    df: Any,
    truth_col: str = "code_true",
    pred_cols: list[str] | None = None,
) -> dict:
    """Compute top-1, top-k, and chapter accuracy. Returns a readable summary dict."""
    df, return_pandas = _to_polars(df)
    pred_cols = pred_cols or ["code_1", "code_2"]
    pred_col = pred_cols[0]
    n = len(df)

    # top-1: first prediction matches ground truth
    top1_hits = int((df[pred_col] == df[truth_col]).sum())
    top1 = top1_hits / n if n else 0.0

    # top-k: any prediction matches ground truth
    match = pl.lit(False)
    for col in pred_cols:
        match = match | (pl.col(col) == pl.col(truth_col))
    topk_hits = int(df.select(match.alias("hit"))["hit"].sum())
    topk = topk_hits / n if n else 0.0

    # chapter: compare first 2 digits only
    pred_ch = df[pred_col].cast(pl.String).str.slice(0, 2)
    true_ch = df[truth_col].cast(pl.String).str.slice(0, 2)
    chapter_hits = int((pred_ch == true_ch).sum())
    chapter = chapter_hits / n if n else 0.0

    correctness_summary = pl.DataFrame(
        {
            "metric": ["top1", "topk", "chapter"],
            "correct": [top1_hits, topk_hits, chapter_hits],
            "incorrect": [n - top1_hits, n - topk_hits, n - chapter_hits],
            "count": [f"{top1_hits}/{n}", f"{topk_hits}/{n}", f"{chapter_hits}/{n}"],
        }
    )

    return {
        "n": n,
        "top1_count": f"{top1_hits}/{n}",
        "topk_count": f"{topk_hits}/{n}",
        "chapter_count": f"{chapter_hits}/{n}",
        "summary_text": (
            f"Top-1 matched on {top1_hits}/{n} rows, "
            f"top-k matched on {topk_hits}/{n}, "
            f"and chapter matched on {chapter_hits}/{n}."
        ),
        "top1_accuracy": top1,
        "topk_accuracy": topk,
        "chapter_accuracy": chapter,
        "correctness_summary": (
            correctness_summary.to_pandas() if return_pandas else correctness_summary
        ),
    }

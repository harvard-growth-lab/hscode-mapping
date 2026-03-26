"""Evaluate pipeline predictions against ground truth HS codes."""

import polars as pl


def evaluation_report(
    df: pl.DataFrame,
    truth_col: str = "code_true",
    pred_cols: list[str] = ["code_1", "code_2"],
) -> dict:
    """Compute top-1, top-k, and chapter accuracy. Returns a summary dict."""
    pred_col = pred_cols[0]

    # top-1: first prediction matches ground truth
    top1 = (df[pred_col] == df[truth_col]).mean()

    # top-k: any prediction matches ground truth
    match = pl.lit(False)
    for col in pred_cols:
        match = match | (pl.col(col) == pl.col(truth_col))
    topk = df.select(match.alias("hit"))["hit"].mean()

    # chapter: compare first 2 digits only
    pred_ch = df[pred_col].cast(pl.String).str.slice(0, 2)
    true_ch = df[truth_col].cast(pl.String).str.slice(0, 2)
    chapter = (pred_ch == true_ch).mean()

    # confusion matrix: true chapter (rows) vs predicted chapter (columns)
    confusion = (
        df.with_columns(
            true_ch.alias("true_chapter"),
            pred_ch.alias("pred_chapter"),
        )
        .group_by("true_chapter", "pred_chapter")
        .agg(pl.len().alias("count"))
        .pivot(on="pred_chapter", index="true_chapter", values="count")
        .fill_null(0)
        .sort("true_chapter")
    )

    return {
        "n": len(df),
        "top1_accuracy": top1,
        "topk_accuracy": topk,
        "chapter_accuracy": chapter,
        "confusion_matrix": confusion,
    }

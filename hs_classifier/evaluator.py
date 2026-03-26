"""Evaluate pipeline predictions against ground truth HS codes.

Computes standard classification metrics at both the 4-digit (HS4) and
2-digit (chapter) levels. Results are returned as a dict and can be
rendered as a markdown report.
"""

import polars as pl


def top1_accuracy(
    df: pl.DataFrame,
    pred_col: str = "code_first",
    truth_col: str = "code_true",
) -> float:
    """Fraction of predictions where code_first exactly matches ground truth."""
    return (df[pred_col] == df[truth_col]).mean()


def topk_accuracy(
    df: pl.DataFrame,
    pred_cols: list[str] = ["code_first", "code_second"],
    truth_col: str = "code_true",
) -> float:
    """Fraction where ground truth matches any of the predicted codes."""
    match = pl.lit(False)
    for col in pred_cols:
        match = match | (pl.col(col) == pl.col(truth_col))
    return df.select(match.alias("hit"))["hit"].mean()


def chapter_accuracy(
    df: pl.DataFrame,
    pred_col: str = "code_first",
    truth_col: str = "code_true",
) -> float:
    """Accuracy at the 2-digit chapter level (first 2 digits of HS code)."""
    pred_ch = df[pred_col].cast(pl.String).str.slice(0, 2)
    true_ch = df[truth_col].cast(pl.String).str.slice(0, 2)
    return (pred_ch == true_ch).mean()


def per_chapter_accuracy(
    df: pl.DataFrame,
    pred_col: str = "code_first",
    truth_col: str = "code_true",
) -> pl.DataFrame:
    """Per-chapter breakdown: chapter code, n samples, accuracy."""
    return (
        df.with_columns(
            pl.col(truth_col).cast(pl.String).str.slice(0, 2).alias("chapter"),
            (pl.col(pred_col) == pl.col(truth_col)).alias("correct"),
        )
        .group_by("chapter")
        .agg(
            pl.len().alias("n"),
            pl.col("correct").mean().alias("accuracy"),
        )
        .sort("chapter")
    )


def confusion_matrix_chapter(
    df: pl.DataFrame,
    pred_col: str = "code_first",
    truth_col: str = "code_true",
) -> pl.DataFrame:
    """Confusion matrix at the 2-digit chapter level.

    Returns a long-form DataFrame with columns: true_chapter, pred_chapter, count.
    """
    return (
        df.with_columns(
            pl.col(truth_col).cast(pl.String).str.slice(0, 2).alias("true_chapter"),
            pl.col(pred_col).cast(pl.String).str.slice(0, 2).alias("pred_chapter"),
        )
        .group_by("true_chapter", "pred_chapter")
        .agg(pl.len().alias("count"))
        .sort("true_chapter", "pred_chapter")
    )


def evaluation_report(
    df: pl.DataFrame,
    truth_col: str = "code_true",
    pred_cols: list[str] = ["code_first", "code_second"],
) -> dict:
    """Run all metrics and return a summary dict."""
    return {
        "n": len(df),
        "top1_accuracy": top1_accuracy(df, pred_col=pred_cols[0], truth_col=truth_col),
        "topk_accuracy": topk_accuracy(df, pred_cols=pred_cols, truth_col=truth_col),
        "chapter_accuracy": chapter_accuracy(df, pred_col=pred_cols[0], truth_col=truth_col),
        "per_chapter": per_chapter_accuracy(df, pred_col=pred_cols[0], truth_col=truth_col),
        "confusion_matrix": confusion_matrix_chapter(
            df, pred_col=pred_cols[0], truth_col=truth_col
        ),
    }


def report_to_markdown(report: dict) -> str:
    """Render an evaluation_report() dict as a markdown string."""
    lines = [
        "# Evaluation Report",
        "",
        f"**Samples:** {report['n']}",
        "",
        "## Overall Metrics",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Top-1 Accuracy (HS4) | {report['top1_accuracy']:.1%} |",
        f"| Top-K Accuracy (HS4) | {report['topk_accuracy']:.1%} |",
        f"| Chapter Accuracy (HS2) | {report['chapter_accuracy']:.1%} |",
        "",
        "## Per-Chapter Breakdown",
        "",
        "| Chapter | Samples | Accuracy |",
        "|---|---|---|",
    ]

    for row in report["per_chapter"].iter_rows(named=True):
        lines.append(f"| {row['chapter']} | {row['n']} | {row['accuracy']:.1%} |")

    lines.extend([
        "",
        "## Confusion Matrix (HS2 chapters, non-zero only)",
        "",
        "| True | Predicted | Count |",
        "|---|---|---|",
    ])

    for row in report["confusion_matrix"].iter_rows(named=True):
        lines.append(f"| {row['true_chapter']} | {row['pred_chapter']} | {row['count']} |")

    lines.append("")
    return "\n".join(lines)

"""Evaluate pipeline predictions against ground truth HS codes."""

import pandas as pd


def top1_accuracy(df: pd.DataFrame, pred_col: str = "code_first", truth_col: str = "code_true") -> float:
    """Fraction of predictions where code_first exactly matches ground truth."""
    # TODO
    raise NotImplementedError


def top2_accuracy(df: pd.DataFrame, pred1_col: str = "code_first", pred2_col: str = "code_second", truth_col: str = "code_true") -> float:
    """Fraction where ground truth matches either code_first or code_second."""
    # TODO
    raise NotImplementedError


def chapter_accuracy(df: pd.DataFrame, pred_col: str = "code_first", truth_col: str = "code_true") -> float:
    """Hierarchical accuracy at the 2-digit chapter level.

    Partial credit: correct chapter even if full 4-digit code is wrong.
    HS chapters are the first 2 digits of the code.
    """
    # TODO: compare df[pred_col].str[:2] vs df[truth_col].str[:2]
    raise NotImplementedError


def confusion_matrix_chapter(df: pd.DataFrame, pred_col: str = "code_first", truth_col: str = "code_true") -> pd.DataFrame:
    """Confusion matrix at the 2-digit chapter level.

    Full 4-digit confusion matrix is ~1900x1900 and unreadable;
    chapter level (~100x100) is more interpretable.
    """
    # TODO: pd.crosstab on first-2-digit slices
    raise NotImplementedError


def evaluation_report(df: pd.DataFrame, truth_col: str = "code_true") -> dict:
    """Run all metrics and return a summary dict."""
    return {
        "top1_accuracy": top1_accuracy(df, truth_col=truth_col),
        "top2_accuracy": top2_accuracy(df, truth_col=truth_col),
        "chapter_accuracy": chapter_accuracy(df, truth_col=truth_col),
    }

"""Ground truth management and Label Studio integration.

Handles two cases:
  - Rows that already have a ground truth HS code (e.g. from an existing labeled dataset)
  - Rows that need manual annotation via Label Studio

Workflow:
  1. export_for_annotation()  → JSON file importable into Label Studio
  2. <human labels in Label Studio>
  3. import_annotations()     → parquet with code_true column
  4. merge_ground_truth()     → combine existing + newly labeled rows
"""

import pandas as pd


def export_for_annotation(
    df: pd.DataFrame,
    output_path: str,
    text_col: str = "name",
    id_col: str = "unique_id",
) -> None:
    """Export unlabeled rows to Label Studio JSON task format.

    Args:
        df: DataFrame of rows that need labeling.
        output_path: Path to write the Label Studio import JSON.
        text_col: Column containing the product description string.
        id_col: Column containing a unique row identifier.
    """
    # TODO: build Label Studio task list and write JSON
    # tasks = [{"id": row[id_col], "data": {"text": row[text_col]}} for _, row in df.iterrows()]
    raise NotImplementedError


def import_annotations(annotations_path: str) -> pd.DataFrame:
    """Parse a Label Studio export file and return a DataFrame with code_true column.

    Args:
        annotations_path: Path to Label Studio export (JSON).

    Returns:
        DataFrame with columns [unique_id, code_true, label_source="label_studio"].
    """
    # TODO: parse Label Studio export format
    raise NotImplementedError


def merge_ground_truth(
    existing: pd.DataFrame,
    annotated: pd.DataFrame,
    id_col: str = "unique_id",
    truth_col: str = "code_true",
) -> pd.DataFrame:
    """Merge existing ground truth with newly annotated rows.

    Tracks label provenance in a label_source column:
      - "existing"      for pre-existing ground truth
      - "label_studio"  for newly annotated rows

    Args:
        existing: DataFrame with known ground truth (must have id_col and truth_col).
        annotated: Output of import_annotations().

    Returns:
        Combined DataFrame deduplicated on id_col.
    """
    # TODO: concat, dedup, track source
    raise NotImplementedError

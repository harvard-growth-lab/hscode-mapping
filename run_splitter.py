"""Run the splitter on a labeled CSV to produce a stratified eval sample.

Usage:
  uv run run_splitter.py --csv_path data/raw/test_labeled_sample.csv \
      --text_col product_description --truth_col hs_code --sample_frac 0.10
"""

import logging

logging.basicConfig(level=logging.INFO, format="%(name)s | %(message)s")
for name in ("httpx", "faiss.loader", "sentence_transformers", "numba"):
    logging.getLogger(name).setLevel(logging.WARNING)

import os

from dotenv import load_dotenv

load_dotenv()

import fire
import polars as pl
from sentence_transformers import SentenceTransformer

from hs_classifier.splitter import prepare_eval_sample

logger = logging.getLogger(__name__)

EMBEDDING_MODEL = os.environ["EMBEDDING_MODEL"]


def split(
    csv_path: str,
    text_col: str = "product_description",
    truth_col: str = "hs_code",
    sample_frac: float = 0.02,
    min_cluster_size: int = 10,
    output_path: str | None = None,
) -> None:
    """Cluster and sample a labeled CSV for evaluation.

    Args:
        csv_path: Path to the input CSV with text and ground truth columns.
        text_col: Column with product descriptions.
        truth_col: Column with ground truth HS codes.
        sample_frac: Fraction to sample (e.g. 0.01 = 1%, 0.10 = 10%).
        min_cluster_size: Minimum points for HDBSCAN cluster formation.
        output_path: Where to save the sample CSV. Defaults to
            <csv_path>_sample_<frac>.csv.
    """
    logger.info(f"Loading {csv_path}")
    df = pl.read_csv(csv_path, schema_overrides={truth_col: pl.String})
    logger.info(f"Loaded {len(df)} rows, columns: {df.columns}")

    logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL, local_files_only=True)

    sample = prepare_eval_sample(
        df,
        text_col=text_col,
        truth_col=truth_col,
        model=model,
        sample_frac=sample_frac,
        min_cluster_size=min_cluster_size,
    )

    if output_path is None:
        frac_str = f"{sample_frac:.0%}".replace("%", "pct")
        output_path = csv_path.replace(".csv", f"_sample_{frac_str}.csv")

    sample.write_csv(output_path)
    logger.info(f"Saved {len(sample)} rows to {output_path}")

    # print summary
    print(f"\n--- Eval sample: {len(sample)} / {len(df)} rows ({len(sample)/len(df):.1%}) ---")
    print(sample.group_by("cluster", truth_col).agg(pl.len().alias("n")).sort("cluster", truth_col))


if __name__ == "__main__":
    fire.Fire(split)

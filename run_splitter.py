"""Run the splitter on a CSV to produce a stratified eval sample.

Usage:
  uv run run_splitter.py --csv_path data/raw/test_labeled_sample.csv
  uv run run_splitter.py --csv_path data/raw/my_data.csv --text_col name --sample_frac 0.05
  uv run run_splitter.py --csv_path data/raw/my_data.csv --output_path results/sample.csv
"""

import logging
import warnings
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(name)s | %(message)s")
for name in ("httpx", "faiss.loader", "sentence_transformers", "numba"):
    logging.getLogger(name).setLevel(logging.WARNING)

warnings.filterwarnings(
    "ignore",
    message=r"n_jobs value 1 overridden to 1 by setting random_state\..*",
    category=UserWarning,
    module="umap",
)

import os

from dotenv import load_dotenv

load_dotenv()

import fire
import polars as pl
from sentence_transformers import SentenceTransformer

from hs_classifier.splitter import prepare_eval_sample

logger = logging.getLogger(__name__)

EMBEDDING_MODEL = os.environ["EMBEDDING_MODEL"]
INTERMEDIATE_DATA_DIR = Path(os.environ.get("INTERMEDIATE_DATA_DIR", "data/intermediate"))


def split(
    csv_path: str,
    text_col: str = "product_description",
    sample_frac: float = 0.02,
    min_cluster_size: int = 10,
    output_path: str | None = None,
) -> None:
    """Cluster and sample a CSV for evaluation.

    Only the text column is used for clustering. All other columns
    (ground truth, metadata, etc.) are carried through to the output.

    The sample is saved under INTERMEDIATE_DATA_DIR by default:
      data/raw/my_data.csv → data/intermediate/samples/my_data_sample_2pct.csv

    Args:
        csv_path: Path to the input CSV.
        text_col: Column with product descriptions to embed.
        sample_frac: Fraction to sample (e.g. 0.01 = 1%, 0.10 = 10%).
        min_cluster_size: Minimum points for HDBSCAN cluster formation.
        output_path: Override where to save the sample CSV.
    """
    csv_path = Path(csv_path)
    logger.info(f"Loading {csv_path}")
    df = pl.read_csv(csv_path)
    logger.info(f"Loaded {len(df)} rows, columns: {df.columns}")

    logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL, local_files_only=True)

    sample = prepare_eval_sample(
        df,
        text_col=text_col,
        model=model,
        sample_frac=sample_frac,
        min_cluster_size=min_cluster_size,
    )

    if output_path is None:
        frac_str = f"{sample_frac:.0%}".replace("%", "pct")
        output_path = INTERMEDIATE_DATA_DIR / "samples" / f"{csv_path.stem}_sample_{frac_str}.csv"
    else:
        output_path = Path(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    sample.write_csv(output_path)

    # print summary
    print(f"\n--- Eval sample: {len(sample)} / {len(df)} rows ({len(sample)/len(df):.1%}) ---")
    print(sample.group_by("cluster").agg(pl.len().alias("n")).sort("cluster"))
    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    fire.Fire(split)

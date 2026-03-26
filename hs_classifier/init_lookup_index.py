"""HS code lookup index — one-time setup.

Loads HS code descriptions from the Atlas DB, encodes them with S-BERT,
and saves everything to a single .parquet. Run once before using the pipeline.

Flow:
  1. Connect to Atlas DB and load HS code descriptions (ordered by code)
  2. Encode descriptions with S-BERT and L2-normalize
  3. Save code, description, and embedding vector to .parquet
"""

import logging
import os
from pathlib import Path

import numpy as np
import polars as pl
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()

logger = logging.getLogger(__name__)


# --- Database ---


def _build_db_uri() -> str:
    """Build a PostgreSQL connection URI from environment variables."""
    host = os.environ["ATLAS_HOST"]
    port = os.environ.get("ATLAS_PORT", "5432")
    user = os.environ["ATLAS_USER"]
    password = os.environ["ATLAS_PASSWORD"]
    db = os.environ["ATLAS_DB"]
    return f"postgresql://{user}:{password}@{host}:{port}/{db}"


def load_hs_data(level: int = 4) -> pl.DataFrame:
    """Load HS code descriptions from the Atlas database, ordered by code."""
    query = (
        "SELECT code, name_en AS description, name_short_en AS short_name "
        "FROM classification.product_hs12 "
        f"WHERE product_level = {level} "
        "ORDER BY code"
    )
    return pl.read_database_uri(query=query, uri=_build_db_uri())


# --- Embeddings ---


def normalized_embeddings(texts: list[str], model) -> np.ndarray:
    """Encode texts with a SentenceTransformer and L2-normalize for cosine similarity."""
    embeddings = model.encode(texts, show_progress_bar=False)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / norms


def save_hs_chapters(output_path: Path, force: bool = False) -> None:
    """Save HS2 chapter descriptions to a parquet (used as LLM prompt context)."""
    if output_path.exists() and not force:
        logger.info(f"HS chapters already exist at {output_path}, skipping")
        return

    data = load_hs_data(level=2)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data.write_parquet(output_path)
    logger.info(f"Saved {len(data)} HS chapters to {output_path}")


def build_index(
    output_path: Path,
    level: int = 4,
    model_name: str = "dell-research-harvard/lt-un-data-fine-fine-en",
    force: bool = False,
) -> None:
    """Generate and save HS code + description + embeddings to a single parquet."""
    if output_path.exists() and not force:
        logger.info(f"Index already exists at {output_path}, skipping (use force=True to rebuild)")
        return

    data = load_hs_data(level)
    logger.info(f"Generating embeddings for {len(data)} HS level-{level} codes from Atlas DB")

    model = SentenceTransformer(model_name)
    embeddings = normalized_embeddings(data["description"].to_list(), model)

    # add embedding vectors as a list column
    data = data.with_columns(pl.Series("embedding", embeddings.tolist()))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    data.write_parquet(output_path)
    logger.info(f"Saved index ({len(data)} codes) to {output_path}")

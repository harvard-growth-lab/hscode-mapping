"""Initialize the lookup index — run once before using the pipeline.

Loads HS code descriptions from the Atlas DB, generates S-BERT embeddings,
and saves code + description + embedding to a single parquet.

Usage:
  uv run run_init.py              # skips if parquet already exists
  uv run run_init.py --force      # rebuild even if it exists
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

import fire

from hs_classifier.init_lookup_index import build_index, save_hs_chapters

INDEX_PATH = Path("data/intermediate/hs12_4_index.parquet")
CHAPTERS_PATH = Path("data/intermediate/hs2_chapters.parquet")
EMBEDDING_MODEL = os.environ["EMBEDDING_MODEL"]


def init(force: bool = False) -> None:
    """Generate HS code index and chapter reference. Skips if already built unless --force."""
    save_hs_chapters(output_path=CHAPTERS_PATH, force=force)
    build_index(
        output_path=INDEX_PATH,
        level=4,
        model_name=EMBEDDING_MODEL,
        force=force,
    )


if __name__ == "__main__":
    fire.Fire(init)

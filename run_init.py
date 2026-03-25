"""Initialize the lookup index — run once before using the pipeline.

Loads HS code descriptions from the Atlas DB, generates S-BERT embeddings,
and saves code + description + embedding to a single parquet.

Usage:
  uv run run_init.py              # skips if parquet already exists
  uv run run_init.py --force      # rebuild even if it exists
"""

from pathlib import Path

import fire

from linkages.init_lookup_index import build_index, save_hs_chapters

INDEX_PATH = Path("data/intermediate/hs12_4_index.parquet")
CHAPTERS_PATH = Path("data/intermediate/hs2_chapters.parquet")
DEFAULT_MODEL = "dell-research-harvard/lt-un-data-fine-fine-en"


def init(
    model: str = DEFAULT_MODEL,
    force: bool = False,
) -> None:
    """Generate HS code index and chapter reference. Skips if already built unless --force."""
    save_hs_chapters(output_path=CHAPTERS_PATH, force=force)
    build_index(
        output_path=INDEX_PATH,
        level=4,
        model_name=model,
        force=force,
    )


if __name__ == "__main__":
    fire.Fire(init)

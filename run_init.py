"""Initialize the lookup index — run once before using the pipeline.

Loads HS code descriptions from the Atlas DB, generates S-BERT embeddings,
and saves code + description + embedding to a single parquet.

Usage:
  uv run run_init.py              # skips if parquet already exists
  uv run run_init.py --force      # rebuild even if it exists
"""

from pathlib import Path

import fire

from linkages.init_lookup_index import build_index

DEFAULT_OUTPUT = Path("data/intermediate/hs12_4_index.parquet")
DEFAULT_MODEL = "dell-research-harvard/lt-un-data-fine-fine-en"


def init(
    output: str = str(DEFAULT_OUTPUT),
    level: int = 4,
    model: str = DEFAULT_MODEL,
    force: bool = False,
) -> None:
    """Generate HS code index. Skips if already built unless --force is set."""
    build_index(
        output_path=Path(output),
        level=level,
        model_name=model,
        force=force,
    )


if __name__ == "__main__":
    fire.Fire(init)

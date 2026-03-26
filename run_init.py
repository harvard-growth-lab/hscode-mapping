"""Initialize the lookup index — run once before using the pipeline.

Usage:
  uv run run_init.py              # skips if parquet already exists
  uv run run_init.py --force      # rebuild even if it exists
"""

import logging

logging.basicConfig(level=logging.INFO, format="%(name)s | %(message)s")

from dotenv import load_dotenv

load_dotenv()

import fire

from hs_classifier import init_index

if __name__ == "__main__":
    fire.Fire(init_index)

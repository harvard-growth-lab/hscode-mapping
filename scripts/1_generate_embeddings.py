"""Generate HS code embeddings.

One-time step (or re-run when changing HS level or model).

Original: 1_hs_embeddings.py.
"""

import argparse
from pathlib import Path

from modules.config import Settings
from modules.lookup_index import generate_embeddings


def main():
    parser = argparse.ArgumentParser(description="Generate HS code embeddings")
    parser.add_argument("--level", type=str, default="4", help="HS level to embed (default: 4)")
    parser.add_argument("--sheet", type=str, default="HS12", help="Excel sheet name (default: HS12)")
    parser.add_argument("--output", type=str, default=None, help="Output .npy path")
    args = parser.parse_args()

    settings = Settings()
    output_path = Path(args.output) if args.output else settings.embeddings_path

    generate_embeddings(
        hs_table_path=settings.hs_table_path,
        output_path=output_path,
        sheet_name=args.sheet,
        level=args.level,
        model_name=settings.embedding_model_name,
    )


if __name__ == "__main__":
    main()

"""HS code embedding generation and loading.

Generates S-BERT embeddings for HS code descriptions and saves as .npy files.
Also provides loading utilities used by retrieval.py at startup.

Original sources: 1_hs_embeddings.py, helper.py generate_embeddings().
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from linkages.utils import normalized_embeddings


def load_embedding_model(model_name: str) -> SentenceTransformer:
    """Load a SentenceTransformer model. Called once at pipeline startup."""
    return SentenceTransformer(model_name)


def generate_embeddings(
    hs_table_path: Path,
    output_path: Path,
    sheet_name: str = "HS12",
    level: str = "4",
    model_name: str = "dell-research-harvard/lt-un-data-fine-fine-en",
) -> None:
    """Generate and save L2-normalized embeddings for HS code descriptions.

    Reads the HS code Excel table, filters to the specified level,
    encodes descriptions with S-BERT, and saves the result as a .npy file.

    Original: 1_hs_embeddings.py (entire file).
    """
    data = pd.read_excel(hs_table_path, sheet_name=sheet_name, dtype=str, engine="openpyxl")
    data = data[data["Level"] == level].copy()
    data.reset_index(drop=True, inplace=True)

    print(f"Generating embeddings for {len(data)} HS{level} codes from {hs_table_path}")

    model = SentenceTransformer(model_name)
    descriptions = data["Description"].tolist()
    embeddings = normalized_embeddings(descriptions, model)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, embeddings)
    print(f"Saved embeddings to {output_path}")


def load_embeddings(embeddings_path: Path) -> np.ndarray:
    """Load pre-computed embeddings from a .npy file."""
    if not embeddings_path.exists():
        raise FileNotFoundError(f"Embeddings file not found: {embeddings_path}")
    return np.load(embeddings_path)


def load_hs_data(
    hs_table_path: Path,
    sheet_name: str = "HS12",
    level: str = "4",
) -> pd.DataFrame:
    """Load and filter the HS code reference table.

    Previously done inside standard_retrieval() on every single query.
    Now called once at startup.

    Original: run_pipe.py lines 59-63.
    """
    data = pd.read_excel(hs_table_path, sheet_name=sheet_name, dtype=str, engine="openpyxl")
    data = data[data["Level"] == level].copy()
    data.reset_index(drop=True, inplace=True)
    return data

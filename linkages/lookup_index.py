"""HS code lookup index: load descriptions, generate embeddings, build FAISS index.

Combines embedding generation/loading (was embeddings.py) with index construction
(was HSIndex.__init__ in retrieval.py). Everything about setting up the HS side
happens here once at startup.

TODO: descriptions are currently loaded from a static Excel file — we have a data
source that provides up-to-date HS vintages, so load_hs_data() should be updated
to pull from there instead.

TODO: replace FAISS with a modern vector DB (e.g. Qdrant, LanceDB) for persistent
storage and easier updates when descriptions change.
"""

from pathlib import Path

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


def normalized_embeddings(texts: list[str], model) -> np.ndarray:
    """Encode texts with a SentenceTransformer and L2-normalize for cosine similarity."""
    embeddings = model.encode(texts, show_progress_bar=False)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / norms


def load_embedding_model(model_name: str) -> SentenceTransformer:
    """Load a SentenceTransformer model. Called once at pipeline startup."""
    return SentenceTransformer(model_name)


def load_hs_data(
    hs_table_path: Path,
    sheet_name: str = "HS12",
    level: str = "4",
) -> pd.DataFrame:
    """Load and filter the HS code reference table."""
    data = pd.read_excel(hs_table_path, sheet_name=sheet_name, dtype=str, engine="openpyxl")
    data = data[data["Level"] == level].copy()
    data.reset_index(drop=True, inplace=True)
    return data


def generate_embeddings(
    hs_table_path: Path,
    output_path: Path,
    sheet_name: str = "HS12",
    level: str = "4",
    model_name: str = "dell-research-harvard/lt-un-data-fine-fine-en",
) -> None:
    """Generate and save L2-normalized S-BERT embeddings for HS code descriptions.

    One-time step. Run via scripts/1_generate_embeddings.py.
    """
    data = load_hs_data(hs_table_path, sheet_name, level)
    print(f"Generating embeddings for {len(data)} HS{level} codes from {hs_table_path}")

    model = SentenceTransformer(model_name)
    embeddings = normalized_embeddings(data["Description"].tolist(), model)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, embeddings)
    print(f"Saved embeddings to {output_path}")


def load_embeddings(embeddings_path: Path) -> np.ndarray:
    """Load pre-computed embeddings from a .npy file."""
    if not embeddings_path.exists():
        raise FileNotFoundError(f"Embeddings file not found: {embeddings_path}")
    return np.load(embeddings_path)


class HSIndex:
    """Holds the HS code data, embeddings, and FAISS index. Built once at startup."""

    def __init__(
        self,
        hs_table_path: Path,
        embeddings_path: Path,
        embedding_model: SentenceTransformer,
        sheet_name: str = "HS12",
        level: str = "4",
    ):
        self.data = load_hs_data(hs_table_path, sheet_name, level)
        self.model = embedding_model

        embeddings = load_embeddings(embeddings_path).astype("float32")
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(embeddings)

        print(f"HSIndex ready: {len(self.data)} codes, {dimension}d embeddings")

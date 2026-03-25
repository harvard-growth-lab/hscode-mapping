"""Embed search terms, query the FAISS index, aggregate and deduplicate results."""

from pathlib import Path

import faiss
import numpy as np
import polars as pl
from sentence_transformers import SentenceTransformer

from hs_classifier.init_lookup_index import normalized_embeddings


# --- Index loading ---


def load_index(index_path: Path) -> tuple[pl.DataFrame, list[str], faiss.IndexFlatIP]:
    """Load the parquet index and build a FAISS index from the embedding column.

    Returns:
        data: DataFrame with code + description for lookups.
        codes: List of codes aligned with FAISS vector positions.
        index: FAISS inner-product index.
    """
    if not index_path.exists():
        raise FileNotFoundError(f"Index not found at {index_path} — run run_init.py first")

    full = pl.read_parquet(index_path)

    # codes list stays aligned with FAISS vector positions
    codes = full["code"].to_list()

    # extract embeddings into a numpy array and build FAISS index
    embeddings = np.stack(full["embedding"].to_list()).astype("float32")
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    # keep only code + description for lookups
    data = full.select("code", "description")

    print(f"Index loaded: {len(data)} codes, {embeddings.shape[1]}d embeddings")
    return data, codes, index


# --- Search ---


def search(
    data: pl.DataFrame,
    codes: list[str],
    index: faiss.IndexFlatIP,
    model: SentenceTransformer,
    query: str,
    top_k: int,
) -> pl.DataFrame:
    """Embed a single query string and return the top_k nearest HS codes."""
    query_embedding = normalized_embeddings([query], model).astype("float32")
    _, indices = index.search(query_embedding, int(top_k))
    # look up by code rather than position so data ordering doesn't matter
    matched_codes = [codes[i] for i in indices[0]]
    return data.filter(pl.col("code").is_in(matched_codes))


def multi_search(
    data: pl.DataFrame,
    codes: list[str],
    index: faiss.IndexFlatIP,
    model: SentenceTransformer,
    query: str,
    terms: list[str],
    top_k_total: int,
    top_k_bert: int,
) -> pl.DataFrame:
    """Search the original query plus each synonym term, then aggregate and deduplicate.

    The query gets top_k_bert slots; remaining budget is split evenly across terms.
    """
    results = [search(data, codes, index, model, query, top_k_bert)]

    top_k_each = (top_k_total - top_k_bert) // len(terms)
    for term in terms:
        results.append(search(data, codes, index, model, term, max(top_k_each, 1)))

    concatenated = pl.concat(results)
    return concatenated.unique(subset=["code"])

"""Embed search terms, query the FAISS index, aggregate and deduplicate results."""

import numpy as np
import pandas as pd

from linkages.lookup_index import HSIndex, normalized_embeddings


def search(hs_index: HSIndex, query: str, top_k: int) -> pd.DataFrame:
    """Embed a single query string and return the top_k nearest HS codes."""
    top_k = int(top_k)
    query_embedding = normalized_embeddings([query], hs_index.model).astype("float32")
    _, indices = hs_index.index.search(query_embedding, top_k)
    return hs_index.data.iloc[indices[0]]


def multi_search(
    hs_index: HSIndex,
    query: str,
    terms: list[str],
    top_k_total: int,
    top_k_bert: int,
) -> pd.DataFrame:
    """Search the original query plus each synonym term, then aggregate and deduplicate.

    The query gets top_k_bert slots; remaining budget is split evenly across terms.
    """
    results = [search(hs_index, query, top_k_bert)]

    top_k_each = (top_k_total - top_k_bert) // len(terms)
    for term in terms:
        results.append(search(hs_index, term, max(top_k_each, 1)))

    concatenated = pd.concat(results, ignore_index=True)
    return concatenated.drop_duplicates(subset=["Code"])

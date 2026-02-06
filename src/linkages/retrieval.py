"""FAISS-backed semantic search for HS codes.

The critical performance fix: the original code rebuilt the FAISS index on every
single query (~48,000 times for a full run). HSIndex builds it once at startup.

Original sources: run_pipe.py standard_retrieval() (lines 41-84),
multi_semantic_search() (lines 86-109).
"""

from pathlib import Path

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from linkages.utils import normalized_embeddings


class HSIndex:
    """Pre-built FAISS index for HS code semantic search.

    Build once at startup, query many times. Holds the HS data, embeddings,
    FAISS index, and embedding model in memory.
    """

    def __init__(
        self,
        hs_table_path: Path,
        embeddings_path: Path,
        embedding_model: SentenceTransformer,
        sheet_name: str = "HS12",
        level: str = "4",
    ):
        # Load HS data ONCE
        self.data = pd.read_excel(
            hs_table_path, sheet_name=sheet_name, dtype=str, engine="openpyxl"
        )
        self.data = self.data[self.data["Level"] == level].copy()
        self.data.reset_index(drop=True, inplace=True)

        # Load embeddings ONCE
        embeddings = np.load(str(embeddings_path)).astype("float32")

        # Build FAISS index ONCE
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(embeddings)

        # Store model reference for query encoding
        self.model = embedding_model

        print(f"HSIndex ready: {len(self.data)} codes, {dimension}d embeddings")

    def search(self, query: str, top_k: int) -> pd.DataFrame:
        """Semantic search for a single query string. Returns top_k HS code rows."""
        top_k = int(top_k)
        query_embedding = normalized_embeddings([query], self.model).astype("float32")
        _, indices = self.index.search(query_embedding, top_k)
        return self.data.iloc[indices[0]]

    def multi_search(
        self,
        query: str,
        terms: list[str],
        top_k_total: int,
        top_k_bert: int,
    ) -> pd.DataFrame:
        """Multi-query search: original query + Claude-generated terms.

        Distributes the remaining budget (top_k_total - top_k_bert) evenly
        across the generated terms, then deduplicates by HS Code.

        Original: run_pipe.py multi_semantic_search() lines 86-109.
        """
        results = [self.search(query, top_k_bert)]

        top_k_each = (top_k_total - top_k_bert) // len(terms)
        for term in terms:
            results.append(self.search(term, max(top_k_each, 1)))

        concatenated = pd.concat(results, ignore_index=True)
        return concatenated.drop_duplicates(subset=["Code"])

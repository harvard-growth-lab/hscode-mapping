"""Shared utility functions.

Deduplicates helpers that were copy-pasted across run_pipe.py, helper.py,
and 1_llm_linkage/helper.py in the original project.
"""

import numpy as np
import pandas as pd


def extract_text_chunks(paragraphs) -> str:
    """Defensively extract text from nested paragraph structures.

    Handles lists of dicts, lists of strings, plain dicts, and plain strings.
    Original: run_pipe.py lines 191-203 (appeared identically at lines 233-245).
    """
    text_chunks = []
    if isinstance(paragraphs, list):
        for item in paragraphs:
            if isinstance(item, dict):
                text_chunks.extend([str(v) for v in item.values() if v])
            elif isinstance(item, str):
                text_chunks.append(item)
    elif isinstance(paragraphs, dict):
        text_chunks.extend([str(v) for v in paragraphs.values() if v])
    elif isinstance(paragraphs, str):
        text_chunks.append(paragraphs)
    return " ".join(text_chunks)


def format_table_for_llm(df: pd.DataFrame) -> list[str]:
    """Format HS code DataFrame rows as 'Code XXXX: Description' strings.

    Original: run_pipe.py lines 219-224.
    """
    return [f"Code {row['Code']}: {row['Description']}" for _, row in df.iterrows()]


def normalized_embeddings(texts: list[str], model) -> np.ndarray:
    """Encode texts with a SentenceTransformer and L2-normalize for cosine similarity.

    Was defined as an inner function in 4 separate places in the original code.
    """
    embeddings = model.encode(texts, show_progress_bar=False)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / norms

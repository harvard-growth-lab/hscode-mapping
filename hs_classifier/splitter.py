"""Semantic clustering and stratified train/test splits.

Uses S-BERT embeddings + HDBSCAN to cluster product descriptions by
meaning, then splits proportionally so every cluster is represented
in both train and test sets.
"""

import numpy as np
import polars as pl
from hdbscan import HDBSCAN
from sentence_transformers import SentenceTransformer


def cluster_descriptions(
    texts: list[str],
    model: SentenceTransformer,
    min_cluster_size: int = 10,
) -> np.ndarray:
    """Embed product descriptions with S-BERT and cluster with HDBSCAN.

    Args:
        texts: List of product description strings.
        model: A loaded SentenceTransformer instance.
        min_cluster_size: Minimum points to form a cluster.

    Returns:
        Array of integer cluster IDs (same length as texts).
        -1 indicates noise/outlier.
    """
    embeddings = model.encode(texts, show_progress_bar=True)
    clusterer = HDBSCAN(min_cluster_size=min_cluster_size)
    clusterer.fit(embeddings)
    return clusterer.labels_


def stratified_split(
    df: pl.DataFrame,
    cluster_col: str = "cluster",
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Stratified train/test split based on cluster labels.

    Samples proportionally from each cluster so all product types are
    represented in both splits. Noise points (cluster == -1) are split
    randomly at the same ratio.

    Args:
        df: DataFrame with a cluster column from cluster_descriptions().
        cluster_col: Name of the column containing cluster IDs.
        test_size: Fraction of data for the test set (0.0–1.0).
        random_state: Random seed for reproducibility.

    Returns:
        (train_df, test_df)
    """
    rng = np.random.default_rng(random_state)

    train_parts = []
    test_parts = []

    for cluster_id in df[cluster_col].unique().sort().to_list():
        group = df.filter(pl.col(cluster_col) == cluster_id)
        n = len(group)
        n_test = max(1, round(n * test_size)) if n > 1 else 0

        indices = rng.permutation(n)
        test_idx = indices[:n_test]
        train_idx = indices[n_test:]

        test_parts.append(group[test_idx.tolist()])
        train_parts.append(group[train_idx.tolist()])

    train_df = pl.concat(train_parts)
    test_df = pl.concat(test_parts)

    return train_df, test_df

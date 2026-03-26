"""Semantic clustering and stratified train/test splits.

Uses S-BERT embeddings + HDBSCAN to cluster product descriptions by
meaning, then splits proportionally so every cluster is represented
in both train and test sets.

Usage:
    from hs_classifier.splitter import prepare_split

    train, test = prepare_split(
        df,
        text_col="product_description",   # column to embed for clustering
        truth_col="hs_code",              # ground truth column (name varies by dataset)
        model=embed_model,                # loaded SentenceTransformer
        test_size=0.2,
    )

Input requirements:
    - A polars DataFrame with at least a text column and a ground truth column.
    - The text column is used for semantic clustering (S-BERT + HDBSCAN).
    - The truth column is carried through for downstream evaluation.
    - All other columns are preserved as-is.
    - Rows with null text are dropped before clustering.
"""

import logging

import numpy as np
import polars as pl
from hdbscan import HDBSCAN
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


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
    n_clusters = len(set(clusterer.labels_)) - (1 if -1 in clusterer.labels_ else 0)
    n_noise = (clusterer.labels_ == -1).sum()
    logger.info(f"HDBSCAN: {n_clusters} clusters, {n_noise} noise points from {len(texts)} texts")
    return clusterer.labels_


def stratified_split(
    df: pl.DataFrame,
    cluster_col: str = "_cluster",
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
        test_size: Fraction of data for the test set (0.0-1.0).
        random_state: Random seed for reproducibility.

    Returns:
        (train_df, test_df) — cluster column is dropped from both.
    """
    rng = np.random.default_rng(random_state)

    train_parts = []
    test_parts = []

    for cluster_id in df[cluster_col].unique().sort().to_list():
        group = df.filter(pl.col(cluster_col) == cluster_id)
        n = len(group)
        n_test = max(1, round(n * test_size)) if n > 1 else 0

        indices = rng.permutation(n)
        test_parts.append(group[indices[:n_test].tolist()])
        train_parts.append(group[indices[n_test:].tolist()])

    train_df = pl.concat(train_parts).drop(cluster_col)
    test_df = pl.concat(test_parts).drop(cluster_col)

    logger.info(f"Split: {len(train_df)} train, {len(test_df)} test")
    return train_df, test_df


def prepare_split(
    df: pl.DataFrame,
    text_col: str,
    truth_col: str,
    model: SentenceTransformer,
    test_size: float = 0.2,
    min_cluster_size: int = 10,
    random_state: int = 42,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """End-to-end: validate → cluster → stratified split.

    Args:
        df: Input DataFrame. Must contain text_col and truth_col.
        text_col: Column with product descriptions (used for embedding).
        truth_col: Column with ground truth HS codes (passed through).
        model: A loaded SentenceTransformer instance.
        test_size: Fraction of data for the test set.
        min_cluster_size: Minimum points for HDBSCAN cluster formation.
        random_state: Random seed for reproducibility.

    Returns:
        (train_df, test_df) with all original columns preserved.
    """
    # validate required columns exist
    for col in (text_col, truth_col):
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found. Available: {df.columns}")

    # drop rows with null text or null truth
    before = len(df)
    df = df.drop_nulls(subset=[text_col, truth_col])
    dropped = before - len(df)
    if dropped:
        logger.info(f"Dropped {dropped} rows with null {text_col} or {truth_col}")

    # cluster
    labels = cluster_descriptions(
        texts=df[text_col].to_list(),
        model=model,
        min_cluster_size=min_cluster_size,
    )
    df = df.with_columns(pl.Series("_cluster", labels))

    # split
    return stratified_split(
        df,
        cluster_col="_cluster",
        test_size=test_size,
        random_state=random_state,
    )

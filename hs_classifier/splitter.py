"""Semantic clustering and stratified sampling for evaluation.

Embeds product descriptions with S-BERT, reduces with UMAP, clusters
with HDBSCAN, then draws a stratified sample from each cluster.

The splitter only needs the text column — it embeds and clusters based
on product description semantics. All other columns (including any
ground truth HS codes) are simply carried through untouched.

Pipeline:
  1. Embed product descriptions with S-BERT (reuses the model already loaded)
  2. Reduce dimensions with UMAP
  3. Cluster with HDBSCAN
  4. Assign a ``cluster`` column to the DataFrame
  5. Stratified sample (e.g. 1%, 2%) from each cluster → eval sample

Usage:
    from hs_classifier.splitter import assign_clusters, stratified_sample

    # Step-by-step
    df = assign_clusters(df, text_col="product_description", model=embed_model)
    sample = stratified_sample(df, sample_frac=0.02)

    # Or all-in-one
    sample = prepare_eval_sample(
        df,
        text_col="product_description",
        model=embed_model,
        sample_frac=0.02,
    )

Input requirements:
    - A pandas or polars DataFrame with at least a text column (product descriptions).
    - The text column is used for semantic embedding → UMAP → HDBSCAN.
    - Rows with null text are dropped before clustering.
    - All other columns are preserved as-is in the output.
"""

import logging
import warnings
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
from hdbscan import HDBSCAN
from sentence_transformers import SentenceTransformer
from umap import UMAP

logger = logging.getLogger(__name__)

warnings.filterwarnings(
    "ignore",
    message=r"n_jobs value 1 overridden to 1 by setting random_state\..*",
    category=UserWarning,
    module="umap",
)


def _to_polars(df: Any) -> tuple[pl.DataFrame, bool]:
    if isinstance(df, pl.DataFrame):
        return df, False
    if isinstance(df, pd.DataFrame):
        return pl.from_pandas(df), True
    raise TypeError("Expected a pandas or polars DataFrame")


def _from_polars(df: pl.DataFrame, return_pandas: bool) -> Any:
    if return_pandas:
        return df.to_pandas()
    return df


def assign_clusters(
    df: Any,
    text_col: str,
    model: SentenceTransformer,
    min_cluster_size: int = 10,
    umap_n_neighbors: int = 15,
    umap_n_components: int = 10,
    random_state: int = 42,
) -> Any:
    """Embed descriptions, reduce with UMAP, cluster with HDBSCAN.

    Adds a ``cluster`` column (int) to the DataFrame.
    -1 indicates noise/outlier points.

    Args:
        df: Input DataFrame.
        text_col: Column with product descriptions to embed.
        model: A loaded SentenceTransformer instance.
        min_cluster_size: Minimum points to form an HDBSCAN cluster.
        umap_n_neighbors: UMAP locality parameter (higher = more global structure).
        umap_n_components: UMAP output dimensions (fed into HDBSCAN).
        random_state: Random seed for UMAP reproducibility.

    Returns:
        DataFrame with a new ``cluster`` column.
    """
    df, return_pandas = _to_polars(df)

    if text_col not in df.columns:
        raise ValueError(f"Column '{text_col}' not found. Available: {df.columns}")

    # drop nulls in text column
    before = len(df)
    df = df.drop_nulls(subset=[text_col])
    dropped = before - len(df)
    if dropped:
        logger.info(f"Dropped {dropped} rows with null {text_col}")

    texts = df[text_col].to_list()

    # embed
    logger.info(f"Embedding {len(texts)} descriptions...")
    embeddings = model.encode(texts, show_progress_bar=True)

    # UMAP dimensionality reduction
    logger.info(
        f"UMAP: {embeddings.shape[1]}d → {umap_n_components}d "
        f"(n_neighbors={umap_n_neighbors})"
    )
    reducer = UMAP(
        n_neighbors=umap_n_neighbors,
        n_components=umap_n_components,
        random_state=random_state,
        metric="cosine",
    )
    reduced = reducer.fit_transform(embeddings)

    # HDBSCAN clustering
    clusterer = HDBSCAN(min_cluster_size=min_cluster_size)
    clusterer.fit(reduced)
    labels = clusterer.labels_

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = int((labels == -1).sum())
    logger.info(f"HDBSCAN: {n_clusters} clusters, {n_noise} noise points")

    return _from_polars(df.with_columns(pl.Series("cluster", labels)), return_pandas)


def stratified_sample(
    df: Any,
    sample_frac: float = 0.02,
    cluster_col: str = "cluster",
    random_state: int = 42,
) -> Any:
    """Draw a stratified sample proportional to each cluster.

    Takes at least 1 row per cluster (including noise) so all semantic
    groups are represented in the eval sample.

    Args:
        df: DataFrame with a ``cluster`` column from assign_clusters().
        sample_frac: Fraction to sample (e.g. 0.01 = 1%, 0.02 = 2%).
        cluster_col: Name of the cluster column.
        random_state: Random seed for reproducibility.

    Returns:
        Sampled DataFrame (cluster column preserved for inspection).
    """
    df, return_pandas = _to_polars(df)
    rng = np.random.default_rng(random_state)
    parts = []

    for cluster_id in df[cluster_col].unique().sort().to_list():
        group = df.filter(pl.col(cluster_col) == cluster_id)
        n = len(group)
        n_sample = max(1, round(n * sample_frac))

        indices = rng.permutation(n)[:n_sample]
        parts.append(group[indices.tolist()])

    sample = pl.concat(parts)
    logger.info(
        f"Stratified sample: {len(sample)} rows from {len(df)} "
        f"({len(sample)/len(df):.1%})"
    )
    return _from_polars(sample, return_pandas)


def prepare_eval_sample(
    df: Any,
    text_col: str,
    model: SentenceTransformer,
    sample_frac: float = 0.02,
    min_cluster_size: int = 10,
    umap_n_neighbors: int = 15,
    umap_n_components: int = 10,
    random_state: int = 42,
) -> Any:
    """End-to-end: validate → cluster → stratified sample.

    Only needs the text column for clustering. All other columns
    (ground truth, metadata, etc.) are carried through untouched.

    Args:
        df: Input DataFrame. Must contain text_col.
        text_col: Column with product descriptions (used for embedding).
        model: A loaded SentenceTransformer instance.
        sample_frac: Fraction to sample (e.g. 0.01 = 1%, 0.02 = 2%).
        min_cluster_size: Minimum points for HDBSCAN cluster formation.
        umap_n_neighbors: UMAP locality parameter.
        umap_n_components: UMAP output dimensions.
        random_state: Random seed for reproducibility.

    Returns:
        Sampled DataFrame with all original columns + ``cluster`` column.
    """
    # cluster
    df = assign_clusters(
        df,
        text_col=text_col,
        model=model,
        min_cluster_size=min_cluster_size,
        umap_n_neighbors=umap_n_neighbors,
        umap_n_components=umap_n_components,
        random_state=random_state,
    )

    # sample
    return stratified_sample(
        df,
        sample_frac=sample_frac,
        random_state=random_state,
    )

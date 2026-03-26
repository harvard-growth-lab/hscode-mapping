"""Cluster product descriptions with BERTopic and produce stratified train/test splits."""

# from bertopic import BERTopic
import pandas as pd


def cluster_descriptions(texts: list[str]) -> list[int]:
    """Fit BERTopic on product descriptions and return cluster labels.

    Args:
        texts: List of product description strings.

    Returns:
        List of integer cluster/topic IDs (same length as texts).
        -1 indicates outlier (BERTopic default for unclustered points).
    """
    # TODO: fit BERTopic model
    # model = BERTopic()
    # topics, _ = model.fit_transform(texts)
    # return topics
    raise NotImplementedError


def stratified_split(
    df: pd.DataFrame,
    topic_col: str = "topic",
    test_size: float = 0.2,
    val_size: float = 0.0,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None]:
    """Stratified train/test(/val) split based on BERTopic cluster labels.

    Samples proportionally from each cluster so all product types are
    represented in every split.

    Args:
        df: DataFrame with a topic column from cluster_descriptions().
        topic_col: Name of the column containing cluster IDs.
        test_size: Fraction of data for the test set.
        val_size: Fraction for validation set. 0.0 skips the val split.
        random_state: Random seed for reproducibility.

    Returns:
        (train_df, test_df, val_df) — val_df is None if val_size=0.
    """
    # TODO: stratified split on topic_col
    raise NotImplementedError

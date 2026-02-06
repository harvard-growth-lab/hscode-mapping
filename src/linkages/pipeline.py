"""Pipeline orchestration: retrieve -> rerank -> batch classify -> output.

Ties together all modules. Also serves as the CLI entry point via main().

Original sources: run_pipe.py process_row() (lines 186-216),
classify_row() (lines 354-388), batch_classify() (lines 392-421),
classification_pipeline() (lines 426-440).
"""

import argparse
from pathlib import Path

import anthropic
import pandas as pd
import polars as pl
from openai import OpenAI
from tqdm import tqdm

from linkages.config import Settings
from linkages.embeddings import load_embedding_model, load_hs_data
from linkages.rerank import generate_search_terms, rerank_codes
from linkages.retrieval import HSIndex
from linkages.utils import extract_text_chunks


def classify_row(
    row: dict,
    hs_index: HSIndex,
    hs_data: pl.DataFrame,
    hs_descriptions: list[str],
    claude_client,
    gpt_client,
    settings: Settings,
) -> dict:
    """Classify a single product row: retrieve candidates, then rerank.

    Args:
        row: Dict with 'name', 'paragraphs', 'unique_id' keys.
        hs_index: Pre-built FAISS index.
        hs_data: Full HS reference table (polars) for description lookup.
        hs_descriptions: Pre-loaded description list for Claude prompt.
        claude_client: anthropic.Anthropic instance.
        gpt_client: openai.OpenAI instance.
        settings: Pipeline settings.

    Returns:
        Dict with classification results.

    Original: run_pipe.py process_row() + classify_row() merged.
    """
    query = row["name"]
    articles = extract_text_chunks(row["paragraphs"])

    # Stage 1: Claude generates search terms
    terms = generate_search_terms(
        query=query,
        articles=articles,
        hs_descriptions=hs_descriptions,
        client=claude_client,
        model=settings.claude_model,
    )

    # Stage 2: Multi-query FAISS search
    shortlist = hs_index.multi_search(
        query=query,
        terms=terms,
        top_k_total=settings.top_k_total,
        top_k_bert=settings.top_k_bert,
    )

    # Stage 3: GPT reranks the shortlist
    classification = rerank_codes(
        shortlist=shortlist,
        query=query,
        articles=articles,
        hs_data=hs_data,
        client=gpt_client,
        model=settings.gpt_model,
    )

    # Attach metadata
    classification["claude_terms"] = terms
    classification["retrieved_terms"] = [
        f"{r['Code']}: {r['Description']}" for _, r in shortlist.iterrows()
    ]
    classification["unique_id"] = row["unique_id"]

    return classification


def batch_classify(
    df: pl.DataFrame,
    hs_index: HSIndex,
    hs_data: pl.DataFrame,
    hs_descriptions: list[str],
    claude_client,
    gpt_client,
    settings: Settings,
    checkpoint_path: Path | None = None,
    limit: int | None = None,
) -> list[dict]:
    """Classify rows in batch with progress bar and checkpointing.

    Supports resume: if checkpoint_path exists, skips already-classified IDs.

    Original: run_pipe.py batch_classify() lines 392-421.
    """
    row_dicts = df.to_dicts()
    if limit:
        row_dicts = row_dicts[:limit]

    # Resume from checkpoint: skip already-processed rows
    done_ids = set()
    existing_results = []
    if checkpoint_path and checkpoint_path.exists():
        existing_df = pl.read_parquet(checkpoint_path)
        done_ids = set(existing_df["unique_id"].to_list())
        existing_results = existing_df.to_dicts()
        print(f"Resuming: {len(done_ids)} rows already classified, skipping them")

    results = list(existing_results)

    pending = [r for r in row_dicts if r.get("unique_id") not in done_ids]
    iterator = tqdm(pending, desc="Classifying")

    for idx, row in enumerate(iterator):
        try:
            out = classify_row(
                row, hs_index, hs_data, hs_descriptions,
                claude_client, gpt_client, settings,
            )
        except Exception as e:
            print(f"Error on row {row.get('unique_id', idx)}: {e}")
            out = {"unique_id": row.get("unique_id"), "error": str(e)}
        results.append(out)

        # Checkpoint
        if checkpoint_path and (idx + 1) % settings.checkpoint_every == 0:
            pd.DataFrame(results).to_parquet(checkpoint_path)

    return results


def main():
    """CLI entry point for batch classification."""
    parser = argparse.ArgumentParser(description="Classify products to HS codes")
    parser.add_argument("--limit", type=int, default=None, help="Max rows to classify")
    parser.add_argument("--checkpoint-path", type=str, default=None, help="Path for checkpoint parquet")
    parser.add_argument("--output-path", type=str, default=None, help="Path for final output parquet")
    args = parser.parse_args()

    settings = Settings()

    # Resolve paths
    checkpoint_path = Path(args.checkpoint_path) if args.checkpoint_path else settings.intermediate_dir / "checkpoint.parquet"
    output_path = Path(args.output_path) if args.output_path else settings.intermediate_dir / "classified_results.parquet"

    print("Loading embedding model...")
    embedding_model = load_embedding_model(settings.embedding_model_name)

    print("Building FAISS index...")
    hs_index = HSIndex(
        hs_table_path=settings.hs_table_path,
        embeddings_path=settings.embeddings_path,
        embedding_model=embedding_model,
        sheet_name=settings.hs_sheet,
        level=settings.hs_level,
    )

    # Pre-load HS descriptions for Claude prompt (done ONCE, not per row)
    hs_descriptions = load_hs_data(
        settings.hs_table_path, settings.hs_sheet, settings.hs_level
    )["Description"].tolist()

    # Full HS table as polars for description lookup in reranking
    hs_data = pl.read_excel(settings.hs_table_path, sheet_name=settings.hs_sheet)

    # Create API clients with built-in retry (replaces random sleep)
    claude_client = anthropic.Anthropic(
        api_key=settings.anthropic_api_key, max_retries=3
    )
    gpt_client = OpenAI(api_key=settings.openai_api_key, max_retries=3)

    # Load input data
    print(f"Loading data from {settings.base_df_path}...")
    base_df = pl.read_parquet(settings.base_df_path)
    base_df = base_df.with_columns(
        (pl.col("article_id").cast(pl.Utf8) + "__" + pl.col("sub_product_id").cast(pl.Utf8)).alias("unique_id")
    )

    print(f"Starting classification ({len(base_df)} total rows, limit={args.limit})")

    results = batch_classify(
        df=base_df,
        hs_index=hs_index,
        hs_data=hs_data,
        hs_descriptions=hs_descriptions,
        claude_client=claude_client,
        gpt_client=gpt_client,
        settings=settings,
        checkpoint_path=checkpoint_path,
        limit=args.limit,
    )

    results_df = pl.from_dicts(results)
    results_df.write_parquet(output_path)
    print(f"Done. Wrote {len(results_df)} results to {output_path}")

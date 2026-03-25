"""Single-row classifier entry point.

Pipeline flow for one row:
  1. Load a row from the CSV
  2. Build a query string from the row's product description
  3. Detect the language and translate to English
  4. Generate HS-vocabulary search terms via LLM
  5. Retrieve candidate HS codes via FAISS
  6. Rerank candidates and return top 2 HS codes
"""

import os
import warnings

warnings.filterwarnings("ignore", category=SyntaxWarning, module="translators")

from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

import fire
import polars as pl

from sentence_transformers import SentenceTransformer

from linkages.build_query import build_query
from linkages.reranker import rerank_codes
from linkages.retrieval import load_index, multi_search
from linkages.search_terms import generate_search_terms, load_hs_chapters
from linkages.translator import detect_language, translate_eng

INDEX_PATH = Path("data/intermediate/hs12_4_index.parquet")
CHAPTERS_PATH = Path("data/intermediate/hs2_chapters.parquet")

# --- Models (defaults from .env, overridable via CLI flags) ---
EMBEDDING_MODEL = os.environ["EMBEDDING_MODEL"]
SEARCH_TERM_MODEL = os.environ["SEARCH_TERM_MODEL"]
RERANKER_MODEL = os.environ["RERANKER_MODEL"]


def load_sample_row(
    csv_path: str = "data/raw/ecuador_sample.csv",
    row_index: int = 0,
) -> dict:
    """Load one row from the sample CSV as a dict."""
    data = pl.read_csv(csv_path)
    if row_index < 0 or row_index >= data.height:
        raise IndexError(f"Row {row_index} is out of range for {csv_path}")
    return data.row(row_index, named=True)


def translate_text(text: str) -> dict[str, str]:
    """Detect language of the input text and translate it to English."""
    detected_lang = detect_language(text)
    english_text = translate_eng(text, from_lang=detected_lang)
    return {
        "input_text": text,
        "detected_language": detected_lang,
        "english_text": english_text,
    }


def process_row(
    row: dict,
    hs_chapters: list[str],
    index_data: pl.DataFrame,
    index_codes: list[str],
    faiss_index,
    embed_model: SentenceTransformer,
) -> dict:
    """Full pipeline for one CSV row: query → translate → search terms → retrieve → rerank."""
    # extract product description and context fields from the row
    query_input = build_query(row)
    # detect language and translate the product description
    result = translate_text(query_input.query)
    result["context"] = query_input.context
    # generate search terms from the translated text
    terms = generate_search_terms(
        query=result["english_text"],
        context=result["context"],
        hs_chapters=hs_chapters,
        model=SEARCH_TERM_MODEL,
    )
    result["search_terms"] = terms
    # retrieve candidate HS codes via FAISS
    shortlist = multi_search(
        data=index_data,
        codes=index_codes,
        index=faiss_index,
        model=embed_model,
        query=result["english_text"],
        terms=terms,
        top_k_total=25,
        top_k_bert=10,
    )
    result["shortlist"] = shortlist
    # rerank candidates and pick top 2
    reranked = rerank_codes(
        shortlist=shortlist,
        query=result["english_text"],
        context=result["context"],
        model=RERANKER_MODEL,
    )
    result.update(reranked)
    return result


def classify(
    csv_path: str = "data/raw/ecuador_sample.csv",
    row_index: int = 1,
) -> None:
    """Classify a single row from the sample CSV."""
    # load resources once
    hs_chapters = load_hs_chapters(CHAPTERS_PATH)
    index_data, index_codes, faiss_index = load_index(INDEX_PATH)
    embed_model = SentenceTransformer(EMBEDDING_MODEL)

    row = load_sample_row(csv_path=csv_path, row_index=row_index)
    result = process_row(
        row,
        hs_chapters=hs_chapters,
        index_data=index_data,
        index_codes=index_codes,
        faiss_index=faiss_index,
        embed_model=embed_model,
    )
    print(result)


if __name__ == "__main__":
    fire.Fire(classify)

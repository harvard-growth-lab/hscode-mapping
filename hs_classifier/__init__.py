"""HS Code Text Classifier.

Usage:
    from hs_classifier import init_classifier, classify_row

    classifier = init_classifier()
    result = classify_row(row, classifier)
"""

import os
import warnings

warnings.filterwarnings("ignore", category=SyntaxWarning, module="translators")

from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

import polars as pl
from sentence_transformers import SentenceTransformer

from hs_classifier.build_query import build_query
from hs_classifier.reranker import rerank_codes
from hs_classifier.retrieval import load_index, multi_search
from hs_classifier.search_terms import generate_search_terms, load_hs_chapters
from hs_classifier.translator import detect_language, translate_eng

INDEX_PATH = Path("data/intermediate/hs12_4_index.parquet")
CHAPTERS_PATH = Path("data/intermediate/hs2_chapters.parquet")

EMBEDDING_MODEL = os.environ["EMBEDDING_MODEL"]
SEARCH_TERM_MODEL = os.environ["SEARCH_TERM_MODEL"]
RERANKER_MODEL = os.environ["RERANKER_MODEL"]


def init_classifier() -> dict:
    """Load all heavy resources once. Pass the result to classify_row()."""
    hs_chapters = load_hs_chapters(CHAPTERS_PATH)
    index_data, index_codes, faiss_index = load_index(INDEX_PATH)
    embed_model = SentenceTransformer(EMBEDDING_MODEL)

    return {
        "hs_chapters": hs_chapters,
        "index_data": index_data,
        "index_codes": index_codes,
        "faiss_index": faiss_index,
        "embed_model": embed_model,
    }


def _translate_text(text: str) -> dict[str, str]:
    """Detect language and translate to English."""
    detected_lang = detect_language(text)
    english_text = translate_eng(text, from_lang=detected_lang)
    return {
        "input_text": text,
        "detected_language": detected_lang,
        "english_text": english_text,
    }


def classify_row(row: dict, classifier: dict) -> dict:
    """Classify one CSV row using preloaded resources.

    Args:
        row: A single row from the CSV as a dict.
        classifier: Output of init_classifier().

    Returns:
        Dict with translation info, search terms, shortlist, top 2 codes, and reasoning.
    """
    # extract product description and context fields from the row
    query_input = build_query(row)
    # detect language and translate the product description
    result = _translate_text(query_input.query)
    result["context"] = query_input.context
    # generate search terms from the translated text
    terms = generate_search_terms(
        query=result["english_text"],
        context=result["context"],
        hs_chapters=classifier["hs_chapters"],
        model=SEARCH_TERM_MODEL,
    )
    result["search_terms"] = terms
    # retrieve candidate HS codes via FAISS
    shortlist = multi_search(
        data=classifier["index_data"],
        codes=classifier["index_codes"],
        index=classifier["faiss_index"],
        model=classifier["embed_model"],
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

"""HS Code Text Classifier.

Usage:
    from hs_classifier import init_index, init_classifier, classify_row

    init_index()              # one-time: build lookup index from Atlas DB
    classifier = init_classifier()
    result = classify_row(row, classifier)
"""

import logging
import os
import warnings
from dataclasses import dataclass

warnings.filterwarnings("ignore", category=SyntaxWarning, module="translators")

from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from sentence_transformers import SentenceTransformer

from hs_classifier.build_query import build_query
from hs_classifier.init_lookup_index import build_index, save_hs_chapters
from hs_classifier.reranker import rerank_codes
from hs_classifier.retrieval import load_index, multi_search
from hs_classifier.search_terms import generate_search_terms, load_hs_chapters
from hs_classifier.translator import detect_language, translate_eng

logger = logging.getLogger(__name__)

INDEX_PATH = Path("data/intermediate/hs12_4_index.parquet")
CHAPTERS_PATH = Path("data/intermediate/hs2_chapters.parquet")

EMBEDDING_MODEL = os.environ["EMBEDDING_MODEL"]
SEARCH_TERM_MODEL = os.environ["SEARCH_TERM_MODEL"]
RERANKER_MODEL = os.environ["RERANKER_MODEL"]
TOP_K_TOTAL = int(os.environ.get("TOP_K_TOTAL", 25))
TOP_K_BERT = int(os.environ.get("TOP_K_BERT", 10))
LLM_TEMPERATURE = float(os.environ.get("LLM_TEMPERATURE", 0.1))


@dataclass
class ClassificationResult:
    """Result of classifying a single product row."""

    code_first: str
    desc_first: str
    code_second: str
    desc_second: str
    reason: str
    search_terms: list[str]
    detected_language: str


def init_index(force: bool = False) -> None:
    """Build the HS code lookup index and chapter reference.

    Connects to the Atlas DB, generates S-BERT embeddings, and saves
    parquet files to data/intermediate/. Skips if files already exist
    unless force=True.

    Must be run once before init_classifier().
    """
    save_hs_chapters(output_path=CHAPTERS_PATH, force=force)
    build_index(
        output_path=INDEX_PATH,
        level=4,
        model_name=EMBEDDING_MODEL,
        force=force,
    )
    logger.info("Index initialization complete")


def init_classifier() -> dict:
    """Load all heavy resources once. Pass the result to classify_row()."""
    logger.info("Loading HS chapters and FAISS index...")
    hs_chapters = load_hs_chapters(CHAPTERS_PATH)
    index_data, index_codes, faiss_index = load_index(INDEX_PATH)
    embed_model = SentenceTransformer(EMBEDDING_MODEL, local_files_only=True)
    logger.info("Classifier ready")

    return {
        "hs_chapters": hs_chapters,
        "index_data": index_data,
        "index_codes": index_codes,
        "faiss_index": faiss_index,
        "embed_model": embed_model,
    }


def classify_row(row: dict, classifier: dict) -> ClassificationResult:
    """Classify one CSV row using preloaded resources.

    Args:
        row: A single row from the CSV as a dict.
        classifier: Output of init_classifier().

    Returns:
        ClassificationResult with top 2 codes, descriptions, reasoning,
        search terms, and detected language.
    """
    # extract product description and context fields from the row
    query_input = build_query(row)
    # detect language and translate to English (skips translation if already English)
    detected_lang = detect_language(query_input.query)
    english_text = translate_eng(query_input.query, from_lang=detected_lang)
    if detected_lang != "en":
        logger.info(f"Language: {detected_lang} | Original: {query_input.query[:80]}")
        logger.info(f"Translated: {english_text[:80]}")
    else:
        logger.info(f"Language: en | Query: {english_text[:80]}")
    # generate search terms from the translated text
    terms = generate_search_terms(
        query=english_text,
        context=query_input.context,
        hs_chapters=classifier["hs_chapters"],
        model=SEARCH_TERM_MODEL,
        temperature=LLM_TEMPERATURE,
    )
    logger.info(f"Search terms: {terms}")
    # retrieve candidate HS codes via FAISS
    shortlist = multi_search(
        data=classifier["index_data"],
        codes=classifier["index_codes"],
        index=classifier["faiss_index"],
        model=classifier["embed_model"],
        query=english_text,
        terms=terms,
        top_k_total=TOP_K_TOTAL,
        top_k_bert=TOP_K_BERT,
    )
    logger.info(f"Retrieved {len(shortlist)} candidate codes")
    # rerank candidates and pick top 2
    reranked = rerank_codes(
        shortlist=shortlist,
        query=english_text,
        context=query_input.context,
        model=RERANKER_MODEL,
        temperature=LLM_TEMPERATURE,
    )

    return ClassificationResult(
        code_first=reranked["code_first"],
        desc_first=reranked["desc_first"],
        code_second=reranked["code_second"],
        desc_second=reranked["desc_second"],
        reason=reranked["reason"],
        search_terms=terms,
        detected_language=detected_lang,
    )

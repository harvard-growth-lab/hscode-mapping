"""HS Code Text Classifier.

Usage:
    from hs_classifier import init_index, init_classifier, classify_row

    init_index()              # one-time: build lookup index from Atlas DB
    classifier = init_classifier()

    result = classify_row(row, classifier)                    # uses .env defaults
    result = classify_row(row, classifier,                    # override per call
        search_term_model="google/gemini-2.5-flash-lite",
        reranker_model="anthropic/claude-haiku-4-5-20251001",
        temperature=0.2,
        top_k_total=50,
    )
"""

import json
import logging
import os
import warnings
from dataclasses import asdict, dataclass

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


def _require_env(name: str) -> str:
    value = os.environ.get(name)
    if value is None or not value.strip():
        raise RuntimeError(
            f"Missing required environment variable '{name}'. "
            "Set it in your environment or .env before using hs_classifier."
        )
    return value


def _runtime_config() -> dict[str, str | int | float]:
    return {
        "embedding_model": _require_env("EMBEDDING_MODEL"),
        "search_term_model": _require_env("SEARCH_TERM_MODEL"),
        "reranker_model": _require_env("RERANKER_MODEL"),
        "intermediate_data_dir": os.environ.get("INTERMEDIATE_DATA_DIR", "data/intermediate"),
        "top_k_total": int(os.environ.get("TOP_K_TOTAL", 25)),
        "top_k_bert": int(os.environ.get("TOP_K_BERT", 10)),
        "llm_temperature": float(os.environ.get("LLM_TEMPERATURE", 0.1)),
    }


def _intermediate_paths(intermediate_data_dir: str | Path | None = None) -> tuple[Path, Path]:
    base_dir = Path(intermediate_data_dir or _runtime_config()["intermediate_data_dir"])
    return (
        base_dir / "hs12_4_index.parquet",
        base_dir / "hs2_chapters.parquet",
    )


@dataclass
class ClassificationResult:
    """Result of classifying a single product row."""

    codes: list[str]
    descriptions: list[str]
    reason: str
    search_terms: list[str]
    detected_language: str

    def to_dict(self) -> dict:
        return asdict(self)

    def __str__(self) -> str:
        return json.dumps(asdict(self), indent=2, ensure_ascii=False)


def init_index(
    force: bool = False,
    intermediate_data_dir: str | Path | None = None,
) -> None:
    """Build the HS code lookup index and chapter reference.

    Connects to the Atlas DB, generates S-BERT embeddings, and saves
    parquet files to the configured intermediate data directory. Skips
    if files already exist unless force=True.

    Must be run once before init_classifier().
    """
    config = _runtime_config()
    index_path, chapters_path = _intermediate_paths(intermediate_data_dir)
    save_hs_chapters(output_path=chapters_path, force=force)
    build_index(
        output_path=index_path,
        level=4,
        model_name=config["embedding_model"],
        force=force,
    )
    logger.info("Index initialization complete")


def init_classifier(
    intermediate_data_dir: str | Path | None = None,
) -> dict:
    """Load all heavy resources once. Pass the result to classify_row()."""
    config = _runtime_config()
    index_path, chapters_path = _intermediate_paths(intermediate_data_dir)
    logger.info("Loading HS chapters and FAISS index...")
    hs_chapters = load_hs_chapters(chapters_path)
    index_data, index_codes, faiss_index = load_index(index_path)
    embed_model = SentenceTransformer(config["embedding_model"], local_files_only=True)
    logger.info("Classifier ready")

    return {
        "hs_chapters": hs_chapters,
        "index_data": index_data,
        "index_codes": index_codes,
        "faiss_index": faiss_index,
        "embed_model": embed_model,
    }


def classify_row(
    row: dict,
    classifier: dict,
    search_term_model: str | None = None,
    reranker_model: str | None = None,
    temperature: float | None = None,
    top_k_total: int | None = None,
    top_k_bert: int | None = None,
    top_n: int = 2,
) -> ClassificationResult:
    """Classify one CSV row using preloaded resources.

    All keyword arguments override the .env defaults for this call only,
    making it easy to compare models or tune retrieval without editing config.

    Args:
        row: A single row from the CSV as a dict.
        classifier: Output of init_classifier().
        search_term_model: LLM for search term generation (default: SEARCH_TERM_MODEL from .env).
        reranker_model: LLM for reranking (default: RERANKER_MODEL from .env).
        temperature: LLM temperature (default: LLM_TEMPERATURE from .env).
        top_k_total: Total FAISS candidates to retrieve (default: TOP_K_TOTAL from .env).
        top_k_bert: Candidates allocated to the raw query (default: TOP_K_BERT from .env).
        top_n: Number of top HS codes to return (default: 2).

    Returns:
        ClassificationResult with top N codes, descriptions, reasoning,
        search terms, and detected language.
    """
    config = _runtime_config()
    # resolve defaults from .env
    _search_term_model = search_term_model or config["search_term_model"]
    _reranker_model = reranker_model or config["reranker_model"]
    _temperature = temperature if temperature is not None else config["llm_temperature"]
    _top_k_total = top_k_total if top_k_total is not None else config["top_k_total"]
    _top_k_bert = top_k_bert if top_k_bert is not None else config["top_k_bert"]

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
        model=_search_term_model,
        temperature=_temperature,
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
        top_k_total=_top_k_total,
        top_k_bert=_top_k_bert,
    )
    logger.info(f"Retrieved {len(shortlist)} candidate codes")
    # rerank candidates and pick top N
    reranked = rerank_codes(
        shortlist=shortlist,
        query=english_text,
        context=query_input.context,
        model=_reranker_model,
        temperature=_temperature,
        top_n=top_n,
    )

    return ClassificationResult(
        codes=reranked["codes"],
        descriptions=reranked["descriptions"],
        reason=reranked["reason"],
        search_terms=terms,
        detected_language=detected_lang,
    )

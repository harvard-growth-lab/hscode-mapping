"""Classify a product description string to HS codes.

Usage:
    from pipeline import Classifier
    clf = Classifier()
    result = clf.classify("solar panel inverter")
    # {"code_first": "8504", "desc_first": "...", "code_second": "8541", "desc_second": "...", "reason": "..."}
"""

import anthropic
import polars as pl
from openai import OpenAI

from modules.config import Settings
from modules.lookup_index import HSIndex, load_embedding_model, load_hs_data
from modules.reranker import rerank_codes
from modules.retrieval import multi_search
from modules.search_terms import generate_search_terms
from modules.translator import translate_to_english


def _extract_text(paragraphs) -> str:
    """Flatten nested paragraph structures into a plain string."""
    chunks = []
    if isinstance(paragraphs, list):
        for item in paragraphs:
            if isinstance(item, dict):
                chunks.extend([str(v) for v in item.values() if v])
            elif isinstance(item, str):
                chunks.append(item)
    elif isinstance(paragraphs, dict):
        chunks.extend([str(v) for v in paragraphs.values() if v])
    elif isinstance(paragraphs, str):
        chunks.append(paragraphs)
    return " ".join(chunks)


class Classifier:
    """Loads all models and indexes once, then classifies product strings on demand."""

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or Settings()

        print("Loading embedding model...")
        embedding_model = load_embedding_model(self.settings.embedding_model_name)

        print("Building FAISS index...")
        self.hs_index = HSIndex(
            hs_table_path=self.settings.hs_table_path,
            embeddings_path=self.settings.embeddings_path,
            embedding_model=embedding_model,
            sheet_name=self.settings.hs_sheet,
            level=self.settings.hs_level,
        )

        self.hs_descriptions = load_hs_data(
            self.settings.hs_table_path, self.settings.hs_sheet, self.settings.hs_level
        )["Description"].tolist()

        self.hs_data = pl.read_excel(
            self.settings.hs_table_path, sheet_name=self.settings.hs_sheet
        )

        self.claude = anthropic.Anthropic(
            api_key=self.settings.anthropic_api_key, max_retries=3
        )
        self.gpt = OpenAI(api_key=self.settings.openai_api_key, max_retries=3)

    def classify(self, text: str, context: str = "") -> dict:
        """Classify a product description string to HS codes.

        Args:
            text: Product name or short description.
            context: Optional article/document text about the product.
                     Can also be a nested paragraph structure (list of dicts etc.)
                     which will be flattened automatically.

        Returns:
            Dict with code_first, desc_first, code_second, desc_second, reason,
            claude_terms, retrieved_codes.
        """
        if not isinstance(context, str):
            context = _extract_text(context)

        # Stage 0: Detect language and translate to English if needed
        text, detected_lang = translate_to_english(text)

        # Stage 1: Claude generates HS-vocabulary search terms
        terms = generate_search_terms(
            query=text,
            articles=context,
            hs_descriptions=self.hs_descriptions,
            client=self.claude,
            model=self.settings.claude_model,
        )

        # Stage 2: Embed each term, search FAISS, aggregate
        shortlist = multi_search(
            hs_index=self.hs_index,
            query=text,
            terms=terms,
            top_k_total=self.settings.top_k_total,
            top_k_bert=self.settings.top_k_bert,
        )

        # Stage 3: GPT reranks the shortlist
        result = rerank_codes(
            shortlist=shortlist,
            query=text,
            articles=context,
            hs_data=self.hs_data,
            client=self.gpt,
            model=self.settings.gpt_model,
        )

        result["detected_lang"] = detected_lang
        result["claude_terms"] = terms
        result["retrieved_codes"] = [
            f"{r['Code']}: {r['Description']}" for _, r in shortlist.iterrows()
        ]

        return result

"""Run the HS classification pipeline end to end or by individual stage."""

import argparse
import json
from pathlib import Path

import pandas as pd
import polars as pl

from linkages.config import Settings
from linkages.lookup_index import HSIndex, load_embedding_model, load_hs_data
from linkages.reranker import rerank_codes
from linkages.retrieval import multi_search
from linkages.search_terms import generate_search_terms
from linkages.translator import normalize_text, translate_to_english


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
    return normalize_text(" ".join(chunks))


def _shortlist_to_records(shortlist: pd.DataFrame) -> list[dict]:
    return shortlist[["Code", "Description"]].to_dict(orient="records")


def _load_row_text(
    csv_path: str | Path,
    row: int,
    text_column: str,
    context_column: str | None = None,
) -> tuple[str, str]:
    data = pl.read_csv(csv_path)
    if row < 0 or row >= data.height:
        raise IndexError(f"Row {row} is out of range for {csv_path} with {data.height} rows")

    record = data.row(row, named=True)
    text = normalize_text(record.get(text_column, ""))
    context = normalize_text(record.get(context_column, "")) if context_column else ""
    return text, context


class Classifier:
    """Loads shared resources once and exposes stage-level runners."""

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or Settings()
        self._embedding_model = None
        self._hs_index = None
        self._hs_descriptions = None
        self._hs_data = None
        self._term_client = None
        self._rerank_client = None

    def _ensure_index(self) -> HSIndex:
        if self._hs_index is None:
            print("Loading embedding model...")
            self._embedding_model = load_embedding_model(self.settings.embedding_model_name)

            print("Building FAISS index...")
            self._hs_index = HSIndex(
                hs_table_path=self.settings.hs_table_path,
                embeddings_path=self.settings.embeddings_path,
                embedding_model=self._embedding_model,
                sheet_name=self.settings.hs_sheet,
                level=self.settings.hs_level,
            )
        return self._hs_index

    def _ensure_hs_descriptions(self) -> list[str]:
        if self._hs_descriptions is None:
            self._hs_descriptions = load_hs_data(
                self.settings.hs_table_path,
                self.settings.hs_sheet,
                self.settings.hs_level,
            )["Description"].tolist()
        return self._hs_descriptions

    def _ensure_hs_data(self) -> pl.DataFrame:
        if self._hs_data is None:
            self._hs_data = pl.read_excel(
                self.settings.hs_table_path,
                sheet_name=self.settings.hs_sheet,
            )
        return self._hs_data

    def _build_llm_client(self):
        provider = self.settings.llm_provider
        if provider == "anthropic":
            import anthropic

            return anthropic.Anthropic(api_key=self.settings.llm_api_key, max_retries=3)

        if provider == "openai":
            from openai import OpenAI

            return OpenAI(api_key=self.settings.llm_api_key, max_retries=3)

        raise NotImplementedError(
            f"LLM provider '{provider}' is not wired into pipeline.py yet"
        )

    def _ensure_term_client(self):
        if self._term_client is None:
            self._term_client = self._build_llm_client()
        return self._term_client

    def _ensure_rerank_client(self):
        if self._rerank_client is None:
            self._rerank_client = self._build_llm_client()
        return self._rerank_client

    def run_translation(self, text: str) -> dict:
        translated_text, detected_lang = translate_to_english(text)
        return {
            "input_text": normalize_text(text),
            "translated_text": translated_text,
            "detected_lang": detected_lang,
        }

    def run_term_generation(self, text: str, context: str = "") -> dict:
        translated_text, detected_lang = translate_to_english(text)
        context_text = context if isinstance(context, str) else _extract_text(context)
        terms = generate_search_terms(
            query=translated_text,
            articles=context_text,
            hs_descriptions=self._ensure_hs_descriptions(),
            client=self._ensure_term_client(),
            model=self.settings.term_generation_model,
        )
        return {
            "input_text": normalize_text(text),
            "translated_text": translated_text,
            "detected_lang": detected_lang,
            "terms": terms,
        }

    def run_retrieval(self, text: str, context: str = "", terms: list[str] | None = None) -> dict:
        translated_text, detected_lang = translate_to_english(text)
        if terms is None:
            terms = self.run_term_generation(text, context)["terms"]

        shortlist = multi_search(
            hs_index=self._ensure_index(),
            query=translated_text,
            terms=terms,
            top_k_total=self.settings.top_k_total,
            top_k_bert=self.settings.top_k_bert,
        )
        return {
            "input_text": normalize_text(text),
            "translated_text": translated_text,
            "detected_lang": detected_lang,
            "terms": terms,
            "shortlist": _shortlist_to_records(shortlist),
        }

    def run_reranking(self, text: str, context: str = "", terms: list[str] | None = None) -> dict:
        retrieval = self.run_retrieval(text, context, terms=terms)
        shortlist = pd.DataFrame(retrieval["shortlist"])
        result = rerank_codes(
            shortlist=shortlist,
            query=retrieval["translated_text"],
            articles=context if isinstance(context, str) else _extract_text(context),
            hs_data=self._ensure_hs_data(),
            client=self._ensure_rerank_client(),
            model=self.settings.reranker_model,
        )
        result["detected_lang"] = retrieval["detected_lang"]
        result["terms"] = retrieval["terms"]
        result["retrieved_codes"] = [
            f"{row['Code']}: {row['Description']}" for row in retrieval["shortlist"]
        ]
        return result

    def classify(self, text: str, context: str = "") -> dict:
        return self.run_reranking(text=text, context=context)

    def run_stage(self, stage: str, text: str, context: str = "") -> dict:
        if stage == "translate":
            return self.run_translation(text)
        if stage == "terms":
            return self.run_term_generation(text, context)
        if stage == "retrieve":
            return self.run_retrieval(text, context)
        if stage in {"rerank", "classify"}:
            return self.run_reranking(text, context)
        raise ValueError(f"Unknown stage: {stage}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the HS pipeline by stage")
    parser.add_argument(
        "--stage",
        choices=["translate", "terms", "retrieve", "rerank", "classify"],
        default="translate",
        help="Pipeline stage to run",
    )
    parser.add_argument("--text", help="Direct product description input")
    parser.add_argument("--context", default="", help="Optional direct context text")
    parser.add_argument(
        "--csv-path",
        default="data/raw/ecuador_sample.csv",
        help="CSV file to read sample rows from when --text is omitted",
    )
    parser.add_argument("--row", type=int, default=0, help="CSV row index to run")
    parser.add_argument(
        "--text-column",
        default="product_description",
        help="CSV column used as the product text",
    )
    parser.add_argument(
        "--context-column",
        default="container_description",
        help="CSV column used as optional context",
    )
    args = parser.parse_args()

    if args.text:
        text = normalize_text(args.text)
        context = normalize_text(args.context)
    else:
        text, context = _load_row_text(
            csv_path=args.csv_path,
            row=args.row,
            text_column=args.text_column,
            context_column=args.context_column,
        )

    classifier = Classifier()
    result = classifier.run_stage(args.stage, text=text, context=context)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

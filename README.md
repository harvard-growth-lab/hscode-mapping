# HS Code Classifier

Takes a product description string and returns the best-matching Harmonized System (HS) trade codes.

## How it works

```mermaid
flowchart TD
    subgraph setup ["Setup (run once)"]
        DB[("Atlas DB\nclassification.product_hs12")]
        EMB["init_lookup_index.py\nS-BERT embeddings"]
        IDX[("hs12_4_index.parquet\ncode + description + embedding")]
        CH[("hs2_chapters.parquet\n97 HS chapter descriptions")]
        DB --> EMB --> IDX
        DB --> CH
    end

    subgraph classify ["run_pipeline.py"]
        A["CSV row"] --> BQ["build_query.py\nextract product description\n+ shipping context"]
        BQ --> T["translator.py\nlanguage detection"]
        T -->|not English| TR["translate to English"]
        T -->|already English| ST
        TR --> ST["search_terms.py\nLLM + HS2 chapters\n→ 5-8 search terms"]
        CH --> ST
        ST --> R["retrieval.py\nembed each term\n+ FAISS search"]
        IDX --> R
        R --> AGG["aggregate +\ndeduplicate\n~25 candidates"]
        AGG --> RR["reranker.py"]
        RR --> OUT["top 2 HS codes\n+ reasoning"]
    end
```

**Stage 0 — Language detection** (`linkages/translator.py`)
Input text is detected for language using Lingua. Non-English text is translated via the `translators` package (Google backend).

**Stage 1 — Search term generation** (`linkages/search_terms.py`)
The LLM receives the product string, shipping context, and the 97 HS2 chapter descriptions as guidance. It generates 5-8 search terms using HS vocabulary that will match well in the embedding space. Uses Instructor with a Pydantic model for structured output. Provider-agnostic via `instructor.from_provider()`.

**Stage 2 — Retrieval** (`linkages/retrieval.py`)
The original query and each generated term are independently embedded and searched against a FAISS index of HS code descriptions. Results are pooled and deduplicated, yielding ~25 candidate codes.

**Stage 3 — Reranking** (`linkages/reranker.py`)
The LLM receives the shortlist and selects the top 2 HS codes with a short justification. Uses Instructor with a Pydantic model for structured output. Provider-agnostic via `instructor.from_provider()`.

## Project structure

```
run_init.py               # One-time setup: build lookup index from Atlas DB
run_pipeline.py           # Classify a single CSV row (Fire CLI)

linkages/
├── init_lookup_index.py  # DB connection, S-BERT encoding, save index parquet
├── build_query.py        # Build one classifier query from one raw row
├── translator.py         # Lingua language detection + Google translation backend
├── search_terms.py       # LLM search term generation (Instructor + Pydantic)
├── retrieval.py          # Load index parquet, FAISS search, aggregate and deduplicate
└── reranker.py           # LLM reranking of candidates (Instructor + Pydantic)

data/
├── raw/                  # Sample CSV data (e.g. ecuador_sample.csv)
└── intermediate/         # hs12_4_index.parquet + hs2_chapters.parquet
```

## Branches

- `main` keeps the MVP classifier: load HS data, retrieve candidates, rerank, and return top HS codes. Provider abstraction (Instructor + Pydantic) is complete.
- `evals` is for split construction, labeling, metrics, notebooks, and benchmark workflow.

## Setup

```bash
uv sync
cp .env.example .env  # fill in API keys, Atlas DB credentials, and model choices
```

### Initialize the lookup index

Pulls HS code descriptions from the Atlas DB, generates S-BERT embeddings, and saves two parquets (HS4 index with embeddings + HS2 chapters for prompt context):

```bash
uv run run_init.py            # skips if parquet already exists
uv run run_init.py --force    # rebuild
```

### Classify a row

```bash
uv run run_pipeline.py                          # default: row 1 from ecuador_sample
uv run run_pipeline.py --row_index 5            # different row
uv run run_pipeline.py --csv_path data/raw/other.csv --row_index 0
```

## Models

All models are configured in `.env` (see `.env.example`):

| `.env` variable | Role | Default |
|---|---|---|
| `EMBEDDING_MODEL` | S-BERT embeddings for FAISS index | `dell-research-harvard/lt-un-data-fine-fine-en` |
| `SEARCH_TERM_MODEL` | LLM for search term generation | `google/gemini-2.5-flash-lite` |
| `RERANKER_MODEL` | LLM for reranking candidates | `google/gemini-2.5-flash-lite` |

LLM models use `instructor.from_provider()`, so any supported provider string works (e.g. `anthropic/claude-sonnet-4-20250514`, `cohere/command-r-plus`).

## Future improvements

- **DeepL for translation (optional):** The current translator uses the `translators` package with the Google backend. A potential upgrade is to use the DeepL API directly (free plan available) for better translation quality, especially on trade/product descriptions.
- **Vector DB (optional):** FAISS works well at the current scale (~1,200 HS4 codes). A managed vector DB like Qdrant or LanceDB would only be worth it if we need persistence, filtering, or incremental updates at much larger scale.

## Notes

This is a rewrite of an earlier monolithic script. Key differences:

- **FAISS built once** — the original rebuilt the index on every query (~48,000 times per full run)
- **No hardcoded secrets** — API keys now loaded from `.env`
- **HS data from Atlas DB** — no longer depends on a local Excel file
- **Flat structure** — original was a single 440-line script; now split into focused modules

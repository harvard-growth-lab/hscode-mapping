# Linkages: Product-to-HS-Code Classifier

Takes a product description string and returns the best-matching Harmonized System (HS) trade codes.

```python
from pipeline import Classifier

clf = Classifier()  # loads models once
result = clf.classify("solar panel inverter", context="...article text...")
# {
#   "code_first":      "8504",
#   "desc_first":      "Electrical transformers, static converters and inductors",
#   "code_second":     "8541",
#   "desc_second":     "Semiconductor devices",
#   "reason":          "...",
#   "detected_lang":   "en",
#   "claude_terms":    ["inverter", "power converter", ...],
#   "retrieved_codes": ["8504: ...", "8541: ...", ...]
# }
```

## How it works

```mermaid
flowchart TD
    subgraph setup ["Setup (once)"]
        HS[("HSCodeandDescription.xlsx")]
        EMB["lookup_index.py\nS-BERT embeddings"]
        IDX[("FAISS index\n~1,900 HS codes")]
        HS --> EMB --> IDX
    end

    subgraph classify [".classify(text, context)"]
        A["product string"] --> T["translator.py\nlanguage detection"]
        T -->|not English| TR["translate to English"]
        T -->|already English| ST
        TR --> ST["search_terms.py\nClaude Haiku\n→ 5-8 search terms"]
        ST --> R["retrieval.py\nembed each term\n+ FAISS search"]
        IDX --> R
        R --> AGG["aggregate +\ndeduplicate\n~25 candidates"]
        AGG --> RR["reranker.py\nGPT-4o-mini"]
        RR --> OUT["top 2 HS codes\n+ reasoning"]
    end
```

**Stage 0 — Language detection** (`modules/translator.py`)
Input text is detected for language. If not English it is translated before entering the pipeline.

**Stage 1 — Thesaurus / search term generation** (`modules/search_terms.py`)
Claude receives the product string and optional context and generates 5-8 search terms drawn from the HS vocabulary — generic product class names that will match well in the embedding space.

**Stage 2 — Retrieval** (`modules/retrieval.py`)
The original query and each generated term are independently embedded and searched against a FAISS index of HS code descriptions. Results are pooled and deduplicated, yielding ~25 candidate codes.

**Stage 3 — Reranking** (`modules/reranker.py`)
GPT-4o-mini receives the shortlist and selects the top 2 HS codes with a short justification.

The FAISS index is built once when `Classifier()` is initialised and reused across all `.classify()` calls.

## Project structure

```
pipeline.py               # Classifier class — the public interface

modules/
├── config.py             # Settings dataclass: API keys, model names, paths, parameters
├── lookup_index.py       # Load HS descriptions, generate/load S-BERT embeddings, build FAISS index
├── translator.py         # Language detection and translation to English (langdetect)
├── llm.py                # Unified LLM call interface (provider-agnostic, swap here)
├── search_terms.py       # Prompt + tool schema for search term generation
├── retrieval.py          # Embed terms, search FAISS, aggregate and deduplicate
├── reranker.py           # Prompt + tool schema for reranking
├── splitter.py           # BERTopic clustering + stratified train/test split  [placeholder]
├── evaluator.py          # Accuracy metrics, chapter-level confusion matrix    [placeholder]
└── labeler.py            # Label Studio export/import + ground truth management [placeholder]

scripts/
├── 0_load_data.py        # Pull article data from MongoDB → parquet (one-time)
├── 1_generate_embeddings.py  # Encode HS descriptions → .npy (one-time)
└── run_classification.sh     # SLURM job script

data/
├── raw/                  # HSCodeandDescription.xlsx
└── intermediate/         # hs12_4_embeddings.npy, base_df.parquet
```

## Setup

```bash
uv sync
cp .env.example .env  # fill in OPENAI_API_KEY and ANTHROPIC_API_KEY
```

### Required data files

```
data/raw/HSCodeandDescription.xlsx        # HS code reference table (sheet "HS12")
data/intermediate/hs12_4_embeddings.npy   # pre-computed S-BERT embeddings
```

Generate the embeddings once after placing the Excel file:

```bash
uv run python scripts/1_generate_embeddings.py
```

## Evaluation workflow (in progress)

The planned evaluation flow:

1. **Cluster** — `splitter.py` uses BERTopic to group product descriptions semantically, then produces a stratified train/test split so all product types are represented in both sets.
2. **Label** — `labeler.py` exports unlabeled rows to Label Studio for annotation and imports completed labels back. Tracks label provenance (existing ground truth vs newly annotated).
3. **Evaluate** — `evaluator.py` computes top-1 accuracy, top-2 accuracy (either of the two returned codes matches), and hierarchical accuracy at the 2-digit chapter level, plus a chapter-level confusion matrix.

## Models

| Role | Model |
|---|---|
| Embeddings | `dell-research-harvard/lt-un-data-fine-fine-en` (S-BERT, trade concordance fine-tune) |
| Term generation | Claude 3.5 Haiku |
| Reranking | GPT-4o-mini |

Provider switching is isolated to `modules/llm.py`. Both term generation and reranking route through a single `call()` function there, so swapping to a different provider (or using one model for both) only requires changes in that file. The roadmap is to replace the current Anthropic/OpenAI branches with [LiteLLM](https://github.com/BerriAI/litellm) for a unified interface across providers.

## Notes

This is a rewrite of an earlier monolithic script. Key differences:

- **FAISS built once** — the original rebuilt the index on every query (~48,000 times per full run)
- **No hardcoded secrets** — API keys now loaded from `.env`
- **Flat structure** — original was a single 440-line script; now split into focused modules

# Linkages: LLM-Powered Product-to-HS-Code Classification

Maps company product descriptions to Harmonized System (HS) trade codes using a retrieve-then-rerank pipeline: Claude generates search terms, FAISS finds candidate HS codes via semantic similarity, and GPT reranks to select the best match.

## Setup

```bash
uv sync
cp .env.example .env  # Fill in your API keys
```

### Data

Place the following files in `data/`:
```
data/
├── raw/HSCodeandDescription.xlsx    # HS code reference table
└── intermediate/
    ├── base_df.parquet              # Input data (from MongoDB via scripts/0_load_data.py)
    └── hs12_4_embeddings.npy        # Pre-computed HS4 embeddings (via scripts/1_generate_embeddings.py)
```

These are copied from the original project at `/n/holylfs05/LABS/hausmann_lab/Lab/kdaryanani/linkages/`.

## Usage

### Classify products
```bash
# Classify all rows
linkages-classify

# Classify first 10 rows with checkpointing
linkages-classify --limit 10 --checkpoint-path data/intermediate/checkpoint.parquet
```

### Generate embeddings (one-time)
```bash
uv run python scripts/1_generate_embeddings.py
```

### Load data from MongoDB (one-time)
```bash
uv run --group scripts python scripts/0_load_data.py
```

### Run on SLURM
```bash
sbatch run_classification.sh
```

## Package Structure

```
src/linkages/
├── config.py       # Settings dataclass, loads .env, resolves paths
├── utils.py        # Text extraction, formatting helpers
├── embeddings.py   # Generate and load HS code embeddings (S-BERT)
├── retrieval.py    # HSIndex class — FAISS index built ONCE, queried many times
├── rerank.py       # Claude term generation + GPT reranking via tool use
└── pipeline.py     # Orchestration: retrieve -> rerank -> batch classify -> output
```

### Pipeline Flow

1. **Claude term generation**: For each product, Claude Haiku receives the product name + article context and generates 5-8 search terms via tool use.
2. **FAISS semantic search**: The search terms + original query drive a multi-query FAISS search over S-BERT embeddings of HS code descriptions, returning ~25 candidate codes.
3. **GPT reranking**: GPT-4o-mini selects the top 2 HS codes from the shortlist with reasoning (structured output via tool use).
4. **Batch processing**: Results are checkpointed to parquet every 10 rows for crash recovery.

### Key Improvements Over Original

- **FAISS index built once** — the original rebuilt it on every query (~48,000 times). Now built once at startup.
- **API keys in .env** — no more hardcoded secrets
- **SDK retry** — replaces `time.sleep(random.randint(7, 25))` with built-in exponential backoff on rate limits
- **Checkpoint resume** — skips already-classified rows on restart
- **Clean dependencies** — dropped unused packages (networkx, langchain, connectorx, fastexcel)

## Models

- **Embeddings**: `dell-research-harvard/lt-un-data-fine-fine-en` (S-BERT fine-tuned on concordance data)
- **Term generation**: Claude 3.5 Haiku
- **Reranking**: GPT-4o-mini

## Original Project

This is a clean rewrite of `/n/holylfs05/LABS/hausmann_lab/Lab/kdaryanani/linkages/`. The original directory is left untouched.

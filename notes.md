# Linkages Package - Development Notes

## Status: In Progress

### What's done
- Full package skeleton created at `/n/holylfs05/LABS/hausmann_lab/Lab/kdaryanani/linkages-pkg/`
- `pyproject.toml` with src layout, trimmed deps, hatchling build
- `.gitignore`, `.env` template, `.env.example`
- All 6 source modules written:
  - `src/linkages/__init__.py`
  - `src/linkages/config.py` — Settings dataclass, .env loading, path resolution
  - `src/linkages/utils.py` — extract_text_chunks, format_table_for_llm, normalized_embeddings
  - `src/linkages/embeddings.py` — generate/load embeddings, load HS data
  - `src/linkages/retrieval.py` — HSIndex class (FAISS built once, not 48k times)
  - `src/linkages/rerank.py` — Claude term gen + GPT reranking, tool schemas cleaned up
  - `src/linkages/pipeline.py` — batch classify with checkpoint resume, CLI entry point
- All 3 scripts written:
  - `scripts/0_load_data.py` — MongoDB loader, URI from env var
  - `scripts/1_generate_embeddings.py` — CLI wrapper
  - `scripts/2_classify.py` — thin wrapper for pipeline.main()
- `run_classification.sh` — SLURM script, updated paths, reduced mem to 32G, 8h wall time
- `README.md` — full docs
- Data files copied to `data/raw/` and `data/intermediate/`
- `uv sync` completed (103 packages resolved)
- Git initialized on main branch

### What's NOT done yet
- **No git commit yet** — all files are unstaged
- **Smoke test not run** — haven't verified `from linkages.config import Settings` works
- **.env has placeholder keys** — need real API keys to test end-to-end
- **No remote repo** — git init only, no GitHub remote
- **API key rotation** — the old project has exposed keys in plain text across 4+ files. Those keys should be rotated (new keys generated, old ones revoked)

### Key decisions made
- Fresh directory (old `linkages/` left untouched)
- No config.yaml — just .env + Settings dataclass + argparse
- Dropped networkx (graph traversal was already removed in run_pipe.py)
- Dropped langchain, connectorx, fastexcel (never imported)
- Claude tool schema: term6-8 made optional (were incorrectly all required)
- SDK built-in retry replaces `time.sleep(random.randint(7, 25))`

### Source mapping
All code extracted from `run_pipe.py` (441 lines) — the most current version.

| New module | Source lines in run_pipe.py |
|---|---|
| config.py | 24-36, 186, 354-363 (hardcoded values) |
| utils.py | 45-48, 191-203, 219-224 |
| embeddings.py | 56-63 + 1_hs_embeddings.py |
| retrieval.py | 41-109 |
| rerank.py | 112-183, 219-342 |
| pipeline.py | 186-216, 345-440 |
| scripts/0_load_data.py | 0_load_bruegel.py |

### Known issues in original code
- API keys hardcoded in plain text in run_pipe.py:24-26, helper.py:18-19, 1_llm_linkage/helper.py (3 different key sets)
- MongoDB URI with password in 0_load_bruegel.py:7
- FAISS index rebuilt on every query call (~48k rebuilds per full run)
- HS Excel table re-read on every query in standard_retrieval() AND in terms_generator_claude_prompt()
- `to_country` field in panjiva data is messy for US/UY (contains port names, not country names)
- `in_2023.parquet` in panjiva data is corrupted (skip it)

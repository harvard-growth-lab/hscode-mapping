# Known Issues

## Dependency Warnings

- **tqdm**: `IProgress not found` warning. Fix with `pip install ipywidgets`.
- **translators**: `SyntaxWarning: invalid escape sequence '\.'` in `server.py` and `server_async.py`. Bug in the `translators` package (unescaped regex). Harmless, needs upstream fix.
- **hdbscan**: `SyntaxWarning: invalid escape sequence '\{'` in `robust_single_linkage_.py:175`. Unescaped LaTeX in a docstring. Harmless, needs upstream fix.

## Gotchas

- `run_splitter.py` now writes default eval samples under `INTERMEDIATE_DATA_DIR/samples/`. If you bypass the CLI and call `write_csv()` yourself, you are still responsible for creating whatever output directory you choose.
- HS code columns get read as integer by default (both Polars and pandas), stripping leading zeros (e.g. `0714` → `714`). HS codes must be read as strings. In Polars: `schema_overrides={"hs_code": pl.Utf8}`. In pandas: `dtype={"hs_code": str}`.
- `init_index()` creates the configured `INTERMEDIATE_DATA_DIR` if it does not exist and writes parquet artifacts there. This is now configurable, but still a write-time side effect users should be aware of.
- `generate_search_terms()` is assumed to return at least one term. The retrieval code now guards against an empty list, but an empty term set still reduces recall because only the raw query search runs.
- `_build_db_uri()` in `hs_classifier/init_lookup_index.py` interpolates credentials directly into the PostgreSQL URI. If the username or password contains reserved URL characters like `@`, `:`, or `/`, the DB connection can fail. Proper fix: URL-encode credentials before building the URI.

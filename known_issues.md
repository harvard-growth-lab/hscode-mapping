# Known Issues

## Dependency Warnings

- **tqdm**: `IProgress not found` warning. Fix with `pip install ipywidgets`.
- **translators**: `SyntaxWarning: invalid escape sequence '\.'` in `server.py` and `server_async.py`. Bug in the `translators` package (unescaped regex). Harmless, needs upstream fix.
- **hdbscan**: `SyntaxWarning: invalid escape sequence '\{'` in `robust_single_linkage_.py:175`. Unescaped LaTeX in a docstring. Harmless, needs upstream fix.
- **umap**: `UserWarning: n_jobs value 1 overridden to 1 by setting random_state`. Harmless, just informational about parallelism being disabled when a seed is set.

## Gotchas

- `prepare_eval_sample` + `write_csv` requires `data/raw/` to exist beforehand. The package doesn't create it automatically.
- HS code columns get read as integer by default (both Polars and pandas), stripping leading zeros (e.g. `0714` → `714`). HS codes must be read as strings. In Polars: `schema_overrides={"hs_code": pl.Utf8}`. In pandas: `dtype={"hs_code": str}`.
- `init_index()` silently creates a `data/intermediate/` folder with intermediate files. Should document this side effect or at least log a message when the directory is created.

## Documentation

- README evaluation example only captures `codes` from `ClassificationResult` but drops `reason`, `descriptions`, `search_terms`, and `detected_language`. The example should show how to include at least `reason` in `results_df` so users know the reasoning is available.
- README examples use Polars, but most users will expect pandas. Should provide pandas examples (or both).

## Connection Errors

The `translators` package leaks HTTP connections. It creates `requests.Session` objects but never calls `.close()` on them. Sessions auto-rotate every 1000 queries or 25 minutes, abandoning the old session without cleanup. This causes `ConnectionError: Connection max age expired` when the remote server times out a stale connection.

`hs_classifier/translator.py` calls `ts.translate_text()` with no retry or session management on its side.

**Workaround:** Re-run the failing cell — the stale session gets replaced on the next call.

**Proper fix:** Add retry logic around translation calls in `hs_classifier`, and/or file an issue upstream on the `translators` package for the missing `.close()` calls.

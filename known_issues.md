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
- `evaluation_report` output is hard to interpret. The "confusion matrix" is a full multi-class cross-tabulation (true chapter × predicted chapter), not a standard 2×2 correct/incorrect matrix. With small samples (e.g. 18 rows) and many classes (15+ chapters), the resulting sparse matrix is mostly zeros and not very useful. Consider a user-friendly summary instead, e.g.:

1. **Plain counts**: "8/18 top-1 matches, 10/18 top-2 matches, 10/18 correct chapter"
2. **Miss table**: Show only the rows that missed — product description, predicted code, truth code — so users can see where the classifier struggles.
3. **2×2 confusion matrix** per level (code and chapter):

```
            Correct  Incorrect
Top-1          8        10
Top-2         10         8
Chapter       10         8
```

This is far more intuitive than a sparse multi-class cross-tabulation at any sample size.

## Connection Errors

The `translators` package leaks HTTP connections. It creates `requests.Session` objects but never calls `.close()` on them. Sessions auto-rotate every 1000 queries or 25 minutes, abandoning the old session without cleanup. This causes `ConnectionError: Connection max age expired` when the remote server times out a stale connection.

`hs_classifier/translator.py` calls `ts.translate_text()` with no retry or session management on its side.

**Workaround:** Re-run the failing cell — the stale session gets replaced on the next call.

**Proper fix:** Add retry logic around translation calls in `hs_classifier`, and/or file an issue upstream on the `translators` package for the missing `.close()` calls.

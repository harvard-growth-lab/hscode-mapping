# `search_terms.py` Instructions

## Purpose

`search_terms.py` should take one classifier-ready query string and optional row context, then generate a small set of HS-style search terms for retrieval.

This module is not the retriever itself. Its job is query expansion.

## Current role

Input:
- one translated product string
- optional same-row context
- HS description reference text

Output:
- 5 to 8 search terms that are generic product-class terms
- terms should be suitable for vector retrieval against HS descriptions

## Recommended refactor

The next version of `search_terms.py` should stop relying on ad hoc tool-schema dicts alone and instead use a Pydantic response model.

Preferred shape:

```python
from pydantic import BaseModel, field_validator

class SearchTerms(BaseModel):
    terms: list[str]

    @field_validator("terms")
    @classmethod
    def validate_terms(cls, terms: list[str]) -> list[str]:
        cleaned = []
        seen = set()
        for term in terms:
            value = " ".join(term.split()).strip()
            if not value:
                continue
            key = value.lower()
            if key in seen:
                continue
            seen.add(key)
            cleaned.append(value)

        if not 5 <= len(cleaned) <= 8:
            raise ValueError("Expected 5 to 8 unique search terms")
        return cleaned
```

## Why use Pydantic

Benefits:
- validates the LLM output before retrieval uses it
- avoids fragile `term1`, `term2`, `term3` field handling
- makes provider swaps easier
- gives one canonical `.terms` list for retrieval

## Output requirements

The generated terms should:
- be noun-like product classes
- match HS vocabulary as closely as possible
- avoid company names
- avoid adjectives unless essential to product identity
- avoid process words
- avoid repeated synonyms that add no retrieval value

Good examples:
- `cleaning articles`
- `plastic sheets`
- `air conditioners`
- `baking powder`

Bad examples:
- `high quality cleaning`
- `3M product`
- `used in industry`
- `container goods`

## Module boundary

`search_terms.py` should own:
- the prompt
- the response model
- parsing and validation of LLM output

It should not own:
- row assembly
- translation
- retrieval
- reranking

## Integration target

The intended row flow is:
1. `build_query.py` builds one row-level query
2. `translator.py` detects language and translates to English
3. `search_terms.py` generates HS-style expansion terms
4. `retrieval.py` searches using the translated query plus generated terms

## Provider direction

Keep the module provider-agnostic.

That means:
- do not hard-wire Anthropic-specific response handling into the long-term design
- accept a generic LLM client or wrapper
- return validated `SearchTerms` data to the caller

## Immediate next task

Refactor `generate_search_terms(...)` so it returns a validated Pydantic model or a validated `list[str]` derived from that model.

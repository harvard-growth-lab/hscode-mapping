"""Rerank a shortlist of HS code candidates using GPT.

TODO: replace direct OpenAI API call in rerank_codes() with llm.call()
      so the provider can be swapped without touching this file.
      This module should only own RERANK_TOOL, format_table_for_llm(),
      and the prompt string.
"""

import json

import pandas as pd
import polars as pl

RERANK_TOOL = {
    "type": "function",
    "function": {
        "name": "labeller",
        "description": "Return the top 2 HS code labels and reasoning for the product classification.",
        "parameters": {
            "type": "object",
            "properties": {
                "code_first": {
                    "type": "string",
                    "description": "The HS code (e.g. '0101') that is your first choice.",
                },
                "code_second": {
                    "type": "string",
                    "description": "The HS code (e.g. '0101') that is your second choice.",
                },
                "reason": {
                    "type": "string",
                    "description": (
                        "A 50 word reason for your choices. If none match, "
                        "write 'None of the provided codes' for each label."
                    ),
                },
            },
            "required": ["code_first", "code_second", "reason"],
        },
    },
}


def format_table_for_llm(df: pd.DataFrame) -> list[str]:
    """Format HS code DataFrame rows as 'Code XXXX: Description' strings for LLM prompts."""
    return [f"Code {row['Code']}: {row['Description']}" for _, row in df.iterrows()]


def rerank_codes(
    shortlist: pd.DataFrame,
    query: str,
    articles: str,
    hs_data: pl.DataFrame,
    client,
    model: str,
) -> dict:
    """Use GPT to select the top 2 HS codes from the retrieved shortlist.

    Args:
        shortlist: DataFrame of ~25 candidate HS codes from retrieval.
        query: Product name or description.
        articles: Optional article/document context.
        hs_data: Full HS reference table (polars) for description lookup.
        client: openai.OpenAI instance.
        model: GPT model name.

    Returns:
        Dict with code_first, desc_first, code_second, desc_second, reason.
    """
    formatted_codes = format_table_for_llm(shortlist)
    search_return = [line for line in formatted_codes[1:]]

    content = (
        f"You are an expert annotator helping a country map the descriptions of "
        f"company products to a broader product classification. We want to align "
        f"these internal product descriptions to the right HS Codes, which we have "
        f"provided descriptions for.\n"
        f"The product that requires classification is: {query}, and has some related "
        f"press coverage here {articles}.\n\n"
        f"Your goal is to match the product (domain-specific name) to its relevant "
        f"product classification. Work through your reasoning step by step, and then "
        f"return a) the two most relevant HS Codes for the product in order, and "
        f"b) a 50 word reason for your choices, as a JSON object. What follows are "
        f"the codes you need to review to assign the label accordingly: {search_return}.\n"
        f"If none of the codes match, please write 'None of the provided codes' in "
        f"place of the label and explain your reasoning as usual.\n"
        f"Make sure your json format clearly follows the structure of the function "
        f"provided to you."
    )

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": content}],
        temperature=0.1,
        tool_choice="required",
        tools=[RERANK_TOOL],
    )

    args_json = response.choices[0].message.tool_calls[0].function.arguments
    results = json.loads(args_json)

    code_first = results["code_first"]
    code_second = results["code_second"]

    desc_first = hs_data.filter(pl.col("Code") == code_first).get_column("Description")
    desc_second = hs_data.filter(pl.col("Code") == code_second).get_column("Description")

    return {
        "code_first": code_first,
        "desc_first": desc_first[0] if len(desc_first) else "(not found)",
        "code_second": code_second,
        "desc_second": desc_second[0] if len(desc_second) else "(not found)",
        "reason": results.get("reason", ""),
    }

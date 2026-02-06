"""LLM-based term generation and reranking.

Stage 1: Claude generates 5-8 search terms for a product description.
Stage 2: GPT selects the top 2 HS codes from a FAISS-retrieved shortlist.

Original sources: run_pipe.py terms_generator_claude_prompt() (lines 112-134),
retrieve_with_claude() (lines 136-183), rerank_codes() (lines 228-298),
process_rerank() (lines 302-342).
"""

import json

import pandas as pd
import polars as pl

from linkages.utils import extract_text_chunks, format_table_for_llm

# Claude tool schema for term generation
TERM_GENERATION_TOOL = {
    "name": "request_terms",
    "description": (
        "Return 5-8 unique product description terms from the reference list "
        "that are most relevant to the product described. These will be used for "
        "vector embedding search, so specific terms from the corpus are most useful."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "term1": {"type": "string", "description": "First term"},
            "term2": {"type": "string", "description": "Second term"},
            "term3": {"type": "string", "description": "Third term"},
            "term4": {"type": "string", "description": "Fourth term"},
            "term5": {"type": "string", "description": "Fifth term"},
            "term6": {"type": "string", "description": "Sixth term (optional)"},
            "term7": {"type": "string", "description": "Seventh term (optional)"},
            "term8": {"type": "string", "description": "Eighth term (optional)"},
        },
        "required": ["term1", "term2", "term3", "term4", "term5"],
    },
}

# GPT tool schema for reranking
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


def generate_search_terms(
    query: str,
    articles: str,
    hs_descriptions: list[str],
    client,
    model: str,
) -> list[str]:
    """Use Claude to generate 5-8 search terms for a product.

    Args:
        query: Product name.
        articles: Concatenated article text about the product.
        hs_descriptions: Pre-loaded list of HS code description strings.
        client: anthropic.Anthropic instance.
        model: Claude model name.

    Returns:
        List of generated search terms.

    Original: run_pipe.py terms_generator_claude_prompt() + retrieve_with_claude().
    """
    prompt = (
        f"TASK: I have a description of a very specific product, namely: '{query}', "
        f"that I would like you to generate 5-8 product description variations, or "
        f"closely related search terms that would maximize our chance of finding this "
        f"product in official product lists.\n\n"
        f"This is some coverage of the product, which should help you decide what it is.\n\n"
        f"'{articles}'\n\n"
        f"Generate these synonyms as input for the request_terms function you have been "
        f"provided, and I would want you to closely try to match the terms from this "
        f"product list below. Return only words that are generic product classes that "
        f"specifically tell me what the product is. DO NOT include company names, "
        f"adjectives, process terms, or non-product phrases. Most importantly, return "
        f"your response as a function call to the request_terms function, for the 5 to 8 "
        f"unique (non repeated) from the list you feel are the most relevant! If there "
        f"are multiple, just share the one common term from the group.\n\n"
        f"Product list to reference: {hs_descriptions}"
    )

    response = client.messages.create(
        model=model,
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
        tools=[TERM_GENERATION_TOOL],
        tool_choice={"type": "tool", "name": "request_terms"},
    )

    # Extract terms from tool call response
    tool_input = response.content[0].to_dict()["input"]
    return list(tool_input.values())


def rerank_codes(
    shortlist: pd.DataFrame,
    query: str,
    articles: str,
    hs_data: pl.DataFrame,
    client,
    model: str,
) -> dict:
    """Use GPT to select the top 2 HS codes from the FAISS-retrieved shortlist.

    Args:
        shortlist: DataFrame of candidate HS codes (with Code + Description columns).
        query: Product name.
        articles: Concatenated article text.
        hs_data: Full HS code reference table (polars) for description lookup.
        client: openai.OpenAI instance.
        model: GPT model name.

    Returns:
        Dict with code_first, desc_first, code_second, desc_second, reason.

    Original: run_pipe.py rerank_codes() + process_rerank() merged into one function.
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

    # Parse GPT tool call output
    args_json = response.choices[0].message.tool_calls[0].function.arguments
    results = json.loads(args_json)

    # Look up descriptions from the full HS table
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

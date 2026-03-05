"""Generate HS-vocabulary search terms for a product string using Claude."""

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


def generate_search_terms(
    query: str,
    articles: str,
    hs_descriptions: list[str],
    client,
    model: str,
) -> list[str]:
    """Use Claude to generate 5-8 HS-vocabulary search terms for a product string.

    Args:
        query: Product name or description.
        articles: Optional article/document context about the product.
        hs_descriptions: Full list of HS code description strings (for Claude to reference).
        client: anthropic.Anthropic instance.
        model: Claude model name.

    Returns:
        List of search terms.
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

    tool_input = response.content[0].to_dict()["input"]
    return list(tool_input.values())

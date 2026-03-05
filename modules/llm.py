"""Unified LLM call interface.

Single place to swap providers. search_terms.py and reranker.py define
their schemas and prompts as data and call through here.

Currently supports:
  - Anthropic (claude-*)
  - OpenAI (gpt-*)

To add a provider (e.g. LiteLLM, Mistral, local):
  1. Add a branch in call() detecting the client/model type
  2. Translate the tool schema if needed (Anthropic vs OpenAI format differ)
  3. Normalise the response into the standard return dict

TODO: replace provider-specific branches with LiteLLM for a single
      unified interface (litellm.completion supports 100+ providers
      via OpenAI-compatible API).
"""


def call(
    client,
    model: str,
    prompt: str,
    tool_schema: dict,
    tool_name: str,
) -> dict:
    """Call an LLM with a tool/function and return the parsed tool input as a dict.

    Args:
        client: An anthropic.Anthropic or openai.OpenAI instance.
        model: Model name string (e.g. "claude-3-5-haiku-latest", "gpt-4o-mini").
        prompt: The user/system prompt string.
        tool_schema: Tool definition in either Anthropic or OpenAI format.
        tool_name: Name of the tool to force the model to call.

    Returns:
        Dict of the tool call arguments as returned by the model.
    """
    # TODO: detect provider from client type or model string prefix,
    #       translate schema format if needed, normalise response

    # Anthropic branch
    # response = client.messages.create(
    #     model=model,
    #     max_tokens=1024,
    #     messages=[{"role": "user", "content": prompt}],
    #     tools=[tool_schema],
    #     tool_choice={"type": "tool", "name": tool_name},
    # )
    # return response.content[0].to_dict()["input"]

    # OpenAI branch
    # response = client.chat.completions.create(
    #     model=model,
    #     messages=[{"role": "system", "content": prompt}],
    #     temperature=0.1,
    #     tool_choice="required",
    #     tools=[tool_schema],
    # )
    # import json
    # return json.loads(response.choices[0].message.tool_calls[0].function.arguments)

    raise NotImplementedError

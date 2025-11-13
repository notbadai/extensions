import json
import time

from notbadai_ide import api
from openai import OpenAI

from common.models import MODELS
from common.settings import LLM_PROVIDERS

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_repo_files",
            "description": "Search for a query string across all repository files and return matching lines with file paths and line numbers.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The string to search for in the repository files",
                    },
                    "file_extensions": {
                        "type": "array",
                        "items": {
                            "type": "string"
                        },
                        "description": "Optional list of file extensions to limit search (e.g., ['.py', '.js'])",
                    },
                },
                "required": ["api", "query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the content of a file from the repository.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to the file to read",
                    },
                },
                "required": ["api", "file_path"],
            },
        },
    },
]

MESSAGES = [
    {
        "role": "user",
        "content": "What's the temperature in San Francisco now? How about tomorrow? Current Date: 2024-09-30.",
    },
]

tools = TOOLS


def call_llm(
        model_id: str,
        messages,
        *,
        push_to_chat: bool = True,
        temperature: float = 1.0,
        top_p: float = 1.0,
        n_outputs: int = 1,
        max_tokens: int = None,
):
    """Streams responses from the LLM and sends them to the chat UI in real-time."""

    model_info = MODELS[model_id]
    provider = None
    model_name = None
    for p in LLM_PROVIDERS:
        if p["name"] in model_info:
            model_name = model_info[p["name"]]
            provider = p
            break

    start_time = time.time()

    client = OpenAI(api_key=provider["api_key"], base_url=provider["base_url"])
    # client = OpenAI(api_key=api.api_key, base_url="https://api.deepinfra.com/v1/openai")

    stream = client.chat.completions.create(
        model=model_name,
        messages=messages,
        tools=tools,
        stream=True,
        temperature=temperature,
        top_p=top_p,
        n=n_outputs,
        max_tokens=max_tokens,
    )

    thinking = False
    usage = None
    content = ""
    response = []

    for chunk in stream:
        delta = chunk.choices[0].delta
        if push_to_chat:
            if getattr(delta, "reasoning", None):
                if not thinking:
                    api.start_block("think")
                    thinking = True
                api.push_to_chat(content=delta.reasoning)

        if delta.content:
            if push_to_chat:
                if thinking:
                    api.end_block()
                    thinking = False
                api.push_to_chat(content=delta.content)
            content += delta.content

        if delta.tool_calls:
            api.log(chunk.choices[0].delta.json())
            if content.strip():
                response.append({
                    "role": "assistant",
                    "content": content,
                })
            content = ''
            tool_calls = [json.loads(t.json()) for t in delta.tool_calls]

            for t in tool_calls:
                args = json.loads(t['function']['arguments'])
                args = ', '.join(f'{k}={repr(v)}' for k, v in args.items())
                api.push_meta('<strong>' + t['function']['name'] + '</strong>(' +
                              args + ')')

            response.append({
                'role': 'assistant',
                'tool_calls': tool_calls,
            })

        if chunk.usage is not None:
            assert usage is None
            usage = chunk.usage

    elapsed = time.time() - start_time
    meta_data = f"Time: {elapsed:.2f}s"
    if usage is not None:
        api.log(str(usage))

        meta_data += f' Prompt tokens: {usage.prompt_tokens :,} Completion tokens {usage.completion_tokens :,}, Model: {model_name} @ {provider["name"]}'

    if push_to_chat:
        api.push_meta(meta_data.strip())

        api.terminate_chat()

    if not content.strip():
        response.append({
            "role": "assistant",
            "content": content,
        })

    return response

import json
import time
from typing import Dict, List

from openai import OpenAI
from notbadai_ide import api, START_METADATA, END_METADATA, START_THINK, END_THINK

from ..common.models import MODELS
from ..common.settings import LLM_PROVIDERS

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
                "required": ["query"],
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
                "required": ["file_path"],
            },
        },
    },
]

tools = TOOLS


def call_llm(model_id: str,
             messages: List[Dict[str, str]],
             *,
             push_to_chat: bool = True,
             temperature: float = 1.0,
             top_p: float = 1.0,
             n_outputs: int = 1,
             max_tokens: int = None,
             ):
    """Streams responses from the LLM and sends them to the chat UI in real-time."""

    model_info = MODELS[model_id]

    api_keys = api.get_api_keys()
    if len(api_keys) == 0:
        raise ValueError('API provider required. Configure at least one in Extensions → Management.')

    default_key = None
    for k in api_keys:
        if k.default:
            default_key = k

    api_key, model_name = None, None
    if default_key.provider in model_info:
        model_name = model_info[default_key.provider]
        api_key = default_key
    else:
        for k in api_keys:
            if k.provider in model_info:
                model_name = model_info[k.provider]
                api_key = k
                break

    if api_key is None:
        raise ValueError(f"The API provider does not support {model_id} model")

    provider = None
    for p in LLM_PROVIDERS:
        if p['name'] == api_key.provider:
            provider = p
            break

    start_time = time.time()

    client = OpenAI(api_key=api_key.key, base_url=provider['base_url'])

    stream = client.chat.completions.create(
        model=model_name,
        messages=messages,
        stream=True,
        temperature=temperature,
        tools=tools,
        top_p=top_p,
        n=n_outputs,
        max_tokens=max_tokens,
    )

    thinking = False
    usage = None
    content = ''
    response = []
    accumulated_tool_calls = {}  # Track tool calls by index

    for chunk in stream:
        delta = chunk.choices[0].delta
        if push_to_chat:
            if getattr(delta, 'reasoning', None):
                if not thinking:
                    api.chat(START_THINK)
                    thinking = True
                api.chat(content=delta.reasoning)

        if delta.content:
            if push_to_chat:
                if thinking:
                    api.chat(END_THINK)
                    thinking = False
                api.chat(content=delta.content)
            content += delta.content

        if delta.tool_calls:
            # accumulate tool calls across stream chunks
            for tool_call_delta in delta.tool_calls:
                index = tool_call_delta.index

                if index not in accumulated_tool_calls:
                    accumulated_tool_calls[index] = {
                        'id': tool_call_delta.id or '',
                        'type': 'function',
                        'function': {
                            'name': tool_call_delta.function.name or '',
                            'arguments': ''
                        }
                    }

                # accumulate the function arguments
                if tool_call_delta.function.arguments:
                    accumulated_tool_calls[index]['function']['arguments'] += tool_call_delta.function.arguments

                # update other fields if present
                if tool_call_delta.id:
                    accumulated_tool_calls[index]['id'] = tool_call_delta.id
                if tool_call_delta.function.name:
                    accumulated_tool_calls[index]['function']['name'] = tool_call_delta.function.name

        if chunk.usage is not None:
            assert usage is None
            usage = chunk.usage

    # after streaming is complete, process accumulated tool calls
    if accumulated_tool_calls:
        if content.strip():
            response.append({
                "role": "assistant",
                "content": content,
            })
        content = ''

        # convert accumulated_tool_calls dict to list and display
        tool_calls = [accumulated_tool_calls[i] for i in sorted(accumulated_tool_calls.keys())]

        for t in tool_calls:
            args = json.loads(t['function']['arguments'])
            args = ', '.join(f'{k}={repr(v)}' for k, v in args.items())
            api.chat(START_METADATA + '<strong>' + t['function']['name'] + '</strong>(' +
                     args + ')' + END_METADATA)

        response.append({
            'role': 'assistant',
            'tool_calls': tool_calls,
        })

    elapsed = time.time() - start_time
    meta_data = f'Time: {elapsed:.2f}s'
    if usage is not None:
        meta_data += f' Prompt tokens: {usage.prompt_tokens :,} Completion tokens {usage.completion_tokens :,}, Model: {model_name} @ {provider["name"]}'

    if push_to_chat:
        api.chat(f'{START_METADATA}{meta_data.strip()}{END_METADATA}')

        api.end_chat()

    if not content.strip():
        response.append({
            "role": "assistant",
            "content": content,
        })

    return response

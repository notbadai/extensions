import time
from typing import Dict, List

from openai import OpenAI
from notbadai_ide import api, START_METADATA, END_METADATA, START_THINK, END_THINK

from .models import MODELS
from .settings import LLM_PROVIDERS


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
        top_p=top_p,
        n=n_outputs,
        max_tokens=max_tokens,
    )

    thinking = False
    usage = None
    content = ''

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

        if chunk.usage is not None:
            assert usage is None
            usage = chunk.usage

    elapsed = time.time() - start_time
    meta_data = f'Time: {elapsed:.2f}s'
    if usage is not None:
        api.log(str(usage))

        meta_data += f' Prompt tokens: {usage.prompt_tokens :,} Completion tokens {usage.completion_tokens :,}, Model: {model_name} @ {provider["name"]}'

    if push_to_chat:
        api.chat(f'{START_METADATA}{meta_data.strip()}{END_METADATA}')

        api.end_chat()

    return content

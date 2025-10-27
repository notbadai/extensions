import asyncio
import time
import json
from string import Template
from typing import List
from pathlib import Path

from notbadai_ide import api, START_METADATA, END_METADATA

from .common.llm import call_llm
from .common.utils import parse_prompt, extract_code_block
from .common.prompt import build_context
from .common.formatting import markdown_section
from .websearch import WebSearch
from .crawler import Crawler


def websearch(query: str) -> List[str]:
    start_time = time.time()
    urls = WebSearch(query).search()
    api.chat(f'{START_METADATA}{len(urls)} search results ({int(time.time() - start_time)}s): {", ".join(urls)}{END_METADATA}')
    results = asyncio.run(Crawler(urls, query).run())

    res = []
    for result in results:
        if result.markdown:
            api.log(result.markdown.fit_markdown)
            res.append(result.markdown.fit_markdown)
        else:
            continue

    return res


def get_prompt_template(template_path: str, **kwargs) -> str:
    path = Path(__file__).parent / f'{template_path}.md'
    with open(str(path)) as f:
        template = Template(f.read())

    return template.substitute(kwargs)


def parse_query_json(content: str) -> str:
    try:
        data = json.loads(content.strip())
        if 'query' not in data:
            raise ValueError("JSON response missing 'query' field")
        return data['query']
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse JSON response: {e}")


def start():
    """Main extension function that handles chat interactions with the AI assistant."""
    command, model, prompt = parse_prompt()
    selection = api.get_selection()
    chat_history = api.get_chat_history()
    prompt = api.get_prompt()

    api.chat(f'{START_METADATA}model: {model}, command: {command}{END_METADATA}')

    context = build_context()

    api.chat(f'{START_METADATA}With context: {len(context) :,} characters,'
             f' selection: {bool(selection)}{END_METADATA}')

    messages = [
        {'role': 'system', 'content': get_prompt_template('query.system', model='qwen')},
        {'role': 'user', 'content': context},
        {'role': 'user', 'content': f'Prompt:\n\n```\n{prompt}\n```'},
    ]

    content = call_llm('qwen', messages, push_to_chat=False)
    api.log(content)
    search_query = parse_query_json(extract_code_block(content))

    api.chat(f'{START_METADATA}Search Query: {search_query}{END_METADATA}')

    results = websearch(search_query)
    context += '\n\n' + markdown_section('Websearch Results', "\n\n".join(results))
    api.chat(context)

    messages = [
        {'role': 'system', 'content': get_prompt_template('chat.system', model=model)},
        {'role': 'user', 'content': context},
        *[m.to_dict() for m in chat_history],
        {'role': 'user', 'content': prompt},
    ]

    call_llm(model, messages)

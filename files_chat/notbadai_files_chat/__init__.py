from typing import List
from string import Template
from pathlib import Path

from notbadai_ide import api, START_METADATA, END_METADATA

from .common.llm import call_llm
from .common.utils import parse_prompt
from .common.terminal import get_terminal_snapshot
from .common.prompt import build_context

module_dir = Path(__file__).parent


def get_prompt_template(template_path: str, **kwargs) -> str:
    path = module_dir / f'{template_path}.md'
    with open(str(path)) as f:
        template = Template(f.read())

    return template.substitute(kwargs)


def get_yaml(response) -> List[str]:
    """Extract YAML code blocks from markdown response.

    Args:
        response: String containing markdown response

    Returns:
        List of paths extracted from YAML code blocks in markdown
    """
    yaml_blocks = []
    lines = response.split('\n')
    in_yaml_block = False

    current_block = []

    for line in lines:
        if line.strip().startswith('```yaml'):
            in_yaml_block = True
            current_block = []
            continue
        elif line.strip().startswith('```') and in_yaml_block:
            in_yaml_block = False
            yaml_blocks.append('\n'.join(current_block))
            continue

        if in_yaml_block:
            current_block.append(line)

    return yaml_blocks


def extract_paths_from_yaml(yaml_blocks: List[str]) -> List[str]:
    paths = []
    for block in yaml_blocks:
        for line in block.split('\n'):
            line = line.split('#')[0]
            line = line.strip()
            if line.startswith('-'):
                path = line[1:].strip()
                paths.append(path)

    return paths


def start():
    """Main extension function that handles chat interactions with the AI assistant."""
    command, model, prompt = parse_prompt()
    selection = api.get_selection()
    chat_history = api.get_chat_history()
    repo_files = api.get_repo_files()
    prompt = api.get_prompt()

    api.chat(f'{START_METADATA}model: {model}, command: {command}{END_METADATA}')
    repo_paths = {f.path: f for f in repo_files}

    if command == 'context':
        api.log('Normal context')
        context = build_context()

        api.chat(f'{START_METADATA}With context: {len(context) :,},'
                 f' selection: {bool(selection)}{END_METADATA}')
        # api.log(context)
        messages = [
            {'role': 'system', 'content': get_prompt_template('files.list.system', model=model)},
            {'role': 'user', 'content': context},
            *[m.to_dict() for m in chat_history],
            {'role': 'user', 'content': f'Prompt:\n\n```\n{prompt}\n```'},
        ]
    else:
        raise ValueError(f'Unknown command: {command}')

    api.log(f'messages {len(messages)}')
    api.log(f'prompt {prompt}')
    # api.log(context)

    response = call_llm(model, messages)

    files = get_yaml(response)
    files = extract_paths_from_yaml(files)

    # api.chat('<metadata>' + f'Files:\n' + '\n'.join(files) + '</metadata>')

    files = [repo_paths[f] for f in files if f in repo_paths]
    bad = [f for f in files if not f.exists()]
    if bad:
        api.chat(
            f'{START_METADATA}The following files do not exist: {"".join(f"<code>{f.path}</code>" for f in bad)}{END_METADATA}')

    files = [f for f in files if f.exists()]
    context = build_context(files=files)

    api.chat(f'{START_METADATA}With context: {len(context) :,} characters,'
             f' selection: {bool(selection)}{END_METADATA}')
    # api.log(context)
    messages = [
        {'role': 'system', 'content': get_prompt_template('chat.system', model=model)},
        {'role': 'user', 'content': context},
        *[m.to_dict() for m in chat_history],
        {'role': 'user', 'content': prompt},
    ]

    call_llm(model, messages)

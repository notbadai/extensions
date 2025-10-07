from string import Template
from pathlib import Path

from notbadai_ide import api, START_METADATA, END_METADATA

from .common.llm import call_llm
from .common.utils import parse_prompt
from .common.prompt import build_context

module_dir = Path(__file__).parent


def get_prompt_template(template_path: str, **kwargs) -> str:
    path = module_dir / f'{template_path}.md'
    with open(str(path)) as f:
        template = Template(f.read())

    return template.substitute(kwargs)


def start():
    """Main extension function that handles chat interactions with the AI assistant."""
    command, model, prompt = parse_prompt()
    selection = api.get_selection()
    chat_history = api.get_chat_history()
    prompt = api.get_prompt()

    api.chat(f'{START_METADATA}model: {model}, command: {command}{END_METADATA}')

    if command == '':
        api.chat(f'{START_METADATA}Without context{END_METADATA}')
        messages = [
            {'role': 'system', 'content': get_prompt_template('chat.system', model=model)},
            *[m.to_dict() for m in chat_history],
            {'role': 'user', 'content': prompt},
        ]
    elif command == 'here':
        context = build_context()

        api.chat(f'{START_METADATA}With context: {len(context) :,} characters,'
                               f' selection: {bool(selection)}{END_METADATA}')
        api.log(context)
        messages = [
            {'role': 'system', 'content': get_prompt_template('chat.system', model=model)},
            {'role': 'user', 'content': context},
            *[m.to_dict() for m in chat_history],
            {'role': 'user', 'content': prompt},
        ]
    elif command == 'context':
        context = build_context()

        api.chat(f'{START_METADATA}With context: {len(context) :,} characters,'
                               f' selection: {bool(selection)}{END_METADATA}')
        # api.log(context)
        messages = [
            {'role': 'system', 'content': get_prompt_template('chat.system', model=model)},
            {'role': 'user', 'content': context},
            *[m.to_dict() for m in chat_history],
            {'role': 'user', 'content': prompt},
        ]
    else:
        raise ValueError(f'Unknown command: {command}')

    api.log(f'messages {len(messages)}')
    api.log(f'prompt {prompt}')
    # api.log(context)

    content = call_llm(model, messages)

    api.log(content)

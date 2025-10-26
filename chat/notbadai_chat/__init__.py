from string import Template
from pathlib import Path

from notbadai_ide import api

from .common.llm import call_llm
from .common.formatting import markdown_section, markdown_code_block


def start():
    """Main extension function that handles chat interactions with the AI assistant."""
    model = 'qwen'
    context = []

    open_files = [f for f in api.get_repo_files() if f.is_open]
    if open_files:
        context.append(markdown_section("Relevant files", "\n\n".join(
            f'Path: `{f.path}`\n\n{markdown_code_block(f.get_content())}' for f in open_files)))

    if current_file := api.get_current_file():
        file_content = markdown_code_block(current_file.get_content())
        context.append(markdown_section("Current File", f"Path: `{current_file.path}`\n\n{file_content}"))

    if terminal := api.get_current_terminal().get_snapshot():
        context.append(markdown_section("Terminal output", markdown_code_block(terminal[-40000:])))

    if selection := api.get_selection().strip():
        selection_content = markdown_code_block(selection)
        context.append(markdown_section("Selection", f"This is the code snippet that I'm referring to\n\n{selection_content}"))

    with open(Path(__file__).parent / 'chat.system.md') as f:
        system_prompt = Template(f.read()).substitute(model=model)

    messages = [
        {'role': 'system', 'content': system_prompt},
        {'role': 'user', 'content': "\n\n".join(context)},
        {'role': 'user', 'content': api.get_prompt()},
    ]

    call_llm(model, messages)
from notbadai_ide import api, START_METADATA, END_METADATA

from .common.llm import call_llm
from .common.utils import extract_code_block
from .common.formatting import markdown_section, markdown_code_block

SYSTEM_PROMPT = """
You are an expert programmer assisting a colleague in adding code to an existing file.

Your colleague will give you:
* Relevant code files if applicable
* Current code file contents
* Insertion point marked by `INSERT_YOUR_CODE` inside the current code file
* If applicable, some contents that should be part of the line/block that should be added (usually a prefix).

Your task:
* Provide a suggestion for the next line or block of code
* Match the file's indentation, style, and conventions
* Wrap your response in triple backticks with the appropriate language identifier.
* Only provide the completion text within the code block, no explanations outside of it
* Do NOT repeat the text that's already before the current line
""".strip()


def make_prompt(prefix: str, suffix: str, next_line: str):
    context = []

    other_files = []
    for file in api.get_repo_files():
        if file.is_open:
            other_files.append(file)
    if other_files:
        api.chat(f'{START_METADATA}Relevant files: {", ".join(f.path for f in other_files)}{END_METADATA}')

        opened_files = [f'Path: `{f.path}`\n\n' + markdown_code_block(f.get_content()) for f in other_files]
        context.append(markdown_section("Relevant files", "\n\n".join(opened_files)))

    prompt = '\n\n'.join(context)

    prompt += '\n\n# Current File\n\n```python\n' + prefix + '\n#INSERT_YOUR_CODE'

    if suffix.strip():
        prompt += '\n' + suffix

    prompt += '\n```'

    if next_line.strip():
        prompt += f'\n\nThe next line begins with `{next_line}`.'

    api.log(prompt)

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt}
    ]


def start():
    # Get file content and cursor position
    lines = api.get_current_file().get_content().splitlines()
    cursor = api.get_cursor()
    idx = cursor.row - 1
    current_line = lines[idx][:cursor.column - 1]

    messages = make_prompt('\n'.join(lines[:idx]),
                           '\n'.join(lines[idx + 1:]),
                           current_line
                           )

    model = 'qwen'
    api.start_chat()

    content = call_llm(model, messages)
    content = extract_code_block(content)

    if content.startswith(current_line):
        content = content[len(current_line):]

    # need to press tab to accept and esc to reject
    api.inline_completion(content)

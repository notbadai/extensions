from notbadai_ide import api, START_METADATA, END_METADATA

from .common.llm import call_llm
from .common.utils import extract_code_block
from .common.formatting import markdown_section, markdown_code_block

SYSTEM_PROMPT = """
You are an expert programmer assisting a colleague in updating code in an existing file.

Your colleague will give you:
* Relevant code files if applicable
* Current code file contents
* A segment of code marked by `UPDATE_START` and `UPDATE_END`.
* The start of the segment usually will contain a comment describing the update he wants.

Your task:
* Suggest an update for the code block
* Match the file's indentation, style, and conventions
* Wrap your response in triple backticks with the appropriate language identifier.
* Only provide the code within the code block, no explanations outside of it
* If the segment started with a instructive comment about the code change, do not include the same comment in your suggestion. If applicable, suggest new descriptive comment(s) about the suggested code.
""".strip()


def make_prompt(prefix, suffix, block):
    context = []

    repo_files = api.get_repo_files()
    other_files = []
    for file in repo_files:
        if file.is_open:
            other_files.append(file)

    if other_files:
        api.chat(f'{START_METADATA}Relevant files: {", ".join(f.path for f in other_files)}{END_METADATA}')

        opened_files = [f'Path: `{f.path}`\n\n' + markdown_code_block(f.get_content()) for f in other_files]
        context.append(markdown_section("Relevant files", "\n\n".join(opened_files)))

    prompt = '\n\n'.join(context)

    prompt += '\n\n# Current File\n\n```python\n' + prefix + f'\n#UPDATE_START\n{block}\n#UPDATE_END'

    if suffix.strip():
        prompt += '\n' + suffix

    prompt += '\n```'

    api.log(prompt)

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt}
    ]


def start():
    # Get file content and cursor position
    lines = api.get_current_file().get_content().splitlines()
    selection_lines = api.get_selection().splitlines()

    if not selection_lines:
        return

    idx = None
    for i in range(len(lines) - len(selection_lines)):
        found = True
        for j in range(len(selection_lines)):
            if lines[i + j] != selection_lines[j]:
                found = False
                break
        if found:
            idx = i
            break

    if idx is None:
        return

    messages = make_prompt('\n'.join(lines[:idx]),
                           '\n'.join(lines[idx + len(selection_lines):]),
                           '\n'.join(selection_lines))

    model = 'qwen'
    api.start_chat()

    content = call_llm(model, messages)
    content = extract_code_block(content)

    # need to press tab to accept and esc to reject
    api.inline_completion(content)

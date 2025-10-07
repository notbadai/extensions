from typing import List
from notbadai_ide import api, File, START_METADATA, END_METADATA

from .terminal import get_terminal_snapshot
from .formatting import markdown_section, markdown_code_block, add_line_comment


def build_context(files: List['File'] = None) -> str:
    """Builds the context string from the current file and selection."""
    context = []

    current_file = api.get_current_file()
    context_files = api.get_context_files()
    selection = api.get_selection()
    terminal = get_terminal_snapshot()
    cursor = api.get_cursor()

    other_files = []
    if files is None:
        file_list = api.get_repo_files()
        for file in file_list:
            if file.is_open:
                other_files.append(file)
    else:
        file_list = files
        other_files = file_list

    if context_files:
        api.chat(f'{START_METADATA}Context paths: {", ".join(list(context_files.keys()))}{END_METADATA}')

    if file_list:
        repo_files = [f'{f.path}`' for f in file_list]
        context.append(markdown_section("List of Files", "\n".join(repo_files)))

    if other_files:
        api.chat(f'{START_METADATA}Opened files: {", ".join(f.path for f in other_files)}{END_METADATA}')

    # combine other_files with files from context_files, removing duplicates
    all_files = (other_files or []) + [f for files_list in context_files.values() for f in files_list]
    relevant_files = list({f.path: f for f in all_files}.values())

    if relevant_files:
        api.chat(f'{START_METADATA}Relevant files: {", ".join(f.path for f in relevant_files)}{END_METADATA}')
        relevant_files = [f'Path: `{f.path}`\n\n' + markdown_code_block(f.get_content()) for f in relevant_files]
        context.append(markdown_section("Relevant files", "\n\n".join(relevant_files)))

    if current_file:
        api.chat(f'{START_METADATA}Current file: {current_file.path}{END_METADATA}')
        context.append(
            markdown_section("Current File",
                             f"Path: `{current_file.path}`\n\n" +
                             markdown_code_block(current_file.get_content()))
        )

    if terminal:
        if len(terminal) > 40_000:
            pre_text = f'Terminal output is {len(terminal)} chars long, and here is the last 40k chars of it.\n\n'
        else:
            pre_text = f'Terminal output is {len(terminal)} chars long.'

        context.append(
            markdown_section("Terminal output",
                             f"{pre_text}\n\n" + markdown_code_block(terminal[-40000:]))
        )

    if selection and selection.strip():
        context.append(
            markdown_section("Selection",
                             "This is the code snippet that I'm referring to\n\n" +
                             markdown_code_block(selection))
        )

    if current_file and cursor:
        cursor_row, cursor_column = cursor.row - 1, cursor.column - 1
        block = current_file.get_content().split('\n')
        assert len(block) > cursor_row, f'Cursor row {cursor_row} block of length {len(block)}'
        prefix = block[cursor_row - 3: cursor_row]
        line = block[cursor_row]
        line = add_line_comment(current_file, line, f'Cursor is here: `{line[:cursor_column].strip()}`')
        suffix = block[cursor_row + 1:cursor_row + 4]

        block = prefix + [line] + suffix

        context.append(markdown_section("Cursor position",
                                        markdown_code_block('\n'.join(block))))

    return "\n\n".join(context)

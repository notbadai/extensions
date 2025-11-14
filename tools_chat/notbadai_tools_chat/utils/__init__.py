import json
from string import Template
from pathlib import Path
from typing import List, Dict

from notbadai_ide import api, START_METADATA, END_METADATA

module_dir = Path(__file__).parent.parent


def get_prompt_template(template_path: str, **kwargs) -> str:
    path = module_dir / f'{template_path}.md'
    with open(str(path)) as f:
        template = Template(f.read())

    return template.substitute(kwargs)


def search_repo_files(query: str, file_extensions: List[str] = None) -> List[Dict]:
    """Search for a query string across all repository files.

    Args:
        query: The string to search for
        file_extensions: Optional list of file extensions to limit search (e.g., ['.py', '.js'])

    Returns:
        List of dictionaries containing file path, line number, and line content
    """
    results = []

    for file in api.get_repo_files():
        # Filter by file extensions if specified
        if file_extensions:
            if not any(file.path.endswith(ext) for ext in file_extensions):
                continue

        try:
            content = file.get_content()
            lines = content.split('\n')

            for line_num, line in enumerate(lines, 1):
                if query.lower() in line.lower():
                    results.append({
                        "file_path": file.path,
                        "line_number": line_num,
                        "content": line.strip()
                    })
        except Exception as e:
            # Skip files that can't be read
            continue

    api.chat(START_METADATA + '<strong>Search Results</strong><br/>' + json.dumps(results, indent=4) + END_METADATA)

    return results


def read_file(file_path: str) -> str:
    """Read the content of a file from the repository.

    Args:
        file_path: Path to the file to read

    Returns:
        String content of the file
    """
    try:
        # Find the file in repo_files
        for file in api.get_repo_files():
            if file.path == file_path:
                return file.get_content()
        # If not found in repo_files, try to read directly
        with open(file_path, 'r') as f:
            return f.read()
    except Exception as e:
        return f"Error reading file {file_path}: {str(e)}"


def get_function_by_name(name):
    if name == "search_repo_files":
        return search_repo_files
    elif name == "read_file":
        return read_file

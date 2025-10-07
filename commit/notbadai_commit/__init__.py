from notbadai_ide import api

from .common.git_client import GitClient
from .common.llm import call_llm


def get_system_prompt() -> str:
    return """
You are an expert at writing clear, concise git commit messages. 

Given a git diff, write a commit message that:
1. Follows conventional commit format when appropriate (feat:, fix:, docs:, etc.)
2. Is concise but descriptive
3. Explains what was changed and why
4. Uses imperative mood (e.g., "Add", "Fix", "Update")

Return only the commit message, no additional text or formatting.
""".strip()


def generate_commit_message(diff: str) -> str:
    """Generate a commit message based on the git diff."""

    if not diff.strip():
        return "Empty commit"

    messages = [
        {"role": "system", "content": get_system_prompt()},
        {"role": "user", "content": f"Generate a commit message for this diff:\n\n```diff\n{diff}\n```"}
    ]

    commit_message = call_llm('qwen',
                              messages,
                              push_to_chat=False,
                              )

    commit_message = commit_message.strip().strip('"').strip("'")

    return commit_message


def start():
    """Extension that provides a commit and push interface."""
    client = GitClient(api.get_repo_path())
    ui_action = api.get_ui_action()

    state = ui_action['state']

    if len(state) == 0:
        diff = client.get_commit_diff()
        api.log(f"Git diff:\n{diff}")

        has_changes = bool(diff.strip())

        if has_changes:
            generated_message = generate_commit_message(diff)
            disabled = ''
        else:
            generated_message = "No changes to commit"
            disabled = 'disabled'

        form_content = f"""
        <form id="form">
          <textarea name="message" rows="6">{generated_message}</textarea>
          <button name="intent" value="commit" {disabled}>Commit and Push</button>
        </form>
        """

        api.ui_form('Source Control', form_content)

    elif state['intent'] == 'commit':
        commit_message = state['message'].strip()
        client.commit_push(commit_message)
        api.ui_form('', '')
    else:
        raise ValueError('Invalid tool action')

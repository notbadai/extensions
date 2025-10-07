from notbadai_ide import api

from .common.diff import get_matches
from .common.llm import call_llm


# https://docs.morphllm.com/api-reference/endpoint/apply
def start():
    code_apply_change = api.get_code_apply_change()
    patch_text = code_apply_change.patch_text
    target_file = code_apply_change.target_file
    current_file = api.get_current_file()

    prompt = patch_text.rstrip()

    if current_file and target_file.path == current_file.path:
        content = current_file.get_content()
    else:
        content = target_file.get_content()

    instruction = ''
    messages = [
        {
            "role": "user",
            "content": f"<instructions>{instruction}</instructions>\n<code>{content}</code>\n<update>{prompt}</update>"
        }
    ]

    merged_code = call_llm('morph_large',
                           messages,
                           push_to_chat=False,
                           )

    matches, cleaned_patch = get_matches(content, merged_code)
    api.update_file(cleaned_patch, matches)

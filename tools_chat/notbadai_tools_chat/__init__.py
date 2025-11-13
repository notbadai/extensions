import json

from notbadai_ide import api, START_METADATA, END_METADATA

from .common.utils import parse_prompt
from .common.prompt import build_context

from .utils import get_prompt_template, get_function_by_name
from .utils.llm import call_llm


def start():
    """Main extension function that handles chat interactions with the AI assistant."""

    model = 'qwen'

    command, _, prompt = parse_prompt()
    selection = api.get_selection()
    chat_history = api.get_chat_history()

    api.chat(f'{START_METADATA}model: {model}, command: {command}{END_METADATA}')

    if command == 'context':
        context = build_context()

        api.chat(f'{START_METADATA}With context: {len(context) :,},'
                 f' selection: {bool(selection)}{END_METADATA}')

        messages = [
            {'role': 'system', 'content': get_prompt_template('tools.system', model=model)},
            {'role': 'user', 'content': context},
            *[m.to_dict() for m in chat_history],
            {'role': 'user', 'content': prompt},
        ]
    else:
        raise ValueError(f'Unknown command: {command}')

    while True:
        res = call_llm(model, messages)
        api.log('#' * 20 + 'Call')
        has_calls = False
        for r in res:
            messages.append(r)
        for r in res:
            if 'tool_calls' in r:
                has_calls = True
                # Handle tool calls
                tool_calls = r['tool_calls']
                tool_results = []

                # Execute each tool call
                for tool_call in tool_calls:
                    function_name = tool_call['function']['name']
                    function_args = json.loads(tool_call['function']['arguments'])

                    # Get the function and execute it
                    function = get_function_by_name(function_name)
                    if function:
                        result = function(**function_args)
                        tool_results.append({
                            "role": "tool",
                            "name": function_name,
                            "content": json.dumps(result),
                            "tool_call_id": tool_call['id']
                        })

                # Add tool results to messages and call LLM again to get final response
                messages.extend(tool_results)  # Add tool results
        if not has_calls:
            break

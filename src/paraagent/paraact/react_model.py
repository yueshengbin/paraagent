
import ast
import json
import os
import re
import uuid

import httpx
from openai import OpenAI


_TOOL_CALL_BLOCK_RE = re.compile(
    r'<tool_call\b[^>]*/>|<tool_call\b[^>]*>.*?</tool_call\s*>',
    flags=re.DOTALL | re.IGNORECASE,
)


def parse_react_action(content):

    if not content:
        return None, None
    matches = list(re.finditer(r'(?m)^[ \t]*Action:[ \t]*(\S[^\n]*)', content))
    if not matches:
        return None, None
    action_match = matches[-1]
    action_name = action_match.group(1).strip().rstrip(',;').strip('`"\' ')
    if not action_name:
        return None, None
    remainder = content[action_match.end():]
    input_match = re.search(r'(?m)^[ \t]*Action Input:[ \t]*(.*)', remainder, re.DOTALL)
    if not input_match:
        return action_name, None
    raw_input = input_match.group(1)
    marker = re.search(r'(?m)^[ \t]*(?:Observation|Thought|Action)\s*:', raw_input)
    if marker:
        raw_input = raw_input[:marker.start()]
    for stop_token in ('</s>', '<|im_end|>', '<|endoftext|>'):
        if stop_token in raw_input:
            raw_input = raw_input.split(stop_token, 1)[0]
    return action_name, raw_input.strip()


def normalize_react_arguments(raw_input):

    if raw_input is None:
        return '{}'
    if isinstance(raw_input, (dict, list)):
        return json.dumps(raw_input, ensure_ascii=False)
    text = str(raw_input).strip()
    if not text:
        return '{}'
    try:
        value, _ = json.JSONDecoder(strict=False).raw_decode(text)
        if isinstance(value, (dict, list)):
            return json.dumps(value, ensure_ascii=False)
    except Exception:
        pass
    try:
        value = ast.literal_eval(text)
        if isinstance(value, (dict, list)):
            return json.dumps(value, ensure_ascii=False, default=str)
    except Exception:
        pass
    return json.dumps({'value': text}, ensure_ascii=False)


class ReActChatModel:


    def __init__(self, model='gpt-4.1-2025-04-14', base_url=None, api_key=None):
        timeout = httpx.Timeout(600.0, connect=60.0)
        self.client = OpenAI(
            api_key=api_key or os.environ.get('OPENAI_API_KEY'),
            base_url=base_url or os.environ.get('OPENAI_BASE_URL'),
            timeout=timeout,
        )
        self.model_name = model
        self.conversation_history = []

    def change_messages(self, messages):
        self.conversation_history = messages

    def parse(self, tools, process_id, **kwargs):
        del tools
        wire_messages = []
        for message in self.conversation_history:
            if message.get('valid') is False:
                continue
            role = message.get('role', 'user')
            content = message.get('content') or ''
            if role == 'tool':
                wire_message = {'role': 'user', 'content': f'Observation: {content}'}
            else:
                wire_message = {'role': role, 'content': content}
            wire_messages.append(wire_message)
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=wire_messages,
            max_tokens=int(os.environ.get('REACT_MAX_TOKENS', '2048')),
            temperature=0.0,
            **kwargs,
        )
        content = response.choices[0].message.content or ''
        completion_tokens = getattr(getattr(response, 'usage', None), 'completion_tokens', None)
        if type(completion_tokens) is not int or completion_tokens < 0:
            completion_tokens = None
        action_name, raw_input = parse_react_action(content)
        if action_name is None:
            action_name = 'Finish'
            arguments = json.dumps({
                'return_type': 'give_answer',
                'final_answer': content,
            }, ensure_ascii=False)
        else:
            arguments = normalize_react_arguments(raw_input)
        clean_content = _TOOL_CALL_BLOCK_RE.sub('', content).strip()
        message = {
            'role': 'assistant',
            'content': clean_content,
            'tool_calls': [{
                'id': uuid.uuid4().hex[:8],
                'type': 'function',
                'function': {'name': action_name, 'arguments': arguments},
            }],
        }
        if process_id == 0:
            print(f'[process({process_id})] completion tokens: {completion_tokens}, total chars: {len(content)}')
        return message, 0, completion_tokens

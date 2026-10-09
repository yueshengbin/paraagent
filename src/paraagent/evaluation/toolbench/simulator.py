import os
import yaml
import json
import re
import threading
from typing import List
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
SFT_SYSTEM = 'You are an API simulator acting as a backend server. Your task is to handle API requests and return realistic, logically consistent responses that strictly follow the API documentation and provided input parameters.\n\n### RESPONSE RULES\n\n1. **Output Format**\n   - Only return valid, well-formed JSON (no markdown, no explanations, no comments, no extra text).\n   - Response schema:\n     {\n       "error": "none" | { "type": "<error type>", "msg": "<error message>" },\n       "response": <object | string | number | array | "none">\n     }\n\n2. **Error Handling**\n   - Use "error": "none" for successful executions.\n   - For failures, return:\n     {\n       "error": { "type": "<error type>", "msg": "<brief message>" },\n       "response": "none"\n     }\n   - Possible error types include (but are not limited to): InvalidRequestError, NetworkError, NotFoundError, PermissionError, ToolExecutionError.\n\n3. **Data Generation**\n   - Generate realistic, type-correct, domain-appropriate data fully aligned with the API documentation and parameters (e.g., timestamps, valid URLs, unique numeric IDs, correct currency codes, usable email formats).\n\n4. **Logical Consistency**\n   - Maintain meaningful and coherent relationships between fields. Avoid contradictions or obviously artificial data.\n\n5. **Quality Requirements**\n   - Do not use placeholders, meaningless filler data, or repetitive patterns.\n   - Ensure outputs appear production-grade and believable.\n\n6. **Final Output Restriction**\n   - Return only the JSON object — no additional formatting, commentary, explanation, or wrapping.'
_DEFAULT_CONFIG_FILE = 'configs/eval/toolbench-simulator.yaml'

class MirrorApiSimulator:

    def __init__(self, config_file=None):
        if config_file is None:
            config_file = os.environ.get('MIRRORAPI_CONFIG', _DEFAULT_CONFIG_FILE)
        self._config_dir = os.path.dirname(os.path.abspath(config_file))
        self.CONFIG = yaml.load(open(config_file, 'r'), Loader=yaml.FullLoader)
        self.OPENAI_API_BASE = os.environ.get('TOOLENV_BASE_URL', self.CONFIG.get('api_base', 'http://127.0.0.1:12345/v1'))
        self.OPENAI_API_KEY = os.environ.get('TOOLENV_API_KEY', self.CONFIG.get('api_key', 'EMPTY'))
        self.client = None
        self.SFT_SYSTEM = SFT_SYSTEM
        cache_folder = self.CONFIG.get('cache_folder')
        if cache_folder:
            if not os.path.isabs(cache_folder):
                cache_folder = os.path.join(self._config_dir, cache_folder)
        self.cache_folder = cache_folder
        self.is_save = self.CONFIG.get('is_save', False)
        self._cache_lock = threading.Lock()

    def _get_client(self):
        if self.client is None:
            import httpx
            self.client = OpenAI(base_url=self.OPENAI_API_BASE, api_key=self.OPENAI_API_KEY, http_client=httpx.Client(trust_env=False, timeout=60))
        return self.client

    def _standard_category(self, category: str) -> str:
        s = category.replace(' ', '_').replace(',', '_').replace('/', '_')
        s = re.sub('_{2,}', '_', s)
        return s

    def _get_cache_path(self, info: dict):

        if not self.cache_folder:
            return (None, None)
        lookup_name = info.get('_lookup_name', '')
        category = info.get('category', '')
        if not lookup_name or '-' not in lookup_name or (not category):
            return (None, None)
        tool_name_std, api_name_std = lookup_name.split('-', 1)
        std_cat = self._standard_category(category)
        tool_name_for = f'{tool_name_std}_for_{std_cat}'
        cache_file = os.path.join(self.cache_folder, std_cat, tool_name_for, api_name_std + '.json')
        cache_key = str(info.get('tool_input', {}))
        return (cache_file, cache_key)

    def _load_from_cache(self, info: dict):

        cache_file, cache_key = self._get_cache_path(info)
        if not cache_file:
            return None
        try:
            if not os.path.exists(cache_file):
                return None
            cache = json.load(open(cache_file, 'r'))
            if cache_key in cache:
                print(f'[Cache] hit: {os.path.relpath(cache_file, self.cache_folder)}')
                return cache[cache_key]
        except Exception as e:
            print(f'[Cache] load error: {e}')
        return None

    def _save_to_cache(self, info: dict, result: dict):

        if not self.is_save:
            return
        cache_file, cache_key = self._get_cache_path(info)
        if not cache_file:
            return
        try:
            with self._cache_lock:
                cache = {}
                if os.path.exists(cache_file):
                    cache = json.load(open(cache_file, 'r'))
                if isinstance(result, dict):
                    cache[cache_key] = result
                elif isinstance(result, str):
                    cache[cache_key] = json.loads(result)
                os.makedirs(os.path.dirname(cache_file), exist_ok=True)
                json.dump(cache, open(cache_file, 'w'), indent=4, ensure_ascii=False)
        except Exception as e:
            print(f'[Cache] save error: {e}')

    def standardize(self, s: str):
        return s.lower().replace(' ', '_').replace('-', '_')

    def extract_attributes_json(self, output):
        try:
            output_dict = json.loads(output)
            error_content, response_content = (output_dict.get('error'), output_dict.get('response'))
        except:
            error_pattern = '"error"\\s*:\\s*(null|"[^"]*")'
            error_match = re.search(error_pattern, output)
            error_content = None
            if error_match:
                val = error_match.group(1)
                if val == 'null':
                    error_content = None
                else:
                    error_content = val.strip('"')
            if '"response":' in output:
                response_part = output.split('"response":', 1)[1].strip()
                if response_part.startswith('{'):
                    response_content = response_part
                elif response_part.startswith('"'):
                    response_content = response_part[1:]
                else:
                    response_content = response_part
            else:
                response_content = None
        return (None, error_content, response_content)

    def call_api(self, messages):
        try:
            generate_texts = self._get_client().chat.completions.create(model=self.CONFIG.get('model', 'toolenv-sft'), messages=messages, temperature=self.CONFIG['temperature'], max_tokens=512, response_format={'type': 'json_object'})
            generate_text = generate_texts.choices[0].message.content
            _, error, response = self.extract_attributes_json(generate_text)
            if error is not None or response is not None:
                return {'content': {'error': error, 'response': response}, 'success': True}
            else:
                return {'content': {'error': 'API failed error', 'response': response}, 'success': False}
        except Exception as e:
            print(f'OpenAI call failed: {e}')
            return {'content': {'error': 'API not working error...', 'response': 'none'}, 'success': False}

    def _build_messages(self, info: dict) -> list:
        USER_PROMPT = '## API Documentation:\n{api_doc}\n\n## Input Parameters:\n{request}\n'
        tool_input = info.get('tool_input')
        if info.get('platform_overview', '') != '':
            single_api_doc = {'- Category': info.get('category'), '- API Name': info.get('name'), '- API Description': info.get('description'), '- Parameters': info.get('parameters'), '- Platform Overview': info.get('platform_overview')}
        else:
            single_api_doc = {'- Category': info.get('category'), '- API Name': info.get('name'), '- API Description': info.get('description'), '- Parameters': info.get('parameters')}
        single_api_doc = '\n'.join((f'{k}: {v}' for k, v in single_api_doc.items()))
        instruction = USER_PROMPT.format(api_doc=single_api_doc, request={**tool_input})
        return [{'role': 'system', 'content': self.SFT_SYSTEM}, {'role': 'user', 'content': instruction}]

    def fake_response_batch(self, tool_infos: List[dict]) -> List[dict]:
        results: List[dict] = [None] * len(tool_infos)
        llm_pending = []
        for idx, info in enumerate(tool_infos):
            cached = self._load_from_cache(info)
            if cached is not None:
                results[idx] = {'content': cached, 'success': True}
            else:
                llm_pending.append((idx, self._build_messages(info), info))
        if not llm_pending:
            return results
        max_workers = min(128, len(llm_pending))
        try:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_pending = {executor.submit(self.call_api, messages): (orig_idx, info) for orig_idx, messages, info in llm_pending}
                for future in as_completed(future_to_pending):
                    orig_idx, info = future_to_pending[future]
                    try:
                        llm_result = future.result()
                    except Exception as e:
                        print(f'call_api future failed at idx={orig_idx}: {e}')
                        llm_result = {'content': {'error': 'API not working error...', 'response': 'none'}, 'success': False}
                    results[orig_idx] = llm_result
                    if llm_result.get('success') and isinstance(llm_result.get('content'), dict):
                        self._save_to_cache(info, llm_result['content'])
        except Exception as e:
            print(f'ThreadPoolExecutor failed: {e}. Fallback to serial execution.')
            for orig_idx, messages, info in llm_pending:
                llm_result = self.call_api(messages)
                results[orig_idx] = llm_result
                if llm_result.get('success') and isinstance(llm_result.get('content'), dict):
                    self._save_to_cache(info, llm_result['content'])
        return results

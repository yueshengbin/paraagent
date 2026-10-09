
import json
import copy
import importlib.util
import os
import requests
from paraagent.evaluation.base_env import BaseEnvironment
FINISH_FUNC = {'type': 'function', 'function': {'name': 'Finish', 'description': 'If you believe that you have obtained a result that can answer the task, please call this function to provide the final answer. Alternatively, if you recognize that you are unable to proceed with the task in the current state, call this function to restart. Remember: you must ALWAYS call this function at the end of your attempt, and the only part that will be shown to the user is the final answer, so it should contain sufficient information.', 'parameters': {'type': 'object', 'properties': {'return_type': {'type': 'string', 'enum': ['give_answer', 'give_up_and_restart']}, 'final_answer': {'type': 'string', 'description': 'The final answer you want to give the user. You should have this field if "return_type"=="give_answer"'}}, 'required': ['return_type']}}}

def _apibank_type_to_openai(t):
    return {'int': 'integer', 'float': 'number', 'bool': 'boolean', 'list': 'array', 'str': 'string'}.get(t, 'string')

def _apibank_schema_to_openai(api_json):

    properties = {}
    required = []
    for param_name, param_info in api_json.get('input_parameters', {}).items():
        ptype = _apibank_type_to_openai(param_info.get('type', 'str'))
        prop = {'type': ptype, 'description': param_info.get('description', '')}
        if ptype == 'array':
            prop['items'] = {'type': 'string'}
        properties[param_name] = prop
        required.append(param_name)
    return {'type': 'function', 'function': {'name': api_json['name'], 'description': api_json.get('description', ''), 'parameters': {'type': 'object', 'properties': properties, 'required': required}}}

def _extract_api_list(query_json):

    api_list = query_json.get('api_list')
    if api_list:
        return api_list
    extracted = []
    seen = set()
    for call in query_json.get('apis', []) or []:
        if not isinstance(call, dict) or call.get('api_name') != 'ToolSearcher':
            continue
        output = call.get('output', {})
        if isinstance(output, dict):
            output = output.get('output', output)
        if not isinstance(output, dict):
            continue
        name = output.get('name')
        if not name or name in seen:
            continue
        if 'input_parameters' not in output:
            continue
        extracted.append(output)
        seen.add(name)
    return extracted

class _SimpleExecutor:


    def __init__(self, apis_dir, database_dir=None):
        self.apis_dir = apis_dir
        self._module_cache = {}
        self._init_databases = {}
        if database_dir and os.path.isdir(database_dir):
            for f in os.listdir(database_dir):
                if f.endswith('.json'):
                    try:
                        with open(os.path.join(database_dir, f)) as fh:
                            self._init_databases[f[:-5]] = json.load(fh)
                    except Exception:
                        pass
        import sys
        if apis_dir and apis_dir not in sys.path:
            sys.path.insert(0, apis_dir)
        self._token_checker = self._init_token_checker()

    def _init_token_checker(self):

        if not self.apis_dir or not os.path.isdir(self.apis_dir):
            return None
        skip = {'__init__.py', 'api.py', 'tool_search.py'}
        for fname in os.listdir(self.apis_dir):
            if not fname.endswith('.py') or fname in skip:
                continue
            try:
                mod_key = fname[:-3]
                path = os.path.join(self.apis_dir, fname)
                spec = importlib.util.spec_from_file_location(mod_key, path)
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                self._module_cache[mod_key] = mod
                check_cls = getattr(mod, 'CheckToken', None)
                if check_cls is not None:
                    init_kwargs = {}
                    if hasattr(check_cls, 'database_name'):
                        db = self._init_databases.get(getattr(check_cls, 'database_name'))
                        if db is not None:
                            init_kwargs['init_database'] = db
                    return check_cls(**init_kwargs) if init_kwargs else check_cls()
            except Exception:
                pass
        return None

    def execute(self, tool_name, arguments):

        import inspect
        if not self.apis_dir or not os.path.isdir(self.apis_dir):
            return {'error': f'apis_dir not found: {self.apis_dir}'}
        skip = {'__init__.py', 'api.py', 'tool_search.py'}
        for fname in os.listdir(self.apis_dir):
            if not fname.endswith('.py') or fname in skip:
                continue
            mod_key = fname[:-3]
            if mod_key not in self._module_cache:
                try:
                    path = os.path.join(self.apis_dir, fname)
                    spec = importlib.util.spec_from_file_location(mod_key, path)
                    mod = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(mod)
                    self._module_cache[mod_key] = mod
                except Exception:
                    self._module_cache[mod_key] = None
            mod = self._module_cache.get(mod_key)
            if mod is None:
                continue
            cls = getattr(mod, tool_name, None)
            if cls and isinstance(cls, type) and hasattr(cls, 'call'):
                try:
                    init_kwargs = {}
                    if hasattr(cls, 'database_name'):
                        db = self._init_databases.get(getattr(cls, 'database_name'))
                        if db is not None:
                            init_kwargs['init_database'] = db
                    needs_token = False
                    if hasattr(cls, 'input_parameters'):
                        needs_token = 'token' in cls.input_parameters
                    if needs_token and self._token_checker is not None:
                        try:
                            if 'token_checker' in inspect.signature(cls.__init__).parameters:
                                init_kwargs['token_checker'] = self._token_checker
                        except Exception:
                            pass
                    instance = cls(**init_kwargs) if init_kwargs else cls()
                    return {'result': instance.call(**arguments)}
                except Exception as e:
                    return {'error': str(e)}
        return {'error': f"Tool '{tool_name}' not found in {self.apis_dir}"}
SEARCH_FUNC = {'type': 'function', 'function': {'name': 'search_tool', 'description': 'Searches for relevant tools in the library based on a natural-language query and returns a ranked list of matching tools.', 'parameters': {'type': 'object', 'properties': {'query': {'type': 'string', 'description': 'A natural-language query describing the tool capability you need.'}}, 'required': ['query']}}}

class ApiBankEnvironment(BaseEnvironment):


    def __init__(self, query_json, args, process_id=0, paradigm=None):
        super().__init__()
        self.max_observation_length = args.max_observation_length
        self.process_id = process_id
        self.apis_dir = getattr(args, 'apibank_apis_dir', None)
        self.database_dir = getattr(args, 'apibank_database_dir', None)
        self.paradigm = paradigm or getattr(args, 'paradigm', 'EaE')
        self.input_description = query_json.get('query') or query_json.get('requirement', '')
        self.task_description = self.input_description
        self.expected_output = query_json.get('expected_output', query_json.get('response', ''))
        self._executor = _SimpleExecutor(self.apis_dir, self.database_dir) if self.apis_dir else None
        self._api_list = _extract_api_list(query_json)
        self.allowed_tool_names = None
        if self.paradigm == 'EaE':
            self.functions = [SEARCH_FUNC, FINISH_FUNC]
            self.allowed_tool_names = {'search_tool', 'Finish'}
        elif self.paradigm == 'ETE':
            self.functions = []
            self.allowed_tool_names = set()
            top_k = getattr(args, 'f_top_k', 3)
            api_url = os.environ.get('TOOL_SEARCH_API_URL', 'http://127.0.0.1:30402')
            try:
                resp = requests.post(f'{api_url}/search', json={'queries': [self.input_description], 'top_k': top_k}, proxies={'http': '', 'https': ''}, timeout=15)
                if resp.status_code == 200:
                    for qr in resp.json().get('query_results', []):
                        for result in qr.get('results', []):
                            tool = result.get('tools', {})
                            fn_name = tool.get('name') or tool.get('tool_name')
                            if not fn_name or fn_name in self.allowed_tool_names:
                                continue
                            self.functions.append({'type': 'function', 'function': {'name': fn_name, 'description': tool.get('description', ''), 'parameters': tool.get('parameters', {'type': 'object', 'properties': {}})}})
                            self.allowed_tool_names.add(fn_name)
                    if process_id == 0 and len(self.allowed_tool_names) < top_k:
                        print(f'[WARNING] API-Bank ETE: requested top_k={top_k}, got {len(self.allowed_tool_names)} unique tools after dedup')
                elif process_id == 0:
                    print(f'[WARNING] API-Bank ETE retrieval returned HTTP {resp.status_code}')
            except Exception as e:
                if process_id == 0:
                    print(f'[WARNING] API-Bank ETE retrieval failed: {e}')
            self.functions.append(FINISH_FUNC)
            self.allowed_tool_names.add('Finish')
        else:
            raise ValueError("paradigm must be 'ETE' or 'EaE'.")
        self.tool_names = [f['function']['name'] for f in self.functions]
        self.success = 0
        self.final_answer = ''
        self.called_apis = []

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        result.max_observation_length = self.max_observation_length
        result.process_id = self.process_id
        result.apis_dir = self.apis_dir
        result.database_dir = self.database_dir
        result.paradigm = self.paradigm
        result.input_description = self.input_description
        result.task_description = self.task_description
        result.expected_output = self.expected_output
        result._executor = self._executor
        result._api_list = self._api_list
        result.functions = copy.deepcopy(self.functions, memo)
        result.tool_names = self.tool_names.copy()
        result.allowed_tool_names = self.allowed_tool_names.copy() if self.allowed_tool_names is not None else None
        result.success = self.success
        result.final_answer = self.final_answer
        result.called_apis = self.called_apis.copy()
        return result

    def check_success(self):
        return self.success

    def to_json(self):
        return {'expected_output': self.expected_output, 'final_answer': self.final_answer, 'called_apis': self.called_apis}

    def restart(self):
        self.success = 0
        self.final_answer = ''
        self.called_apis = []

    def get_score(self):
        return 1.0 if self.success == 1 else 0.0

    def step(self, action_name='', action_input=''):
        obs, code = self._step(action_name=action_name, action_input=action_input)
        if len(obs) > self.max_observation_length:
            obs = obs[:self.max_observation_length] + '...'
        return (obs, code)

    def _step(self, action_name='', action_input=''):
        if isinstance(action_input, str):
            try:
                action_input = json.loads(action_input)
            except Exception:
                import ast
                try:
                    action_input = ast.literal_eval(action_input)
                except Exception:
                    pass
        if not isinstance(action_input, dict):
            action_input = {}
        if action_name == 'Finish':
            return_type = action_input.get('return_type', 'give_up_and_restart')
            if return_type == 'give_answer':
                self.final_answer = str(action_input.get('final_answer', ''))
                self.success = 1
                return (json.dumps({'response': 'successfully giving the final answer.'}), 3)
            else:
                return (json.dumps({'response': 'chose to give up and restart'}), 4)
        if self.allowed_tool_names is not None and action_name not in self.allowed_tool_names:
            return (json.dumps({'name': action_name, 'result': {'error': {'type': 'NotFoundError', 'msg': f'Tool is outside the available tool set: {action_name}'}}}, ensure_ascii=False), 1)
        if self._executor:
            result = self._executor.execute(action_name, action_input)
        else:
            result = {'result': f'(no executor) Called {action_name} with {action_input}'}
        self.called_apis.append({'tool': action_name, 'input': action_input, 'output': result})
        error_value = result.get('error') if isinstance(result, dict) else None
        exception_value = result.get('exception') if isinstance(result, dict) else None
        if error_value or exception_value:
            error_msg = str(error_value or exception_value)
            if 'not found' in error_msg.lower():
                return (json.dumps({'name': action_name, 'result': {'error': {'type': 'NotFoundError', 'msg': f'No such tool: {action_name}'}}}, ensure_ascii=False), 1)
            return (json.dumps({'name': action_name, 'result': {'error': {'type': 'ToolExecutionError', 'msg': error_msg}}}, ensure_ascii=False), 2)
        return (json.dumps({'name': action_name, 'result': result.get('result', result)}, ensure_ascii=False), 0)

import os
import json
import time
import threading
import copy
import requests
from termcolor import colored
from paraagent.evaluation.toolbench.simulator import MirrorApiSimulator
from paraagent.evaluation.toolbench.service import ToolBenchServiceClient, ToolBenchServiceError
from jsonschema import validate, ValidationError
import pandas as pd
from paraagent.paraact.naming import standardize, change_name
from paraagent.evaluation.base_env import BaseEnvironment
import ast
_API_NAME_REFLECT_CACHE = None
_API_NAME_STD_TO_KEY_CACHE = None
_API_NAME_REFLECT_LOCK = threading.Lock()
_SIMULATOR = None
_SIMULATOR_LOCK = threading.Lock()

def get_api_name_reflect_cached(tsv_path):
    global _API_NAME_REFLECT_CACHE, _API_NAME_STD_TO_KEY_CACHE
    if _API_NAME_REFLECT_CACHE is not None:
        return _API_NAME_REFLECT_CACHE
    with _API_NAME_REFLECT_LOCK:
        if _API_NAME_REFLECT_CACHE is not None:
            return _API_NAME_REFLECT_CACHE
        api_name_reflect = {}
        df = pd.read_csv(tsv_path, sep='\t')
        for item in df.itertuples():
            if ' : ' in item.name:
                tool_name, api_name = item.name.split(' : ', 1)
                name = f'{standardize(tool_name)}-{change_name(standardize(api_name))}'[:64]
            else:
                name = item.name[:64]
            api_name_reflect[name] = json.loads(item.document_content)
        _API_NAME_STD_TO_KEY_CACHE = {standardize(k): k for k in api_name_reflect}
        _API_NAME_REFLECT_CACHE = api_name_reflect
    return api_name_reflect

def get_std_to_key_cached():
    if _API_NAME_STD_TO_KEY_CACHE is None:
        raise RuntimeError('get_api_name_reflect_cached must be called before get_std_to_key_cached')
    return _API_NAME_STD_TO_KEY_CACHE

def normalize_tool_name(name):

    if _API_NAME_STD_TO_KEY_CACHE is None:
        return name
    return _API_NAME_STD_TO_KEY_CACHE.get(standardize(name), name)

def get_simulator():
    global _SIMULATOR
    if _SIMULATOR is not None:
        return _SIMULATOR
    with _SIMULATOR_LOCK:
        if _SIMULATOR is None:
            _SIMULATOR = MirrorApiSimulator()
    return _SIMULATOR

class ToolBenchEnvironment(BaseEnvironment):

    def __init__(self, query_json, tool_descriptions, retriever, args, process_id=0):
        super().__init__()
        self.tool_root_dir = args.tool_root_dir
        self.max_observation_length = args.max_observation_length
        self.observ_compress_method = args.observ_compress_method
        self.retriever = retriever
        self.process_id = process_id
        self.paradigm = args.paradigm
        self.backend = getattr(args, 'toolbench_backend', 'simulator')
        if self.backend not in {'simulator', 'live', 'virtual', 'mirrorapi'}:
            raise ValueError(f'Unknown ToolBench backend: {self.backend}')
        self.service_client = None
        if self.backend in {'live', 'virtual', 'mirrorapi'}:
            self.service_client = ToolBenchServiceClient(
                args.toolbench_service_url, args.toolbench_key, backend=self.backend)
        self.tool_names = []
        self.cate_names = []
        self.input_description = query_json['query']
        self.functions = []
        self.name_reflect = get_api_name_reflect_cached(args.tool_root_dir)
        self.allowed_tool_names = None
        search_func = {'type': 'function', 'function': {'name': 'search_tool', 'description': 'Searches for relevant tools in library based on a natural-language query and returns a ranked list of matching tools.', 'parameters': {'type': 'object', 'properties': {'query': {'type': 'string', 'description': 'A natural-language query to match against available tools by name, capability, and metadata.'}}, 'required': ['query']}}}
        finish_func = {'type': 'function', 'function': {'name': 'Finish', 'description': 'If you believe that you have obtained a result that can answer the task, please call this function to provide the final answer. Alternatively, if you recognize that you are unable to proceed with the task in the current state, call this function to restart. Remember: you must ALWAYS call this function at the end of your attempt, and the only part that will be shown to the user is the final answer, so it should contain sufficient information.', 'parameters': {'type': 'object', 'properties': {'return_type': {'type': 'string', 'enum': ['give_answer', 'give_up_and_restart']}, 'final_answer': {'type': 'string', 'description': 'The final answer you want to give the user. You should have this field if "return_type"=="give_answer"'}}, 'required': ['return_type']}}}
        if self.paradigm == 'ETE':
            from paraagent.paraact.retrieval import ToolSearch
            top_k = getattr(args, 'f_top_k', 3)
            tool_searcher = ToolSearch()
            self.allowed_tool_names = set()
            try:
                payload = {'queries': [self.input_description], 'top_k': top_k}
                resp = requests.post(f'{tool_searcher.api_url}/search', json=payload, proxies={'http': '', 'https': ''}, timeout=15)
                if resp.status_code == 200:
                    for qr in resp.json().get('query_results', []):
                        for result in qr.get('results', []):
                            schema = tool_searcher.api_json_to_openai_json(result['tools'])
                            func_json = {'type': 'function', **schema}
                            fn_name = func_json['function']['name']
                            if fn_name not in self.allowed_tool_names:
                                self.functions.append(func_json)
                                self.allowed_tool_names.add(fn_name)
                    if process_id == 0 and len(self.allowed_tool_names) < top_k:
                        print(f'[WARNING] ETE: requested top_k={top_k}, got {len(self.allowed_tool_names)} unique tools after dedup')
                elif process_id == 0:
                    print(f'[WARNING] ETE retrieval returned HTTP {resp.status_code}')
            except Exception as e:
                if process_id == 0:
                    print(f'[WARNING] ETE one-time retrieval failed: {e}')
            self.functions.append(finish_func)
            self.allowed_tool_names.add('Finish')
        elif self.paradigm == 'EaE':
            self.allowed_tool_names = {'search_tool', 'Finish'}
            self.functions.append(search_func)
            self.functions.append(finish_func)
        else:
            raise ValueError("paradigm must be 'ETE' or 'EaE'.")
        self.CALL_MAX_TIME = 3
        self.task_description = f"""You should use functions to help handle the real time user querys. Remember:\n1.ALWAYS call "Finish" function at the end of the task. And the final answer should contain enough information to show to the user,If you can't handle the task, or you find that function calls always fail(the function is not valid now), use function Finish->give_up_and_restart.\n2.Do not use origin tool names, use only subfunctions' names.\nYou have access of the following tools:\n"""
        unduplicated_reflection = {}
        for standardize_tool_name, tool_des in tool_descriptions:
            unduplicated_reflection[standardize_tool_name] = tool_des
        for k, (standardize_tool_name, tool_des) in enumerate(unduplicated_reflection.items()):
            try:
                stripped = tool_des[:512].replace('\n', '').strip()
            except:
                stripped = ''
            if stripped == '':
                stripped = 'None'
            self.task_description += f'{k + 1}.{standardize_tool_name}: {stripped}\n'
        self.success = 0

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        result.tool_root_dir = self.tool_root_dir
        result.max_observation_length = self.max_observation_length
        result.observ_compress_method = self.observ_compress_method
        result.retriever = self.retriever
        result.process_id = self.process_id
        result.paradigm = self.paradigm
        result.backend = self.backend
        result.service_client = self.service_client
        result.input_description = self.input_description
        result.task_description = self.task_description
        result.name_reflect = self.name_reflect
        result.allowed_tool_names = self.allowed_tool_names.copy() if self.allowed_tool_names is not None else None
        result.CALL_MAX_TIME = self.CALL_MAX_TIME
        result.tool_names = self.tool_names.copy()
        result.cate_names = self.cate_names.copy()
        result.functions = copy.deepcopy(self.functions, memo)
        result.success = self.success
        return result

    def retrieve_tools(self, query, top_k, jsons_path):
        retrieved_tools = self.retriever.retrieving(query, top_k=top_k)
        query_json = {'api_list': []}
        for tool_dict in retrieved_tools:
            if len(query_json['api_list']) == top_k:
                break
            category = tool_dict['category']
            tool_name = tool_dict['tool_name']
            api_name = tool_dict['api_name']
            if os.path.exists(jsons_path):
                if os.path.exists(os.path.join(jsons_path, category)):
                    if os.path.exists(os.path.join(jsons_path, category, tool_name + '.json')):
                        query_json['api_list'].append({'category_name': category, 'tool_name': tool_name, 'api_name': api_name})
        return query_json

    def api_json_to_openai_json(self, api_json):
        function_template = {'type': 'function', 'function': {'name': '', 'description': '', 'parameters': {'type': 'object', 'properties': {}, 'required': []}}}
        template = function_template['function']
        map_type = {'NUMBER': 'integer', 'STRING': 'string', 'BOOLEAN': 'boolean'}
        standard_tool_name = standardize(api_json['tool_name'])
        pure_api_name = change_name(standardize(api_json['api_name']))
        template['name'] = f'{standard_tool_name}-{pure_api_name}'[:64]
        template['description'] = f'This is the subfunction for tool "{standard_tool_name}", you can use this tool.'
        if api_json['api_description'].strip() != '':
            truncated_description = api_json['api_description'].strip()
            template['description'] = template['description'] + f'The description of this function is: "{truncated_description}"'
        if 'required_parameters' in api_json.keys() and len(api_json['required_parameters']) > 0:
            for para in api_json['required_parameters']:
                name = standardize(para['name'])
                name = change_name(name)
                if para['type'] in map_type:
                    param_type = map_type[para['type']]
                else:
                    param_type = 'string'
                prompt = {'type': param_type, 'description': para['description']}
                default_value = para['default']
                if len(str(default_value)) != 0:
                    prompt = {'type': param_type, 'description': para['description'], 'example_value': default_value}
                else:
                    prompt = {'type': param_type, 'description': para['description']}
                template['parameters']['properties'][name] = prompt
                template['parameters']['required'].append(name)
            for para in api_json['optional_parameters']:
                name = standardize(para['name'])
                name = change_name(name)
                if para['type'] in map_type:
                    param_type = map_type[para['type']]
                else:
                    param_type = 'string'
                default_value = para['default']
                if len(str(default_value)) != 0:
                    prompt = {'type': param_type, 'description': para['description'], 'example_value': default_value}
                else:
                    prompt = {'type': param_type, 'description': para['description']}
                template['parameters']['properties'][name] = prompt
        template['parameters']['required'] = list(dict.fromkeys(template['parameters']['required']))
        return (function_template, api_json['category_name'], standard_tool_name)

    def check_success(self):
        return self.success

    def to_json(self):
        return {}

    def restart(self):
        pass

    def get_score(self):
        return 0.0

    def validate_schema(self, input1, schema) -> bool:
        try:
            validate(instance=input1, schema=schema)
            return True
        except ValidationError:
            return False

    def step(self, **args):
        obs, code = self._step(**args)
        if '<tools>' not in obs:
            if len(obs) > self.max_observation_length:
                obs = obs[:self.max_observation_length] + '...'
            return (obs, code)
        else:
            return (obs, code)

    def _step(self, action_name='', action_input=''):


        def make_return(data, code):
            return (json.dumps(data, ensure_ascii=False), code)

        def parse_action_input_lenient(raw_input):
            if not isinstance(raw_input, str):
                return raw_input
            try:
                return json.loads(raw_input)
            except Exception:
                try:
                    return ast.literal_eval(raw_input)
                except Exception:
                    return raw_input
        if isinstance(action_input, str):
            action_input = parse_action_input_lenient(action_input)
        if action_name == 'Error':
            res = {'name': action_name, 'result': action_input}
            return make_return(res, 1)
        if action_name == 'Finish':
            try:
                if isinstance(action_input, str):
                    json_data = json.loads(action_input, strict=False)
                elif isinstance(action_input, (dict, list)):
                    json_data = action_input
            except:
                json_data = {}
                if '"return_type": "' in action_input or 'return_type' in action_input:
                    if '"return_type": "give_answer"' in action_input:
                        return_type = 'give_answer'
                    elif '"return_type": "give_up_and_restart"' in action_input:
                        return_type = 'give_up_and_restart'
                    else:
                        return_type = action_input[action_input.find('"return_type": "') + len('"return_type": "'):action_input.find('",')]
                    json_data['return_type'] = return_type
                if '"final_answer": "' in action_input:
                    final_answer = action_input[action_input.find('"final_answer": "') + len('"final_answer": "'):]
                    json_data['final_answer'] = final_answer
            if not isinstance(json_data, dict):
                return (f'{{error: {json_data}}}', 2)
            if 'return_type' not in json_data.keys():
                return ('{error:"must have "return_type""}', 2)
            if json_data['return_type'] == 'give_up_and_restart':
                return ('{"response":"chose to give up and restart"}', 4)
            elif json_data['return_type'] == 'give_answer':
                if 'final_answer' not in json_data.keys():
                    return ('{error:"must have "final_answer""}', 2)
                self.success = 1
                return ('{"response":"successfully giving the final answer."}', 3)
            else:
                return ('{error:""return_type" is not a valid choice"}', 2)
        else:
            if action_name not in self.name_reflect:
                action_name = normalize_tool_name(action_name)
            if self.allowed_tool_names is not None and action_name not in self.allowed_tool_names:
                return make_return({'name': action_name, 'result': {'error': {'type': 'InvalidRequestError', 'msg': f'Tool is outside the available tool set: {action_name}'}}}, 1)
            if action_name in self.name_reflect:
                function = self.name_reflect[action_name]
                platform_overview = function.get('platform_overview', '')
                function['parameters']['required'] = list(dict.fromkeys(function['parameters']['required']))
                payload = {'category': function['category'], 'name': function['name'], 'description': function['description'], 'parameters': function['parameters'], 'platform_overview': platform_overview, 'tool_input': action_input, '_lookup_name': action_name}
                if self.process_id == 0:
                    print(colored(f"query to {function['category']}-->{action_name}", color='yellow'))
                if not self.validate_schema(action_input, function['parameters']):
                    keys_str = 'Invalid Input'
                    if isinstance(action_input, dict):
                        keys_str = ','.join(action_input.keys())
                    return make_return({'error': {'type': 'InvalidRequestError', 'msg': f'Invalid tool parameters: {keys_str}'}, 'response': ''}, 12)
                try:
                    if self.backend in {'live', 'virtual', 'mirrorapi'}:
                        response = self.service_client.call(function, action_input, self.observ_compress_method)
                    else:
                        response = get_simulator().fake_response_batch([payload])[0]['content']
                except requests.exceptions.Timeout:
                    return (json.dumps({'name': function['name'], 'result': {'error': {'type': 'NetworkError', 'msg': f'Timeout error...'}}}, ensure_ascii=False), 5)
                except (ToolBenchServiceError, KeyError, IndexError, TypeError) as exc:
                    return make_return({'error': {'type': 'ToolExecutionError', 'msg': str(exc)}}, 12)
                if not isinstance(response, dict):
                    print(f'[WARNING] MirrorAPI returned non-object content: {response!r}')
                    return make_return({'error': {'type': 'ToolExecutionError', 'msg': 'MirrorAPI returned malformed response content'}, 'response': ''}, 12)
                if response['error'] == 'API not working error...':
                    status_code = 6
                elif response['error'] == 'Unauthorized error...':
                    status_code = 7
                elif response['error'] == 'Unsubscribed error...':
                    status_code = 8
                elif response['error'] == 'Too many requests error...':
                    status_code = 9
                elif response['error'] == 'Rate limit per minute error...':
                    print('Reach api calling limit per minute, sleeping...')
                    time.sleep(10)
                    status_code = 10
                elif response['error'] == 'Message error...':
                    status_code = 11
                else:
                    status_code = 0
                response_error = response.get('error')
                no_error = response_error is None or (isinstance(response_error, str) and response_error.strip().lower() in ('none', 'null', ''))
                if no_error:
                    final_result = response['response']
                else:
                    final_result = {'error': response_error}
                return make_return({'name': action_name, 'result': final_result}, status_code)
            return make_return({'name': action_name, 'result': {'error': {'type': 'InvalidRequestError', 'msg': f'No such tool : {action_name}'}}}, 1)

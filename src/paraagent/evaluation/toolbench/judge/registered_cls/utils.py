import os
from openai import OpenAI
__registered_evaluators__ = {}

def register_evaluator(cls):

    __registered_evaluators__[cls.__name__] = cls
    return cls

def get_evaluator_cls(clsname):

    evaluator_cls = __registered_evaluators__.get(clsname)
    if evaluator_cls is None:
        raise ModuleNotFoundError(f'Cannot find evaluator class {clsname}')
    return evaluator_cls

class OpenaiPoolRequest:

    def __init__(self, api_config):
        if api_config is None:
            raise ValueError("ToolBench judge requires an explicit API config")
        self.api_config = api_config

    def request(self, messages, **kwargs):
        client_kwargs = {'api_key': self.api_config['api_key'], 'base_url': self.api_config['base_url']}
        if os.environ.get('OPENAI_MAX_RETRIES') is not None:
            client_kwargs['max_retries'] = int(os.environ['OPENAI_MAX_RETRIES'])
        client = OpenAI(**client_kwargs)
        response = client.chat.completions.create(messages=messages, **kwargs)
        return response

    def __call__(self, messages, **kwargs):
        return self.request(messages, **kwargs)

import random
from copy import deepcopy
from typing import List, Union, Dict, Any, Callable
import os
import yaml
from .utils import register_evaluator

def process_answer(answer: Dict):
    answer = deepcopy(answer)
    answer['final_answer'] = answer['final_answer'][:1000]
    answer['answer_details'] = answer['answer_details'][:3000]
    answer.pop('method', None)
    return answer

def process_tools(tools: List[Dict]):
    tools = deepcopy(tools)
    for tool in tools:
        tool.pop('description', None)
        tool.pop('parameters', None)
    return tools

@register_evaluator
class BaseEvaluator:


    def __init__(self, fn_completions: Callable[[Dict, List[Dict]], int]=None, *args, **kwargs):
        self.fn_completions = fn_completions

    def annotate_preference(self, query: str, available_tools: List[Dict[Any, Any]], answers: List[Dict], multisample=False, sample_n=4, task_status=None, answer_statuses=None) -> Union[List[int], int]:

        answers_processed = [process_answer(ans) for ans in answers]
        available_tools = process_tools(available_tools)
        if answer_statuses is None:
            answer_statuses = [None] * len(answers_processed)

        def shuffle_run() -> int:
            indices = list(range(len(answers_processed)))
            random.shuffle(indices)
            answers_projected = [answers_processed[idx] for idx in indices]
            statuses_projected = [answer_statuses[idx] for idx in indices]
            preferred_index = self.fn_completions({'query': query, 'available_tools': available_tools}, answers_projected, task_status, statuses_projected)
            if 0 <= preferred_index < len(indices):
                return indices[preferred_index]
            raise ValueError(f'Preferred index {preferred_index} is invalid!')
        if not multisample:
            return shuffle_run()
        else:
            prefers = [shuffle_run() for _ in range(sample_n)]
            return prefers

@register_evaluator
class ToolEvalEvaluator(BaseEvaluator):


    def __init__(self, cfg_path: str=None):
        eval_config = yaml.load(open(os.path.join(cfg_path, 'config.yaml')), Loader=yaml.FullLoader)
        template = open(os.path.join(cfg_path, eval_config['prompt_template'])).read()
        super().__init__(fn_completions=getattr(self, eval_config['fn_completions']))
        self.eval_config = eval_config
        self.template = template

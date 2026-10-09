from .registered_cls import BaseEvaluator, get_evaluator_cls
__all__ = ['get_evaluator_cls', 'BaseEvaluator', 'load_evaluator']

def load_evaluator(profile_name, profiles_root, api_config=None) -> BaseEvaluator:
    import os
    import yaml
    if api_config is None:
        raise ValueError('ToolBench evaluator requires an explicit API config')
    cfg_path = os.path.join(profiles_root, profile_name)
    with open(os.path.join(cfg_path, 'config.yaml')) as config_file:
        cls_name = yaml.safe_load(config_file)['registered_cls_name']
    return get_evaluator_cls(cls_name)(cfg_path, api_config=api_config)



def load_registered_automatic_evaluator(config=None, evaluator_name=None, evaluators_cfg_path=None):
    config = {} if config is None else config
    profile_name = config.get('evaluator') if evaluator_name is None else evaluator_name
    profiles_root = config.get('evaluators_cfg_path') if evaluators_cfg_path is None else evaluators_cfg_path
    if not profile_name or not profiles_root:
        raise ValueError('evaluator_name and evaluators_cfg_path are required')
    return load_evaluator(profile_name, profiles_root, config.get('api_config'))

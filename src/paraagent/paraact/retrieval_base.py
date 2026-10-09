from abc import ABC, abstractmethod
from typing import Dict, List, Any
import jsonschema
from jsonschema import validate, ValidationError


def is_tool_schema(obj: dict) -> bool:

    try:
        assert set(obj.keys()) == {'name', 'description', 'parameters'}
        assert isinstance(obj['name'], str)
        assert obj['name'].strip()
        assert isinstance(obj['description'], str)
        assert isinstance(obj['parameters'], dict)
        assert set(obj['parameters'].keys()) == {'type', 'properties', 'required'}
        assert obj['parameters']['type'] == 'object'
        assert isinstance(obj['parameters']['properties'], dict)
        assert isinstance(obj['parameters']['required'], list)
        assert set(obj['parameters']['required']).issubset(set(obj['parameters']['properties'].keys()))
    except AssertionError:
        return False
    try:
        jsonschema.validate(instance={}, schema=obj['parameters'])
    except jsonschema.exceptions.SchemaError:
        return False
    except jsonschema.exceptions.ValidationError:
        pass
    return True


class BaseTool(ABC):
    name: str = ''
    description: str = ''
    parameters: dict = {}

    def __init__(self):
        if not self.name:
            raise ValueError('Tool name must be provided')
        if not is_tool_schema({'name': self.name, 'description': self.description, 'parameters': self.parameters}):
            raise ValueError('The parameters, when provided as a dict, must confirm to a valid openai-compatible JSON schema.')

    @abstractmethod
    def execute(self, args: Dict, **kwargs) -> Dict[str, Any]:
        pass

    def batch_execute(self, args_list: List[Dict], **kwargs) -> List[Dict[str, Any]]:
        return [self.execute(args, **kwargs) for args in args_list]

    @property
    def tool_info(self) -> Dict:
        return {'name': self.name, 'description': self.description, 'parameters': self.parameters}

    @property
    def tool_description(self) -> Dict:
        return {'type': 'function', 'function': {'name': self.name, 'description': self.description, 'parameters': self.parameters}}

    def validate_args(self, args: Dict) -> bool:
        try:
            validate(instance=args, schema=self.parameters)
            return True
        except ValidationError:
            return False



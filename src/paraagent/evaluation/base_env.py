class BaseEnvironment:

    def __init__(self):
        self.task_description = ''
        self.input_description = ''
        self.tool_names = []
        self.functions = []

    def restart(self):

        raise NotImplementedError

    def get_score(self):

        raise NotImplementedError

    def step(self, action, input_str):

        raise NotImplementedError

    def check_success(self):

        raise NotImplementedError

    def to_json(self):
        raise NotImplementedError

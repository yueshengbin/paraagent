from termcolor import colored
from copy import deepcopy

class my_tree:

    def __init__(self):
        self.root = tree_node()
        self.now_deal_node = self.root

    def to_json_recursive(self, use_messages=False):
        tree_structure = self.root.to_json_recursive(use_messages=use_messages)
        js_obj = {'size': self.root.get_size(), 'max_length': self.root.get_max_depth(), 'tree': tree_structure}
        return js_obj

class tree_node:

    def __init__(self):
        self.is_terminal = False
        self.pruned = False
        self.finished = False
        self.node_type = None
        self.description = ''
        self.observation = ''
        self.observation_code = None
        self.children = []
        self.father = None
        self.io_state = None
        self.expand_num = 0
        self.Elo = 1000.0
        self.messages = []


    def get_max_depth(self):

        max_depth = 0
        for child in self.children:
            max_depth = max(max_depth, child.get_max_depth())
        return max_depth + 1

    def get_depth(self):
        if self.father == None:
            return 0
        return self.father.get_depth() + 1

    def get_size(self):

        size = 1
        for child in self.children:
            size += child.get_size()
        return size

    def prune(self):

        self.pruned = True
        for child in self.children:
            child.prune()

    def print(self, process_id=0):
        if process_id != 0:
            return
        color_converter = {'Thought': 'red', 'Action': 'blue', 'Action Input': 'cyan', 'Final Answer': 'green', 'Reflection': 'blue'}
        print(colored(f'{self.node_type}: {self.description}', color=color_converter.get(self.node_type, 'white')))
        if self.observation != '':
            if len(self.observation) < 1536:
                print(colored(f'Observation: {self.observation}', color='yellow'))
            else:
                print(colored(f'Observation: {self.observation[:1536]}......(len={len(self.observation)})', color='yellow'))


    def to_json_recursive(self, use_messages=False):
        js_obj = self.to_json(use_messages=use_messages)
        js_obj['children'] = []
        for child in self.children:
            js_obj['children'].append(child.to_json_recursive())
        return js_obj

    def make_finish(self, inter_val=1):

        self.finished = True
        if self.node_type == 'Action Input':
            inter_val -= 1
        if self.father != None and inter_val >= 0:
            self.father.make_finish(inter_val)

    def get_train_messages_from_this_node(self):


        def sift_first_invalid_message(messages):
            use_messages = []
            flag = True
            for message_id in range(len(messages))[::-1]:
                if not ('valid' in messages[message_id].keys() and messages[message_id]['valid'] == False):
                    use_messages = [messages[message_id]] + use_messages
                elif flag:
                    flag = False
                    use_messages = [messages[message_id]] + use_messages
            return use_messages
        now_node = self
        result = []
        while now_node.father != None:
            if now_node.node_type == 'Action Input':
                use_messages = deepcopy(now_node.messages)
                while use_messages[-1]['role'] != 'assistant':
                    use_messages = use_messages[:-1]
                use_messages = sift_first_invalid_message(use_messages)
                result = [use_messages] + result
            elif now_node.node_type == 'Thought':
                use_messages = deepcopy(now_node.messages)
                while use_messages[-1]['role'] == 'user':
                    use_messages = use_messages[:-1]
                use_messages = sift_first_invalid_message(use_messages)
                if use_messages[-1]['role'] == 'assistant':
                    result = [use_messages] + result
            now_node = now_node.father
        return result

    def get_chain_result_from_this_node(self, use_messages=False):

        now_node = self
        result = []
        while now_node.father != None:
            result = [now_node.to_json(use_messages=use_messages)] + result
            now_node = now_node.father
        return result


    def to_json(self, use_messages=False):
        json_obj = {}
        json_obj['is_terminal'] = False
        json_obj['pruned'] = self.pruned
        json_obj['finished'] = self.finished
        json_obj['depth'] = self.get_depth()
        json_obj['node_type'] = self.node_type
        json_obj['description'] = self.description
        json_obj['Elo'] = self.Elo
        if self.observation != '':
            json_obj['observation'] = self.observation
        if self.observation_code != None:
            json_obj['observation_code'] = self.observation_code
        json_obj['child_count'] = len(self.children)
        json_obj['expand_num'] = self.expand_num
        if self.io_state != None and self.node_type == 'Action Input':
            json_obj['io_state'] = self.io_state.to_json()
        if use_messages:
            json_obj['messages'] = []
            for message in self.messages:
                if not ('valid' in message.keys() and message['valid'] == False):
                    json_obj['messages'].append(message['role'])
                else:
                    json_obj['messages'].append(message['role'] + '_invalid')
        return json_obj

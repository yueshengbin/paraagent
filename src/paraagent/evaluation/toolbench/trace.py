
from .trace_graph import ExecutionGraph, ExecutionNode

def generate_init_message_node(eg: ExecutionGraph, functions, query):
    init_node = ExecutionNode(role='system', message='You can use many tools(functions) to do the following task.\nFirst I will give you the task description, and your task start.\nAt each step, you need to give your thought to analyze the status now and what to do next, with a function call to actually excute your step.\nAfter the call, you will get the call result, and you are now in a new state.\nThen you will analyze your status now, then decide what to do next...\nAfter many (Thought-call) pairs, you finally perform the task, then you can give your finial answer.\nRemember: \n1.the state change is irreversible, you can\'t go back to one of the former state, if you want to restart the task, say "I give up and restart".\n2.All the thought is short, at most in 5 sentence.\n3.You can do more then one trys, so if your plan is to continusly try some conditions, you can do one of the conditions per try.\nLet\'s Begin!\nTask description: You should use functions to help handle the real time user querys. Remember to ALWAYS call "Finish" function at the end of the task. And the final answer should contain enough information to show to the user.\nSpecifically, you have access to the following functions: ' + str(functions))
    eg.set_init_node(init_node)
    node = ExecutionNode(role='user', message=query)
    eg.add_node(node)
    eg[init_node, node] = None
    return node

def process_valid_data(method, answer_generation):
    conversation = answer_generation['train_messages'][-1]
    functions = answer_generation['function']
    query = answer_generation['query']
    eg = ExecutionGraph()
    last_node = generate_init_message_node(eg, functions, query)
    index = 2
    while index < len(conversation):
        message = conversation[index]
        role = message['role']
        if role == 'system' or role == 'user' or role == 'function' or (role == 'tool'):
            index = index + 1
            continue
        elif role == 'assistant':
            if 'function_call' in message and message['function_call']:
                node = ExecutionNode(role='tool', message={'name': message['function_call']['name'], 'arguments': message['function_call']['arguments'], 'response': conversation[index + 1]['content'] if message['function_call']['name'] != 'Finish' else ''})
                index = index + 1
            elif 'tool_calls' in message and message['tool_calls'] is not None and (len(message['tool_calls']) != 0):
                calls = message['tool_calls']
                responses = []
                for message2 in conversation[index + 1:]:
                    if message2['role'] != 'tool':
                        break
                    responses.append(message2)
                used_responses = set()
                for tc in calls:
                    function, name = (tc['function'], tc['function']['name'])
                    name, arguments = (function['name'], function['arguments'])
                    tc_id = tc.get('id')
                    if name == 'Finish':
                        node = ExecutionNode(role='tool', message={'name': name, 'arguments': arguments, 'response': ''})
                        break
                    else:
                        matched_response = None
                        # IDs are authoritative; name fallback supports older traces without IDs.
                        candidates = [i for i, response in enumerate(responses)
                                      if i not in used_responses and tc_id and response.get('tool_call_id') == tc_id]
                        if not candidates:
                            candidates = [i for i, response in enumerate(responses)
                                          if i not in used_responses and response.get('name') == name
                                          and (not tc_id or not response.get('tool_call_id'))]
                        if candidates:
                            response_index = candidates[0]
                            matched_response = responses[response_index]['content']
                            used_responses.add(response_index)
                        response = matched_response if matched_response is not None else ''
                        node = ExecutionNode(role='tool', message={'name': name, 'arguments': arguments, 'response': response})
                        eg.add_node(node)
                        eg[last_node, node] = None
                        last_node = node
            else:
                node = ExecutionNode(role='assistant', message=message['content'])
        else:
            raise NotImplementedError(f'Unkown role {role}')
        index = index + 1
        try:
            if last_node != node:
                eg.add_node(node)
                eg[last_node, node] = None
                last_node = node
        except Exception as e:
            print(e)
            print('⚠️ Assistant message has no content/tool_call/tool_calls:', message)
    eg = eg.reduce_graph_to_sequence()
    return {'query': query, 'available_tools': functions, 'answer': {'method': method, 'total_steps': eg.node_count, 'final_answer': answer_generation['final_answer'], 'answer_details': eg.convert_to_dict()}}

def convert_result(result):
    answer = result['answer_generation']
    if answer.get('valid_data') and answer.get('train_messages'):
        return process_valid_data('Agent@1', answer)


    graph = ExecutionGraph()
    last = generate_init_message_node(graph, answer['function'], answer['query'])
    chain = result['trys'][0]['chain']
    index = 0
    while index < len(chain):
        entry = chain[index]
        if entry['node_type'] == 'Action':
            node = ExecutionNode(role='tool', message={'name': entry['description'],
                'arguments': chain[index + 1]['description'], 'response': chain[index + 1]['observation']})
            index += 1
        elif entry['node_type'] == 'Thought':
            node = ExecutionNode(role='assistant', message=entry['description'])
        else:
            raise ValueError(f"Unknown trace node: {entry['node_type']}")
        index += 1
        graph.add_node(node)
        graph[last, node] = None
        last = node
    graph = graph.reduce_graph_to_sequence()
    return {'query': answer['query'], 'available_tools': answer['function'], 'answer': {
        'method': 'Agent@1', 'total_steps': graph.node_count,
        'final_answer': answer['final_answer'], 'answer_details': graph.convert_to_dict()}}

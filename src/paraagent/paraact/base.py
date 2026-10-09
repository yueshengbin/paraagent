class BaseSearchMethod:


    def __init__(self, llm, io_func, process_id=0, callbacks=None):

        pass

    @staticmethod
    def compute_chain_stats(terminal_node):

        path = []
        n = terminal_node
        while n is not None:
            path.append(n)
            n = getattr(n, 'father', None)
        path.reverse()
        search_calls = 0
        tool_calls = 0
        failed_calls = 0
        rounds = 0
        action_names = []
        for node in path:
            node_type = getattr(node, 'node_type', None)
            if node_type == 'Action':
                rounds += 1
                name = (node.description or '').strip()
                action_names.append(name)
                for individual in (n.strip() for n in name.split(' + ')):
                    if not individual:
                        continue
                    if individual == 'search_tool':
                        search_calls += 1
                    elif individual != 'Finish':
                        tool_calls += 1
            elif node_type == 'Action Input':
                code = getattr(node, 'observation_code', None)
                if code is not None and code not in (0, 3, 4):
                    failed_calls += 1
        last_action = action_names[-1] if action_names else ''
        if last_action == 'Finish':
            success = 0
            try:
                success = int(getattr(terminal_node.io_state, 'success', 0))
            except Exception:
                pass
            terminal_reason = 'give_answer' if success == 1 else 'give_up'
        elif getattr(terminal_node, 'pruned', False):
            terminal_reason = 'max_step'
        else:
            terminal_reason = 'unknown'
        discovered_tools_count = None
        try:
            allowed = getattr(terminal_node.io_state, 'allowed_tool_names', None)
            if isinstance(allowed, (set, list, tuple)):
                allowed_set = set(allowed)
                discovered_tools_count = max(0, len(allowed_set) - len({'search_tool', 'Finish'} & allowed_set))
        except Exception:
            pass
        return {'search_calls': search_calls, 'tool_calls': tool_calls, 'failed_calls': failed_calls, 'rounds': rounds, 'terminal_reason': terminal_reason, 'discovered_tools_count': discovered_tools_count}

    def to_json(self, answer=False, process=True):

        raise NotImplementedError

    def start(self, **args):

        raise NotImplementedError

import json
import os
from paraagent.paraact.trace import my_tree, tree_node
from paraagent.paraact.prompts import (
    FORMAT_INSTRUCTIONS_SYSTEM_FUNCTION_PARAAGENT,
    FORMAT_INSTRUCTIONS_SYSTEM_FUNCTION_REACT_EAE,
    FORMAT_INSTRUCTIONS_SYSTEM_FUNCTION_REACT_ETE,
)
from paraagent.paraact.base import BaseSearchMethod
from paraagent.paraact.retrieval import ToolSearch
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
import ast


def _search_query_input(raw_args):
    parsed = raw_args
    if isinstance(raw_args, str):
        try:
            parsed = json.loads(raw_args)
        except ValueError:
            try:
                parsed = ast.literal_eval(raw_args)
            except (SyntaxError, ValueError):
                pass
    if isinstance(parsed, dict):
        query = parsed.get("query")
        if isinstance(query, str):
            return parsed
        if query is not None:
            return {**parsed, "query": str(query)}
    return {"query": str(parsed)}


class ParaAct(BaseSearchMethod):
    def __init__(
        self, llm, io_func, extra_prefix="", process_id=0, protocol="paraagent", start_message_list=None
    ):
        super().__init__(llm, io_func, process_id, callbacks=None)
        self.io_func = io_func
        self.llm = llm
        self.extra_prefix = extra_prefix
        self.start_message_list = start_message_list
        self.process_id = process_id
        if protocol not in {"paraagent", "react"}:
            raise ValueError("protocol must be 'paraagent' or 'react'.")
        self.protocol = protocol
        self.retriever = ToolSearch()
        self.restart()

    def restart(self):
        self.status = 0
        self.try_list = []
        self.terminal_node = []
        self.query_count = 0
        self.total_tokens = 0
        self.total_characters = 0
        self.token_usage_missing_calls = 0
        self.success_count = 0

    def to_json(self, answer=False, process=True):
        if process:
            json_obj = {
                "win": self.status == 1,
                "try_count": len(self.try_list),
                "trys": self.try_list,
                "compare_candidates": [],
                "forward_args": self.forward_args,
            }
            for node in self.terminal_node:
                if node.pruned == False:
                    json_obj["compare_candidates"].append(
                        node.get_chain_result_from_this_node(use_messages=False)
                    )
        else:
            json_obj = {}
        if answer:
            json_obj["answer_generation"] = {
                "valid_data": False,
                "final_answer": "",
                "function": self.io_func.functions,
                "query_count": self.query_count,
                "total_tokens": self.total_tokens,
                "train_messages": [],
                "chain": [],
            }
            json_obj["answer_generation"]["total_characters"] = self.total_characters
            json_obj["answer_generation"]["token_usage_missing_calls"] = self.token_usage_missing_calls
            stats_node = None
            for node in self.terminal_node:
                if node.pruned == False:
                    stats_node = node
                    json_obj["answer_generation"]["valid_data"] = True
                    json_obj["answer_generation"]["final_answer"] = node.description
                    json_obj["answer_generation"]["train_messages"] = node.get_train_messages_from_this_node()
                    break
            if stats_node is None and self.terminal_node:
                stats_node = self.terminal_node[-1]
            if stats_node is not None:
                json_obj["answer_generation"]["stats"] = self.compute_chain_stats(stats_node)
        return json_obj

    def to_json_single(self):
        json_obj = {}
        tree_obj = self.terminal_node[-1].get_chain_result_from_this_node()
        json_obj["chain"] = tree_obj
        json_obj["win"] = self.status == 1
        return json_obj

    def start(self, max_depth, pass_at=1, answer=1):
        self.forward_args = locals()
        self.forward_args.pop("self", None)
        initial_functions = deepcopy(self.io_func.functions)
        initial_allowed = (
            None
            if getattr(self.io_func, "allowed_tool_names", None) is None
            else self.io_func.allowed_tool_names.copy()
        )
        for i in range(pass_at):
            if self.process_id == 0:
                print(f"[{self.protocol}] run {i + 1}/{pass_at}")
            self.io_func.functions[:] = deepcopy(initial_functions)
            if initial_allowed is not None:
                self.io_func.allowed_tool_names.clear()
                self.io_func.allowed_tool_names.update(initial_allowed)
            self.tree = my_tree()
            self.tree.root.node_type = "Action Input"
            self.tree.root.io_state = deepcopy(self.io_func)
            out_node = self.do_chain(self.tree.root, max_depth)
            self.terminal_node.append(out_node)
            self.try_list.append(self.to_json_single())
            if out_node.io_state.check_success() == 1:
                self.status = 1
                self.success_count += 1
                if self.success_count >= answer:
                    return 1
        return 0

    def _add_node(
        self, parent_node, node_type, description, observation=None, observation_code=0, share_state=False
    ):

        temp_node = tree_node()
        temp_node.node_type = node_type
        temp_node.description = description
        if share_state:
            temp_node.io_state = parent_node.io_state
        else:
            temp_node.io_state = deepcopy(parent_node.io_state)
        temp_node.is_terminal = temp_node.io_state.check_success() != 0
        temp_node.messages = parent_node.messages.copy()
        temp_node.father = parent_node
        parent_node.children.append(temp_node)
        if observation is not None:
            temp_node.observation = observation
            temp_node.observation_code = observation_code
        temp_node.print(self.process_id)
        return temp_node

    def do_chain(self, now_node, max_depth):

        def parse_tool_arguments_lenient(raw_args):
            if not isinstance(raw_args, str):
                return raw_args
            try:
                return json.loads(raw_args, strict=False)
            except Exception:
                pass
            try:
                return ast.literal_eval(raw_args)
            except Exception:
                return raw_args

        def format_search_observation(result_tool):
            if getattr(self.io_func, "search_result_as_functions", True):
                schema = self.retriever.api_json_to_openai_json(result_tool)
                return (
                    json.dumps({"type": "function", **schema}, ensure_ascii=False),
                    {"type": "function", **schema},
                )
            lightweight = {
                "name": result_tool.get("name") or result_tool.get("tool_name", ""),
                "description": result_tool.get("description", ""),
                "parameters": result_tool.get("parameters", {"type": "object", "properties": {}}),
            }
            return (json.dumps(lightweight, ensure_ascii=False), None)

        if self.start_message_list is None:
            paradigm = getattr(self.io_func, "paradigm", "EaE")
            if self.protocol == "paraagent":
                system = FORMAT_INSTRUCTIONS_SYSTEM_FUNCTION_PARAAGENT.rstrip("\n")
            elif paradigm == "ETE":
                system = FORMAT_INSTRUCTIONS_SYSTEM_FUNCTION_REACT_ETE.rstrip("\n")
            else:
                system = FORMAT_INSTRUCTIONS_SYSTEM_FUNCTION_REACT_EAE.rstrip("\n")
            user_content = self.io_func.input_description
            inject_tool_block = os.environ.get("REACT_INJECT_TOOL_BLOCK", "1") == "1"
            if (
                self.protocol == "paraagent"
                or not inject_tool_block
                or getattr(self.llm, "embeds_tools_in_prompt", False)
            ):
                system_with_tools = system
            else:
                tool_block_lines = ["## Available tools"]
                for f in self.io_func.functions or []:
                    fn = f.get("function", f)
                    tool_block_lines.append(
                        f"- **{fn.get('name', '?')}** — {fn.get('description', '').strip()}\n  Parameters: {json.dumps(fn.get('parameters', {}), ensure_ascii=False)}"
                    )
                system_with_tools = system.rstrip() + "\n\n" + "\n".join(tool_block_lines)
            self.tree.root.messages.append({"role": "system", "content": system_with_tools})
            self.tree.root.messages.append({"role": "user", "content": user_content})
        else:
            self.tree.root.messages = self.start_message_list
        now_node = self.tree.root
        while True:
            self.llm.change_messages(now_node.messages)
            new_message, error_code, completion_tokens = self.llm.parse(
                tools=self.io_func.functions, process_id=self.process_id
            )
            if completion_tokens is None:
                self.token_usage_missing_calls += 1
                self.total_tokens = None
            elif self.total_tokens is not None:
                self.total_tokens += completion_tokens
            self.total_characters += len(new_message.get("content") or "")
            self.query_count += 1
            assert new_message["role"] == "assistant"
            if new_message.get("content") is not None:
                thought_text = new_message["content"]
                if "\nAction:" in thought_text:
                    thought_text = thought_text[: thought_text.index("\nAction:")].strip()
                if thought_text.startswith("Thought:"):
                    thought_text = thought_text[len("Thought:") :].strip()
                now_node = self._add_node(now_node, "Thought", thought_text, share_state=True)
                if error_code != 0:
                    now_node.observation_code = error_code
                    now_node.pruned = True
            tool_calls = new_message.get("tool_calls")
            if tool_calls:

                def get_action_type(tool_call):
                    return tool_call.get("action_type") or (
                        "finish"
                        if tool_call["function"]["name"] == "Finish"
                        else "search_tool"
                        if tool_call["function"]["name"] == "search_tool"
                        else "tool_call"
                    )

                if any((get_action_type(tc) == "finish" for tc in tool_calls)):
                    tool_calls = [tc for tc in tool_calls if get_action_type(tc) == "finish"][:1]
                if len(tool_calls) == 1:
                    tool_call = tool_calls[0]
                    action_type = get_action_type(tool_call)
                    function_name = tool_call["function"]["name"]
                    if action_type == "search_tool":
                        now_node = self._add_node(now_node, "Action", function_name, share_state=True)
                        raw_args = tool_call["function"]["arguments"]
                        query_input = _search_query_input(raw_args)
                        if hasattr(self.io_func, "executable_tools") and getattr(
                            self.io_func, "executable_tools", None
                        ):
                            query_input["executable_tools"] = self.io_func.executable_tools
                        retriever_input = query_input
                        retriever_input["top_k"] = int(os.environ.get("TOOL_SEARCH_DEFAULT_TOP_K", "3"))
                        raw_results = self.retriever.batch_execute_no_form([retriever_input])
                        obs_lines = []
                        new_funcs = []
                        for qr in raw_results:
                            if isinstance(qr, dict) and "results" in qr:
                                for result in qr["results"]:
                                    if hasattr(self.io_func, "register_discovered_tool"):
                                        self.io_func.register_discovered_tool(result["tools"])
                                    obs_text, full_schema = format_search_observation(result["tools"])
                                    if full_schema is not None:
                                        new_funcs.append(full_schema)
                                    obs_lines.append(obs_text)
                        observation = "<tools>\n" + "\n".join(obs_lines) + "\n</tools>"
                        status = 0 if new_funcs else 6
                        if new_funcs and isinstance(
                            getattr(now_node.io_state, "allowed_tool_names", None), set
                        ):
                            for f in new_funcs:
                                now_node.io_state.allowed_tool_names.add(f["function"]["name"])
                        if new_funcs and isinstance(getattr(self.io_func, "allowed_tool_names", None), set):
                            for f in new_funcs:
                                self.io_func.allowed_tool_names.add(f["function"]["name"])
                        if getattr(self.io_func, "append_discovered_functions", True):
                            existing_names = {f["function"]["name"] for f in self.io_func.functions}
                            for f in new_funcs:
                                if f["function"]["name"] not in existing_names:
                                    self.io_func.functions.append(f)
                                    existing_names.add(f["function"]["name"])
                            if "Finish" not in existing_names:
                                self.io_func.functions.append(
                                    {
                                        "type": "function",
                                        "function": {
                                            "name": "Finish",
                                            "description": "If you believe that you have obtained a result that can answer the task, please call this function to provide the final answer. Alternatively, if you recognize that you are unable to proceed with the task in the current state, call this function to restart. Remember: you must ALWAYS call this function at the end of your attempt, and the only part that will be shown to the user is the final answer, so it should contain sufficient information.",
                                            "parameters": {
                                                "type": "object",
                                                "properties": {
                                                    "return_type": {
                                                        "type": "string",
                                                        "enum": ["give_answer", "give_up_and_restart"],
                                                    },
                                                    "final_answer": {
                                                        "type": "string",
                                                        "description": 'The final answer you want to give the user. You should have this field if "return_type"=="give_answer"',
                                                    },
                                                },
                                                "required": ["return_type"],
                                            },
                                        },
                                    }
                                )
                        now_node = self._add_node(
                            now_node,
                            "Action Input",
                            query_input,
                            observation=observation,
                            observation_code=status,
                        )
                        if status == 4:
                            now_node.pruned = True
                        now_node.messages.append(new_message)
                        now_node.messages.append(
                            {
                                "role": "tool",
                                "name": function_name,
                                "content": now_node.observation,
                                "tool_call_id": tool_call["id"],
                            }
                        )
                    else:
                        now_node = self._add_node(now_node, "Action", function_name, share_state=True)
                        function_input = tool_call["function"]["arguments"]
                        observation = ""
                        status = 0
                        try:
                            function_input = parse_tool_arguments_lenient(function_input)
                            observation, status = now_node.io_state.step(
                                action_name=function_name, action_input=function_input
                            )
                        except Exception as e:
                            print(e)
                            observation = str(e)
                            status = -1
                        now_node = self._add_node(
                            now_node,
                            "Action Input",
                            function_input,
                            observation=observation,
                            observation_code=status,
                        )
                        if status == 4:
                            now_node.pruned = True
                        elif status == 1:
                            tool_call["function"]["name"] = "invalid_hallucination_function_name"
                        now_node.messages.append(new_message)
                        now_node.messages.append(
                            {
                                "role": "tool",
                                "name": tool_call["function"]["name"],
                                "content": str(now_node.observation),
                                "tool_call_id": tool_call["id"],
                            }
                        )
                else:
                    action_names = " + ".join((tc["function"]["name"] for tc in tool_calls))
                    now_node = self._add_node(now_node, "Action", action_names, share_state=True)

                    def _exec_one(tc):
                        fname = tc["function"]["name"]
                        fargs = tc["function"]["arguments"]
                        if fname == "search_tool":
                            query_input = _search_query_input(fargs)
                            if hasattr(self.io_func, "executable_tools") and getattr(
                                self.io_func, "executable_tools", None
                            ):
                                query_input["executable_tools"] = self.io_func.executable_tools
                            retriever_input = query_input
                            retriever_input["top_k"] = int(os.environ.get("TOOL_SEARCH_DEFAULT_TOP_K", "3"))
                            raw_results = self.retriever.batch_execute_no_form([retriever_input])
                            obs_lines, new_funcs = ([], [])
                            for qr in raw_results:
                                if isinstance(qr, dict) and "results" in qr:
                                    for result in qr["results"]:
                                        if hasattr(self.io_func, "register_discovered_tool"):
                                            self.io_func.register_discovered_tool(result["tools"])
                                        obs_text, full_schema = format_search_observation(result["tools"])
                                        if full_schema is not None:
                                            new_funcs.append(full_schema)
                                        obs_lines.append(obs_text)
                            obs = "<tools>\n" + "\n".join(obs_lines) + "\n</tools>"
                            stat = 0 if new_funcs else 6
                            return (tc, fname, obs, stat, new_funcs)
                        else:
                            try:
                                fargs = parse_tool_arguments_lenient(fargs)
                                obs, stat = now_node.io_state.step(action_name=fname, action_input=fargs)
                            except Exception as e:
                                obs, stat = (str(e), -1)
                            display_name = fname
                            if stat == 1:
                                display_name = "invalid_hallucination_function_name"
                                tc["function"]["name"] = display_name
                            return (tc, display_name, obs, stat, [])

                    order = {id(tc): i for i, tc in enumerate(tool_calls)}
                    with ThreadPoolExecutor(max_workers=len(tool_calls)) as pool:
                        futures = [pool.submit(_exec_one, tc) for tc in tool_calls]
                        raw_exec = sorted(
                            (f.result() for f in as_completed(futures)), key=lambda r: order[id(r[0])]
                        )
                    exec_results = []
                    for tc, display_name, obs, stat, new_funcs in raw_exec:
                        if new_funcs and isinstance(
                            getattr(now_node.io_state, "allowed_tool_names", None), set
                        ):
                            for f in new_funcs:
                                now_node.io_state.allowed_tool_names.add(f["function"]["name"])
                        if new_funcs and isinstance(getattr(self.io_func, "allowed_tool_names", None), set):
                            for f in new_funcs:
                                self.io_func.allowed_tool_names.add(f["function"]["name"])
                        if new_funcs and getattr(self.io_func, "append_discovered_functions", True):
                            existing_names = {f["function"]["name"] for f in self.io_func.functions}
                            for f in new_funcs:
                                if f["function"]["name"] not in existing_names:
                                    self.io_func.functions.append(f)
                                    existing_names.add(f["function"]["name"])
                        exec_results.append((tc, display_name, obs, stat))
                    if len(exec_results) == 1:
                        combined_obs = exec_results[0][2]
                        final_status = exec_results[0][3]
                    elif all((name == "search_tool" for _, name, _, _ in exec_results)):
                        merged_lines = []
                        seen_lines = set()
                        for _, _, obs, _ in exec_results:
                            text = str(obs)
                            if "<tools>" in text and "</tools>" in text:
                                inner = text.split("<tools>", 1)[1].split("</tools>", 1)[0]
                            else:
                                inner = text
                            for line in inner.splitlines():
                                line = line.strip()
                                if not line:
                                    continue
                                if line not in seen_lines:
                                    seen_lines.add(line)
                                    merged_lines.append(line)
                        combined_obs = "<tools>\n" + "\n".join(merged_lines) + "\n</tools>"
                        statuses = [s for _, _, _, s in exec_results]
                        final_status = next((s for s in statuses if s != 0), 0)
                    else:
                        parts = [f"[{name}]: {obs}" for _, name, obs, _ in exec_results]
                        combined_obs = "\n\n".join(parts)
                        statuses = [s for _, _, _, s in exec_results]
                        final_status = (
                            4 if 4 in statuses else next((s for s in statuses if s not in (0, 1)), 0)
                        )
                    now_node = self._add_node(
                        now_node,
                        "Action Input",
                        [{tc["function"]["name"]: tc["function"]["arguments"]} for tc, *_ in exec_results],
                        observation=combined_obs,
                        observation_code=final_status,
                    )
                    if final_status == 4:
                        now_node.pruned = True
                    now_node.messages.append(new_message)
                    for tc, display_name, obs, _ in exec_results:
                        now_node.messages.append(
                            {
                                "role": "tool",
                                "name": display_name,
                                "content": str(obs),
                                "tool_call_id": tc["id"],
                            }
                        )
            else:
                now_node.messages.append(new_message)
            if now_node.get_depth() >= max_depth and (not now_node.is_terminal):
                now_node.pruned = True
            if now_node.pruned or now_node.is_terminal:
                return now_node

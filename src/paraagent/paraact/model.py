import os
import json
import re
import time
import random
import string
from openai import OpenAI
from typing import List, Any, Tuple, Dict, Optional
from requests.exceptions import ConnectionError, Timeout, RequestException

_RE_TOOL_QUERY = re.compile("<tool_query>\\s*(.*?)\\s*</tool_query>", re.DOTALL)
_RE_SEARCH_TOOL = re.compile("<search_tool>\\s*(.*?)\\s*</search_tool>", re.DOTALL)
_RE_TOOL_CALL = re.compile("<tool_call>\\s*(.*?)\\s*</tool_call>", re.DOTALL)
_RE_ANSWER = re.compile("<answer>\\s*(.*?)\\s*</answer>", re.DOTALL)


def extract_tool_calls(raw_response: str, tool_list: List[str]) -> Tuple[str, List[Dict[str, Any]]]:

    tool_calls = []
    has_think = "<think>" in raw_response and "</think>" in raw_response
    has_tool_query = "<tool_query>" in raw_response and "</tool_query>" in raw_response
    has_search_tool = "<search_tool>" in raw_response and "</search_tool>" in raw_response
    has_tool_call = "<tool_call>" in raw_response and "</tool_call>" in raw_response
    has_answer = "<answer>" in raw_response and "</answer>" in raw_response
    if isinstance(tool_list, str):
        available_tool_names = set()
        for item in re.findall('"name"\\s*:\\s*"([^"]+)"', tool_list):
            if item != "search_tool":
                available_tool_names.add(item)
    else:
        available_tool_names = set(tool_list)

    def _infer_action_type(name: str) -> str:
        if name == "search_tool":
            return "search_tool"
        if name == "Finish":
            return "finish"
        if name == "Error":
            return "error"
        return "tool_call"

    def _pack_result(name: str, args: Any) -> Dict[str, Any]:
        random_id = "".join([random.choice(string.ascii_letters + string.digits) for _ in range(8)])
        if isinstance(args, dict):
            args = json.dumps(args, ensure_ascii=False)
        return {
            "type": "function",
            "id": random_id,
            "action_type": _infer_action_type(name),
            "function": {"name": name, "arguments": args},
        }

    def _pack_error(error_type: str, msg: str) -> Dict[str, Any]:
        return _pack_result("Error", {"error": {"type": error_type, "msg": msg}})

    action_type_count = sum([has_tool_query or has_search_tool, has_tool_call, has_answer])
    if action_type_count > 1:
        tool_calls.append(
            _pack_result("Finish", {"return_type": "give_answer", "final_answer": raw_response})
        )
        return ("tool_calls", tool_calls)
    elif has_think and (has_tool_query or has_search_tool):
        query_blocks = _RE_TOOL_QUERY.findall(raw_response) + _RE_SEARCH_TOOL.findall(raw_response)
        for query_content in query_blocks:
            tool_calls.append(_pack_result("search_tool", query_content))
        return ("tool_calls", tool_calls)
    elif has_think and has_tool_call:
        for json_str in _RE_TOOL_CALL.findall(raw_response):
            try:
                call_data = json.loads(json_str)
                if "name" not in call_data:
                    tool_calls.append(_pack_error("InvalidRequestError", "No tool name"))
                    continue
                if "arguments" not in call_data:
                    tool_calls.append(_pack_error("InvalidRequestError", "No tool arguments"))
                    continue
                tool_name = call_data["name"]
                tool_calls.append(_pack_result(tool_name, call_data["arguments"]))
            except json.JSONDecodeError:
                tool_calls.append(_pack_error("InvalidRequestError", "JSONDecodeError"))
            except Exception:
                tool_calls.append(_pack_error("InvalidRequestError", "Unknown processing error"))
        return ("tool_calls", tool_calls)
    elif has_think and has_answer:
        answer_matches = _RE_ANSWER.findall(raw_response)
        if answer_matches:
            tool_calls.append(
                _pack_result("Finish", {"return_type": "give_answer", "final_answer": answer_matches[-1]})
            )
        return ("tool_calls", tool_calls)
    else:
        tool_calls.append(
            _pack_result(
                "Finish", {"return_type": "give_answer", "final_answer": raw_response, "error_answer": ""}
            )
        )
        return ("tool_calls", tool_calls)


class AgentVLLMFunction:
    def __init__(
        self,
        model: str,
        base_url: str = "http://127.0.0.1:8000/v1",
        openai_key: str = "EMPTY",
        template: str = "agent-tool",
        max_sequence_length: int = 8192,
    ) -> None:
        self.model_name = model
        self.template = template
        self.max_sequence_length = max_sequence_length
        self.client = OpenAI(api_key=openai_key, base_url=base_url)
        self.conversation_history = []

    @staticmethod
    def _render_prompt(messages: List[dict]) -> str:

        parts = []
        for m in messages:
            parts.append(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n")
        parts.append("<|im_start|>assistant\n")
        return "".join(parts)

    def prediction(self, prompt: List[dict], stop: Optional[List[str]] = None) -> Tuple[str, Optional[int]]:

        prompt_text = self._render_prompt(prompt)
        max_retries = 13
        request_max_tokens = int(os.environ.get("AGENT_MAX_TOKENS", "2048"))
        last_error_type = None
        for attempt in range(max_retries):
            try:
                response = self.client.completions.create(
                    model=self.model_name,
                    prompt=prompt_text,
                    temperature=0.0,
                    max_tokens=request_max_tokens,
                    stop=["<|im_end|>"],
                )
                prediction = response.choices[0].text
                completion_tokens = getattr(getattr(response, "usage", None), "completion_tokens", None)
                if type(completion_tokens) is not int or completion_tokens < 0:
                    completion_tokens = None
                return prediction, completion_tokens
            except (ConnectionError, Timeout) as e:
                print(f"Network error ({type(e).__name__}). Retrying {attempt + 1}/{max_retries}...")
                time.sleep(60)
                if attempt == max_retries - 1:
                    raise
            except RequestException as e:
                print(f"Request error ({type(e).__name__})")
                raise
            except Exception as e:
                last_error_type = type(e).__name__
                message = str(e)
                context_match = re.search(
                    "maximum context length is (\\d+) tokens.*?prompt contains at least (\\d+) input tokens",
                    message,
                    flags=re.IGNORECASE | re.DOTALL,
                )
                if context_match:
                    context_limit = int(context_match.group(1))
                    input_tokens = int(context_match.group(2))
                    reduced_max_tokens = request_max_tokens // 2
                    if 0 < reduced_max_tokens < request_max_tokens:
                        request_max_tokens = reduced_max_tokens
                        print(
                            f"Context limit reached; retrying with max_tokens={request_max_tokens} (context={context_limit}, input={input_tokens})."
                        )
                        continue
                if attempt < max_retries - 1:
                    time.sleep(30)
                print(f"Policy request failed ({last_error_type}); attempt {attempt + 1}/{max_retries}")
        raise RuntimeError(
            f"Policy request failed after {max_retries} attempts ({last_error_type}); no model answer was produced"
        ) from None

    def add_message(self, message):
        self.conversation_history.append(message)

    def change_messages(self, messages):
        self.conversation_history = messages

    def parse(self, tools, process_id, **args):
        if self.template == "qwen-tool":
            roles = {
                "system": "system",
                "user": "user",
                "function": "user",
                "tool": "user",
                "assistant": "assistant",
            }
        elif self.template == "agent-tool":
            roles = {
                "system": "system",
                "user": "user",
                "function": "user",
                "tool": "user",
                "assistant": "assistant",
            }
        else:
            roles = {
                "system": "system",
                "user": "user",
                "function": "user",
                "tool": "user",
                "assistant": "assistant",
            }
        conversation_history = self.conversation_history
        tool_response = ""
        search_tool_buffer = ""
        prompt = []
        tool_list = ""
        for k, message in enumerate(conversation_history):
            target_role = roles.get(message["role"], "user")
            if message["role"] == "system":
                content = message["content"]
                prompt.append({"role": target_role, "content": content})
            elif message["role"] == "tool" and message.get("name") != "search_tool":
                if k + 1 < len(conversation_history) and conversation_history[k + 1]["role"] == "tool":
                    tool_response += "<tool_response>\n" + str(message["content"]) + "\n</tool_response>\n"
                else:
                    tool_response += "<tool_response>\n" + str(message["content"]) + "\n</tool_response>"
                    prompt.append({"role": target_role, "content": tool_response})
                    tool_response = ""
            elif message["role"] == "tool" and message.get("name") == "search_tool":
                next_is_search = (
                    k + 1 < len(conversation_history)
                    and conversation_history[k + 1]["role"] == "tool"
                    and (conversation_history[k + 1].get("name") == "search_tool")
                )
                if next_is_search:
                    search_tool_buffer += message["content"] + "\n"
                else:
                    search_tool_buffer += message["content"]
                    prompt.append({"role": target_role, "content": search_tool_buffer})
                    search_tool_buffer = ""
            else:
                content = message["content"]
                prompt.append({"role": target_role, "content": content})
        predictions, completion_tokens = self.prediction(prompt)
        if predictions is None:
            predictions = ""
        if process_id == 0:
            print(
                f"[process({process_id})] completion tokens: {completion_tokens}, total chars: {len(predictions)}"
            )
        name, tool = extract_tool_calls(predictions, tool_list)
        message = {"role": "assistant", "content": predictions, name: tool}
        return (message, 0, completion_tokens)

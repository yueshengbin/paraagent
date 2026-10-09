from typing import Dict, List, Any
import requests
import os
import json
from paraagent.paraact.retrieval_base import BaseTool
from paraagent.paraact.naming import standardize, change_name


class ToolSearch(BaseTool):
    name = "SearchTools"
    description = "get a set of candidate tools from the tool library."
    parameters = {
        "type": "object",
        "properties": {"query": {"type": "string", "description": "Search query"}},
        "required": ["query"],
    }

    def __init__(self):
        super().__init__()
        self.api_url = os.environ.get("TOOL_SEARCH_API_URL", "http://127.0.0.1:30400")
        self.default_limit = int(os.environ.get("TOOL_SEARCH_DEFAULT_TOP_K", "3"))

    def execute(self, args: Dict) -> Dict[str, Any]:

        query = args.get("query", "").strip()
        limit = args.get("limit", self.default_limit)
        try:
            response = requests.get(
                f"{self.api_url}/search",
                params={"query": query, "top_k": limit},
                proxies={"http": "", "https": ""},
                timeout=30,
            )
            if response.status_code == 200:
                result = response.json()
                formatted_result = self._format_results(result)
                return {"content": formatted_result, "success": True}
            else:
                error_msg = f"Search API returned error: {response.status_code}"
                if response.text:
                    error_msg += f" - {response.text}"
                print(f"[WARNING] {error_msg}")
                return {"content": error_msg, "success": False}
        except Exception as e:
            error_msg = f"Failed to execute search: {str(e)}"
            print(f"[WARNING] {error_msg}")
            return {"content": error_msg, "success": False}

    def batch_execute(self, args_list: List[Dict]) -> List[Dict[str, Any]]:
        if not args_list:
            return []
        queries = [args.get("query", "").strip() for args in args_list]
        limits = [args.get("top_k", args.get("limit", self.default_limit)) for args in args_list]
        max_limit = max(limits)
        executable_tools = None
        for args in args_list:
            if isinstance(args, dict) and args.get("executable_tools") is not None:
                executable_tools = args.get("executable_tools")
                break
        try:
            payload = {"queries": queries, "top_k": max_limit}
            if executable_tools is not None:
                payload["executable_tools"] = executable_tools
            response = requests.post(
                f"{self.api_url}/search", json=payload, proxies={"http": "", "https": ""}, timeout=30
            )
            if response.status_code == 200:
                batch_result = response.json()
                results = []
                for i, query_result in enumerate(batch_result["query_results"]):
                    limited_results = {
                        "query": query_result["query"],
                        "results": query_result["results"][: limits[i]],
                    }
                    formatted_result = self._format_results(limited_results)
                    results.append({"content": formatted_result, "success": True})
                return results
            else:
                error_msg = f"Batch search API returned error: {response.status_code}"
                if response.text:
                    error_msg += f" - {response.text}"
                print(f"[WARNING] {error_msg}")
                return [{"content": error_msg, "success": False} for _ in queries]
        except Exception as e:
            error_msg = f"Failed to execute batch search: {str(e)}"
            print(f"[WARNING] {error_msg}")
            return [{"content": error_msg, "success": False} for _ in queries]

    def batch_execute_no_form(self, args_list: List[Dict]) -> List[Dict[str, Any]]:
        if not args_list:
            return []
        queries = [args.get("query", "").strip() for args in args_list]
        limits = [args.get("top_k", self.default_limit) for args in args_list]
        max_limit = max(limits)
        executable_tools = None
        for args in args_list:
            if isinstance(args, dict) and args.get("executable_tools") is not None:
                executable_tools = args.get("executable_tools")
                break
        try:
            payload = {"queries": queries, "top_k": max_limit}
            if executable_tools is not None:
                payload["executable_tools"] = executable_tools
            response = requests.post(
                f"{self.api_url}/search", json=payload, proxies={"http": "", "https": ""}, timeout=30
            )
            if response.status_code == 200:
                batch_result = response.json()
                results = []
                for i, query_result in enumerate(batch_result["query_results"]):
                    results.append({**query_result, "results": query_result["results"][: limits[i]]})
                return results
            else:
                error_msg = f"Batch search API returned error: {response.status_code}"
                if response.text:
                    error_msg += f" - {response.text}"
                print(f"[WARNING] {error_msg}")
                return [{"content": error_msg, "success": False} for _ in queries]
        except Exception as e:
            error_msg = f"Failed to execute batch search: {str(e)}"
            print(f"[WARNING] {error_msg}")
            return [{"content": error_msg, "success": False} for _ in queries]

    def _format_results(self, api_result) -> str:

        if "error" in api_result:
            return json.dumps(api_result, ensure_ascii=False)
        if "query_results" in api_result:
            if len(api_result["query_results"]) > 0:
                query_result = api_result["query_results"][0]
                results_list = []
                for result in query_result["results"]:
                    opena_result = result["tools"]
                    results_list.append({"type": "function", "function": opena_result})
                resul_str = "\n".join([json.dumps(item, ensure_ascii=False) for item in results_list])
                return resul_str
            else:
                return json.dumps([], ensure_ascii=False)
        else:
            results_list = []
            for result in api_result.get("results", []):
                if "tools" in result:
                    opena_result = self.api_json_to_openai_json(result["tools"])
                    results_list.append({"type": "function", **opena_result})
                else:
                    results_list.append(result)
            result_str = "\n".join([json.dumps(item, ensure_ascii=False) for item in results_list])
            return result_str

    def truncate_descriptions(self, data, max_len=20):

        if isinstance(data, dict):
            for key, value in data.items():
                if key == "description" and isinstance(value, str):
                    if len(value) > max_len:
                        data[key] = value[:max_len] + "..."
                elif isinstance(value, (dict, list)):
                    self.truncate_descriptions(value, max_len)
        elif isinstance(data, list):
            for item in data:
                self.truncate_descriptions(item, max_len)
        return data

    def api_json_to_openai_json(self, api_json):
        raw_name = api_json.get("name", "")
        function_template = {
            "function": {
                "name": raw_name[:64],
                "description": "",
                "parameters": self.truncate_descriptions(api_json["parameters"], max_len=256),
            }
        }
        raw_desc = api_json.get("description", "").strip()
        if " : " in raw_name:
            tool_name, api_name = [s.strip() for s in raw_name.split(" : ", 1)]
            pure_api_name = change_name(standardize(api_name))
            standard_tool_name = standardize(tool_name)
            standardized_name = f"{standard_tool_name}-{pure_api_name}"
            function_template["function"]["name"] = standardized_name[:64]
            base_desc = f'This is the subfunction for tool "{standard_tool_name}", you can use this tool.'
            if raw_desc:
                full_desc = f'{base_desc} The description of this function is: "{raw_desc}"'
            else:
                full_desc = base_desc
            function_template["function"]["description"] = full_desc
        else:
            function_template["function"]["description"] = raw_desc
        return function_template

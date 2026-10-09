import asyncio
from copy import deepcopy
import json
import logging
import os
import re
import time
from functools import lru_cache
from typing import Any

import httpx
from jsonschema.exceptions import SchemaError
from jsonschema.validators import validator_for

from paraagent.toolenv.runtime.utils import change_name, normalize_tool_name, standardize

from ..base import BaseTool
from ..schema import ToolResponse

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

_PARAMETER_EXAMPLE_CUE_RE = re.compile(
    r"\bexamples?\b|(?<!\w)e\.g\.(?=\s|$)|\be\.g\b|\bsuch as\b|\bfor instance\b",
    re.IGNORECASE,
)

AUTONOMOUS_RETAIL_TOOL_DESCRIPTIONS = {
    "cancel_pending_order": (
        "Cancel an order that is still pending. Orders that have already been processed or delivered "
        "cannot be cancelled. A successful cancellation changes the order status to 'cancelled' and "
        "refunds the payment."
    ),
    "exchange_delivered_order_items": (
        "Exchange one or more items from a delivered order for different variants of the same products. "
        "A delivered order can be returned or exchanged only once."
    ),
    "modify_pending_order_address": "Update the shipping address for an order that is still pending.",
    "modify_pending_order_items": (
        "Replace one or more items in a pending order with different variants of the same products. "
        "This operation can be performed only once per pending order."
    ),
    "modify_pending_order_payment": "Change the payment method for an order that is still pending.",
    "modify_user_address": "Update a user's default address.",
    "return_delivered_order_items": (
        "Request a return for one or more items from a delivered order. A successful request changes "
        "the order status to 'return requested'."
    ),
}

def _iter_schema_refs(node: Any):
    """Yield every ``$ref`` string found anywhere inside a JSON Schema document."""
    if isinstance(node, dict):
        ref = node.get("$ref")
        if isinstance(ref, str):
            yield ref
        for value in node.values():
            yield from _iter_schema_refs(value)
    elif isinstance(node, list):
        for item in node:
            yield from _iter_schema_refs(item)

def _local_json_pointer_resolves(root: dict[str, Any], reference: str) -> bool:
    """Check local JSON Pointer targets before exposing the schema to the model."""
    if reference == "#":
        return True
    if not reference.startswith("#/"):

        return False
    target: Any = root
    for raw_part in reference[2:].split("/"):
        part = raw_part.replace("~1", "/").replace("~0", "~")
        if isinstance(target, list):
            if not part.isdigit() or int(part) >= len(target):
                return False
            target = target[int(part)]
        elif isinstance(target, dict):
            if part not in target:
                return False
            target = target[part]
        else:
            return False
    return True

@lru_cache(maxsize=4)
def _get_search_async_client(timeout: float) -> httpx.AsyncClient:
    return httpx.AsyncClient(trust_env=False, timeout=timeout)

@lru_cache(maxsize=8)
def _get_search_async_semaphore(limit: int) -> asyncio.Semaphore:
    return asyncio.Semaphore(limit)

@BaseTool.register("SearchTools")
class ToolSearchTool(BaseTool):
    name = "SearchTools"
    description = "Retrieve candidate tools from a tool library based on a natural language query."
    TOOL_DESCRIPTION_MAX_LEN = 256
    PARAMETER_DESCRIPTION_MAX_LEN = 4096
    PARAMETER_EXAMPLE_MAX_LEN = 2048
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query"},
            "limit": {"type": "integer", "description": "Maximum number of tool candidates to return"},
        },
        "required": ["query"],
    }

    def __init__(self):
        super().__init__()
        self.api_url = os.environ.get("TOOL_SEARCH_API_URL", "http://localhost:30400")
        self.timeout = float(os.environ.get("TOOL_SEARCH_API_TIMEOUT", "30"))
        self.default_limit = int(os.environ.get("TOOL_SEARCH_DEFAULT_TOP_K", "3"))
        self.max_concurrent = int(os.environ.get("TOOL_SEARCH_MAX_CONCURRENT_PER_PROCESS", "0") or "0")

        self.max_retries = int(os.environ.get("TOOL_SEARCH_MAX_RETRIES", "0") or "0")
        self.retry_backoff = float(os.environ.get("TOOL_SEARCH_RETRY_BACKOFF", "0.5") or "0.5")

    def _truncate(self, text: Any, limit: int = 500) -> str:
        value = "" if text is None else str(text)
        if len(value) <= limit:
            return value
        return value[:limit] + "..."

    def _truncate_text(self, text: Any, limit: int) -> str:
        value = "" if text is None else str(text)
        if len(value) <= limit:
            return value
        return value[:limit] + "..."

    def _truncate_parameter_text(self, text: str, limit: int) -> str:
        """Keep about 70% of the retained text at the start and 30% at the end."""
        if len(text) <= limit:
            return text
        marker = "....."
        if limit <= len(marker):
            return marker[:max(0, limit)]
        available = limit - len(marker)
        head = available * 7 // 10
        tail = available - head
        return text[:head] + marker + (text[-tail:] if tail else "")

    def _format_parameter_example(self, value: Any) -> str:
        rendered = json.dumps(value, ensure_ascii=False)
        limit = self.PARAMETER_EXAMPLE_MAX_LEN
        if len(rendered) <= limit:
            return rendered
        if not isinstance(value, str):

            return self._truncate_parameter_text(rendered, limit)

        low, high = 5, min(len(value) - 1, limit - 2)
        best = json.dumps(".....", ensure_ascii=False)
        while low <= high:
            middle = (low + high) // 2
            candidate = json.dumps(self._truncate_parameter_text(value, middle), ensure_ascii=False)
            if len(candidate) <= limit:
                best = candidate
                low = middle + 1
            else:
                high = middle - 1
        return best

    def _format_parameter_schema(self, schema: Any) -> Any:
        """Add examples to model-visible descriptions and bound text length.
        Preserve both ends of truncated text and reserve space for examples. Traverse
        only schema-valued fields; retain literal values, property names, and raw candidates.
        """
        if not isinstance(schema, dict):
            return deepcopy(schema)
        result = deepcopy(schema)

        if "example_value" in result:
            example_value = result.pop("example_value")
            example = self._format_parameter_example(example_value)
            description = (result.get("description") or "").rstrip()
            if self._description_already_contains_example(description, example_value, example):
                result["description"] = self._truncate_parameter_text(
                    description, self.PARAMETER_DESCRIPTION_MAX_LEN,
                )
            else:
                suffix = f"e.g. {example}"
                description = self._truncate_parameter_text(
                    description, self.PARAMETER_DESCRIPTION_MAX_LEN - len(suffix) - 1,
                )
                result["description"] = f"{description} {suffix}" if description else suffix
        elif isinstance(result.get("description"), str):
            result["description"] = self._truncate_parameter_text(
                result["description"], self.PARAMETER_DESCRIPTION_MAX_LEN,
            )

        for key in ("properties", "patternProperties", "$defs", "definitions", "dependentSchemas", "dependencies"):
            if isinstance(result.get(key), dict):
                result[key] = {
                    name: self._format_parameter_schema(child)
                    for name, child in result[key].items()
                }
        for key in ("items", "additionalItems", "additionalProperties", "contains", "propertyNames",
                    "not", "if", "then", "else", "unevaluatedItems", "unevaluatedProperties"):
            if key in result:
                value = result[key]
                result[key] = (
                    [self._format_parameter_schema(child) for child in value]
                    if isinstance(value, list) else self._format_parameter_schema(value)
                )
        for key in ("allOf", "anyOf", "oneOf", "prefixItems"):
            if isinstance(result.get(key), list):
                result[key] = [self._format_parameter_schema(child) for child in result[key]]
        return result

    @staticmethod
    def _description_already_contains_example(
        description: str,
        example_value: Any,
        serialized_example: str,
    ) -> bool:
        """Detect an existing example by token boundaries and a nearby example cue."""
        if not description:
            return False
        candidates = [serialized_example]
        if isinstance(example_value, str):
            if example_value:
                candidates.append(example_value)
        else:
            candidates.append(json.dumps(example_value, ensure_ascii=False))
        for cue in _PARAMETER_EXAMPLE_CUE_RE.finditer(description):

            clause = description[cue.end():cue.end() + 256]
            stops = [
                match.start()
                for pattern in (r"(?i)\bdefault(?:\s+value)?\b", r"[.!?](?:\s|$)")
                if (match := re.search(pattern, clause)) is not None
            ]
            if stops:
                clause = clause[:min(stops)]
            for candidate in dict.fromkeys(candidates):
                if not candidate:
                    continue
                if re.search(

                    rf"(?<![\w.]){re.escape(candidate)}(?![\w.])",
                    clause,
                    re.IGNORECASE,
                ):
                    return True
        return False

    def _sanitize_openai_tool_schema(self, schema: dict[str, Any]) -> dict[str, Any] | None:
        if not isinstance(schema, dict):
            return None
        if schema.get("type") != "function" or not isinstance(schema.get("function"), dict):
            return None

        function = schema["function"]
        normalized_name = normalize_tool_name(function.get("name", ""))
        description = AUTONOMOUS_RETAIL_TOOL_DESCRIPTIONS.get(
            normalized_name,
            function.get("description", ""),
        )
        return {
            "type": "function",
            "function": {
                "name": normalized_name,
                "description": self._truncate_text(
                    description, limit=self.TOOL_DESCRIPTION_MAX_LEN
                ),
                "parameters": self._format_parameter_schema(function.get("parameters", {})),
            },
        }

    def _build_search_error(
        self,
        message: str,
        *,
        error_type: str | None = None,
        elapsed_ms: float | None = None,
        attempts: int | None = None,
        status_code: int | None = None,
    ) -> tuple[ToolResponse, float | None, dict]:
        logger.warning("ToolSearchTool.execute error=%s", message)
        return ToolResponse(text=message), None, {
            "success": False,
            "error": message,
            "tool_candidates": [],
            "error_type": error_type,
            "elapsed_ms": elapsed_ms,
            "attempts": attempts,
            "status_code": status_code,
        }

    def _normalize_query_args(self, args: dict[str, Any]) -> tuple[str, int]:
        query = str(args.get("query", "")).strip()
        limit = int(args.get("limit", self.default_limit) or self.default_limit)
        return query, limit

    async def execute(self, args: dict[str, Any], **kwargs) -> tuple[ToolResponse, float | None, dict]:
        results = await self.execute_many([args], **kwargs)
        return results[0]

    async def execute_many(
        self,
        args_list: list[dict[str, Any]],
        **kwargs,
    ) -> list[tuple[ToolResponse, float | None, dict]]:
        results: list[tuple[ToolResponse, float | None, dict] | None] = [None] * len(args_list)
        grouped_queries: dict[int, list[tuple[int, str]]] = {}

        for idx, args in enumerate(args_list):
            query, limit = self._normalize_query_args(args)
            logger.debug(
                "ToolSearchTool.execute_many request api_url=%s query=%s limit=%s",
                self.api_url,
                query,
                limit,
            )
            if not query:
                results[idx] = self._build_search_error("Failed to execute search: empty query")
                continue
            grouped_queries.setdefault(limit, []).append((idx, query))

        if not grouped_queries:
            return [result or self._build_search_error("Failed to execute search: empty query") for result in results]

        logger.debug(
            "ToolSearchTool.execute_many grouped_limits=%s total_queries=%s",
            sorted(grouped_queries.keys()),
            sum(len(items) for items in grouped_queries.values()),
        )

        client = _get_search_async_client(self.timeout)
        semaphore = (
            _get_search_async_semaphore(self.max_concurrent)
            if self.max_concurrent and self.max_concurrent > 0
            else None
        )
        for limit, indexed_queries in grouped_queries.items():
            queries = [query for _, query in indexed_queries]

            async def _do_post(_queries=queries, _limit=limit):
                if semaphore is None:
                    return await client.post(f"{self.api_url}/search", json={"queries": _queries, "top_k": _limit})
                async with semaphore:
                    return await client.post(f"{self.api_url}/search", json={"queries": _queries, "top_k": _limit})

            api_result = None
            last_exc: Exception | None = None
            status_code: int | None = None
            attempts = 0
            t_group0 = time.perf_counter()
            
            for attempt in range(self.max_retries + 1):
                attempts = attempt + 1
                try:
                    response = await _do_post()
                    status_code = response.status_code
                    logger.debug(
                        "ToolSearchTool.execute_many response status=%s body_head=%s",
                        status_code,
                        self._truncate(response.text),
                    )
                    response.raise_for_status()
                    api_result = response.json()
                    break
                except Exception as exc:
                    last_exc = exc
                    if attempt < self.max_retries:
                        await asyncio.sleep(self.retry_backoff * (attempt + 1))

            if api_result is None:
                message = f"Failed to execute search: {last_exc}"
                elapsed_ms = (time.perf_counter() - t_group0) * 1000.0
                for idx, _ in indexed_queries:
                    results[idx] = self._build_search_error(
                        message,
                        error_type=type(last_exc).__name__ if last_exc is not None else "unknown",
                        elapsed_ms=elapsed_ms,
                        attempts=attempts,
                        status_code=status_code,
                    )
                continue

            query_results = api_result.get("query_results") or []
            if len(query_results) != len(indexed_queries):
                message = (
                    "Failed to execute search: query_results length mismatch "
                    f"(expected {len(indexed_queries)}, got {len(query_results)})"
                )
                for idx, _ in indexed_queries:
                    results[idx] = self._build_search_error(message)
                continue

            for local_idx, (idx, query) in enumerate(indexed_queries):

                tool_blobs = self._extract_tool_blobs(api_result, query_index=local_idx)
                if isinstance(api_result, dict) and "error" in api_result:
                    formatted = json.dumps(api_result, ensure_ascii=False)
                else:
                    formatted = self._format_tool_blobs(tool_blobs)
                logger.debug(
                    "ToolSearchTool.execute_many formatted_results query=%s head=%s",
                    query,
                    self._truncate(formatted),
                )
                results[idx] = (
                    ToolResponse(text=formatted),
                    None,
                    {
                        "success": True,
                        "tool_candidates": tool_blobs,
                    },
                )

        return [result or self._build_search_error("Failed to execute search: unknown error") for result in results]

    def _parameter_schema_is_valid(self, tool_blob: dict[str, Any]) -> tuple[bool, str | None]:
        """Return (valid, reason) for an optional inline parameter schema.

        Check schema syntax, ref targets, and the shapes supported by
        api_json_to_openai_json; this does not validate individual call arguments.
        """
        if tool_blob.get("type") == "function" and isinstance(tool_blob.get("function"), dict):
            schema = tool_blob["function"].get("parameters")
        elif "tool_name" in tool_blob and "api_name" in tool_blob:

            return True, None
        elif "name" in tool_blob and "parameters" in tool_blob:
            schema = tool_blob.get("parameters")
        else:
            return False, "blob does not match any recognized tool contract"

        if schema is None:
            return True, None
        if not isinstance(schema, dict):
            return False, "parameters is not a JSON object"
        try:
            validator_for(schema).check_schema(schema)
        except SchemaError as exc:
            return False, exc.message
        for ref in _iter_schema_refs(schema):
            if not _local_json_pointer_resolves(schema, ref):
                return False, f"unresolved $ref: {ref}"
        return True, None

    def _extract_tool_blobs(self, api_result: dict[str, Any], query_index: int = 0) -> list[dict[str, Any]]:
        if not isinstance(api_result, dict) or "error" in api_result:
            return []
        if "query_results" in api_result:
            query_results = api_result.get("query_results") or []
            if query_index >= len(query_results):
                return []
            results = query_results[query_index].get("results", [])
        else:
            results = api_result.get("results", [])

        tool_blobs: list[dict[str, Any]] = []
        for result in results:
            tool_blob = result.get("tools", result)
            if not isinstance(tool_blob, dict):
                continue
            is_valid, reason = self._parameter_schema_is_valid(tool_blob)
            if not is_valid:
                name = tool_blob.get("name") or tool_blob.get("function", {}).get("name")
                logger.warning(
                    "ToolSearchTool dropping retrieval candidate name=%s: invalid parameter schema (%s)",
                    name,
                    reason,
                )
                continue
            tool_blobs.append(tool_blob)
        return tool_blobs

    def _format_tool_blobs(self, tool_blobs: list[dict[str, Any]]) -> str:
        formatted = []
        for tool_blob in tool_blobs:
            openai_tool = self.api_json_to_openai_json(tool_blob)
            if openai_tool is not None:
                formatted.append(openai_tool)
        return "\n".join(json.dumps(item, ensure_ascii=False) for item in formatted)

    def api_json_to_openai_json(self, api_json: dict[str, Any]) -> dict[str, Any] | None:
        if not isinstance(api_json, dict):
            return None

        if api_json.get("type") == "function" and isinstance(api_json.get("function"), dict):
            return self._sanitize_openai_tool_schema(api_json)

        if "tool_name" in api_json and "api_name" in api_json:
            return self._sanitize_openai_tool_schema(self._convert_legacy_tool_schema(api_json))

        if "name" in api_json and "parameters" in api_json:
            return self._sanitize_openai_tool_schema(self._convert_flat_tool_schema(api_json))

        return None

    def _convert_legacy_tool_schema(self, api_json: dict[str, Any]) -> dict[str, Any]:
        function_template = {
            "type": "function",
            "function": {
                "name": "",
                "description": "",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            }
        }

        template = function_template["function"]
        map_type = {

            "NUMBER": "number",
            "STRING": "string",
            "BOOLEAN": "boolean",
            "ARRAY": "array",
            "OBJECT": "object",
        }

        standard_tool_name = standardize(api_json["tool_name"])
        pure_api_name = change_name(standardize(api_json["api_name"]))
        template["name"] = f"{standard_tool_name}-{pure_api_name}"[:64]
        template["description"] = f'This is the subfunction for tool "{standard_tool_name}", you can use this tool.'

        api_description = (api_json.get("api_description") or "").strip()
        if api_description:
            normalized_desc = api_description.replace(api_json["api_name"], template["name"])
            template["description"] += f' The description of this function is: "{normalized_desc}"'

        for para in api_json.get("required_parameters", []):
            if not isinstance(para, dict) or not para.get("name"):

                continue
            name = change_name(standardize(para["name"]))
            param_type = map_type.get(para.get("type"), "string")
            template["parameters"]["properties"][name] = {
                "type": param_type,
                "description": para.get("description") or "",
            }
            if "example_value" in para:
                template["parameters"]["properties"][name]["example_value"] = para["example_value"]
            if name not in template["parameters"]["required"]:
                template["parameters"]["required"].append(name)

        for para in api_json.get("optional_parameters", []):
            if not isinstance(para, dict) or not para.get("name"):
                continue
            name = change_name(standardize(para["name"]))
            param_type = map_type.get(para.get("type"), "string")
            template["parameters"]["properties"][name] = {
                "type": param_type,
                "description": para.get("description") or "",
            }
            if "example_value" in para:
                template["parameters"]["properties"][name]["example_value"] = para["example_value"]

        return function_template

    def _convert_flat_tool_schema(self, api_json: dict[str, Any]) -> dict[str, Any]:
        raw_name = str(api_json.get("name", "")).strip()
        raw_desc = str(api_json.get("description", "") or "").strip()
        parameters = api_json.get("parameters", {})

        function_template = {
            "type": "function",
            "function": {
                "name": raw_name[:64],
                "description": raw_desc,
                "parameters": parameters,
            },
        }

        if " : " in raw_name:
            tool_name, api_name = [s.strip() for s in raw_name.split(" : ", 1)]
            standard_tool_name = standardize(tool_name)
            pure_api_name = change_name(standardize(api_name))
            normalized_name = f"{standard_tool_name}-{pure_api_name}"[:64]
            base_desc = f'This is the subfunction for tool "{standard_tool_name}", you can use this tool.'
            function_template["function"]["name"] = normalized_name
            function_template["function"]["description"] = (
                f'{base_desc} The description of this function is: "{raw_desc}"' if raw_desc else base_desc
            )

        return function_template

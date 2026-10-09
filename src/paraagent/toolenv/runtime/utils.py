import re
from typing import Any

BUSINESS_LEVEL_TOOL_FAILURE_MARKERS = (
    "not_found",
    "not found",
    "notfound",
    "notfounderror",
    "missing_resource",
    "resource_missing",
    "resource_not_found",
    "resourcenotfound",
    "empty_result",
    "empty results",
    "emptyresult",
    "no_result",
    "no_results",
    "no results",

    "businessrule",
    "business_rule",
    "business rule",
    "domain_error",
    "domainerror",
)

NON_BUSINESS_LEVEL_TOOL_FAILURE_MARKERS = (
    "param",
    "argument",
    "schema",
    "json",
    "parse",
    "validation",
    "connection",
    "network",
    "timeout",
    "auth",
    "permission",
    "ratelimit",
    "rate_limit",
    "rate limit",
    "serviceunavailable",
    "service_unavailable",
    "service unavailable",
    "tool_execution",
    "toolexecution",
    "tool_not_found",
    "toolnotfound",
    "unsupported",
)

RETRYABLE_EXTERNAL_TOOL_ERROR_MARKERS = (
    "auth",
    "permission",
    "network",
    "connection",
    "timeout",
    "ratelimit",
    "rate_limit",
    "rate limit",
    "serviceunavailable",
    "service_unavailable",
    "service unavailable",
    "tool_execution",
    "toolexecution",
    "emptyresponse",
    "empty_response",
    "empty response",
    "malformedresponse",
    "malformed_response",
    "malformed response",
)

def normalize_tool_error_type(error_type: Any) -> str:
    if not isinstance(error_type, str):
        return ""
    return error_type.strip().lower()

def is_business_level_tool_failure(error_type: Any) -> bool:
    normalized = normalize_tool_error_type(error_type)
    if not normalized:
        return False
    if any(marker in normalized for marker in NON_BUSINESS_LEVEL_TOOL_FAILURE_MARKERS):
        return False
    return any(marker in normalized for marker in BUSINESS_LEVEL_TOOL_FAILURE_MARKERS)

def is_retryable_external_tool_error(error_type: Any) -> bool:
    normalized = normalize_tool_error_type(error_type)
    if not normalized:
        return False
    return any(marker in normalized for marker in RETRYABLE_EXTERNAL_TOOL_ERROR_MARKERS)

def is_effective_tool_observation(success: bool, error_type: Any, *, injected_error: bool = False) -> bool:
    if injected_error:
        return False
    if bool(success):
        return True
    return is_business_level_tool_failure(error_type)

def standardize_category(category: str) -> str:
    save_category = category.replace(" ", "_").replace(",", "_").replace("/", "_")
    while " " in save_category or "," in save_category:
        save_category = save_category.replace(" ", "_").replace(",", "_")
    save_category = save_category.replace("__", "_")
    return save_category

def standardize(string: str) -> str:
    s1 = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", string)
    s2 = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", s1)
    string = s2

    string = re.sub(r"[^\u4e00-\u9fa5a-zA-Z0-9_]", "_", string)
    string = re.sub(r"(_)\1+", "_", string).lower()

    while string and string[0] == "_":
        string = string[1:]
    while string and string[-1] == "_":
        string = string[:-1]

    if not string:
        return string
    if string[0].isdigit():
        string = "get_" + string
    return string

def change_name(name: str) -> str:
    change_list = ["from", "class", "return", "false", "true", "id", "and"]
    if name in change_list:
        return "is_" + name
    return name

def normalize_tool_name(raw_name: str) -> str:
    name = str(raw_name or "").strip()
    if not name:
        return ""
    if " : " in name:
        tool_name, api_name = [s.strip() for s in name.split(" : ", 1)]
        standard_tool_name = standardize(tool_name)
        pure_api_name = change_name(standardize(api_name))
        return f"{standard_tool_name}-{pure_api_name}"[:64]
    return name[:64]

def get_tool_display_name(tool_payload: dict) -> str:
    if not isinstance(tool_payload, dict):
        return ""
    if tool_payload.get("type") == "function" and isinstance(tool_payload.get("function"), dict):
        return str(tool_payload["function"].get("name", "")).strip()
    if "name" in tool_payload:
        return str(tool_payload.get("name", "")).strip()
    if "tool_name" in tool_payload and "api_name" in tool_payload:
        tool_name = str(tool_payload.get("tool_name", "")).strip()
        api_name = str(tool_payload.get("api_name", "")).strip()
        return f"{tool_name} : {api_name}".strip(" :")
    return ""

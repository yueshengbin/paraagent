"""Build environment-neutral execution metadata using only the standard library.
Keep it separate from model-visible observations; reward code decides node completion.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

OUTCOME_VERSION = 1

CATEGORY_SUCCESS = "success"
CATEGORY_INVALID_REQUEST = "invalid_request"
CATEGORY_TOOL_NOT_FOUND = "tool_not_found"
CATEGORY_RESOURCE_NOT_FOUND = "resource_not_found"
CATEGORY_STATE_CONFLICT = "state_conflict"
CATEGORY_CAPACITY_UNAVAILABLE = "capacity_unavailable"
CATEGORY_AUTHORIZATION_ERROR = "authorization_error"
CATEGORY_RATE_LIMITED = "rate_limited"
CATEGORY_TRANSIENT_FAILURE = "transient_failure"
CATEGORY_EXECUTOR_FAILURE = "executor_failure"
CATEGORY_UNKNOWN_FAILURE = "unknown_failure"

OUTCOME_CATEGORIES = frozenset(
    {
        CATEGORY_SUCCESS,
        CATEGORY_INVALID_REQUEST,
        CATEGORY_TOOL_NOT_FOUND,
        CATEGORY_RESOURCE_NOT_FOUND,
        CATEGORY_STATE_CONFLICT,
        CATEGORY_CAPACITY_UNAVAILABLE,
        CATEGORY_AUTHORIZATION_ERROR,
        CATEGORY_RATE_LIMITED,
        CATEGORY_TRANSIENT_FAILURE,
        CATEGORY_EXECUTOR_FAILURE,
        CATEGORY_UNKNOWN_FAILURE,
    }
)

STAGE_COMPLETED = "completed"
STAGE_DISPATCH = "dispatch"
STAGE_REQUEST = "request"
STAGE_DOMAIN = "domain"
STAGE_AUTHORIZATION = "authorization"
STAGE_TRANSPORT = "transport"
STAGE_EXECUTOR = "executor"
STAGE_UNKNOWN = "unknown"

OUTCOME_STAGES = frozenset(
    {
        STAGE_COMPLETED,
        STAGE_DISPATCH,
        STAGE_REQUEST,
        STAGE_DOMAIN,
        STAGE_AUTHORIZATION,
        STAGE_TRANSPORT,
        STAGE_EXECUTOR,
        STAGE_UNKNOWN,
    }
)

BASIS_NATIVE_RETURN_SITE = "native_return_site"
BASIS_NATIVE_MESSAGE_REGISTRY = "native_message_registry"
BASIS_LOCAL_VALIDATION = "local_validation"
BASIS_UNKNOWN = "unknown"

OUTCOME_BASES = frozenset(
    {
        BASIS_NATIVE_RETURN_SITE,
        BASIS_NATIVE_MESSAGE_REGISTRY,
        BASIS_LOCAL_VALIDATION,
        BASIS_UNKNOWN,
    }
)

RETRY_NOT_APPLICABLE = "not_applicable"
RETRY_NEVER = "never"
RETRY_RETRY = "retry"
RETRY_BACKOFF = "backoff"
RETRY_UNKNOWN = "unknown"

RETRY_HINTS = frozenset(
    {
        RETRY_NOT_APPLICABLE,
        RETRY_NEVER,
        RETRY_RETRY,
        RETRY_BACKOFF,
        RETRY_UNKNOWN,
    }
)

_CODE_COMPONENT_RE = re.compile(r"[^A-Za-z0-9]+")

def _text_or_empty(value: Any) -> str:
    """Keep raw evidence exact when it is a string; reject implicit coercion."""

    return value if isinstance(value, str) else ""

def _normalise_code(value: Any) -> str:
    if not isinstance(value, str):
        return "UNKNOWN"
    code = _CODE_COMPONENT_RE.sub("_", value.strip()).strip("_").upper()
    return code or "UNKNOWN"

def _normalise_causal_inputs(values: Iterable[str] | None) -> list[str]:
    if values is None:
        return []
    if isinstance(values, str):
        values = (values,)
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        if not isinstance(value, str):
            continue
        value = value.strip()
        if value and value not in seen:
            result.append(value)
            seen.add(value)
    return result

def _require_member(name: str, value: str, allowed: frozenset[str]) -> str:
    if value not in allowed:
        choices = ", ".join(sorted(allowed))
        raise ValueError(f"invalid {name} {value!r}; expected one of: {choices}")
    return value

def make_tool_outcome(
    success: bool,
    *,
    stage: str,
    category: str,
    code: str,
    basis: str,
    retry_hint: str,
    causal_inputs: Iterable[str] | None = (),
    raw_type: Any = None,
    raw_message: Any = None,
    source: Any = None,
) -> dict[str, Any]:
    """Build a JSON-serializable outcome with validated semantic enums.
    causal_inputs lists argument paths identifying the failed resource, never values
    or proof of grounding.
    """

    success = bool(success)
    category = _require_member("category", category, OUTCOME_CATEGORIES)
    stage = _require_member("stage", stage, OUTCOME_STAGES)
    basis = _require_member("basis", basis, OUTCOME_BASES)
    retry_hint = _require_member("retry_hint", retry_hint, RETRY_HINTS)
    if success and category != CATEGORY_SUCCESS:
        raise ValueError("a successful outcome must use category='success'")
    if not success and category == CATEGORY_SUCCESS:
        raise ValueError("a failed outcome cannot use category='success'")

    return {
        "v": OUTCOME_VERSION,
        "success": success,
        "stage": stage,
        "category": category,
        "code": _normalise_code(code),
        "basis": basis,
        "retry_hint": retry_hint,
        "causal_inputs": _normalise_causal_inputs(causal_inputs),
        "raw_type": _text_or_empty(raw_type),
        "raw_message": _text_or_empty(raw_message),
        "source": _text_or_empty(source),
    }

__all__ = [name for name in globals() if name.startswith(("BASIS_", "CATEGORY_", "RETRY_", "STAGE_"))] + [
    "OUTCOME_BASES",
    "OUTCOME_CATEGORIES",
    "OUTCOME_STAGES",
    "OUTCOME_VERSION",
    "make_tool_outcome",
]

"""Project tool schemas into retrieval text with document_to_ir_text.
Preserve false-y annotations and non-scalar schema types.
"""

from __future__ import annotations

import json
from typing import Any

_TYPE_LABELS = {
    "integer": "NUMBER",
    "number": "NUMBER",
    "boolean": "BOOLEAN",
    "string": "STRING",
    "array": "ARRAY",
    "object": "OBJECT",
    "null": "NULL",
}


def schema_type_label(schema: Any) -> str:
    """Return a compact label without pretending unknown schemas are strings."""
    if not isinstance(schema, dict):
        return "UNKNOWN"
    declared = schema.get("type")
    if isinstance(declared, str):
        label = _TYPE_LABELS.get(declared, declared.upper())
        if declared == "array" and isinstance(schema.get("items"), dict):
            return f"ARRAY<{schema_type_label(schema['items'])}>"
        return label
    if isinstance(declared, list):
        return "|".join(_TYPE_LABELS.get(item, str(item).upper()) for item in declared)
    for combinator in ("anyOf", "oneOf"):
        branches = schema.get(combinator)
        if isinstance(branches, list):
            labels = list(dict.fromkeys(schema_type_label(branch) for branch in branches))
            return "|".join(labels) if labels else combinator.upper()
    if "$ref" in schema:
        return "REFERENCE"
    if "allOf" in schema:
        return "ALL_OF"
    return "UNKNOWN"


def annotation_value(schema: Any) -> Any:
    """Select an example/default by key presence, preserving 0 and false."""
    if not isinstance(schema, dict):
        return ""
    if "example_value" in schema:
        return schema["example_value"]
    if "default" in schema:
        return schema["default"]
    return ""


def _document_to_ir_text(document: dict[str, Any]) -> str:
    name = document.get("name", "")
    if " : " in name:
        tool_name, api_name = name.split(" : ", 1)
    else:
        tool_name, api_name = "", name

    parameters = document.get("parameters", {}) or {}
    properties = parameters.get("properties", {}) or {}
    required = set(parameters.get("required", []) or [])

    required_parameters = []
    optional_parameters = []
    for parameter_name, parameter_info in properties.items():
        parameter_info = parameter_info or {}
        type_label = schema_type_label(parameter_info)
        default = annotation_value(parameter_info)
        item = {
            "name": parameter_name,
            "description": parameter_info.get("description", ""),
            "type": type_label,
            "default": default,
        }
        target = required_parameters if parameter_name in required else optional_parameters
        target.append(item)

    return (
        document.get("category", "")
        + ", "
        + tool_name
        + ", "
        + api_name
        + ", "
        + document.get("description", "")
        + ", required_parameters: "
        + json.dumps(required_parameters)
        + ", optional_parameters: "
        + json.dumps(optional_parameters)
    )


def document_to_ir_text(document: dict[str, Any]) -> str:
    return _document_to_ir_text(document)

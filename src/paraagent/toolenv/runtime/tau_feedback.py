"""TAU-only, deterministic request/response contract. No reward or GT inputs."""
from copy import deepcopy
from functools import lru_cache
import json
import math
import re

from jsonschema import validators
from referencing import Registry
from referencing.exceptions import NoSuchResource

def mounted_schema(info):
    """The same schema is advertised to the model and enforced before execution."""
    info = deepcopy(info)

    def close(schema):
        if schema.get("type") == "object":
            schema["additionalProperties"] = False
            for child in schema.get("properties", {}).values():
                close(child)
        if isinstance(schema.get("items"), dict):
            close(schema["items"])

    close(info["function"]["parameters"])
    if info["function"]["name"] == "update_reservation_flights":

        fields = info["function"]["parameters"]["properties"]["flights"]["items"]["properties"]
        for field, kind in (("origin", "string"), ("destination", "string"), ("price", "number")):
            fields[field] = {"type": kind, "description": "Optional read-only echo of an existing reservation segment; must match the current stored value. Omit for a new segment."}
    if info["function"]["name"] == "send_certificate":
        info["function"]["parameters"]["properties"]["amount"]["exclusiveMinimum"] = 0
    return info

def _no_remote_schema(uri):

    raise NoSuchResource(ref=uri)

@lru_cache(maxsize=128)
def _schema_validator(schema_json):
    schema = json.loads(schema_json)
    
    if isinstance(schema, dict) and isinstance(schema.get("required"), list):
        schema = {**schema, "required": list(dict.fromkeys(schema["required"]))}
    cls = validators.validator_for(schema)
    cls.check_schema(schema)
    return cls(schema, registry=Registry(retrieve=_no_remote_schema))

def _finite_errors(value, path):
    errors = []
    if isinstance(value, float) and not math.isfinite(value):
        return [f"{path}: expected a finite JSON number"]
    if isinstance(value, dict):
        for key, child in value.items():
            errors += _finite_errors(child, f"{path}.{key}")
    elif isinstance(value, list):
        for i, child in enumerate(value):
            errors += _finite_errors(child, f"{path}[{i}]")
    return errors

def argument_errors(value, schema, path="arguments"):
    """Use jsonschema, as non-TAU does, retaining paths and finite JSON inputs.

    No FormatChecker is enabled, matching non-TAU jsonschema.validate. Invalid
    schemas/unresolved references raise an executor fault, not a user mistake.
    Validation is cached by schema content, with no model arguments retained.
    """
    validator = _schema_validator(json.dumps(schema, sort_keys=True, allow_nan=False))
    errors = _finite_errors(value, path)
    if errors:
        return errors
    for error in validator.iter_errors(value):
        field = path + "".join(f"[{key}]" if isinstance(key, int) else f".{key}" for key in error.absolute_path)
        errors.append(f"{field}: {error.message}")
        if len(errors) >= 8:
            break
    return errors

def parse_calls(text):
    """Keep one slot per block, including malformed and unclosed blocks."""
    slots = []
    start = None

    def append(body=None, error=None):
        call = None
        if error is None:
            try:
                call = json.loads(body, parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
            except (ValueError, TypeError, RecursionError):
                error = "Invalid tool call JSON. Provide one JSON object inside a complete <tool_call> block."
        if error is None and not isinstance(call, dict):
            error = "Invalid tool call: expected a JSON object."
        name = call.get("name") if isinstance(call, dict) else None
        if error is None and (not isinstance(name, str) or not name.strip()):
            error = "Invalid tool name: expected a non-empty string."
        slot = {"name": name if isinstance(name, str) else "", "arguments": call.get("arguments", {}) if isinstance(call, dict) else {}}
        if error:
            slot["_tau_parse_error"] = error
        slots.append(slot)

    for token in re.finditer(r"</?tool_call>", text or ""):
        if token.group() == "<tool_call>":
            if start is not None:
                append(error="Unclosed tool_call block. Close it with </tool_call> before starting another call.")
            start = token.end()
        elif start is None:
            append(error="Unexpected </tool_call>: missing opening <tool_call> tag.")
        else:
            append(body=text[start:token.start()])
            start = None
    if start is not None:
        append(error="Unclosed tool_call block. Add </tool_call> and retry the complete call.")
    return slots

def certificate_error(state, arguments):
    user = state.get("users", {}).get(arguments["user_id"])
    if user is None:
        return None  
    if all(f"certificate_{i}" in user["payment_methods"] for i in (3221322, 3221323, 3221324)):
        return "All three certificate IDs are occupied; no certificate was issued."
    return None

def retained_flight_error(state, arguments):
    reservation = state.get("reservations", {}).get(arguments["reservation_id"])
    if reservation is None:
        return None
    existing = {(s["flight_number"], s["date"]): s for s in reservation["flights"]}
    for index, segment in enumerate(arguments["flights"]):
        old = existing.get((segment["flight_number"], segment["date"]))
        for key in ("origin", "destination", "price"):
            if key in segment and (old is None or segment[key] != old.get(key)):
                return f"arguments.flights[{index}].{key}: read-only metadata must match an existing reservation segment; omit it for a new segment."
    return None

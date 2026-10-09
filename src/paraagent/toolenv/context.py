"""Compact, task-supplied TAU simulation time. Never consult the host clock."""
from datetime import datetime
import re

VERSION = "tau_prompt_time_v1"
_VALUE = re.compile(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) (EST|EDT|UTC|GMT)\.?")

def _valid_value(value):
    if not isinstance(value, str):
        return False
    match = _VALUE.fullmatch(value)
    if not match:
        return False
    try:
        datetime.strptime(match[1], "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return False
    return True

def extract_tau_prompt_time(messages):
    """Read only leading time headers in original user messages, not answers.

    Preserve the literal simulator timezone; unsupported/malformed/conflicting
    declarations are explicit invalid context, never a guessed current date.
    """
    out = {"version": VERSION, "status": "absent", "value": "", "source": "initial_user_prompt"}
    values = []
    if not isinstance(messages, (list, tuple)):
        return out
    for message in messages:
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        content = message.get("content")
        if not isinstance(content, str) or not content.lstrip().startswith("[Current Time]"):
            continue
        lines = content.lstrip().splitlines()
        value = lines[1].strip() if len(lines) > 1 and lines[0].strip() == "[Current Time]" else ""
        if not _valid_value(value):
            return {**out, "status": "invalid"}
        values.append(value.rstrip("."))
    if len(set(values)) > 1:
        return {**out, "status": "invalid"}
    if values:
        out.update(status="valid", value=values[0])
    return out

def validate_tau_prompt_time(context):
    """Validate a compact environment sidecar before forwarding it to a judge."""
    if context is None:
        return extract_tau_prompt_time([])
    if (not isinstance(context, dict) or context.get("version") != VERSION
            or context.get("source") != "initial_user_prompt"
            or not isinstance(context.get("status"), str)
            or context.get("status") not in {"valid", "absent"}
            or (context.get("status") == "valid" and not _valid_value(context.get("value")))
            or (context.get("status") == "absent" and context.get("value") != "")):
        return {**extract_tau_prompt_time([]), "status": "invalid"}
    return {key: context[key] for key in ("version", "status", "value", "source")}

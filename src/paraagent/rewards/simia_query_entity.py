"""Bind Simia inquiry answers to dataset target contracts and exact read evidence.
Answers may omit IDs; explicit claims about another record cannot borrow target evidence.
"""

from __future__ import annotations
import re

CONTRACT_VERSION = "tau_query_target_v1"
QUERY_ROLES = {
    "airline": {
        "flight_type",
        "cabin",
        "baggage_count",
        "checked_baggage_count",
        "passenger_count",
        "first_flight_number",
        "total_paid",
    },
    "retail": {
        "status",
        "item_count",
        "shipping_zip",
        "tracking",
        "tracking_id",
        "total_paid",
        "original_credit_card",
    },
}


def _id(value):
    return value.strip().lstrip("#").lower() if isinstance(value, str) else ""


def validate_contract(atom):
    contract = atom.get("query_target_contract")
    if not isinstance(contract, dict) or contract.get("version") != CONTRACT_VERSION:
        return None, "missing_or_invalid_query_target_contract"
    domain = contract.get("domain")
    if domain not in ("airline", "retail"):
        return None, "invalid_query_target_domain"
    rid = contract.get("record_id")
    pattern = r"[A-Z0-9]{6}" if domain == "airline" else r"#W\d{7}"
    if not isinstance(rid, str) or not re.fullmatch(pattern, rid):
        return None, "invalid_query_target_record_id"
    if not isinstance(contract.get("owner_user_id"), str) or not contract["owner_user_id"]:
        return None, "missing_query_target_owner"
    if (
        not isinstance(atom.get("semantic_role"), str)
        or contract.get("semantic_role") != atom["semantic_role"]
    ):
        return None, "query_target_role_mismatch"
    if contract["semantic_role"] not in QUERY_ROLES[domain]:
        return None, "unsupported_query_target_role"
    tool, arg = (
        ("get_reservation_details", "reservation_id")
        if domain == "airline"
        else ("get_order_details", "order_id")
    )
    expected = {"tool": tool, "arguments": {arg: rid}}
    if contract.get("required_read_call") != expected:
        return None, "query_target_read_contract_mismatch"
    evidence = atom.get("required_evidence_calls")
    if not isinstance(evidence, list) or expected not in evidence:
        return None, "query_target_not_in_gt_evidence"
    if contract["semantic_role"] == "original_credit_card":
        supporting = {"tool": "get_user_details", "arguments": {"user_id": contract["owner_user_id"]}}
        if domain != "retail" or supporting not in evidence:
            return None, "query_card_supporting_evidence_missing_from_contract"
    return contract, ""


def _mentions(text, contract):
    if contract["domain"] == "retail":
        found = [(m.start(), m.end(), m[0]) for m in re.finditer(r"(?<!\w)#?W\d{7}\b", text, re.I)]
    else:
        token = r"(?:[A-Z0-9]{6}|(?=[A-Za-z0-9]{0,5}\d)[A-Za-z0-9]{6})"
        explicit = (
            r"(?i:reservation(?:\s+(?:id|code))?|booking(?:\s+(?:id|code))?)\s*[:#`*]*\s*("
            + token
            + r")(?![A-Za-z0-9_])"
        )
        found = [(m.start(1), m.end(1), m[1]) for m in re.finditer(explicit, text)]
        found += [(m.start(1), m.end(1), m[1]) for m in re.finditer(r"(?<!\w)([A-Z0-9]{6})\s*:", text)]
        target = re.escape(contract["record_id"])
        found += [
            (m.start(), m.end(), m[0]) for m in re.finditer(r"(?<!\w)" + target + r"(?!\w)", text, re.I)
        ]

    mentions = []
    for start, end, value in sorted(set(found)):
        before = text[max(text.rfind("\n", 0, start) + 1, start - 65) : start].lower()
        if re.search(r"\b(?:not|unrelated|wrong|mistaken|previously|initially|earlier)\b[^.;!?]*$", before):
            continue
        mentions.append((start, end, value))
    return mentions


def answer_scope(text, contract, match_text):
    mentions = _mentions(text, contract)
    if not mentions:
        return True, "no_explicit_record_claim"
    target = _id(contract["record_id"])
    matching = [i for i, m in enumerate(mentions) if _id(m[2]) == target]
    if not matching:
        return False, "answer_names_wrong_target_record"
    if all(_id(m[2]) == target for m in mentions):
        return True, "only_target_record_named"

    for i in matching:
        start = mentions[i][0]
        line_start = text.rfind("\n", 0, start) + 1
        if i == 0 or line_start >= mentions[i - 1][1]:
            start = line_start
        end = mentions[i + 1][0] if i + 1 < len(mentions) else len(text)
        if match_text(text[start:end]):
            return True, "required_value_in_target_record_region"
    return False, "required_value_not_bound_to_target_record"


def check_target(text, atom, succeeded_calls, match_text):
    contract, error = validate_contract(atom)
    if contract is None:
        return {
            "checked": True,
            "spec_valid": False,
            "read_pass": False,
            "answer_pass": False,
            "reason": error,
        }
    required = contract["required_read_call"]
    minimal_reads = [required]

    if contract["semantic_role"] == "original_credit_card":
        minimal_reads.append(
            {"tool": "get_user_details", "arguments": {"user_id": contract["owner_user_id"]}}
        )
    missing = []
    for call in minimal_reads:
        arg = next(iter(call["arguments"]))
        if not any(
            tool == call["tool"]
            and isinstance(args, dict)
            and isinstance(args.get(arg), str)
            and _id(args[arg]) == _id(call["arguments"][arg])
            for tool, args in succeeded_calls
        ):
            missing.append(call)
    read_pass = not missing
    answer_pass, reason = answer_scope(text, contract, match_text)
    return {
        "checked": True,
        "spec_valid": True,
        "record_id": contract["record_id"],
        "required_read_call": required,
        "read_pass": read_pass,
        "answer_pass": answer_pass,
        "minimal_evidence_calls": minimal_reads,
        "missing_evidence_calls": missing,
        "reason": reason if not answer_pass or read_pass else "query_target_or_field_evidence_missing",
    }

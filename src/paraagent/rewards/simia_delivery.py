"""Bind Simia answer assertions to their target entities."""

from __future__ import annotations

import re

from paraagent.rewards import simia_delivery_matcher as matcher

VERSION = "delivery_v3"
ORDER_ID = re.compile(r"(?<!\w)#?w\d{7}\b", re.I)
ORDER_REASONS = {
    "order_not_pending_for_cancellation": (
        r"cancel(?:led|ed|lation|ing)?",
        {"processed", "delivered", "cancelled", "not pending"},
    ),
    "order_not_pending_for_address_change": (
        r"(?:address|requested change)",
        {"processed", "delivered", "cancelled", "not pending"},
    ),
    "order_not_pending_for_item_change": (
        r"(?:items?|products?|variants?|order)",
        {"processed", "delivered", "cancelled", "not pending"},
    ),
    "order_already_cancelled_for_modification": (
        r"(?:modify|modified|change|changed|update|updated)",
        {"cancelled"},
    ),
    "order_not_delivered_for_return": (r"return(?:ed|ing)?", {"pending", "processed", "not delivered"}),
    "order_not_delivered_for_exchange": (r"exchange(?:d|ing)?", {"pending", "processed", "not delivered"}),
}


def _id(value):
    return str(value).lstrip("#").lower()


def _without_attributed_quotes(text):
    """Ignore quoted user/history claims, not quoted current field values."""
    text = re.sub(r"(?m)^\s*>.*$", "", text)
    return re.sub(
        r"\b(?:you\s+(?:wrote|said)|earlier\s+I\s+said|previously\s+I\s+said)\s*:?\s*"
        r'(?:"[^"\n]*"|“[^”]*”|`[^`]*`)',
        "",
        text,
        flags=re.I,
    )


def assertion_units(text):
    """Separate assertions, dropping quotes and explicitly superseded claims."""
    text = text.replace("’", "'")
    text = _without_attributed_quotes(text)
    text = re.sub(r'["“”`]', "", text)
    units = re.split(r"(?<=[!?;])\s*|(?<!\d)\.\s+|\n|\b(?:but|however)\b", text, flags=re.I)
    return [
        u.strip().lower()
        for u in units
        if u.strip()
        and not re.search(
            r"\b(?:earlier|previously|initially|mistakenly|incorrectly|quoted|hypothetically)\b", u, re.I
        )
    ]


def refusal_binding(text, atom, reason_code, question=""):
    """Require an asserted target status and a nearby corresponding refusal."""
    if reason_code not in ORDER_REASONS:
        return None
    target = _id(atom.get("source_entity", ""))
    if not re.fullmatch(r"w\d{7}", target):
        ids = {_id(m[0]) for m in ORDER_ID.finditer(question)}
        if len(ids) != 1:
            return False, "refusal_target_not_resolved"
        target = ids.pop()
    action, allowed = ORDER_REASONS[reason_code]
    units = assertion_units(text)
    status_hits, refusal_hits, context = [], [], target
    for index, unit in enumerate(units):
        ids = {_id(m[0]) for m in ORDER_ID.finditer(unit)}
        if ids:
            context = next(iter(ids)) if len(ids) == 1 else ""

        if ids and ids != {target}:
            continue
        if context != target:
            continue
        denial = re.search(
            r"\b(?:cannot|can't|could not|couldn't|unable to|not allowed|not permitted|"
            r"not possible|not eligible|must decline|could not be|was not)\b",
            unit,
        )
        if denial and re.search(r"\b(?:" + action + r")\b", unit):
            if not re.search(
                r"\b(?:confirm|verify|determine|tell|know)\b", unit[denial.end() : denial.end() + 65]
            ):
                refusal_hits.append(index)
        if re.search(
            r"\b(?:whether|if|might|may be|perhaps|possibly|uncertain|unknown)\b|"
            r"\b(?:cannot|can't|could not|unable to)\s+(?:confirm|verify|determine|tell)\b",
            unit,
        ):
            continue
        subject = r"(?:(?:(?:the|your|this|that)\s+)?order(?:\s+#?w\d{7})?|#?w\d{7}|it)"
        predicate = (
            r"(?:\s+(?:is|was|has been|had been|remains)(?:\s+(?:already|currently|now|still))?\s+|\s*:\s*)"
        )
        status = r"(not\s+(?:yet\s+)?delivered|has not been delivered|not pending|non-pending|processed|delivered|cancelled|canceled|pending)\b"
        matches = list(re.finditer(r"\b" + subject + predicate + status, unit))

        matches += list(re.finditer(r"\b(?:order\s+)?status\s*(?:is\s+|was\s+|[:=]\s*)" + status, unit))
        for match in matches:
            before = re.split(r"\b(?:because|since|as|therefore|so)\b|[,;:]", unit[: match.start()])[-1]
            if re.search(
                r"\b(?:request|attempt|payment|refund|change|update)\b.*\b(?:for|to|of)\s*$", before
            ):
                continue

            if re.search(
                r"^(?:the|your|this|a)\s+(?:request|attempt|payment|refund|change|update)\b", before.strip()
            ):
                continue
            value = (
                match.group(match.lastindex)
                .replace("canceled", "cancelled")
                .replace("non-pending", "not pending")
            )
            value = re.sub(r"not yet delivered|has not been delivered", "not delivered", value)
            if value in allowed:
                status_hits.append(index)
    if any(abs(status - denial) <= 1 for status in status_hits for denial in refusal_hits):
        return True, "target_order_status_and_corresponding_refusal"
    return False, "refusal_status_not_asserted_for_target_order"


def atom_match(text, atom, schema_match, *, question="", reason_code=""):
    result = matcher.atom_match(text, atom, schema_match, question=question)
    if result is None:
        return None
    bound = refusal_binding(text, atom, reason_code, question)
    if bound is None:
        return result

    return (bool(bound[0] and not result[1]), bound[1] if not bound[0] else result[1])


STREET_SUFFIXES = {
    "st": "street",
    "ave": "avenue",
    "av": "avenue",
    "rd": "road",
    "blvd": "boulevard",
    "dr": "drive",
    "ln": "lane",
    "ct": "court",
    "pl": "place",
    "pkwy": "parkway",
    "cir": "circle",
    "ter": "terrace",
}
STREET_WORDS = "street|st|avenue|ave|av|road|rd|boulevard|blvd|drive|dr|lane|ln|court|ct|place|pl|parkway|pkwy|circle|cir|terrace|ter|way"


def normalize_field(value, field):
    value = re.sub(r"[.*`\"“”]", "", str(value).lower()).strip(" ,:;-\n")
    value = re.sub(r"\s+", " ", value)
    if field == "address1":
        words = value.split()
        if words:
            words[-1] = STREET_SUFFIXES.get(words[-1], words[-1])
        return " ".join(words)
    if field == "address2":
        value = re.sub(r"\b(?:suite|ste)\b", "suite", value)
        return re.sub(r"\b(?:apartment|apt)\b", "apartment", value)
    if field == "country" and value in {"us", "u s", "usa", "united states", "united states of america"}:
        return "usa"
    return value


def _address_fields(block):
    fields = {}
    street = re.search(
        r"\b(\d+[a-z]?(?:[-/]\d+)?\s+[a-z0-9][a-z0-9 .'-]{0,70}?\s+(?:" + STREET_WORDS + r"))\b\.?",
        block,
        re.I,
    )
    if street:
        fields["address1"] = street[1]
    unit = re.search(r"\b((?:suite|ste\.?|apartment|apt\.?|unit)\s*[#:]?\s*[a-z0-9-]+)\b", block, re.I)
    if unit:
        fields["address2"] = unit[1]
    for field, label in {
        "address1": r"address(?: line)?\s*1|street address",
        "address2": r"address(?: line)?\s*2",
        "city": r"city",
        "state": r"state",
        "country": r"country",
        "zip": r"zip(?: code)?|postal code",
    }.items():
        match = re.search(r"\b(?:" + label + r")\s*(?:[:=]|is\b|to\b)\s*([^,;\n]+)", block, re.I)
        if match:
            fields[field] = match[1].strip().rstrip(".")

    tail = re.search(
        r"(?:,|\n)\s*([a-z][a-z .'-]{0,45}?)\s*,\s*([a-z]{2})\s+(\d{5}(?:-\d{4})?)\b(?:\s*,\s*(usa|us|united states(?: of america)?))?",
        block,
        re.I,
    )
    if tail:
        fields.update(city=tail[1].strip(), state=tail[2], zip=tail[3])
        if tail[4]:
            fields["country"] = tail[4]
    else:
        zip_match = re.search(r"\b[a-z]{2}\s+(\d{5}(?:-\d{4})?)\b", block, re.I)
        if zip_match:
            fields["zip"] = zip_match[1]
    house = re.search(r"\b(?:house|building|door)\s*(?:number|no\.?)\s*[:=]?\s*(\d+[a-z]?)\b", block, re.I)
    if house:
        fields["house_number"] = house[1]
    return fields


def answer_consistency(text, writes):
    """Veto only explicit contradictory updated-target fields; 'Done' is fine."""
    if not isinstance(writes, list) or any(not isinstance(call, dict) for call in writes):
        return {"pass": False, "reason": "invalid_write_spec"}
    addresses = [
        call for call in writes if call.get("tool") in {"modify_user_address", "modify_pending_order_address"}
    ]
    if not addresses:
        return {"pass": True, "reason": "not_applicable"}

    clean = text.replace("’", "'")
    clean = _without_attributed_quotes(clean)
    clean = re.sub(r'["“”`]', "", clean)
    blocks = re.split(r"\n\s*\n|(?<=[!?])\s+|(?<!\bSt)(?<!\bRd)\.\s+(?=[A-Z])", clean)
    checked = []
    for block in blocks:
        parts = re.split(r"\b(?:but|however|instead)\b|[,;]\s*(?=(?:I|we)\b)", block, flags=re.I)
        for part in parts:
            part = re.sub(r"\bas requested\b", "", part, flags=re.I)
            if re.search(
                r"\b(?:old|previous|previously|formerly|earlier|incorrect(?:ly)?|mistaken(?:ly)?|not|never)\b",
                part,
                re.I,
            ):
                continue
            if not (
                re.search(r"\baddress\b", part, re.I)
                and re.search(r"\b(?:updated|changed|now|new|set to)\b", part, re.I)
            ):
                continue
            if re.search(
                r"\b(?:will|would|could|should|requested|please|want|wish|if)\b|\bunable to\b|\bcannot\b",
                part,
                re.I,
            ):
                continue
            applicable = []
            order_ids = {_id(match[0]) for match in ORDER_ID.finditer(part)}
            user_ids = set(re.findall(r"\b[a-z][a-z0-9]+_[a-z0-9_]+\b", part, re.I))
            for call in addresses:
                args = call.get("args")
                if not isinstance(args, dict):
                    return {"pass": False, "reason": "invalid_write_args"}
                key = "order_id" if call["tool"] == "modify_pending_order_address" else "user_id"
                identity = args.get(key)
                named = order_ids if key == "order_id" else {value.lower() for value in user_ids}
                if named and _id(identity) not in named:
                    continue
                if len(addresses) == 1 or (
                    isinstance(identity, str) and identity.lower().lstrip("#") in {_id(x) for x in named}
                ):
                    applicable.append(args)
            if len(applicable) != 1:
                continue
            args = applicable[0]
            for field, reported in _address_fields(part).items():
                expected = args.get(field)
                if field == "house_number":
                    match = re.match(r"\s*(\d+[a-z]?)", str(args.get("address1", "")), re.I)
                    expected = match[1] if match else None
                if expected is None:
                    continue
                checked.append(field)
                if normalize_field(reported, field) != normalize_field(expected, field):
                    return {
                        "pass": False,
                        "reason": "reported_updated_address_" + field + "_mismatch",
                        "field": field,
                        "expected": expected,
                        "reported": reported,
                    }
    return {
        "pass": True,
        "reason": "no_explicit_updated_address_contradiction",
        "checked_fields": sorted(set(checked)),
    }

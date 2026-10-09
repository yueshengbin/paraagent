"""Match Simia answer text; state and execution evidence are checked separately."""

from __future__ import annotations

import copy
import math
import re


ROLES = {
    "refund": r"refund(?:ed)?|reversal|paid amount|monetary result",
    "total_paid": r"total|(?:paid|charged|collected) amount|amount.{0,25}charged",
    "item_count": r"item.?unit|total item|unit count|item count",
    "shipping_zip": r"(?:shipping.?)?(?:postal.?code|zip)",
    "status": r"(?:order|delivery) status",
    "flight_type": r"trip type|round.?trip|one.?way",
    "checked_baggage_count": r"checked bags?|baggage count",
    "passenger_count": r"passenger count|number of passengers",
}


def clauses(text):

    return re.split(r"(?<=[.!?])\s+|[;\n]|\b(?:but|however|therefore|so)\b", text.lower())


def denial_reason(text, role):
    subject = ROLES.get(role)
    if not subject:
        return ""
    subject = rf"\b(?:{subject})\b"
    if role == "refund" and re.search(
        r"(?:cannot|can't|unable to|could not)\s+refund\b|"
        r"rather than (?:a )?completed refund|"
        r"cannot confirm[^\n]*:\s*\n\s*-\s*(?:a )?refund\b|"
        r"\bno refund\b.{0,65}\bwas returned\b",
        text,
        re.I,
    ):
        return "required_refund_explicitly_denied"
    for clause in clauses(text.replace("’", "'")):
        if role == "refund" and re.search(r"\b(?:bank|settlement|external|received by)\b", clause):
            continue
        active = re.search(
            r"(?:cannot|can't|could not|couldn't|unable to|failed to)\s+"
            r"(?:independently\s+)?(?:confirm|verify|provide|determine|identify|process|execute)\s+(.+)",
            clause,
        )
        if active:
            obj = re.split(r",|\b(?:because|from|based on|due to|only)\b", active[1])[0]
            if re.search(subject, obj):
                return "required_result_explicitly_unconfirmed"
        if re.search(
            rf"(?:{subject}).{{0,65}}?\b(?:is|was|were|could|can|cannot|are|remains?)\s+"
            r"(?:(?:not|never)\s+(?:(?:be|explicitly|fully)\s+)*"
            r"(?:available|visible|confirmed|executed|processed|completed|refunded)|"
            r"not be confirmed|unavailable|unresolved)",
            clause,
        ):
            return "required_result_denied"
        if re.search(rf"(?:does not|did not|doesn't)\s+show.{{0,55}}(?:{subject})", clause):
            return "required_result_reported_missing"
    return ""


def mask_identifier_numbers(text):

    return re.sub(r"\b[a-zA-Z][a-zA-Z0-9]*(?:_[a-zA-Z0-9]+)+\b", lambda m: re.sub(r"\d", "x", m[0]), text)


def entity_regions(text, entity):
    """Bind paragraphs/lists to explicit retail order IDs, not a larger window."""
    if not re.fullmatch(r"#?W\d{7}", str(entity), re.I):
        return [text]
    mentions = list(re.finditer(r"(?<!\w)#?W\d{7}\b", text, re.I))
    if not mentions:
        return [text]
    target = str(entity).lstrip("#").lower()
    regions = []
    for index, mention in enumerate(mentions):
        if mention[0].lstrip("#").lower() != target:
            continue
        start = text.rfind("\n", 0, mention.start()) + 1

        if index and start < mentions[index - 1].end():
            start = mention.start()
        end = mentions[index + 1].start() if index + 1 < len(mentions) else len(text)
        regions.append(text[start:end])
    return regions


def stage_reason(text, stage):
    if stage is None:
        return ""
    if not isinstance(stage, str) or stage not in {
        "return_requested",
        "exchange_requested",
        "refund_recorded",
    }:
        return "invalid_delivery_stage"
    for clause in clauses(text):
        if re.search(r"\b(?:not|never|will|would|expected|once|after|pending|awaiting)\b", clause):
            continue
        if stage == "return_requested" and re.search(
            r"\brefunded\b|\brefund\b.{0,40}\b(?:completed|processed|issued|settled)\b", clause
        ):
            return "return_request_is_not_completed_refund"
        if stage == "exchange_requested" and re.search(
            r"\b(?:replacement|new items?|exchange)\b.{0,40}\b(?:shipped|delivered|received)\b", clause
        ):
            return "exchange_request_is_not_shipment"
        if stage == "refund_recorded" and re.search(
            r"\b(?:refund|money|funds)\b.{0,55}\b(?:settled|arrived|received by.*bank)\b", clause
        ):
            return "recorded_refund_is_not_external_settlement"
    return ""


def atom_match(text, atom, schema_match, *, question=""):
    spec = atom.get("reward_match_spec")
    if not isinstance(spec, dict):
        return None
    original_hit, original_error = schema_match(text, spec)
    if original_error:
        return original_hit, original_error
    role = atom.get("semantic_role", "")
    if not isinstance(role, str) or not isinstance(atom.get("source_tool", ""), str):
        return False, "invalid_delivery_atom_metadata"
    regions = (
        entity_regions(text, atom.get("source_entity", ""))
        if atom.get("source_tool")
        in {"cancel_pending_order", "return_delivered_order_items", "exchange_delivered_order_items"}
        else [text]
    )
    for region in regions:
        reason = denial_reason(region, role) or stage_reason(region, atom.get("delivery_stage"))
        if reason:
            return False, reason
    if spec.get("matcher") == "money":
        target = spec.get("target")
        if isinstance(target, bool):
            return original_hit, original_error
        try:
            target = float(target)
        except (TypeError, ValueError, OverflowError):
            return original_hit, original_error
        if not math.isfinite(target):
            return original_hit, original_error

        amount_pattern = (
            r"[+-]?(?:\d{1,3}(?:,\d{3})+(?:\.\d+)?|"
            r"\d+(?:\.\d+)?|\.\d+)"
        )
        money_regions = []
        for region in regions:
            region = re.sub(r"[*`]", "", region).replace("−", "-")

            region = re.sub(r"([+-])([$£€])\s*", r"\2\1", region)

            if role not in {"item_count", "passenger_count", "baggage_count"}:
                foreign = re.finditer(
                    rf"(?:[£€]|\b(?:GBP|EUR)\s*)\s*({amount_pattern})|"
                    rf"(?<![\w.,+-])({amount_pattern})\s*(?:GBP|EUR|pounds|euros)\b",
                    region,
                    re.I,
                )

                negated_foreign_spans = []
                for match in foreign:
                    if re.search(r"\b(?:not|never)\s*$", region[: match.start()], re.I):
                        negated_foreign_spans.append(match.span())
                        continue
                    amount = match[1] or match[2]
                    if amount and abs(float(amount.replace(",", "")) - target) < 0.005:
                        return False, "required_amount_wrong_currency"
                for start, end in reversed(negated_foreign_spans):
                    region = region[:start] + " " * (end - start) + region[end:]
            for match in re.finditer(rf"\b(?:not|never)\s+(?:\$|USD\s*)?({amount_pattern})", region, re.I):
                if abs(float(match[1].replace(",", "")) - target) < 0.005:
                    return False, "correct_amount_explicitly_negated"
            money_regions.append(region)
        for region in money_regions:
            local_spec = copy.deepcopy(spec)
            groups = local_spec.get("context_any_form_groups") or []

            entity = str(atom.get("source_entity", "")).lstrip("#").lower()
            if re.fullmatch(r"w\d{7}", entity) and re.search(r"#?" + re.escape(entity) + r"\b", region, re.I):
                local_spec["context_any_form_groups"] = [
                    g for g in groups if not any(str(x).lstrip("#").lower() == entity for x in g)
                ]
            hit, error = schema_match(mask_identifier_numbers(region), local_spec)
            if hit and not error:
                return True, ""
        return False, "required_amount_not_asserted_in_entity_region"
    if role == "refusal" and re.search(r"\bshipping address\b", question, re.I):
        if not re.search(r"\b(?:cancel|exchange|return|payment)\b", question, re.I):
            expanded = re.sub(r"\brequested change\b", "requested address change", text, flags=re.I)
            return schema_match(expanded, spec)
    return original_hit, original_error

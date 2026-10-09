"""Validate airline baggage and repeated cancellation without mutating state."""

from __future__ import annotations

VERSION = "guard_v1"
BUSINESS_TOOLS = frozenset({"cancel_reservation", "update_reservation_baggages"})
FREE_BAGS = {
    "regular": {"basic_economy": 0, "economy": 1, "business": 2},
    "silver": {"basic_economy": 1, "economy": 2, "business": 3},
    "gold": {"basic_economy": 2, "economy": 3, "business": 3},
}

def _reject(code, message, *causal_inputs):
    return {"code": code, "message": message, "causal_inputs": list(causal_inputs)}

def check_airline_business(state, tool_name, arguments):
    """Return a violation or None; unknown IDs defer to native validation.
    Baggage allowance uses the current cabin and passenger count. Native tools
    charge added paid bags; cabin changes do not refund purchased baggage.
    """
    if tool_name not in BUSINESS_TOOLS or not isinstance(arguments, dict):
        return None
    rid = arguments.get("reservation_id")
    if not isinstance(rid, str):
        return None
    reservation = state.get("reservations", {}).get(rid)
    if reservation is None:
        return None
    if reservation.get("status") == "cancelled":
        return _reject(
            "AIRLINE_RESERVATION_ALREADY_CANCELLED",
            "This reservation is already cancelled; no new changes or refund entries were made.",
            "reservation_id",
        )
    if tool_name == "cancel_reservation":
        return None

    total, nonfree = arguments.get("total_baggages"), arguments.get("nonfree_baggages")
    invalid = [key for key, value in (("total_baggages", total), ("nonfree_baggages", nonfree))
               if type(value) is not int or value < 0]
    if invalid:
        return _reject("AIRLINE_INVALID_BAGGAGE_COUNTS",
                       "Total and non-free baggage counts must be non-negative integers.", *invalid)
    try:
        old_total = reservation["total_baggages"]
        old_nonfree = reservation["nonfree_baggages"]
        if any(type(value) is not int or value < 0 for value in (old_total, old_nonfree)) or old_nonfree > old_total:
            raise ValueError("Invalid stored baggage counts")
        user = state["users"][reservation["user_id"]]
        passengers = reservation["passengers"]
        if not isinstance(passengers, list) or not passengers:
            raise ValueError("Missing passenger list")
        allowance = FREE_BAGS[user["membership"]][reservation["cabin"]] * len(passengers)
    except (KeyError, TypeError, ValueError, AttributeError):
        return _reject("AIRLINE_BAGGAGE_FACTS_UNAVAILABLE",
                       "Cannot establish the stored baggage counts and free allowance.", "reservation_id")
    if total < old_total:
        return _reject("AIRLINE_BAGGAGE_REMOVAL_FORBIDDEN",
                       "Checked baggage may be added but not removed.", "total_baggages")
    expected = max(0, total - allowance)
    if nonfree != expected:
        return _reject("AIRLINE_NONFREE_BAGGAGE_MISMATCH",
                       f"For a total of {total} bags and free allowance of {allowance}, "
                       f"nonfree_baggages must be {expected}; no charge or update was made.",
                       "total_baggages", "nonfree_baggages")
    return None

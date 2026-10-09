"""Check cancellation and basic-economy eligibility without mutating state.
Insurance alone does not establish a covered reason. Flight timing is checked separately.
"""

from __future__ import annotations

import importlib.util
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path

VERSION = "guard_v2"
SIMULATION_NOW = datetime(2024, 5, 15, 15)
ELIGIBILITY_TOOLS = frozenset({"cancel_reservation", "update_reservation_flights"})
_TEMPORAL = None

def _temporal():
    
    global _TEMPORAL
    if _TEMPORAL is None:
        path = Path(__file__).with_name("airline_temporal.py")
        spec = importlib.util.spec_from_file_location("_tau_eligibility_temporal", path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load temporal facts: {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _TEMPORAL = module
    return _TEMPORAL

def _reject(code, message, *causal_inputs):
    return {"code": code, "message": message, "causal_inputs": list(causal_inputs)}

def _itinerary_keys(segments):
    if not isinstance(segments, list) or not segments:
        raise ValueError("A nonempty complete itinerary is required")
    keys = []
    for segment in segments:
        if not isinstance(segment, Mapping):
            raise ValueError("Each segment must be an object")
        key = (segment.get("flight_number"), segment.get("date"))
        if any(not isinstance(value, str) or not value for value in key):
            raise ValueError("Each segment needs a flight number and date")
        keys.append(key)
    return keys

def check_airline_eligibility(state, tool_name, arguments):
    """Check eligibility against current reservation state without mutating inputs.
    Missing eligibility evidence fails closed; unknown IDs and malformed outer
    arguments defer to native and schema validation.
    """
    if tool_name not in ELIGIBILITY_TOOLS or not isinstance(arguments, Mapping):
        return None
    rid = arguments.get("reservation_id")
    if not isinstance(rid, str):
        return None
    reservations = state.get("reservations") if isinstance(state, Mapping) else None
    if not isinstance(reservations, Mapping):
        return _reject("AIRLINE_ELIGIBILITY_FACTS_UNAVAILABLE",
                       "Cannot establish the reservation registry.", "reservation_id")
    if rid not in reservations:
        return None
    reservation = reservations[rid]
    if not isinstance(reservation, Mapping):
        return _reject("AIRLINE_ELIGIBILITY_FACTS_UNAVAILABLE",
                       "Cannot establish the reservation facts.", "reservation_id")

    if tool_name == "update_reservation_flights":
        cabin = reservation.get("cabin")
        if cabin not in {"basic_economy", "economy", "business"}:
            return _reject("AIRLINE_ELIGIBILITY_FACTS_UNAVAILABLE",
                           "Cannot establish the current cabin.", "reservation_id")
        if cabin != "basic_economy":
            return None
        try:
            old_keys = _itinerary_keys(reservation.get("flights"))
            new_keys = _itinerary_keys(arguments.get("flights"))
        except (TypeError, ValueError) as exc:
            return _reject("AIRLINE_ELIGIBILITY_FACTS_UNAVAILABLE", str(exc), "flights")
        if old_keys != new_keys:
            return _reject(
                "AIRLINE_BASIC_ECONOMY_FLIGHT_CHANGE_FORBIDDEN",
                "Basic economy permits only a cabin change on the same complete, ordered flights and dates.",
                "reservation_id", "flights",
            )

        return None

    if reservation.get("status") == "cancelled":
        return _reject("AIRLINE_RESERVATION_ALREADY_CANCELLED",
                       "This reservation is already cancelled.", "reservation_id")
    try:
        segments = reservation.get("flights")
        _itinerary_keys(segments)
        facts = [_temporal().segment_facts(state, segment, SIMULATION_NOW) for segment in segments]
        created = datetime.fromisoformat(reservation["created_at"])
        if created.tzinfo is not None:
            raise ValueError("Expected fixture-local EST creation time")
        age = (SIMULATION_NOW - created).total_seconds()
    except (KeyError, TypeError, ValueError, AttributeError, OverflowError) as exc:
        return _reject("AIRLINE_ELIGIBILITY_FACTS_UNAVAILABLE",
                       f"Cannot establish cancellation eligibility: {exc}.", "reservation_id")
    if any(fact["started"] for fact in facts):
        return _reject("AIRLINE_CANCEL_TRIP_ALREADY_STARTED",
                       "Cancellation requires a wholly unflown trip; a partly flown trip requires transfer.",
                       "reservation_id")
    if age < 0:
        return _reject("AIRLINE_CANCEL_CREATED_IN_FUTURE",
                       "The reservation creation time is later than the simulated current time.",
                       "reservation_id")
    if age <= 86400 or any(fact["status"] == "cancelled" for fact in facts) or reservation.get("cabin") == "business":
        return None
    return _reject(
        "AIRLINE_CANCEL_NOT_POLICY_ELIGIBLE",
        "Cancellation requires booking within 24 hours, an airline-cancelled flight, or business cabin; "
        "insurance alone does not establish a covered reason.",
        "reservation_id",
    )

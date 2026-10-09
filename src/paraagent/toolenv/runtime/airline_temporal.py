"""Check airline service-time rules using the simulator's fixed EST clock."""

from __future__ import annotations

from datetime import datetime, timedelta

SIMULATION_NOW = datetime(2024, 5, 15, 15)
VERSION = "predeparture_v1"
UPDATE_TOOLS = frozenset({
    "update_reservation_baggages", "update_reservation_passengers",
    "update_reservation_flights",
})
UNFLOWN_STATUSES = frozenset({"available", "on time", "delayed"})

def _timestamp(value):
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is not None:
        raise ValueError("Expected fixture-local EST timestamp")
    return parsed

def _scheduled(date, value):
    """Support the fixture's HH:MM:SS+1 overnight notation."""
    clock, separator, days = value.partition("+")
    return _timestamp(f"{date}T{clock}") + timedelta(days=int(days) if separator else 0)

def segment_facts(state, segment, now=SIMULATION_NOW):
    """Resolve status and actual/estimated departure; raise on ambiguous facts."""
    flight = state["flights"][segment["flight_number"]]
    dated = flight["dates"][segment["date"]]
    status = dated["status"]
    if status not in UNFLOWN_STATUSES | {"landed", "flying", "cancelled"}:
        raise ValueError(f"Unknown flight status: {status}")
    actual = dated.get("actual_departure_time_est")
    arrival = dated.get("actual_arrival_time_est")
    departure = _timestamp(actual) if actual else None
    if actual and departure > now:
        raise ValueError("Actual departure is in the future")
    if status in {"landed", "flying"}:
        if departure is None:
            raise ValueError("Departed flight lacks actual departure")
        if status == "landed" and (not arrival or not departure <= _timestamp(arrival) <= now):
            raise ValueError("Landed flight lacks consistent actual arrival")
        if status == "flying" and arrival:
            raise ValueError("Flying flight already has actual arrival")
    elif actual or arrival:
        raise ValueError("Unflown/cancelled status conflicts with actual timestamps")
    if status in UNFLOWN_STATUSES:
        estimated = dated.get("estimated_departure_time_est")
        departure = _timestamp(estimated) if estimated else _scheduled(
            segment["date"], flight["scheduled_departure_time_est"])
        if departure <= now:
            
            raise ValueError("Unflown status has no future departure time")
    elif status == "cancelled":
        departure = _scheduled(segment["date"], flight["scheduled_departure_time_est"])
    return {
        "flight_number": segment["flight_number"], "date": segment["date"],
        "status": status, "started": status in {"landed", "flying"},
        "unflown": status in UNFLOWN_STATUSES,
        "changeable": status in UNFLOWN_STATUSES or (status == "cancelled" and departure > now),
        "departure": departure.isoformat() if departure else None,
        "actual_arrival": arrival,
    }

def _reject(code, message, *causal_inputs):
    return {"code": code, "message": message, "causal_inputs": list(causal_inputs)}

def check_airline_temporal(state, tool_name, arguments, now=SIMULATION_NOW):
    """Return a structured violation, or None; never mutate state or arguments.

    Unknown reservation IDs/malformed outer arguments remain native/schema errors.
    Missing temporal evidence on an existing target fails closed in this mode.
    """
    if tool_name not in UPDATE_TOOLS or not isinstance(arguments, dict):
        return None
    rid = arguments.get("reservation_id")
    if not isinstance(rid, str):
        return None
    reservation = state.get("reservations", {}).get(rid)
    if reservation is None:
        return None
    if reservation.get("status") == "cancelled":
        return _reject("AIRLINE_RESERVATION_CANCELLED",
                       "A cancelled reservation cannot be updated.", "reservation_id")
    try:
        segments = reservation["flights"]
        if not segments:
            raise ValueError("Empty itinerary")
        facts = [segment_facts(state, segment, now) for segment in segments]
    except (KeyError, TypeError, ValueError, AttributeError, OverflowError) as exc:
        return _reject("AIRLINE_TEMPORAL_FACTS_UNAVAILABLE",
                       f"Cannot establish reservation timing: {exc}.", "reservation_id")
    started = [index for index, fact in enumerate(facts) if fact["started"]]
    unflown = any(fact["unflown"] for fact in facts)
    if tool_name != "update_reservation_flights":
        if started:
            return _reject("AIRLINE_WHOLE_RESERVATION_ALREADY_STARTED",
                           "Baggage and passenger updates affect the whole reservation and "
                           "are allowed only before any segment departs; historical record "
                           "correction is outside this service.", "reservation_id")
        if not unflown:
            return _reject("AIRLINE_NO_UNFLOWN_ACTIVE_SEGMENT",
                           "Baggage and passenger updates require an unflown, non-cancelled "
                           "segment.", "reservation_id")
        return None

    if not any(fact["changeable"] for fact in facts):
        return _reject("AIRLINE_NO_CHANGEABLE_SEGMENT",
                       "No future unstarted segment remains to change; historical itineraries "
                       "cannot be rewritten.", "reservation_id")
    if started and arguments.get("cabin") != reservation.get("cabin"):
        return _reject("AIRLINE_DEPARTED_CABIN_IMMUTABLE",
                       "The reservation-wide cabin cannot change after a segment departs.",
                       "reservation_id", "cabin")
    requested = arguments.get("flights")
    if not isinstance(requested, list) or not requested or any(not isinstance(s, dict) for s in requested):
        return _reject("AIRLINE_INVALID_TEMPORAL_ITINERARY",
                       "Supply a nonempty complete itinerary.", "flights")
    old_keys = [(s["flight_number"], s["date"]) for s in segments]
    new_keys = [(s.get("flight_number"), s.get("date")) for s in requested]
    if any(not isinstance(part, str) for key in new_keys for part in key) or len(set(new_keys)) != len(new_keys):
        return _reject("AIRLINE_INVALID_TEMPORAL_ITINERARY",
                       "Every dated flight must be identified once.", "flights")
    frozen = [old_keys[index] for index in started]
    retained = [key for key in new_keys if key in frozen]
    if retained != frozen:
        return _reject("AIRLINE_DEPARTED_SEGMENT_IMMUTABLE",
                       "Keep every departed segment's flight number, date and relative order.",
                       "reservation_id", "flights")
    last_frozen = max((new_keys.index(key) for key in frozen), default=-1)
    for index, (segment, key) in enumerate(zip(requested, new_keys)):
        
        retained_unchanged = key in old_keys and arguments.get("cabin") == reservation.get("cabin")
        if retained_unchanged:
            if facts[old_keys.index(key)]["changeable"] and index < last_frozen:
                return _reject("AIRLINE_FUTURE_SEGMENT_BEFORE_DEPARTED",
                               "Future segments, including cancelled ones, must remain after "
                               "departed segments.", "flights")
            continue
        try:
            fact = segment_facts(state, segment, now)
        except (KeyError, TypeError, ValueError, AttributeError, OverflowError) as exc:
            return _reject("AIRLINE_TEMPORAL_FACTS_UNAVAILABLE",
                           f"Cannot establish replacement flight timing: {exc}.", "flights")
        if fact["status"] != "available" or not fact["unflown"] or index < last_frozen:
            return _reject("AIRLINE_REPLACEMENT_NOT_FUTURE_AVAILABLE",
                           "Changed or repriced segments must be future available flights "
                           "after the retained departed segments.", "flights")
    return None

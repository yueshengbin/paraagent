"""Track session-local seat deltas from residual availability.
Existing reservations form the allocation baseline; successful writes update it.
Closed flights without seat data never gain fabricated availability.
"""
from collections import Counter

from paraagent.toolenv.runtime.airline_temporal import segment_facts

VERSION = "seats_v1"
TOOLS = {"book_reservation", "update_reservation_flights", "cancel_reservation"}
CABINS = {"basic_economy", "economy", "business"}

class InventoryError(ValueError):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code

def allocations(flights, cabin, passengers):
    if cabin not in CABINS or not isinstance(passengers, list) or not passengers:
        raise InventoryError("AIRLINE_INVENTORY_FACTS_INVALID", "A valid cabin and nonempty passenger list are required")
    if not isinstance(flights, list) or not flights:
        raise InventoryError("AIRLINE_INVENTORY_FACTS_INVALID", "A nonempty itinerary is required")
    out = Counter()
    for segment in flights:
        if not isinstance(segment, dict):
            raise InventoryError("AIRLINE_INVENTORY_FACTS_INVALID", "Invalid itinerary segment")
        key = (segment.get("flight_number"), segment.get("date"), cabin)
        if any(not isinstance(v, str) or not v for v in key):
            raise InventoryError("AIRLINE_INVENTORY_FACTS_INVALID", "Invalid flight/date/cabin key")
        if key in out:
            raise InventoryError("AIRLINE_INVENTORY_DUPLICATE_SEGMENT", "The same flight/date cannot occur twice in an itinerary")
        out[key] = len(passengers)
    return out

def plan(state, tool, args):
    """Pure preflight. Return (key, old_remaining, new_remaining) entries."""
    if tool not in TOOLS:
        return []
    old, new = Counter(), Counter()
    if tool != "book_reservation":
        reservation = state["reservations"].get(args.get("reservation_id"))
        if reservation is None:
            return []  
        if reservation.get("status") == "cancelled":
            raise InventoryError("AIRLINE_INVENTORY_RESERVATION_CANCELLED", "Cancelled reservations hold no seats")
        old = allocations(reservation["flights"], reservation["cabin"], reservation["passengers"])
        if tool == "update_reservation_flights":
            new = allocations(args["flights"], args["cabin"], reservation["passengers"])
    else:
        new = allocations(args["flights"], args["cabin"], args["passengers"])
        for segment in args["flights"]:
            if segment_facts(state, segment)["started"]:
                raise InventoryError("AIRLINE_INVENTORY_FLIGHT_STARTED", "Cannot book an already departed flight")
    changes = []
    for key in sorted(old.keys() | new.keys()):
        number, date, cabin = key
        dated = state["flights"][number]["dates"][date]
        delta = old[key] - new[key]
        if not delta:
            continue
        if new[key] > old[key] and dated.get("status") != "available":
            raise InventoryError("AIRLINE_INVENTORY_FLIGHT_CLOSED", "Cannot allocate additional seats on a closed flight")
        seats = dated.get("available_seats")
        if seats is None and not new[key] and dated.get("status") in {"cancelled", "on time", "delayed", "flying", "landed"}:

            continue
        if not isinstance(seats, dict) or type(seats.get(cabin)) is not int or seats[cabin] < 0:
            raise InventoryError("AIRLINE_INVENTORY_FACTS_INVALID", "Remaining seats must be a nonnegative integer")
        remaining = seats[cabin] + delta
        if remaining < 0:
            raise InventoryError("AIRLINE_INVENTORY_INSUFFICIENT", f"Not enough seats on flight {number} on {date} in {cabin}")
        changes.append((key, seats[cabin], remaining))
    return changes

def commit(state, changes):
    """Apply a validated plan only to the private successful transaction state."""
    for (number, date, cabin), before, after in changes:
        seats = state["flights"][number]["dates"][date]["available_seats"]
        if seats[cabin] != before:
            raise InventoryError("AIRLINE_INVENTORY_UNEXPECTED_MUTATION", "Native tool unexpectedly changed seat inventory")
        seats[cabin] = after

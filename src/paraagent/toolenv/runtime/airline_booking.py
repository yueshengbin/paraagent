"""Check booking ID allocation and repeated payment IDs without mutating inputs.
Payment amounts, balances, and limits are validated separately.
"""

from __future__ import annotations

from collections.abc import Mapping

VERSION = "airline_atomic_v1"
RESERVATION_IDS = ("HATHAT", "HATHAU", "HATHAV")

def _reject(code, message, *causal_inputs):
    return {"code": code, "message": message, "causal_inputs": list(causal_inputs)}

def check_airline_booking(state, arguments):
    """Check allocator and payment-ID hazards without mutating inputs.
    Unknown users defer to native validation.
    """
    if not isinstance(arguments, Mapping):
        return _reject("AIRLINE_INVALID_BOOKING_ARGUMENTS",
                       "Booking arguments must be an object.", "arguments")
    user_id = arguments.get("user_id")
    if not isinstance(user_id, str):
        return _reject("AIRLINE_INVALID_BOOKING_ARGUMENTS",
                       "The booking user_id must be a string.", "user_id")
    if not isinstance(state, Mapping) or not isinstance(state.get("users"), Mapping):
        return _reject("AIRLINE_BOOKING_FACTS_UNAVAILABLE",
                       "Cannot establish the booking user registry.", "user_id")
    if user_id not in state["users"]:
        return None
    reservations = state.get("reservations")
    if not isinstance(reservations, Mapping):
        return _reject("AIRLINE_BOOKING_FACTS_UNAVAILABLE",
                       "Cannot establish the existing reservation IDs.", "user_id")
    if all(reservation_id in reservations for reservation_id in RESERVATION_IDS):
        return _reject(
            "AIRLINE_BOOKING_ID_SPACE_EXHAUSTED",
            "All three native booking reservation IDs are occupied; no existing reservation was overwritten.",
            "user_id",
        )

    payments = arguments.get("payment_methods")
    if not isinstance(payments, list):
        return _reject("AIRLINE_INVALID_BOOKING_PAYMENT_METHODS",
                       "The booking payment_methods must be an array of payment objects.", "payment_methods")
    seen = set()
    for payment in payments:
        if not isinstance(payment, Mapping) or not isinstance(payment.get("payment_id"), str):
            return _reject("AIRLINE_INVALID_BOOKING_PAYMENT_METHODS",
                           "Each booking payment must identify its payment_id as a string.", "payment_methods")
        payment_id = payment["payment_id"]
        if payment_id in seen:
            return _reject(
                "AIRLINE_DUPLICATE_BOOKING_PAYMENT_ID",
                "Each payment_id may appear only once in booking payment_methods; no payments were combined or charged.",
                "payment_methods",
            )
        seen.add(payment_id)
    return None

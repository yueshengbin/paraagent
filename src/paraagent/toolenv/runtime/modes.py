"""Execution modes required by the released Simia environment."""

import os

NATIVE_MODES = {
    "write_schema_mode": "strict_v1",
    "temporal_mode": "predeparture_v1",
    "business_mode": "guard_v2",
    "transaction_mode": "all_atomic_v1",
    "inventory_mode": "seats_v1",
    "isolation_mode": "rollout_v1",
    "feedback_mode": "guard_v1",
    "connection_mode": "connections_v1",
}


def require_current_native_modes(**overrides):
    for name, expected in NATIVE_MODES.items():
        key = "TAU_NATIVE_" + name.upper()
        requested = overrides.get(name)
        if requested is None:
            requested = os.environ.get(key, expected)
        if requested != expected:
            raise ValueError(f"{key} must be {expected!r} for this release, got {requested!r}")
    return dict(NATIVE_MODES)

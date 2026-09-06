from __future__ import annotations
from datetime import datetime
import copy

DEFAULT_STATE = {
    "destination": None,
    "arrival_datetime": None,
    "preference": "Balanced",
    "max_walk_minutes": None,
    "lot_id": None,
    "last_recommended_lot_id": None,
}

def new_state(default_destination: str, default_when: datetime):
    state = copy.deepcopy(DEFAULT_STATE)
    state["destination"] = default_destination
    state["arrival_datetime"] = default_when.isoformat(timespec="minutes")
    return state

def merge_state(current: dict, updates: dict | None):
    out = dict(current or {})
    if updates:
        for key, value in updates.items():
            if value is not None:
                out[key] = value
    return out

def readable_state(state: dict) -> str:
    parts = []

    if state.get("destination"):
        parts.append(state["destination"])

    if state.get("arrival_datetime"):
        try:
            dt = datetime.fromisoformat(state["arrival_datetime"])
            parts.append(dt.strftime("%I:%M %p").lstrip("0"))
        except Exception:
            parts.append(state["arrival_datetime"].replace("T", " "))

    preference_labels = {
        "Balanced": "Best overall",
        "Closest": "Shortest walk",
        "Highest availability": "Most availability",
        "Cheapest": "Lowest cost",
    }
    if state.get("preference"):
        parts.append(preference_labels.get(state["preference"], state["preference"]))

    if state.get("max_walk_minutes") is not None:
        parts.append(f"Max {float(state['max_walk_minutes']):g} min walk")

    return " · ".join(parts)

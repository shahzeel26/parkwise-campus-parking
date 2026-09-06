import pandas as pd
from src.intent_router import parse_message

BUILDINGS = pd.DataFrame({"building": [
    "Engineering Building",
    "Business School",
    "Main Library",
    "Sports Centre",
    "Medical School",
    "Student Central",
]})

LOTS = pd.DataFrame({"lot_id": ["P1","P2","P3","P4","P5","P6"]})

def apply(state, route):
    out = dict(state)
    for k, v in route["updates"].items():
        if v is not None:
            out[k] = v
    return out

def test_multiturn_context_updates():
    state = {
        "destination": "Business School",
        "arrival_datetime": "2026-09-07T09:30",
        "preference": "Balanced",
        "max_walk_minutes": None,
        "last_recommended_lot_id": None,
    }

    r1 = parse_message("I want to go to the library", state, BUILDINGS, LOTS)
    state = apply(state, r1)
    assert state["destination"] == "Main Library"

    r2 = parse_message("I don't want to walk more than 2 minutes", state, BUILDINGS, LOTS)
    state = apply(state, r2)
    assert state["destination"] == "Main Library"
    assert state["max_walk_minutes"] == 2

    r3 = parse_message("What if I arrive at 11 AM?", state, BUILDINGS, LOTS)
    state = apply(state, r3)
    assert state["destination"] == "Main Library"
    assert state["max_walk_minutes"] == 2
    assert "11:00" in state["arrival_datetime"]

    r4 = parse_message("Which one is cheapest?", state, BUILDINGS, LOTS)
    state = apply(state, r4)
    assert state["preference"] == "Cheapest"
    assert state["max_walk_minutes"] == 2

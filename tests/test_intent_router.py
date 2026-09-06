from datetime import datetime
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

STATE = {
    "destination": "Main Library",
    "arrival_datetime": "2026-09-07T09:30",
    "preference": "Balanced",
    "max_walk_minutes": None,
    "last_recommended_lot_id": "P3",
}

def route(q):
    return parse_message(q, STATE, BUILDINGS, LOTS)

def test_general_available():
    assert route("Hi, I am looking for the available parking areas")["intent"] == "general_availability"

def test_general_spaces():
    assert route("Where has spaces available?")["intent"] == "general_availability"

def test_library():
    r = route("I want to go to the library")
    assert r["intent"] == "recommend_parking"
    assert r["updates"]["destination"] == "Main Library"

def test_business():
    assert route("What about business school?")["updates"]["destination"] == "Business School"

def test_engineering():
    assert route("I am going to engineering")["updates"]["destination"] == "Engineering Building"

def test_walk_limit():
    r = route("I don't want to walk more than 2 minutes")
    assert r["intent"] == "recommend_parking"
    assert r["updates"]["max_walk_minutes"] == 2

def test_walk_under():
    assert route("keep it under 5 minutes walking")["updates"]["max_walk_minutes"] == 5

def test_arrival_time():
    r = route("What if I arrive at 11 AM?")
    assert r["intent"] == "recommend_parking"
    assert "11:00" in r["updates"]["arrival_datetime"]

def test_arrival_1130():
    assert "11:30" in route("What about 11:30 am?")["updates"]["arrival_datetime"]

def test_cheapest():
    assert route("Which one is cheapest?")["updates"]["preference"] == "Cheapest"

def test_closest():
    assert route("Anything closer?")["updates"]["preference"] == "Closest"

def test_easiest():
    assert route("Which is easiest?")["updates"]["preference"] == "Closest"

def test_most_spaces():
    assert route("Which has the most spaces?")["updates"]["preference"] == "Highest availability"

def test_specific_lot():
    assert route("Will P3 be full at 11 AM?")["intent"] == "predict_lot"

def test_specific_lot_available():
    assert route("How many spaces are available in P1?")["intent"] == "predict_lot"

def test_why():
    assert route("Why did you recommend that?")["intent"] == "explain_last_recommendation"

def test_peak():
    assert route("Which car park is usually busiest?")["intent"] == "historical_peak"

def test_quietest():
    assert route("What is the quietest parking time?")["intent"] == "historical_peak"

def test_recommend():
    assert route("Where should I park?")["intent"] == "general_availability"

def test_unknown():
    assert route("hello there")["intent"] == "unknown"

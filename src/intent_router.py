from __future__ import annotations

import re
from datetime import datetime, timedelta
import pandas as pd

from .tool_assistant import execute_tool, ask_parkai, llm_configured


DESTINATION_ALIASES = {
    "library": "Main Library",
    "main library": "Main Library",
    "business": "Business School",
    "business school": "Business School",
    "engineering": "Engineering Building",
    "engineering building": "Engineering Building",
    "medical": "Medical School",
    "medical school": "Medical School",
    "sports": "Sports Centre",
    "sports centre": "Sports Centre",
    "student central": "Student Central",
    "student": "Student Central",
}


def _extract_destination(text: str, valid_buildings: list[str]) -> str | None:
    low = text.lower()

    for b in valid_buildings:
        if b.lower() in low:
            return b

    for alias, canonical in DESTINATION_ALIASES.items():
        if alias in low and canonical in valid_buildings:
            return canonical

    return None


def _extract_lot_id(text: str, valid_lots: list[str]) -> str | None:
    low = text.lower()
    for lot in valid_lots:
        if re.search(rf"\b{re.escape(lot.lower())}\b", low):
            return lot
    m = re.search(r"\bp\s?(\d+)\b", low)
    if m:
        lot = f"P{m.group(1)}"
        if lot in valid_lots:
            return lot
    return None


def _extract_max_walk(text: str) -> float | None:
    low = text.lower()

    patterns = [
        r"(?:walk|walking).{0,25}?(?:more than|over|above|max(?:imum)?|within|less than|under)?\s*(\d+(?:\.\d+)?)\s*(?:min|mins|minute|minutes)",
        r"(?:more than|over|above|max(?:imum)?|within|less than|under)\s*(\d+(?:\.\d+)?)\s*(?:min|mins|minute|minutes).{0,20}?(?:walk|walking)?",
        r"(\d+(?:\.\d+)?)\s*(?:min|mins|minute|minutes)\s*(?:walk|walking)"
    ]

    for pattern in patterns:
        m = re.search(pattern, low)
        if m:
            try:
                return float(m.group(1))
            except Exception:
                pass
    return None


def _extract_time(text: str, current_when: datetime) -> datetime | None:
    low = text.lower()
    when = current_when

    if "tomorrow" in low:
        when = when + timedelta(days=1)
    elif "today" in low:
        when = datetime.combine(datetime.now().date(), when.time())

    patterns = [
        r"(?:arrive|arrival|at|around|by)\s*(\d{1,2})(?::(\d{2}))?\s*(am|pm)\b",
        r"\b(\d{1,2})(?::(\d{2}))\s*(am|pm)?\b",
    ]

    found = False
    for pattern in patterns:
        m = re.search(pattern, low)
        if not m:
            continue

        hour = int(m.group(1))
        minute = int(m.group(2) or 0)
        ampm = m.group(3) if len(m.groups()) >= 3 else None

        if ampm:
            if ampm == "pm" and hour < 12:
                hour += 12
            elif ampm == "am" and hour == 12:
                hour = 0

        if 0 <= hour <= 23 and 0 <= minute <= 59:
            when = when.replace(hour=hour, minute=minute, second=0, microsecond=0)
            found = True
            break

    if "tomorrow" in low or "today" in low:
        found = True

    return when if found else None


def _extract_preference(text: str) -> str | None:
    low = text.lower()

    if any(k in low for k in ["cheapest", "cheap", "lowest price", "least expensive"]):
        return "Cheapest"

    if any(k in low for k in ["closest", "easiest", "shortest walk", "nearby", "closer"]):
        return "Closest"

    if any(k in low for k in [
        "most available", "most spaces", "highest availability",
        "best chance", "safest option", "more spaces"
    ]):
        return "Highest availability"

    return None


def parse_message(message: str, state: dict, buildings, lots) -> dict:
    """
    Pure routing decision. No LLM or ML calls.
    """
    low = message.lower().strip()

    current_when = pd.Timestamp(
        state.get("arrival_datetime") or datetime.now()
    ).to_pydatetime()

    destination = _extract_destination(message, buildings["building"].tolist())
    lot_id = _extract_lot_id(message, lots["lot_id"].tolist())
    max_walk = _extract_max_walk(message)
    new_when = _extract_time(message, current_when)
    preference = _extract_preference(message)

    # General availability must be checked before default destination logic.
    general_availability_phrases = [
        "available parking",
        "parking available",
        "available parking areas",
        "where has spaces",
        "where are spaces",
        "where can i park",
        "where should i park",
        "what parking is available",
        "parking areas available",
        "spaces available",
    ]

    if any(p in low for p in general_availability_phrases) and destination is None and lot_id is None:
        return {
            "intent": "general_availability",
            "updates": {
                "arrival_datetime": new_when.isoformat(timespec="minutes") if new_when else None
            },
            "args": {
                "arrival_datetime": new_when.isoformat(timespec="minutes") if new_when else None
            }
        }

    if any(p in low for p in ["why did you recommend", "why this", "why that", "why did you choose", "why recommend"]):
        return {"intent": "explain_last_recommendation", "updates": {}, "args": {}}

    if any(p in low for p in ["busiest", "peak", "usually busiest", "quietest", "historical"]):
        return {"intent": "historical_peak", "updates": {}, "args": {}}

    # Specific lot forecast.
    if lot_id and any(k in low for k in ["full", "available", "spaces", "occupancy", "busy"]):
        return {
            "intent": "predict_lot",
            "updates": {
                "lot_id": lot_id,
                "arrival_datetime": new_when.isoformat(timespec="minutes") if new_when else None,
            },
            "args": {
                "lot_id": lot_id,
                "arrival_datetime": new_when.isoformat(timespec="minutes") if new_when else None,
            }
        }

    updates = {}
    if destination:
        updates["destination"] = destination
    if max_walk is not None:
        updates["max_walk_minutes"] = max_walk
    if new_when is not None:
        updates["arrival_datetime"] = new_when.isoformat(timespec="minutes")
    if preference:
        updates["preference"] = preference

    # Any explicit destination/constraint/time/preference change should re-run recommendation.
    if updates:
        return {
            "intent": "recommend_parking",
            "updates": updates,
            "args": {
                "destination": destination,
                "arrival_datetime": new_when.isoformat(timespec="minutes") if new_when else None,
                "preference": preference,
                "max_walk_minutes": max_walk,
            }
        }

    # Natural recommendation phrases without explicit changes.
    if any(k in low for k in ["recommend", "best option", "where should i park", "where should i go", "best parking"]):
        return {
            "intent": "recommend_parking",
            "updates": {},
            "args": {}
        }

    return {"intent": "unknown", "updates": {}, "args": {}}


def _apply_updates(state: dict, updates: dict) -> dict:
    out = dict(state)
    for k, v in updates.items():
        if v is not None:
            out[k] = v
    return out


def _general_availability_answer(result: dict) -> str:
    rows = result.get("results", [])
    if not rows:
        return "I couldn't find any parking availability results."

    lines = []
    for r in rows[:3]:
        lines.append(
            f"**{r['name']} ({r['lot_id']})** — about **{r['predicted_available']} spaces** "
            f"available ({r['predicted_occupancy_pct']}% occupied)"
        )

    return (
        "Here are the parking areas with the most predicted availability:\n\n"
        + "\n\n".join(lines)
        + "\n\nTell me where on campus you're going and I can recommend the best option based on walking distance as well."
    )


def _recommendation_answer(result: dict) -> str:
    if result.get("status") == "no_match":
        n = result["nearest_option"]
        limit = result.get("max_walk_minutes")
        return (
            f"I couldn't find a parking area within your **{limit:g}-minute walking limit**. "
            f"The closest option is **{n['name']} ({n['lot_id']})**, about **{n['walk_minutes']:.1f} minutes** away, "
            f"with roughly **{n['predicted_available']} spaces available**."
        )

    rows = result.get("results", [])
    if not rows:
        return "I couldn't find a suitable parking option."

    top = rows[0]
    answer = (
        f"I recommend **{top['name']} ({top['lot_id']})** for **{result['destination']}**. "
        f"At **{pd.Timestamp(result['arrival_datetime']).strftime('%I:%M %p')}**, "
        f"it is predicted to have about **{top['predicted_available']} spaces available** "
        f"(likely range **{top['available_range']}**), with approximately "
        f"**{top['walk_minutes']:.1f} minutes of walking** and "
        f"**{top['predicted_occupancy_pct']}% occupancy**."
    )

    if result.get("max_walk_minutes") is not None:
        answer += (
            f"\n\nThis satisfies your **{float(result['max_walk_minutes']):g}-minute maximum walking limit**."
        )

    if len(rows) > 1:
        second = rows[1]
        answer += (
            f"\n\n**Backup:** {second['name']} ({second['lot_id']}) with about "
            f"**{second['predicted_available']} spaces available**."
        )

    return answer


def _prediction_answer(result: dict) -> str:
    return (
        f"**{result['name']} ({result['lot_id']})** is predicted to have about "
        f"**{result['predicted_available']} spaces available** at "
        f"**{pd.Timestamp(result['arrival_datetime']).strftime('%I:%M %p')}**, "
        f"with **{result['predicted_occupancy_pct']}% occupancy** "
        f"(likely availability range **{result['available_range']}**)."
    )


def _historical_answer(result: dict) -> str:
    p = result["peak"]
    q = result["quiet"]
    return (
        f"Historically, the strongest demand is around **{p['hour']:02d}:00** at **{p['lot_id']}**, "
        f"with about **{p['average_occupancy_pct']}% average occupancy**. "
        f"The quietest pattern is around **{q['hour']:02d}:00** at **{q['lot_id']}**, "
        f"at roughly **{q['average_occupancy_pct']}% occupancy**."
    )


def _explain_answer(result: dict) -> str:
    if result.get("status") != "ok":
        return "I don't have a previous recommendation to explain yet."

    answer = (
        f"I recommended **{result['name']} ({result['lot_id']})** because, under your "
        f"**{result['preference']}** preference, it offered a strong trade-off between "
        f"**{result['predicted_available']} predicted spaces**, "
        f"**{result['walk_minutes']:.1f} minutes of walking**, and "
        f"**${result['hourly_rate']:.2f}/h** parking cost."
    )

    if result.get("active_max_walk_minutes") is not None:
        answer += (
            f" It also respects your **{float(result['active_max_walk_minutes']):g}-minute walking limit**."
        )

    return answer


def route_and_answer(
    message: str,
    state: dict,
    model,
    metrics,
    history,
    lots,
    buildings,
    default_destination: str,
    default_when: datetime,
):
    """
    Deterministic-first pipeline.
    LLM is only used when the deterministic parser genuinely cannot classify the request.
    """
    route = parse_message(message, state, buildings, lots)
    new_state = _apply_updates(state, route["updates"])

    if route["intent"] != "unknown":
        result, dataframe, final_state = execute_tool(
            route["intent"],
            route["args"],
            model,
            metrics,
            history,
            lots,
            buildings,
            default_destination,
            default_when,
            new_state,
        )

        if route["intent"] == "general_availability":
            answer = _general_availability_answer(result)
            payload_type = "availability"
        elif route["intent"] == "recommend_parking":
            answer = _recommendation_answer(result)
            payload_type = "recommendation"
        elif route["intent"] == "predict_lot":
            answer = _prediction_answer(result)
            payload_type = "prediction"
        elif route["intent"] == "historical_peak":
            answer = _historical_answer(result)
            payload_type = "historical"
        elif route["intent"] == "explain_last_recommendation":
            answer = _explain_answer(result)
            payload_type = "explanation"
        else:
            answer = "I can help with campus parking."
            payload_type = "help"

        return answer, {
            "mode": "deterministic_router",
            "tool": route["intent"],
            "type": payload_type,
            "dataframe": dataframe,
            "data": dataframe,
            "conversation_state": final_state,
            "tool_result": result,
        }

    # Only use LLM when deterministic parsing really has no route.
    if llm_configured():
        return ask_parkai(
            message,
            model,
            metrics,
            history,
            lots,
            buildings,
            default_destination,
            default_when,
            state,
        )

    return (
        "I can help with parking availability, destinations, walking limits, arrival times, prices, and specific car parks.",
        {
            "mode": "fallback_help",
            "tool": None,
            "type": "help",
            "dataframe": None,
            "data": None,
            "conversation_state": state,
        }
    )

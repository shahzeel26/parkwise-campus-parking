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


# ---------------------------------------------------------------------
# Entity extraction
# ---------------------------------------------------------------------

def _extract_destination(text: str, valid_buildings: list[str]) -> str | None:
    low = text.lower()

    for building in valid_buildings:
        if building.lower() in low:
            return building

    for alias, canonical in DESTINATION_ALIASES.items():
        if alias in low and canonical in valid_buildings:
            return canonical

    return None


def _extract_lot_id(text: str, valid_lots: list[str]) -> str | None:
    low = text.lower()

    for lot in valid_lots:
        if re.search(rf"\b{re.escape(lot.lower())}\b", low):
            return lot

    match = re.search(r"\bp\s?(\d+)\b", low)

    if match:
        lot = f"P{match.group(1)}"

        if lot in valid_lots:
            return lot

    return None


def _extract_max_walk(text: str) -> float | None:
    low = text.lower()

    patterns = [
        (
            r"(?:walk|walking).{0,25}?"
            r"(?:more than|over|above|max(?:imum)?|within|less than|under)?\s*"
            r"(\d+(?:\.\d+)?)\s*(?:min|mins|minute|minutes)"
        ),
        (
            r"(?:more than|over|above|max(?:imum)?|within|less than|under)\s*"
            r"(\d+(?:\.\d+)?)\s*(?:min|mins|minute|minutes)"
            r".{0,20}?(?:walk|walking)?"
        ),
        r"(\d+(?:\.\d+)?)\s*(?:min|mins|minute|minutes)\s*(?:walk|walking)",
    ]

    for pattern in patterns:
        match = re.search(pattern, low)

        if match:
            try:
                return float(match.group(1))
            except Exception:
                pass

    return None


def _extract_time(text: str, current_when: datetime) -> datetime | None:
    low = text.lower()
    when = current_when

    if "tomorrow" in low:
        when = when + timedelta(days=1)

    elif "today" in low:
        when = datetime.combine(
            datetime.now().date(),
            when.time(),
        )

    patterns = [
        r"(?:arrive|arrival|at|around|by)\s*(\d{1,2})(?::(\d{2}))?\s*(am|pm)\b",
        r"\b(\d{1,2})(?::(\d{2}))\s*(am|pm)?\b",
    ]

    found = False

    for pattern in patterns:
        match = re.search(pattern, low)

        if not match:
            continue

        hour = int(match.group(1))
        minute = int(match.group(2) or 0)
        ampm = match.group(3) if len(match.groups()) >= 3 else None

        if ampm:
            if ampm == "pm" and hour < 12:
                hour += 12

            elif ampm == "am" and hour == 12:
                hour = 0

        if 0 <= hour <= 23 and 0 <= minute <= 59:
            when = when.replace(
                hour=hour,
                minute=minute,
                second=0,
                microsecond=0,
            )

            found = True
            break

    if "tomorrow" in low or "today" in low:
        found = True

    return when if found else None


def _extract_preference(text: str) -> str | None:
    low = text.lower()

    if any(
        phrase in low
        for phrase in [
            "cheapest",
            "cheap",
            "lowest price",
            "least expensive",
            "lowest cost",
        ]
    ):
        return "Cheapest"

    if any(
        phrase in low
        for phrase in [
            "closest",
            "easiest",
            "shortest walk",
            "nearby",
            "closer",
            "nearest",
        ]
    ):
        return "Closest"

    if any(
        phrase in low
        for phrase in [
            "most available",
            "most spaces",
            "highest availability",
            "best chance",
            "safest option",
            "more spaces",
        ]
    ):
        return "Highest availability"

    return None


# ---------------------------------------------------------------------
# Comparison intent detection
# ---------------------------------------------------------------------

def _extract_comparison(text: str) -> str | None:
    """
    Detect questions asking to compare current parking options.

    Important distinction:

    "Which option is cheapest?"
        -> comparison

    "I want the cheapest option."
        -> recommendation preference update
    """

    low = text.lower().strip()

    question_style = (
        low.startswith("which")
        or low.startswith("what")
        or low.startswith("show")
        or low.startswith("compare")
        or "which one" in low
        or "which option" in low
        or "what option" in low
        or "what is the" in low
    )

    if not question_style:
        return None

    if any(
        phrase in low
        for phrase in [
            "cheapest",
            "least expensive",
            "lowest cost",
            "lowest price",
            "cheaper",
        ]
    ):
        return "Cheapest"

    if any(
        phrase in low
        for phrase in [
            "closest",
            "nearest",
            "shortest walk",
            "least walking",
            "easiest to walk",
        ]
    ):
        return "Closest"

    if any(
        phrase in low
        for phrase in [
            "most spaces",
            "most available",
            "highest availability",
            "best availability",
            "most parking",
        ]
    ):
        return "Highest availability"

    return None


# ---------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------

def parse_message(
    message: str,
    state: dict,
    buildings,
    lots,
) -> dict:
    """
    Deterministic routing decision.

    No ML or LLM calls happen here.
    """

    low = message.lower().strip()

    current_when = pd.Timestamp(
        state.get("arrival_datetime") or datetime.now()
    ).to_pydatetime()

    destination = _extract_destination(
        message,
        buildings["building"].tolist(),
    )

    lot_id = _extract_lot_id(
        message,
        lots["lot_id"].tolist(),
    )

    max_walk = _extract_max_walk(message)

    new_when = _extract_time(
        message,
        current_when,
    )

    preference = _extract_preference(message)

    comparison = _extract_comparison(message)

    # -------------------------------------------------------------
    # Follow-up to a specific car-park question
    # Example:
    # "Will P1 be full at 9:30?"
    # "What if I arrive at 11 AM?"
    # -------------------------------------------------------------

    if (
        new_when is not None
        and destination is None
        and lot_id is None
        and max_walk is None
        and preference is None
        and comparison is None
        and state.get("lot_id")
    ):
        return {
            "intent": "predict_lot",
            "updates": {
                "arrival_datetime": new_when.isoformat(timespec="minutes"),
            },
            "args": {
                "lot_id": state.get("lot_id"),
                "arrival_datetime": new_when.isoformat(timespec="minutes"),
            },
        }

    # --------------------------------------
    # -----------------------
    # Explain previous recommendation
    # -------------------------------------------------------------

    if any(
        phrase in low
        for phrase in [
            "why did you recommend",
            "why this",
            "why that",
            "why did you choose",
            "why recommend",
            "why did you pick",
        ]
    ):
        return {
            "intent": "explain_last_recommendation",
            "updates": {},
            "args": {},
        }

    # -------------------------------------------------------------
    # Comparison request
    # Must run before preference updates.
    # -------------------------------------------------------------

    if comparison is not None:
        updates = {}

        if destination:
            updates["destination"] = destination

        if new_when is not None:
            updates["arrival_datetime"] = new_when.isoformat(
                timespec="minutes"
            )

        if max_walk is not None:
            updates["max_walk_minutes"] = max_walk

        return {
            "intent": "compare_options",
            "updates": updates,
            "args": {
                "comparison": comparison,
                "destination": destination,
                "arrival_datetime": (
                    new_when.isoformat(timespec="minutes")
                    if new_when
                    else None
                ),
                "max_walk_minutes": max_walk,
            },
        }

    # -------------------------------------------------------------
    # Historical patterns
    # -------------------------------------------------------------

    if any(
        phrase in low
        for phrase in [
            "busiest",
            "peak",
            "usually busiest",
            "quietest",
            "historical",
        ]
    ):
        return {
            "intent": "historical_peak",
            "updates": {},
            "args": {},
        }

    # -------------------------------------------------------------
    # Specific parking-lot prediction
    # -------------------------------------------------------------

    if lot_id and any(
        word in low
        for word in [
            "full",
            "available",
            "spaces",
            "occupancy",
            "busy",
        ]
    ):
        return {
            "intent": "predict_lot",
            "updates": {
                "lot_id": lot_id,
                "arrival_datetime": (
                    new_when.isoformat(timespec="minutes")
                    if new_when
                    else None
                ),
            },
            "args": {
                "lot_id": lot_id,
                "arrival_datetime": (
                    new_when.isoformat(timespec="minutes")
                    if new_when
                    else None
                ),
            },
        }

    # -------------------------------------------------------------
    # General parking availability
    # -------------------------------------------------------------

    general_availability_phrases = [
        "available parking",
        "parking available",
        "available parking areas",
        "where has spaces",
        "where are spaces",
        "where can i park",
        "what parking is available",
        "parking areas available",
        "spaces available",
    ]

    if (
        any(phrase in low for phrase in general_availability_phrases)
        and destination is None
        and lot_id is None
    ):
        return {
            "intent": "general_availability",
            "updates": {
                "arrival_datetime": (
                    new_when.isoformat(timespec="minutes")
                    if new_when
                    else None
                )
            },
            "args": {
                "arrival_datetime": (
                    new_when.isoformat(timespec="minutes")
                    if new_when
                    else None
                )
            },
        }

    # -------------------------------------------------------------
    # Conversation updates
    # -------------------------------------------------------------

    updates = {}

    if destination:
        updates["destination"] = destination

    if max_walk is not None:
        updates["max_walk_minutes"] = max_walk

    if new_when is not None:
        updates["arrival_datetime"] = new_when.isoformat(
            timespec="minutes"
        )

    if preference:
        updates["preference"] = preference

    # A direct state change means:
    # update context and re-run recommendation.
    if updates:
        return {
            "intent": "recommend_parking",
            "updates": updates,
            "args": {
                "destination": destination,
                "arrival_datetime": (
                    new_when.isoformat(timespec="minutes")
                    if new_when
                    else None
                ),
                "preference": preference,
                "max_walk_minutes": max_walk,
            },
        }

    # -------------------------------------------------------------
    # General recommendation phrases
    # -------------------------------------------------------------

    if any(
        phrase in low
        for phrase in [
            "recommend",
            "best option",
            "where should i park",
            "where should i go",
            "best parking",
        ]
    ):
        return {
            "intent": "recommend_parking",
            "updates": {},
            "args": {},
        }

    return {
        "intent": "unknown",
        "updates": {},
        "args": {},
    }


# ---------------------------------------------------------------------
# State
# ---------------------------------------------------------------------

def _apply_updates(state: dict, updates: dict) -> dict:
    out = dict(state)

    for key, value in updates.items():
        if value is not None:
            out[key] = value

    return out


# ---------------------------------------------------------------------
# Response builders
# ---------------------------------------------------------------------

def _general_availability_answer(result: dict) -> str:
    rows = result.get("results", [])

    if not rows:
        return "I couldn't find any parking availability results."

    lines = []

    for row in rows[:3]:
        lines.append(
            f"**{row['name']} ({row['lot_id']})** — about "
            f"**{row['predicted_available']} spaces** available "
            f"({row['predicted_occupancy_pct']}% occupied)"
        )

    return (
        "Here are the parking areas with the most predicted availability:\n\n"
        + "\n\n".join(lines)
        + "\n\nTell me where on campus you're going and I can "
          "recommend the best option based on walking distance as well."
    )


def _recommendation_answer(result: dict) -> str:
    if result.get("status") == "no_match":
        nearest = result["nearest_option"]
        limit = result.get("max_walk_minutes")

        return (
            f"I couldn't find a parking area within your "
            f"**{limit:g}-minute walking limit**. "
            f"The closest option is **{nearest['name']} "
            f"({nearest['lot_id']})**, about "
            f"**{nearest['walk_minutes']:.1f} minutes** away, "
            f"with roughly **{nearest['predicted_available']} "
            f"spaces available**."
        )

    rows = result.get("results", [])

    if not rows:
        return "I couldn't find a suitable parking option."

    top = rows[0]

    answer = (
        f"I recommend **{top['name']} ({top['lot_id']})** for "
        f"**{result['destination']}**. "
        f"At **{pd.Timestamp(result['arrival_datetime']).strftime('%I:%M %p')}**, "
        f"it is predicted to have about "
        f"**{top['predicted_available']} spaces available** "
        f"(likely range **{top['available_range']}**), with approximately "
        f"**{top['walk_minutes']:.1f} minutes of walking** and "
        f"**{top['predicted_occupancy_pct']}% occupancy**."
    )

    if result.get("max_walk_minutes") is not None:
        answer += (
            f"\n\nThis satisfies your "
            f"**{float(result['max_walk_minutes']):g}-minute maximum "
            f"walking limit**."
        )

    if len(rows) > 1:
        second = rows[1]

        answer += (
            f"\n\n**Backup:** {second['name']} "
            f"({second['lot_id']}) with about "
            f"**{second['predicted_available']} spaces available**."
        )

    return answer


def _comparison_answer(result: dict) -> str:
    if result.get("status") == "no_match":
        limit = result.get("max_walk_minutes")

        return (
            f"I couldn't find a parking option within your "
            f"**{float(limit):g}-minute walking limit**."
        )

    rows = result.get("results", [])

    if not rows:
        return "I couldn't find parking options to compare."

    best = rows[0]

    comparison = result["comparison"]

    time_text = pd.Timestamp(
        result["arrival_datetime"]
    ).strftime("%I:%M %p")

    if comparison == "Cheapest":
        answer = (
            f"The cheapest option for **{result['destination']}** at "
            f"**{time_text}** is **{best['name']} "
            f"({best['lot_id']})** at **${best['hourly_rate']:.2f}/h**."
            f"\n\nIt is approximately "
            f"**{best['walk_minutes']:.1f} minutes** away and is predicted "
            f"to have about **{best['predicted_available']} spaces available**."
        )

    elif comparison == "Closest":
        answer = (
            f"The closest option to **{result['destination']}** at "
            f"**{time_text}** is **{best['name']} "
            f"({best['lot_id']})**, approximately "
            f"**{best['walk_minutes']:.1f} minutes** away."
            f"\n\nIt is predicted to have about "
            f"**{best['predicted_available']} spaces available** and costs "
            f"**${best['hourly_rate']:.2f}/h**."
        )

    else:
        answer = (
            f"The option with the most predicted availability for "
            f"**{result['destination']}** at **{time_text}** is "
            f"**{best['name']} ({best['lot_id']})**, with about "
            f"**{best['predicted_available']} spaces available**."
            f"\n\nIt is approximately "
            f"**{best['walk_minutes']:.1f} minutes** away and costs "
            f"**${best['hourly_rate']:.2f}/h**."
        )

    if result.get("max_walk_minutes") is not None:
        answer += (
            f"\n\nThis comparison respects your active "
            f"**{float(result['max_walk_minutes']):g}-minute walking limit**."
        )

    if len(rows) > 1:
        second = rows[1]

        if comparison == "Cheapest":
            answer += (
                f"\n\n**Next cheapest:** {second['name']} "
                f"({second['lot_id']}) at "
                f"**${second['hourly_rate']:.2f}/h**."
            )

        elif comparison == "Closest":
            answer += (
                f"\n\n**Next closest:** {second['name']} "
                f"({second['lot_id']}) at "
                f"**{second['walk_minutes']:.1f} minutes**."
            )

        else:
            answer += (
                f"\n\n**Next highest availability:** {second['name']} "
                f"({second['lot_id']}) with about "
                f"**{second['predicted_available']} spaces**."
            )

    return answer


def _prediction_answer(result: dict) -> str:
    return (
        f"**{result['name']} ({result['lot_id']})** is predicted to have "
        f"about **{result['predicted_available']} spaces available** at "
        f"**{pd.Timestamp(result['arrival_datetime']).strftime('%I:%M %p')}**, "
        f"with **{result['predicted_occupancy_pct']}% occupancy** "
        f"(likely availability range **{result['available_range']}**)."
    )


def _historical_answer(result: dict) -> str:
    peak = result["peak"]
    quiet = result["quiet"]

    return (
        f"Historically, the strongest demand is around "
        f"**{peak['hour']:02d}:00** at **{peak['lot_id']}**, "
        f"with about **{peak['average_occupancy_pct']}% average occupancy**. "
        f"The quietest pattern is around **{quiet['hour']:02d}:00** at "
        f"**{quiet['lot_id']}**, at roughly "
        f"**{quiet['average_occupancy_pct']}% occupancy**."
    )


def _explain_answer(result: dict) -> str:
    if result.get("status") != "ok":
        return "I don't have a previous recommendation to explain yet."

    answer = (
        f"I recommended **{result['name']} ({result['lot_id']})** because, "
        f"under your **{result['preference']}** preference, it offered a "
        f"strong trade-off between "
        f"**{result['predicted_available']} predicted spaces**, "
        f"**{result['walk_minutes']:.1f} minutes of walking**, and "
        f"**${result['hourly_rate']:.2f}/h** parking cost."
    )

    if result.get("active_max_walk_minutes") is not None:
        answer += (
            f" It also respects your "
            f"**{float(result['active_max_walk_minutes']):g}-minute "
            f"walking limit**."
        )

    return answer


# ---------------------------------------------------------------------
# Main routing pipeline
# ---------------------------------------------------------------------

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

    LLM is only used when deterministic routing cannot classify the request.
    """

    route = parse_message(
        message,
        state,
        buildings,
        lots,
    )

    new_state = _apply_updates(
        state,
        route["updates"],
    )

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

        elif route["intent"] == "compare_options":
            answer = _comparison_answer(result)
            payload_type = "comparison"

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

    # Only use the LLM when deterministic routing genuinely has no route.
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
        "I can help with parking availability, destinations, walking limits, "
        "arrival times, prices, and specific car parks.",
        {
            "mode": "fallback_help",
            "tool": None,
            "type": "help",
            "dataframe": None,
            "data": None,
            "conversation_state": state,
        },
    )
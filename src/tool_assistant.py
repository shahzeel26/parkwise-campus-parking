from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any

import pandas as pd
from dotenv import load_dotenv

from .recommender import recommend_parking
from .model import predict_occupancy

load_dotenv()

SYSTEM_PROMPT = """
You are ParkAI, the conversational assistant inside SmartPark AI.

You help users with campus parking and must use SmartPark tools for parking facts.

IMPORTANT:
- Never invent parking availability, occupancy, walking distance, prices, ranking scores, or model results.
- Use the current conversation state for omitted values.
- Only update state values the user explicitly changes or clearly implies.
- Treat explicit walking limits as HARD constraints.
- If the user asks generally for "available parking", "where has spaces", or "what parking is available",
  do NOT assume a destination. Use the general_availability tool.
- Interpret:
  * "closest", "easiest", "shortest walk" -> Closest
  * "cheapest", "cheap" -> Cheapest
  * "most available", "best chance", "most spaces" -> Highest availability
- Follow-ups like "what about the library?", "what if I arrive at 11?", "anything closer?",
  "which is cheaper?", or "why did you recommend that?" should use prior conversation context.
- Keep answers concise, practical, and natural.
"""


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "general_availability",
            "description": "Show parking areas with the most predicted available spaces when the user has not supplied a destination.",
            "parameters": {
                "type": "object",
                "properties": {
                    "arrival_datetime": {"type": ["string", "null"]}
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "recommend_parking",
            "description": "Rank parking options for a destination using current conversational constraints.",
            "parameters": {
                "type": "object",
                "properties": {
                    "destination": {"type": ["string", "null"]},
                    "arrival_datetime": {"type": ["string", "null"]},
                    "preference": {
                        "type": ["string", "null"],
                        "enum": ["Balanced", "Closest", "Highest availability", "Cheapest", None]
                    },
                    "max_walk_minutes": {"type": ["number", "null"]}
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "predict_lot",
            "description": "Predict availability for a specific car park, including follow-ups such as 'will that still have spaces?'",
            "parameters": {
                "type": "object",
                "properties": {
                    "lot_id": {"type": ["string", "null"]},
                    "arrival_datetime": {"type": ["string", "null"]}
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "historical_peak",
            "description": "Summarise historical peak and quiet parking patterns.",
            "parameters": {"type": "object", "properties": {}}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "explain_last_recommendation",
            "description": "Explain why the previously recommended parking lot was selected.",
            "parameters": {"type": "object", "properties": {}}
        }
    }
]


def llm_configured() -> bool:
    return bool(os.getenv("LLM_API_KEY"))


def _client():
    if not llm_configured():
        return None
    from openai import OpenAI
    return OpenAI(
        api_key=os.getenv("LLM_API_KEY"),
        base_url=os.getenv("LLM_BASE_URL") or "https://api.openai.com/v1",
    )


def _model_name() -> str:
    return os.getenv("LLM_MODEL") or "gpt-4o-mini"


def _parse_dt(value: str | None, fallback: datetime) -> datetime:
    if not value:
        return fallback
    try:
        return pd.Timestamp(value).to_pydatetime()
    except Exception:
        return fallback


def _normalise_destination(destination: str | None, buildings: pd.DataFrame, fallback: str) -> str:
    if not destination:
        return fallback

    low = destination.lower().strip()
    exact = buildings[buildings["building"].str.lower() == low]
    if len(exact):
        return exact.iloc[0]["building"]

    for b in buildings["building"]:
        bl = b.lower()
        if low in bl or bl in low:
            return b

    # Friendly aliases
    aliases = {
        "library": "Main Library",
        "business": "Business School",
        "engineering": "Engineering Building",
        "medical": "Medical School",
        "sports": "Sports Centre",
        "student": "Student Central",
    }
    for key, val in aliases.items():
        if key in low and val in buildings["building"].tolist():
            return val

    return fallback


def _normalise_lot(lot_id: str | None, lots: pd.DataFrame) -> str | None:
    if not lot_id:
        return None
    key = str(lot_id).upper().replace(" ", "")
    found = lots[lots["lot_id"].str.upper() == key]
    if len(found):
        return found.iloc[0]["lot_id"]
    return None


def execute_tool(
    name: str,
    args: dict[str, Any],
    model,
    metrics: dict,
    history: pd.DataFrame,
    lots: pd.DataFrame,
    buildings: pd.DataFrame,
    default_destination: str,
    default_when: datetime,
    conversation_state: dict,
):
    state = dict(conversation_state)


    if name == "general_availability":
        when = _parse_dt(
            args.get("arrival_datetime"),
            pd.Timestamp(state.get("arrival_datetime") or default_when).to_pydatetime()
        )

        rows = []
        for _, lot in lots.iterrows():
            p = predict_occupancy(
                model,
                lot["lot_id"],
                int(lot["capacity"]),
                when,
                residual_std=metrics.get("residual_std", 0.04)
            )
            rows.append({
                "lot_id": lot["lot_id"],
                "name": lot["name"],
                "predicted_available": int(p["available"]),
                "available_range": f"{int(p['available_low'])}-{int(p['available_high'])}",
                "predicted_occupancy_pct": round(float(p["occupancy_rate"]) * 100),
                "hourly_rate": round(float(lot["hourly_rate"]), 2),
            })

        df = pd.DataFrame(rows).sort_values(
            ["predicted_available", "predicted_occupancy_pct"],
            ascending=[False, True]
        ).reset_index(drop=True)

        state["arrival_datetime"] = when.isoformat(timespec="minutes")

        return {
            "tool": "general_availability",
            "arrival_datetime": when.isoformat(timespec="minutes"),
            "results": df.head(5).to_dict("records"),
        }, df, state

    if name == "explain_last_recommendation":
        lot_id = state.get("last_recommended_lot_id")
        if not lot_id:
            return {
                "tool": "explain_last_recommendation",
                "status": "no_previous_recommendation"
            }, None, state

        destination = state.get("destination") or default_destination
        when = pd.Timestamp(
            state.get("arrival_datetime") or default_when
        ).to_pydatetime()
        preference = state.get("preference") or "Balanced"

        recs = recommend_parking(
            model, metrics, lots, buildings, destination, when, preference=preference
        )
        row = recs[recs["lot_id"] == lot_id]
        if len(row) == 0:
            return {
                "tool": "explain_last_recommendation",
                "status": "not_found"
            }, None, state

        r = row.iloc[0]
        return {
            "tool": "explain_last_recommendation",
            "status": "ok",
            "lot_id": lot_id,
            "name": r["name"],
            "destination": destination,
            "preference": preference,
            "predicted_available": int(r["available"]),
            "predicted_occupancy_pct": round(float(r["occupancy_rate"]) * 100),
            "walk_minutes": round(float(r["walk_min"]), 1),
            "hourly_rate": round(float(r["hourly_rate"]), 2),
            "recommendation_score": round(float(r["score"]), 1),
            "active_max_walk_minutes": state.get("max_walk_minutes")
        }, None, state

    if name == "recommend_parking":
        destination = _normalise_destination(
            args.get("destination"),
            buildings,
            state.get("destination") or default_destination
        )
        when = _parse_dt(
            args.get("arrival_datetime"),
            pd.Timestamp(state.get("arrival_datetime") or default_when).to_pydatetime()
        )

        preference = args.get("preference") or state.get("preference") or "Balanced"
        if preference not in ["Balanced", "Closest", "Highest availability", "Cheapest"]:
            preference = "Balanced"

        max_walk = args.get("max_walk_minutes")
        if max_walk is None:
            max_walk = state.get("max_walk_minutes")

        recs = recommend_parking(
            model, metrics, lots, buildings, destination, when, preference=preference
        )

        constraint_applied = False
        no_match = False

        if max_walk is not None:
            constraint_applied = True
            filtered = recs[recs["walk_min"] <= float(max_walk)].copy()
            if len(filtered):
                recs = filtered.reset_index(drop=True)
            else:
                no_match = True

        state.update({
            "destination": destination,
            "arrival_datetime": when.isoformat(timespec="minutes"),
            "preference": preference,
            "max_walk_minutes": max_walk,
        })

        if no_match:
            nearest = recommend_parking(
                model, metrics, lots, buildings, destination, when, preference="Closest"
            ).iloc[0]
            state["last_recommended_lot_id"] = nearest["lot_id"]
            return {
                "tool": "recommend_parking",
                "status": "no_match",
                "destination": destination,
                "arrival_datetime": when.isoformat(timespec="minutes"),
                "max_walk_minutes": max_walk,
                "nearest_option": {
                    "lot_id": nearest["lot_id"],
                    "name": nearest["name"],
                    "walk_minutes": round(float(nearest["walk_min"]), 1),
                    "predicted_available": int(nearest["available"]),
                    "predicted_occupancy_pct": round(float(nearest["occupancy_rate"]) * 100),
                }
            }, None, state

        rows = []
        for _, r in recs.head(5).iterrows():
            rows.append({
                "lot_id": r["lot_id"],
                "name": r["name"],
                "predicted_available": int(r["available"]),
                "available_range": f"{int(r['available_low'])}-{int(r['available_high'])}",
                "predicted_occupancy_pct": round(float(r["occupancy_rate"]) * 100),
                "walk_minutes": round(float(r["walk_min"]), 1),
                "hourly_rate": round(float(r["hourly_rate"]), 2),
                "recommendation_score": round(float(r["score"]), 1),
            })

        state["last_recommended_lot_id"] = rows[0]["lot_id"]

        return {
            "tool": "recommend_parking",
            "status": "ok",
            "destination": destination,
            "arrival_datetime": when.isoformat(timespec="minutes"),
            "preference": preference,
            "max_walk_minutes": max_walk,
            "constraint_applied": constraint_applied,
            "results": rows,
        }, recs, state

    if name == "predict_lot":
        requested_lot = args.get("lot_id")
        lot_id = _normalise_lot(requested_lot, lots)
        if lot_id is None:
            lot_id = state.get("last_recommended_lot_id") or state.get("lot_id")

        if not lot_id:
            return {"error": "No car park is available in the conversation context."}, None, state

        when = _parse_dt(
            args.get("arrival_datetime"),
            pd.Timestamp(state.get("arrival_datetime") or default_when).to_pydatetime()
        )

        lot = lots[lots["lot_id"] == lot_id].iloc[0]
        p = predict_occupancy(
            model,
            lot_id,
            int(lot["capacity"]),
            when,
            residual_std=metrics.get("residual_std", 0.04)
        )

        state["lot_id"] = lot_id
        state["arrival_datetime"] = when.isoformat(timespec="minutes")

        return {
            "tool": "predict_lot",
            "lot_id": lot_id,
            "name": lot["name"],
            "arrival_datetime": when.isoformat(timespec="minutes"),
            "predicted_available": int(p["available"]),
            "available_range": f"{int(p['available_low'])}-{int(p['available_high'])}",
            "predicted_occupancy_pct": round(float(p["occupancy_rate"]) * 100),
            "capacity": int(lot["capacity"]),
        }, None, state

    if name == "historical_peak":
        temp = history.copy()
        temp["hour_int"] = temp["timestamp"].dt.hour
        grouped = temp.groupby(["lot_id","hour_int"])["occupancy_rate"].mean().reset_index()
        peak = grouped.loc[grouped["occupancy_rate"].idxmax()]
        quiet = grouped.loc[grouped["occupancy_rate"].idxmin()]

        return {
            "tool": "historical_peak",
            "peak": {
                "lot_id": peak["lot_id"],
                "hour": int(peak["hour_int"]),
                "average_occupancy_pct": round(float(peak["occupancy_rate"]) * 100),
            },
            "quiet": {
                "lot_id": quiet["lot_id"],
                "hour": int(quiet["hour_int"]),
                "average_occupancy_pct": round(float(quiet["occupancy_rate"]) * 100),
            }
        }, None, state

    return {"error": f"Unknown tool: {name}"}, None, state


def ask_parkai(
    question: str,
    model,
    metrics: dict,
    history: pd.DataFrame,
    lots: pd.DataFrame,
    buildings: pd.DataFrame,
    default_destination: str,
    default_when: datetime,
    conversation_state: dict,
):
    client = _client()
    if client is None:
        raise RuntimeError("LLM_API_KEY is not configured.")

    context = {
        "today": datetime.now().strftime("%Y-%m-%d"),
        "valid_buildings": buildings["building"].tolist(),
        "valid_car_parks": lots[["lot_id","name"]].to_dict("records"),
        "current_conversation_state": conversation_state,
        "instruction": (
            "Use the current state for omitted values. "
            "Only change values the user explicitly changes or clearly implies."
        )
    }

    first = client.chat.completions.create(
        model=_model_name(),
        temperature=0,
        messages=[
            {"role":"system","content":SYSTEM_PROMPT},
            {
                "role":"user",
                "content":"Context:\n" + json.dumps(context) + "\n\nUser:\n" + question
            }
        ],
        tools=TOOLS,
        tool_choice="auto",
    )

    msg = first.choices[0].message

    if not msg.tool_calls:
        return msg.content or "I can help with campus parking.", {
            "mode":"llm",
            "tool":None,
            "dataframe":None,
            "conversation_state":conversation_state,
        }

    tool_call = msg.tool_calls[0]
    name = tool_call.function.name

    try:
        args = json.loads(tool_call.function.arguments or "{}")
    except Exception:
        args = {}

    result, dataframe, new_state = execute_tool(
        name,
        args,
        model,
        metrics,
        history,
        lots,
        buildings,
        default_destination,
        default_when,
        conversation_state,
    )

    second = client.chat.completions.create(
        model=_model_name(),
        temperature=0.2,
        messages=[
            {"role":"system","content":SYSTEM_PROMPT},
            {
                "role":"user",
                "content": (
                    "User question:\n" + question +
                    "\n\nUpdated conversation state:\n" + json.dumps(new_state) +
                    "\n\nSmartPark tool result:\n" + json.dumps(result) +
                    "\n\nRespond naturally and concisely. "
                    "If a walking-time constraint is active, explicitly say whether the recommendation satisfies it. "
                    "If no option satisfies the hard constraint, say so clearly and mention the nearest alternative without pretending it qualifies."
                )
            }
        ]
    )

    answer = second.choices[0].message.content or "I found a parking result."

    return answer, {
        "mode":"llm_tool_calling",
        "tool":name,
        "tool_result":result,
        "dataframe":dataframe,
        "conversation_state":new_state,
    }

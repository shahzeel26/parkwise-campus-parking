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
You are the Parking Assistant inside ParkWise.

You help users make campus parking decisions.

IMPORTANT:
- Never invent parking availability, occupancy, walking distance, prices,
  ranking scores, or model results.
- Parking facts must come from ParkWise tools.
- Use the current conversation state for values omitted by the user.
- Only update state values the user explicitly changes or clearly implies.
- Treat explicit walking limits as HARD constraints.
- Do not assume a destination when the user asks generally about available parking.

Distinguish comparison questions from preference changes.

Examples:

"Which option is cheapest?"
-> compare_options with comparison="Cheapest"

"Which one is closest?"
-> compare_options with comparison="Closest"

"Which has the most spaces?"
-> compare_options with comparison="Highest availability"

"I want the cheapest parking."
-> recommend_parking with preference="Cheapest"

"I don't want to walk more than 5 minutes."
-> update max_walk_minutes and recommend again.

"What if I arrive at 11 AM?"
-> preserve previous destination, preference and walking constraint,
   change arrival time and recommend again.

"Why did you recommend that?"
-> explain_last_recommendation.

Keep answers concise, practical and grounded in ParkWise results.
"""


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "general_availability",
            "description": (
                "Show parking areas with the most predicted available spaces "
                "when the user has not supplied a destination."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "arrival_datetime": {
                        "type": ["string", "null"]
                    }
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "recommend_parking",
            "description": (
                "Rank parking options for a destination using current "
                "conversation preferences and constraints."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "destination": {
                        "type": ["string", "null"]
                    },
                    "arrival_datetime": {
                        "type": ["string", "null"]
                    },
                    "preference": {
                        "type": ["string", "null"],
                        "enum": [
                            "Balanced",
                            "Closest",
                            "Highest availability",
                            "Cheapest",
                            None,
                        ],
                    },
                    "max_walk_minutes": {
                        "type": ["number", "null"]
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compare_options",
            "description": (
                "Compare parking options for the current destination and "
                "arrival time by price, walking distance, or predicted "
                "availability. Use this for questions such as "
                "'which option is cheapest?', 'which is closest?' or "
                "'which has the most spaces?'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "comparison": {
                        "type": "string",
                        "enum": [
                            "Cheapest",
                            "Closest",
                            "Highest availability",
                        ],
                    },
                    "destination": {
                        "type": ["string", "null"]
                    },
                    "arrival_datetime": {
                        "type": ["string", "null"]
                    },
                    "max_walk_minutes": {
                        "type": ["number", "null"]
                    },
                },
                "required": ["comparison"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "predict_lot",
            "description": (
                "Predict availability for a specific car park, including "
                "contextual follow-ups."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "lot_id": {
                        "type": ["string", "null"]
                    },
                    "arrival_datetime": {
                        "type": ["string", "null"]
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "historical_peak",
            "description": (
                "Summarise historical peak and quiet parking patterns."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "explain_last_recommendation",
            "description": (
                "Explain why the previously recommended parking lot "
                "was selected."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
    },
]


# ---------------------------------------------------------------------
# LLM configuration
# ---------------------------------------------------------------------

def llm_configured() -> bool:
    return bool(os.getenv("LLM_API_KEY"))


def _client():
    if not llm_configured():
        return None

    from openai import OpenAI

    return OpenAI(
        api_key=os.getenv("LLM_API_KEY"),
        base_url=(
            os.getenv("LLM_BASE_URL")
            or "https://api.openai.com/v1"
        ),
    )


def _model_name() -> str:
    return os.getenv("LLM_MODEL") or "gpt-4o-mini"


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _parse_dt(
    value: str | None,
    fallback: datetime,
) -> datetime:
    if not value:
        return fallback

    try:
        return pd.Timestamp(value).to_pydatetime()

    except Exception:
        return fallback


def _normalise_destination(
    destination: str | None,
    buildings: pd.DataFrame,
    fallback: str,
) -> str:
    if not destination:
        return fallback

    low = destination.lower().strip()

    exact = buildings[
        buildings["building"].str.lower() == low
    ]

    if len(exact):
        return exact.iloc[0]["building"]

    for building in buildings["building"]:
        building_low = building.lower()

        if low in building_low or building_low in low:
            return building

    aliases = {
        "library": "Main Library",
        "business": "Business School",
        "engineering": "Engineering Building",
        "medical": "Medical School",
        "sports": "Sports Centre",
        "student": "Student Central",
    }

    for key, value in aliases.items():
        if (
            key in low
            and value in buildings["building"].tolist()
        ):
            return value

    return fallback


def _normalise_lot(
    lot_id: str | None,
    lots: pd.DataFrame,
) -> str | None:
    if not lot_id:
        return None

    key = str(lot_id).upper().replace(" ", "")

    found = lots[
        lots["lot_id"].str.upper() == key
    ]

    if len(found):
        return found.iloc[0]["lot_id"]

    return None


def _recommendation_rows(
    recs: pd.DataFrame,
) -> list[dict]:
    rows = []

    for _, row in recs.head(5).iterrows():
        rows.append(
            {
                "lot_id": row["lot_id"],
                "name": row["name"],
                "predicted_available": int(row["available"]),
                "available_range": (
                    f"{int(row['available_low'])}-"
                    f"{int(row['available_high'])}"
                ),
                "predicted_occupancy_pct": round(
                    float(row["occupancy_rate"]) * 100
                ),
                "walk_minutes": round(
                    float(row["walk_min"]),
                    1,
                ),
                "hourly_rate": round(
                    float(row["hourly_rate"]),
                    2,
                ),
                "recommendation_score": round(
                    float(row["score"]),
                    1,
                ),
            }
        )

    return rows


# ---------------------------------------------------------------------
# Tool execution
# ---------------------------------------------------------------------

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

    # -------------------------------------------------------------
    # GENERAL AVAILABILITY
    # -------------------------------------------------------------

    if name == "general_availability":
        when = _parse_dt(
            args.get("arrival_datetime"),
            pd.Timestamp(
                state.get("arrival_datetime")
                or default_when
            ).to_pydatetime(),
        )

        rows = []

        for _, lot in lots.iterrows():
            prediction = predict_occupancy(
                model,
                lot["lot_id"],
                int(lot["capacity"]),
                when,
                residual_std=metrics.get(
                    "residual_std",
                    0.04,
                ),
            )

            rows.append(
                {
                    "lot_id": lot["lot_id"],
                    "name": lot["name"],
                    "predicted_available": int(
                        prediction["available"]
                    ),
                    "available_range": (
                        f"{int(prediction['available_low'])}-"
                        f"{int(prediction['available_high'])}"
                    ),
                    "predicted_occupancy_pct": round(
                        float(
                            prediction["occupancy_rate"]
                        )
                        * 100
                    ),
                    "hourly_rate": round(
                        float(lot["hourly_rate"]),
                        2,
                    ),
                }
            )

        dataframe = (
            pd.DataFrame(rows)
            .sort_values(
                [
                    "predicted_available",
                    "predicted_occupancy_pct",
                ],
                ascending=[False, True],
            )
            .reset_index(drop=True)
        )

        state["arrival_datetime"] = when.isoformat(
            timespec="minutes"
        )

        return (
            {
                "tool": "general_availability",
                "arrival_datetime": when.isoformat(
                    timespec="minutes"
                ),
                "results": dataframe.head(5).to_dict(
                    "records"
                ),
            },
            dataframe,
            state,
        )

    # -------------------------------------------------------------
    # COMPARE OPTIONS
    # -------------------------------------------------------------

    if name == "compare_options":
        comparison = (
            args.get("comparison")
            or "Cheapest"
        )

        if comparison not in [
            "Cheapest",
            "Closest",
            "Highest availability",
        ]:
            comparison = "Cheapest"

        destination = _normalise_destination(
            args.get("destination"),
            buildings,
            state.get("destination")
            or default_destination,
        )

        when = _parse_dt(
            args.get("arrival_datetime"),
            pd.Timestamp(
                state.get("arrival_datetime")
                or default_when
            ).to_pydatetime(),
        )

        max_walk = args.get(
            "max_walk_minutes"
        )

        if max_walk is None:
            max_walk = state.get(
                "max_walk_minutes"
            )

        # Generate all candidate options.
        recs = recommend_parking(
            model,
            metrics,
            lots,
            buildings,
            destination,
            when,
            preference="Balanced",
        ).copy()

        if max_walk is not None:
            eligible = recs[
                recs["walk_min"]
                <= float(max_walk)
            ].copy()

            if len(eligible) == 0:
                return (
                    {
                        "tool": "compare_options",
                        "status": "no_match",
                        "comparison": comparison,
                        "destination": destination,
                        "arrival_datetime": when.isoformat(
                            timespec="minutes"
                        ),
                        "max_walk_minutes": max_walk,
                    },
                    None,
                    state,
                )

            recs = eligible

        if comparison == "Cheapest":
            recs = recs.sort_values(
                [
                    "hourly_rate",
                    "walk_min",
                    "available",
                ],
                ascending=[
                    True,
                    True,
                    False,
                ],
            )

        elif comparison == "Closest":
            recs = recs.sort_values(
                [
                    "walk_min",
                    "available",
                    "hourly_rate",
                ],
                ascending=[
                    True,
                    False,
                    True,
                ],
            )

        else:
            recs = recs.sort_values(
                [
                    "available",
                    "walk_min",
                    "hourly_rate",
                ],
                ascending=[
                    False,
                    True,
                    True,
                ],
            )

        recs = recs.reset_index(drop=True)

        rows = _recommendation_rows(recs)

        # Preserve the user's existing recommendation preference.
        # Comparison questions should not silently change it.
        state["destination"] = destination
        state["arrival_datetime"] = when.isoformat(
            timespec="minutes"
        )

        if max_walk is not None:
            state["max_walk_minutes"] = max_walk

        state["last_compared_lot_id"] = (
            rows[0]["lot_id"]
        )

        return (
            {
                "tool": "compare_options",
                "status": "ok",
                "comparison": comparison,
                "destination": destination,
                "arrival_datetime": when.isoformat(
                    timespec="minutes"
                ),
                "max_walk_minutes": max_walk,
                "results": rows,
            },
            recs,
            state,
        )

    # -------------------------------------------------------------
    # EXPLAIN LAST RECOMMENDATION
    # -------------------------------------------------------------

    if name == "explain_last_recommendation":
        lot_id = state.get(
            "last_recommended_lot_id"
        )

        if not lot_id:
            return (
                {
                    "tool": (
                        "explain_last_recommendation"
                    ),
                    "status": (
                        "no_previous_recommendation"
                    ),
                },
                None,
                state,
            )

        destination = (
            state.get("destination")
            or default_destination
        )

        when = pd.Timestamp(
            state.get("arrival_datetime")
            or default_when
        ).to_pydatetime()

        preference = (
            state.get("preference")
            or "Balanced"
        )

        recs = recommend_parking(
            model,
            metrics,
            lots,
            buildings,
            destination,
            when,
            preference=preference,
        )

        row = recs[
            recs["lot_id"] == lot_id
        ]

        if len(row) == 0:
            return (
                {
                    "tool": (
                        "explain_last_recommendation"
                    ),
                    "status": "not_found",
                },
                None,
                state,
            )

        result = row.iloc[0]

        return (
            {
                "tool": (
                    "explain_last_recommendation"
                ),
                "status": "ok",
                "lot_id": lot_id,
                "name": result["name"],
                "destination": destination,
                "preference": preference,
                "predicted_available": int(
                    result["available"]
                ),
                "predicted_occupancy_pct": round(
                    float(
                        result["occupancy_rate"]
                    )
                    * 100
                ),
                "walk_minutes": round(
                    float(result["walk_min"]),
                    1,
                ),
                "hourly_rate": round(
                    float(result["hourly_rate"]),
                    2,
                ),
                "recommendation_score": round(
                    float(result["score"]),
                    1,
                ),
                "active_max_walk_minutes": (
                    state.get(
                        "max_walk_minutes"
                    )
                ),
            },
            None,
            state,
        )

    # -------------------------------------------------------------
    # RECOMMEND PARKING
    # -------------------------------------------------------------

    if name == "recommend_parking":
        destination = _normalise_destination(
            args.get("destination"),
            buildings,
            state.get("destination")
            or default_destination,
        )

        when = _parse_dt(
            args.get("arrival_datetime"),
            pd.Timestamp(
                state.get("arrival_datetime")
                or default_when
            ).to_pydatetime(),
        )

        preference = (
            args.get("preference")
            or state.get("preference")
            or "Balanced"
        )

        if preference not in [
            "Balanced",
            "Closest",
            "Highest availability",
            "Cheapest",
        ]:
            preference = "Balanced"

        max_walk = args.get(
            "max_walk_minutes"
        )

        if max_walk is None:
            max_walk = state.get(
                "max_walk_minutes"
            )

        recs = recommend_parking(
            model,
            metrics,
            lots,
            buildings,
            destination,
            when,
            preference=preference,
        )

        constraint_applied = False
        no_match = False

        if max_walk is not None:
            constraint_applied = True

            filtered = recs[
                recs["walk_min"]
                <= float(max_walk)
            ].copy()

            if len(filtered):
                recs = filtered.reset_index(
                    drop=True
                )

            else:
                no_match = True

        state.update(
    {
        "destination": destination,
        "arrival_datetime": when.isoformat(timespec="minutes"),
        "preference": preference,
        "max_walk_minutes": max_walk,
    }
)
        # A destination recommendation replaces any previous
# specific-car-park conversation context.
        state.pop("lot_id", None)

        if no_match:
            nearest = recommend_parking(
                model,
                metrics,
                lots,
                buildings,
                destination,
                when,
                preference="Closest",
            ).iloc[0]

            state[
                "last_recommended_lot_id"
            ] = nearest["lot_id"]

            return (
                {
                    "tool": "recommend_parking",
                    "status": "no_match",
                    "destination": destination,
                    "arrival_datetime": (
                        when.isoformat(
                            timespec="minutes"
                        )
                    ),
                    "max_walk_minutes": max_walk,
                    "nearest_option": {
                        "lot_id": (
                            nearest["lot_id"]
                        ),
                        "name": nearest["name"],
                        "walk_minutes": round(
                            float(
                                nearest["walk_min"]
                            ),
                            1,
                        ),
                        "predicted_available": int(
                            nearest["available"]
                        ),
                        "predicted_occupancy_pct": (
                            round(
                                float(
                                    nearest[
                                        "occupancy_rate"
                                    ]
                                )
                                * 100
                            )
                        ),
                    },
                },
                None,
                state,
            )

        rows = _recommendation_rows(recs)

        state[
            "last_recommended_lot_id"
        ] = rows[0]["lot_id"]

        return (
            {
                "tool": "recommend_parking",
                "status": "ok",
                "destination": destination,
                "arrival_datetime": (
                    when.isoformat(
                        timespec="minutes"
                    )
                ),
                "preference": preference,
                "max_walk_minutes": max_walk,
                "constraint_applied": (
                    constraint_applied
                ),
                "results": rows,
            },
            recs,
            state,
        )

    # -------------------------------------------------------------
    # SPECIFIC LOT PREDICTION
    # -------------------------------------------------------------

    if name == "predict_lot":
        requested_lot = args.get(
            "lot_id"
        )

        lot_id = _normalise_lot(
            requested_lot,
            lots,
        )

        if lot_id is None:
            lot_id = (
                state.get(
                    "last_recommended_lot_id"
                )
                or state.get("lot_id")
            )

        if not lot_id:
            return (
                {
                    "error": (
                        "No car park is available "
                        "in the conversation context."
                    )
                },
                None,
                state,
            )

        when = _parse_dt(
            args.get("arrival_datetime"),
            pd.Timestamp(
                state.get("arrival_datetime")
                or default_when
            ).to_pydatetime(),
        )

        lot = lots[
            lots["lot_id"] == lot_id
        ].iloc[0]

        prediction = predict_occupancy(
            model,
            lot_id,
            int(lot["capacity"]),
            when,
            residual_std=metrics.get(
                "residual_std",
                0.04,
            ),
        )

        state["lot_id"] = lot_id
        state["arrival_datetime"] = (
            when.isoformat(
                timespec="minutes"
            )
        )

        return (
            {
                "tool": "predict_lot",
                "lot_id": lot_id,
                "name": lot["name"],
                "arrival_datetime": (
                    when.isoformat(
                        timespec="minutes"
                    )
                ),
                "predicted_available": int(
                    prediction["available"]
                ),
                "available_range": (
                    f"{int(prediction['available_low'])}-"
                    f"{int(prediction['available_high'])}"
                ),
                "predicted_occupancy_pct": (
                    round(
                        float(
                            prediction[
                                "occupancy_rate"
                            ]
                        )
                        * 100
                    )
                ),
                "capacity": int(
                    lot["capacity"]
                ),
            },
            None,
            state,
        )

    # -------------------------------------------------------------
    # HISTORICAL PATTERNS
    # -------------------------------------------------------------

    if name == "historical_peak":
        temp = history.copy()

        temp["hour_int"] = (
            temp["timestamp"].dt.hour
        )

        grouped = (
            temp.groupby(
                [
                    "lot_id",
                    "hour_int",
                ]
            )["occupancy_rate"]
            .mean()
            .reset_index()
        )

        peak = grouped.loc[
            grouped[
                "occupancy_rate"
            ].idxmax()
        ]

        quiet = grouped.loc[
            grouped[
                "occupancy_rate"
            ].idxmin()
        ]

        return (
            {
                "tool": "historical_peak",
                "peak": {
                    "lot_id": peak["lot_id"],
                    "hour": int(
                        peak["hour_int"]
                    ),
                    "average_occupancy_pct": (
                        round(
                            float(
                                peak[
                                    "occupancy_rate"
                                ]
                            )
                            * 100
                        )
                    ),
                },
                "quiet": {
                    "lot_id": quiet["lot_id"],
                    "hour": int(
                        quiet["hour_int"]
                    ),
                    "average_occupancy_pct": (
                        round(
                            float(
                                quiet[
                                    "occupancy_rate"
                                ]
                            )
                            * 100
                        )
                    ),
                },
            },
            None,
            state,
        )

    return (
        {
            "error": f"Unknown tool: {name}"
        },
        None,
        state,
    )


# ---------------------------------------------------------------------
# LLM fallback
# ---------------------------------------------------------------------

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
        raise RuntimeError(
            "LLM_API_KEY is not configured."
        )

    context = {
        "today": datetime.now().strftime(
            "%Y-%m-%d"
        ),
        "valid_buildings": (
            buildings["building"].tolist()
        ),
        "valid_car_parks": (
            lots[
                [
                    "lot_id",
                    "name",
                ]
            ].to_dict("records")
        ),
        "current_conversation_state": (
            conversation_state
        ),
        "instruction": (
            "Use current state for omitted values. "
            "Only change values explicitly changed "
            "or clearly implied by the user."
        ),
    }

    first = client.chat.completions.create(
        model=_model_name(),
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": (
                    "Context:\n"
                    + json.dumps(context)
                    + "\n\nUser:\n"
                    + question
                ),
            },
        ],
        tools=TOOLS,
        tool_choice="auto",
    )

    message = first.choices[0].message

    if not message.tool_calls:
        return (
            message.content
            or "I can help with campus parking.",
            {
                "mode": "llm",
                "tool": None,
                "dataframe": None,
                "conversation_state": (
                    conversation_state
                ),
            },
        )

    tool_call = message.tool_calls[0]

    name = tool_call.function.name

    try:
        args = json.loads(
            tool_call.function.arguments
            or "{}"
        )

    except Exception:
        args = {}

    result, dataframe, new_state = (
        execute_tool(
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
    )

    second = (
        client.chat.completions.create(
            model=_model_name(),
            temperature=0.2,
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": (
                        "User question:\n"
                        + question
                        + "\n\nUpdated conversation state:\n"
                        + json.dumps(new_state)
                        + "\n\nParkWise tool result:\n"
                        + json.dumps(result)
                        + "\n\nRespond naturally and concisely. "
                        "Use only facts contained in the tool result. "
                        "If a walking-time constraint is active, "
                        "say whether the result satisfies it. "
                        "If no option satisfies the hard constraint, "
                        "say so clearly."
                    ),
                },
            ],
        )
    )

    answer = (
        second.choices[0].message.content
        or "I found a parking result."
    )

    return (
        answer,
        {
            "mode": "llm_tool_calling",
            "tool": name,
            "tool_result": result,
            "dataframe": dataframe,
            "conversation_state": new_state,
        },
    )
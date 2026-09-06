from __future__ import annotations
import json
import os
import re
from datetime import datetime, timedelta
import pandas as pd

from .recommender import recommend_parking
from .model import predict_occupancy


SYSTEM_PROMPT = """
You are ParkAI, the conversational assistant inside SmartPark AI.

Your job is to understand a user's parking question and convert it into one of the supported actions.

Supported actions:
1. recommend_parking
2. predict_lot
3. historical_peak
4. help

Rules:
- Do not invent parking availability, occupancy, prices, walking times, or lot names.
- The application will calculate all parking facts.
- Extract only what is present or reasonably implied by the user.
- Return JSON only.
- destination must be one of the supplied building names or null.
- lot_id must be one of the supplied lot IDs or null.
- preference must be one of: Balanced, Closest, Highest availability, Cheapest.
- If the user says "closest", use Closest.
- If the user says "most spaces", "safest", or "highest availability", use Highest availability.
- If the user says "cheap" or "cheapest", use Cheapest.
- max_walk_minutes may be a number or null.
- date and time may be null if not specified.
"""


def _client():
    api_key = os.getenv("LLM_API_KEY")
    if not api_key:
        return None
    try:
        from openai import OpenAI
    except Exception:
        return None

    base_url = os.getenv("LLM_BASE_URL") or "https://api.openai.com/v1"
    return OpenAI(api_key=api_key, base_url=base_url)


def llm_enabled():
    return bool(os.getenv("LLM_API_KEY"))


def _extract_json(text: str):
    text = text.strip()
    # direct JSON
    try:
        return json.loads(text)
    except Exception:
        pass

    # fenced or embedded JSON
    match = re.search(r"\{.*\}", text, flags=re.S)
    if match:
        return json.loads(match.group(0))
    raise ValueError("No valid JSON found")


def parse_with_llm(question, buildings, lots):
    client = _client()
    if client is None:
        return None

    model = os.getenv("LLM_MODEL") or "gpt-4o-mini"
    building_names = buildings["building"].tolist()
    lot_ids = lots["lot_id"].tolist()

    user_payload = {
        "question": question,
        "valid_buildings": building_names,
        "valid_lot_ids": lot_ids,
        "today": datetime.now().strftime("%Y-%m-%d"),
        "output_schema": {
            "action": "recommend_parking | predict_lot | historical_peak | help",
            "destination": "string or null",
            "lot_id": "string or null",
            "date": "YYYY-MM-DD or null",
            "time": "HH:MM or null",
            "preference": "Balanced | Closest | Highest availability | Cheapest",
            "max_walk_minutes": "number or null"
        }
    }

    resp = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[
            {"role":"system","content":SYSTEM_PROMPT},
            {"role":"user","content":json.dumps(user_payload)}
        ],
    )
    content = resp.choices[0].message.content
    return _extract_json(content)


def _parse_relative_datetime(question: str, default_when: datetime):
    """Fallback parser for basic relative times when no LLM is configured."""
    q = question.lower()
    when = default_when

    if "tomorrow" in q:
        when = when + timedelta(days=1)
    elif "today" in q:
        when = datetime.combine(datetime.now().date(), when.time())

    # 9:30 am, 10am, 14:00
    m = re.search(r"\b(\d{1,2})(?::(\d{2}))?\s*(am|pm)?\b", q)
    if m:
        h = int(m.group(1))
        minute = int(m.group(2) or 0)
        ampm = m.group(3)
        if ampm == "pm" and h < 12:
            h += 12
        if ampm == "am" and h == 12:
            h = 0
        if 0 <= h <= 23 and 0 <= minute <= 59:
            when = when.replace(hour=h, minute=minute)

    return when


def fallback_parse(question, buildings, lots, default_destination, default_when):
    q = question.lower()

    destination = default_destination
    for b in buildings["building"]:
        if b.lower() in q:
            destination = b
            break

    lot_id = None
    for lid in lots["lot_id"]:
        if re.search(rf"\b{re.escape(lid.lower())}\b", q):
            lot_id = lid
            break

    preference = "Balanced"
    if "closest" in q or "shortest walk" in q:
        preference = "Closest"
    elif "cheapest" in q or "cheap" in q:
        preference = "Cheapest"
    elif any(k in q for k in ["most available","highest availability","most spaces","safest"]):
        preference = "Highest availability"

    if lot_id and any(k in q for k in ["full","available","space","occupancy","busy"]):
        action = "predict_lot"
    elif any(k in q for k in ["busiest","peak","historical","usually","quiet"]):
        action = "historical_peak"
    elif any(k in q for k in ["where","park","recommend","best","closest"]):
        action = "recommend_parking"
    else:
        action = "help"

    max_walk = None
    mw = re.search(r"(?:walk|walking|more than|within)\s*(\d+(?:\.\d+)?)\s*(?:min|minute)", q)
    if not mw:
        mw = re.search(r"(\d+(?:\.\d+)?)\s*(?:min|minute)\s*(?:walk|walking)", q)
    if mw:
        max_walk = float(mw.group(1))

    return {
        "action": action,
        "destination": destination,
        "lot_id": lot_id,
        "when": _parse_relative_datetime(question, default_when),
        "preference": preference,
        "max_walk_minutes": max_walk,
    }


def normalize_plan(plan, default_destination, default_when):
    if plan is None:
        return None

    out = {
        "action": plan.get("action","help"),
        "destination": plan.get("destination") or default_destination,
        "lot_id": plan.get("lot_id"),
        "preference": plan.get("preference") or "Balanced",
        "max_walk_minutes": plan.get("max_walk_minutes"),
    }

    when = default_when
    d = plan.get("date")
    t = plan.get("time")

    try:
        if d:
            day = pd.Timestamp(d).date()
        else:
            day = when.date()
        if t:
            tm = pd.Timestamp(f"2000-01-01 {t}").time()
        else:
            tm = when.time()
        when = datetime.combine(day, tm)
    except Exception:
        pass

    out["when"] = when
    return out


def execute_plan(plan, model, metrics, history, lots, buildings):
    action = plan["action"]
    when = plan["when"]

    if action == "recommend_parking":
        recs = recommend_parking(
            model, metrics, lots, buildings,
            plan["destination"], when,
            preference=plan["preference"]
        )

        if plan.get("max_walk_minutes") is not None:
            filtered = recs[recs["walk_min"] <= float(plan["max_walk_minutes"])]
            if len(filtered):
                recs = filtered.reset_index(drop=True)

        top = recs.iloc[0]
        second = recs.iloc[1] if len(recs) > 1 else None

        answer = (
            f"I recommend **{top['name']} ({top['lot_id']})** for **{plan['destination']}** at "
            f"**{when.strftime('%I:%M %p')}**. It is predicted to have about "
            f"**{int(top['available'])} spaces available** "
            f"(likely range **{int(top['available_low'])}–{int(top['available_high'])}**), "
            f"with approximately **{top['walk_min']:.0f} minutes of walking** and "
            f"**{top['occupancy_rate']:.0%} predicted occupancy**."
        )
        if second is not None:
            answer += (
                f"\n\n**Backup option:** {second['name']} ({second['lot_id']}) with about "
                f"{int(second['available'])} predicted spaces available."
            )

        answer += (
            f"\n\n**Why this recommendation:** the ranking uses your **{plan['preference']}** preference "
            f"and combines predicted availability, walking distance and parking cost."
        )
        return answer, {"type":"recommendation","data":recs}

    if action == "predict_lot" and plan.get("lot_id"):
        lot = lots[lots["lot_id"] == plan["lot_id"]].iloc[0]
        p = predict_occupancy(
            model, lot["lot_id"], int(lot["capacity"]), when,
            residual_std=metrics.get("residual_std",0.04)
        )
        status = (
            "nearly full" if p["occupancy_rate"] >= .90 else
            "busy" if p["occupancy_rate"] >= .75 else
            "moderately busy" if p["occupancy_rate"] >= .50 else
            "relatively quiet"
        )
        answer = (
            f"**{lot['name']} ({lot['lot_id']})** is predicted to be **{status}** at "
            f"**{when.strftime('%I:%M %p')}**, with about **{p['available']} spaces available** "
            f"and **{p['occupancy_rate']:.0%} occupancy**. "
            f"The approximate availability range is **{p['available_low']}–{p['available_high']} spaces**."
        )
        return answer, {"type":"lot_prediction","data":p}

    if action == "historical_peak":
        temp = history.copy()
        temp["hour_int"] = temp["timestamp"].dt.hour
        grouped = temp.groupby(["lot_id","hour_int"])["occupancy_rate"].mean().reset_index()
        peak = grouped.loc[grouped["occupancy_rate"].idxmax()]
        quiet = grouped.loc[grouped["occupancy_rate"].idxmin()]
        answer = (
            f"Historically, the strongest average demand occurs at **{peak['lot_id']} around "
            f"{int(peak['hour_int']):02d}:00**, with about **{peak['occupancy_rate']:.0%} average occupancy**. "
            f"The quietest lot/time pattern is **{quiet['lot_id']} around {int(quiet['hour_int']):02d}:00** "
            f"at roughly **{quiet['occupancy_rate']:.0%} occupancy**."
        )
        return answer, {"type":"historical","data":grouped}

    answer = (
        "I can help with parking recommendations and forecasts. Try asking:\n\n"
        "- **I have class at the Business School tomorrow at 10 AM. Where should I park?**\n"
        "- **I don't want to walk more than 5 minutes. What's the best option?**\n"
        "- **Will P1 be full at 9:30 AM?**\n"
        "- **Which car park is usually busiest?**"
    )
    return answer, {"type":"help","data":None}


def answer_natural_question(question, model, metrics, history, lots, buildings,
                            default_destination, default_when):
    # Prefer LLM parsing when configured.
    plan = None
    mode = "Local parser"

    try:
        raw = parse_with_llm(question, buildings, lots)
        if raw:
            plan = normalize_plan(raw, default_destination, default_when)
            mode = "LLM parser"
    except Exception:
        plan = None

    if plan is None:
        plan = fallback_parse(question, buildings, lots, default_destination, default_when)

    answer, payload = execute_plan(plan, model, metrics, history, lots, buildings)
    payload["parser_mode"] = mode
    payload["plan"] = plan
    return answer, payload

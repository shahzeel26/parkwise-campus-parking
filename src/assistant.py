from __future__ import annotations
import re
import pandas as pd
from .recommender import recommend_parking
from .model import predict_occupancy

def _find_lot(text, lots):
    t = text.lower()
    for _,r in lots.iterrows():
        if str(r["lot_id"]).lower() in t or str(r["name"]).lower() in t:
            return r
    m = re.search(r"\bp\s?([1-9]\d*)\b",t)
    if m:
        key = f"P{m.group(1)}"
        found = lots[lots["lot_id"].str.upper()==key]
        if len(found):
            return found.iloc[0]
    return None

def answer_question(question, model, metrics, history, lots, buildings,
                    default_destination, when, preference="Balanced"):
    q = question.strip()
    low = q.lower()

    lot = _find_lot(q,lots)
    if lot is not None and any(k in low for k in ["full","available","space","occupancy","busy"]):
        p = predict_occupancy(
            model,lot["lot_id"],int(lot["capacity"]),when,
            residual_std=metrics.get("residual_std",0.04)
        )
        status = (
            "nearly full" if p["occupancy_rate"]>=.90 else
            "busy" if p["occupancy_rate"]>=.75 else
            "moderately busy" if p["occupancy_rate"]>=.50 else
            "relatively quiet"
        )
        return (
            f"**{lot['name']} ({lot['lot_id']})** is predicted to be {status} at "
            f"**{pd.Timestamp(when).strftime('%I:%M %p')}**. I estimate about "
            f"**{p['available']} spaces available** ({p['occupancy_rate']:.0%} occupied), "
            f"with an approximate availability range of **{p['available_low']}–{p['available_high']} spaces**."
        )

    if any(k in low for k in ["where","recommend","best","park","closest"]):
        destination = default_destination
        for b in buildings["building"]:
            if b.lower() in low:
                destination = b
                break

        recs = recommend_parking(model,metrics,lots,buildings,destination,when,preference)
        top = recs.iloc[0]
        second = recs.iloc[1]
        return (
            f"I recommend **{top['name']} ({top['lot_id']})** for {destination}. "
            f"It is predicted to have about **{int(top['available'])} spaces available** "
            f"(likely range **{int(top['available_low'])}–{int(top['available_high'])}**) at "
            f"**{pd.Timestamp(when).strftime('%I:%M %p')}**, with an estimated "
            f"**{top['walk_min']:.0f}-minute walk**. "
            f"**Why:** it gives the best balance of predicted availability, walking distance and price. "
            f"A strong backup is **{second['name']} ({second['lot_id']})**."
        )

    if any(k in low for k in ["busiest","peak","usually","historical","quiet"]):
        temp = history.copy()
        temp["hour_int"] = temp["timestamp"].dt.hour
        grp = temp.groupby(["lot_id","hour_int"])["occupancy_rate"].mean().reset_index()
        peak = grp.loc[grp["occupancy_rate"].idxmax()]
        quiet = grp.loc[grp["occupancy_rate"].idxmin()]
        return (
            f"Historically, the strongest average demand occurs at **{peak['lot_id']} around "
            f"{int(peak['hour_int']):02d}:00** ({peak['occupancy_rate']:.0%} occupied on average). "
            f"The quietest lot/time pattern in the dataset is **{quiet['lot_id']} around "
            f"{int(quiet['hour_int']):02d}:00** ({quiet['occupancy_rate']:.0%})."
        )

    return (
        "Try asking **“Where should I park for the Business School?”**, "
        "**“Will P1 be full?”**, **“Which car park is busiest?”**, or "
        "**“Where has the most available spaces?”**."
    )

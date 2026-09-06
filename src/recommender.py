from __future__ import annotations
import math
import pandas as pd
from .model import predict_occupancy

def haversine_m(lat1, lon1, lat2, lon2):
    R = 6371000
    p1,p2 = math.radians(lat1),math.radians(lat2)
    dp = math.radians(lat2-lat1)
    dl = math.radians(lon2-lon1)
    a = math.sin(dp/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
    return 2*R*math.asin(math.sqrt(a))

def recommend_parking(model, metrics, lots, buildings, destination, when,
                      preference="Balanced", rain_mm=0.0, event_flag=0, exam_period=0):
    dest = buildings.loc[buildings["building"]==destination].iloc[0]
    rows = []

    for lot in lots.to_dict("records"):
        pred = predict_occupancy(
            model, lot["lot_id"], lot["capacity"], when,
            rain_mm=rain_mm, event_flag=event_flag, exam_period=exam_period,
            residual_std=metrics.get("residual_std",0.04)
        )

        dist = haversine_m(dest["lat"],dest["lon"],lot["lat"],lot["lon"])
        walk_min = max(1, dist/80.0)

        availability_score = pred["available"]/lot["capacity"]
        distance_score = 1/(1+dist/500)
        price_score = 1/(1+lot["hourly_rate"])

        if preference == "Closest":
            score = .70*distance_score + .25*availability_score + .05*price_score
        elif preference == "Highest availability":
            score = .70*availability_score + .25*distance_score + .05*price_score
        elif preference == "Cheapest":
            score = .60*price_score + .25*availability_score + .15*distance_score
        else:
            score = .48*availability_score + .40*distance_score + .12*price_score

        rows.append({
            **lot, **pred,
            "distance_m":round(dist),
            "walk_min":round(walk_min,1),
            "score":round(score*100,1),
            "occupancy_pct":round(pred["occupancy_rate"]*100),
        })

    return pd.DataFrame(rows).sort_values(["score","available"],ascending=[False,False]).reset_index(drop=True)

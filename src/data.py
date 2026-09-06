from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

# Prototype campus metadata. Replace with official/public coordinates later if desired.
LOT_META = pd.DataFrame([
    {"lot_id":"P1","name":"Engineering North","capacity":180,"lat":-31.98130,"lon":115.81710,"permit_type":"Student","hourly_rate":2.20},
    {"lot_id":"P2","name":"Business School","capacity":140,"lat":-31.97960,"lon":115.81890,"permit_type":"Mixed","hourly_rate":2.50},
    {"lot_id":"P3","name":"Library West","capacity":220,"lat":-31.98050,"lon":115.81530,"permit_type":"Student","hourly_rate":2.00},
    {"lot_id":"P4","name":"Sports Centre","capacity":260,"lat":-31.98300,"lon":115.81980,"permit_type":"Mixed","hourly_rate":1.80},
    {"lot_id":"P5","name":"Medical Precinct","capacity":120,"lat":-31.97790,"lon":115.81600,"permit_type":"Staff","hourly_rate":3.00},
    {"lot_id":"P6","name":"South Campus","capacity":300,"lat":-31.98510,"lon":115.81690,"permit_type":"Student","hourly_rate":1.50},
])

BUILDINGS = pd.DataFrame([
    {"building":"Engineering Building","lat":-31.98100,"lon":115.81790},
    {"building":"Business School","lat":-31.97940,"lon":115.81930},
    {"building":"Main Library","lat":-31.98010,"lon":115.81600},
    {"building":"Sports Centre","lat":-31.98240,"lon":115.82030},
    {"building":"Medical School","lat":-31.97750,"lon":115.81660},
    {"building":"Student Central","lat":-31.98090,"lon":115.81830},
])

def generate_demo_data(days: int = 180, seed: int = 42) -> pd.DataFrame:
    """
    Generate a realistic synthetic university parking dataset.
    The data is deliberately simulated and should be described that way in a portfolio.
    """
    rng = np.random.default_rng(seed)
    end = pd.Timestamp.today().normalize() - pd.Timedelta(days=1)
    start = end - pd.Timedelta(days=days-1)
    stamps = pd.date_range(start, end + pd.Timedelta(hours=23, minutes=30), freq="30min")
    stamps = stamps[(stamps.hour >= 7) & (stamps.hour <= 20)]

    # Lot-specific baseline demand.
    lot_bias = {"P1":0.13,"P2":0.09,"P3":0.04,"P4":-0.01,"P5":0.08,"P6":-0.10}

    rows = []
    for lot in LOT_META.to_dict("records"):
        for ts in stamps:
            hour = ts.hour + ts.minute/60
            dow = ts.dayofweek
            weekend = dow >= 5

            # University demand curve: sharp morning arrival, busy late morning,
            # moderate lunch period, then gradual decline.
            morning = np.exp(-((hour - 9.3) ** 2) / 2.8)
            late_morning = 0.72 * np.exp(-((hour - 11.2) ** 2) / 5.5)
            midday = 0.48 * np.exp(-((hour - 13.5) ** 2) / 8.0)
            afternoon = 0.28 * np.exp(-((hour - 16.0) ** 2) / 7.0)

            demand = 0.10 + 0.46*morning + 0.31*late_morning + 0.18*midday + 0.11*afternoon

            if weekend:
                demand *= 0.42
            else:
                # Slightly stronger Tue-Thu, slightly softer Friday.
                weekday_factor = {0:0.98,1:1.03,2:1.05,3:1.04,4:0.94}.get(dow, 1.0)
                demand *= weekday_factor

            # Contextual factors.
            rain_mm = 0.0
            if rng.random() < 0.18:
                rain_mm = max(0.0, rng.gamma(1.4, 1.2))
            event_flag = int((dow in [1,2,3]) and (16 <= hour <= 19) and rng.random() < 0.08)
            exam_period = int((ts.dayofyear % 120) >= 100)

            # Rain slightly increases closer-campus parking demand.
            rain_effect = min(rain_mm, 8) * 0.008
            event_effect = event_flag * 0.10
            exam_effect = exam_period * 0.06

            ratio = demand + lot_bias[lot["lot_id"]] + rain_effect + event_effect + exam_effect
            ratio += rng.normal(0, 0.025)
            ratio = float(np.clip(ratio, 0.02, 0.98))

            occupied = int(round(ratio * lot["capacity"]))
            rows.append({
                "timestamp":ts,
                "lot_id":lot["lot_id"],
                "capacity":lot["capacity"],
                "occupied":occupied,
                "available":lot["capacity"] - occupied,
                "occupancy_rate":occupied / lot["capacity"],
                "rain_mm":round(float(rain_mm),2),
                "event_flag":event_flag,
                "exam_period":exam_period,
            })
    return add_features(pd.DataFrame(rows))

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"])
    out["hour"] = out["timestamp"].dt.hour + out["timestamp"].dt.minute/60
    out["dow"] = out["timestamp"].dt.dayofweek
    out["month"] = out["timestamp"].dt.month
    out["is_weekend"] = (out["dow"] >= 5).astype(int)

    # Cyclical time features improve smooth forecasting across the day/week.
    out["hour_sin"] = np.sin(2*np.pi*out["hour"]/24)
    out["hour_cos"] = np.cos(2*np.pi*out["hour"]/24)
    out["dow_sin"] = np.sin(2*np.pi*out["dow"]/7)
    out["dow_cos"] = np.cos(2*np.pi*out["dow"]/7)
    return out.sort_values(["timestamp","lot_id"]).reset_index(drop=True)

def load_demo_data() -> pd.DataFrame:
    path = DATA_DIR / "demo_parking.csv"
    if path.exists():
        return add_features(pd.read_csv(path, parse_dates=["timestamp"]))
    DATA_DIR.mkdir(exist_ok=True)
    df = generate_demo_data()
    cols = ["timestamp","lot_id","capacity","occupied","available","occupancy_rate","rain_mm","event_flag","exam_period"]
    df[cols].to_csv(path, index=False)
    return df

def rebuild_demo_csv(days: int = 180, seed: int = 42) -> pd.DataFrame:
    df = generate_demo_data(days=days, seed=seed)
    DATA_DIR.mkdir(exist_ok=True)
    cols = ["timestamp","lot_id","capacity","occupied","available","occupancy_rate","rain_mm","event_flag","exam_period"]
    df[cols].to_csv(DATA_DIR / "demo_parking.csv", index=False)
    return df

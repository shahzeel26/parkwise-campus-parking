from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

FEATURES = [
    "lot_id","capacity","hour","dow","month","is_weekend",
    "hour_sin","hour_cos","dow_sin","dow_cos",
    "rain_mm","event_flag","exam_period"
]
TARGET = "occupancy_rate"

def train_model(df: pd.DataFrame):
    data = df.sort_values("timestamp").dropna(subset=FEATURES+[TARGET]).copy()

    # Time-ordered split to avoid random temporal leakage.
    cutoff = data["timestamp"].quantile(0.80)
    train = data[data["timestamp"] <= cutoff]
    test = data[data["timestamp"] > cutoff]

    categorical = ["lot_id"]
    numeric = [c for c in FEATURES if c not in categorical]

    prep = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
        ("num", StandardScaler(), numeric),
    ])

    rf = RandomForestRegressor(
        n_estimators=180,
        max_depth=16,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=-1,
    )

    pipe = Pipeline([("prep",prep),("model",rf)])
    pipe.fit(train[FEATURES], train[TARGET])

    pred = np.clip(pipe.predict(test[FEATURES]),0,1)
    residuals = test[TARGET].to_numpy() - pred

    metrics = {
        "MAE":float(mean_absolute_error(test[TARGET],pred)),
        "RMSE":float(mean_squared_error(test[TARGET],pred)**0.5),
        "R2":float(r2_score(test[TARGET],pred)),
        "residual_std":float(np.std(residuals)),
    }
    return pipe, metrics

def make_prediction_frame(lot_id: str, capacity: int, when, rain_mm=0.0, event_flag=0, exam_period=0):
    when = pd.Timestamp(when)
    hour = when.hour + when.minute/60
    dow = when.dayofweek
    return pd.DataFrame([{
        "lot_id":str(lot_id),
        "capacity":int(capacity),
        "hour":hour,
        "dow":dow,
        "month":when.month,
        "is_weekend":int(dow >= 5),
        "hour_sin":np.sin(2*np.pi*hour/24),
        "hour_cos":np.cos(2*np.pi*hour/24),
        "dow_sin":np.sin(2*np.pi*dow/7),
        "dow_cos":np.cos(2*np.pi*dow/7),
        "rain_mm":float(rain_mm),
        "event_flag":int(event_flag),
        "exam_period":int(exam_period),
    }])

def predict_occupancy(model, lot_id, capacity, when, rain_mm=0.0, event_flag=0, exam_period=0, residual_std=0.04):
    X = make_prediction_frame(lot_id, capacity, when, rain_mm, event_flag, exam_period)
    rate = float(np.clip(model.predict(X)[0],0,1))
    occupied = int(round(rate*capacity))
    available = max(0, int(capacity-occupied))

    # Approximate 80% predictive range using model holdout residual spread.
    margin_rate = max(0.03, 1.28*float(residual_std))
    low_rate = max(0.0, rate-margin_rate)
    high_rate = min(1.0, rate+margin_rate)
    available_low = max(0, int(round(capacity*(1-high_rate))))
    available_high = min(capacity, int(round(capacity*(1-low_rate))))

    return {
        "occupancy_rate":rate,
        "occupied":occupied,
        "available":available,
        "available_low":available_low,
        "available_high":available_high,
    }

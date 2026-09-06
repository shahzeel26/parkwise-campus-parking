from __future__ import annotations
import os
from pathlib import Path
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[1]

def database_url() -> str | None:
    return os.getenv("DATABASE_URL")

def get_engine() -> Engine | None:
    url = database_url()
    if not url:
        return None
    return create_engine(url, pool_pre_ping=True, future=True)

def db_available() -> bool:
    engine = get_engine()
    if engine is None:
        return False
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception:
        return False

def init_schema(engine: Engine):
    schema_path = PROJECT_ROOT / "sql" / "schema.sql"
    sql = schema_path.read_text(encoding="utf-8")
    statements = [s.strip() for s in sql.split(";") if s.strip()]
    with engine.begin() as conn:
        for stmt in statements:
            conn.execute(text(stmt))

def seed_database(engine: Engine, lots: pd.DataFrame, buildings: pd.DataFrame, history: pd.DataFrame):
    """
    Idempotently load prototype parking lots, buildings and occupancy observations.
    """
    with engine.begin() as conn:
        # parking lots
        for _, r in lots.iterrows():
            conn.execute(text("""
                INSERT INTO parking_lots
                (lot_id, name, latitude, longitude, capacity, permit_type, hourly_rate)
                VALUES (:lot_id, :name, :lat, :lon, :capacity, :permit_type, :hourly_rate)
                ON CONFLICT (lot_id) DO UPDATE SET
                    name = EXCLUDED.name,
                    latitude = EXCLUDED.latitude,
                    longitude = EXCLUDED.longitude,
                    capacity = EXCLUDED.capacity,
                    permit_type = EXCLUDED.permit_type,
                    hourly_rate = EXCLUDED.hourly_rate
            """), {
                "lot_id": r["lot_id"],
                "name": r["name"],
                "lat": float(r["lat"]),
                "lon": float(r["lon"]),
                "capacity": int(r["capacity"]),
                "permit_type": r["permit_type"],
                "hourly_rate": float(r["hourly_rate"]),
            })

        # buildings
        for _, r in buildings.iterrows():
            conn.execute(text("""
                INSERT INTO campus_buildings
                (building_name, latitude, longitude)
                VALUES (:name, :lat, :lon)
                ON CONFLICT (building_name) DO UPDATE SET
                    latitude = EXCLUDED.latitude,
                    longitude = EXCLUDED.longitude
            """), {
                "name": r["building"],
                "lat": float(r["lat"]),
                "lon": float(r["lon"]),
            })

    # bulk occupancy load in chunks, keeping it idempotent
    payload = history[[
        "lot_id","timestamp","occupied","available","occupancy_rate",
        "rain_mm","event_flag","exam_period"
    ]].copy()
    payload = payload.rename(columns={
        "timestamp":"observed_at",
        "occupied":"occupied_spaces",
        "available":"available_spaces",
    })

    rows = payload.to_dict("records")
    insert_sql = text("""
        INSERT INTO occupancy_observations
        (lot_id, observed_at, occupied_spaces, available_spaces,
         occupancy_rate, rain_mm, event_flag, exam_period)
        VALUES
        (:lot_id, :observed_at, :occupied_spaces, :available_spaces,
         :occupancy_rate, :rain_mm, :event_flag, :exam_period)
        ON CONFLICT (lot_id, observed_at) DO NOTHING
    """)

    chunk_size = 2000
    with engine.begin() as conn:
        for i in range(0, len(rows), chunk_size):
            conn.execute(insert_sql, rows[i:i+chunk_size])

def load_from_database(engine: Engine):
    lots = pd.read_sql("""
        SELECT
            lot_id,
            name,
            latitude AS lat,
            longitude AS lon,
            capacity,
            permit_type,
            hourly_rate
        FROM parking_lots
        ORDER BY lot_id
    """, engine)

    buildings = pd.read_sql("""
        SELECT
            building_name AS building,
            latitude AS lat,
            longitude AS lon
        FROM campus_buildings
        ORDER BY building_name
    """, engine)

    history = pd.read_sql("""
        SELECT
            observed_at AS timestamp,
            lot_id,
            occupied_spaces AS occupied,
            available_spaces AS available,
            occupancy_rate,
            rain_mm,
            event_flag::int AS event_flag,
            exam_period::int AS exam_period
        FROM occupancy_observations
        ORDER BY timestamp, lot_id
    """, engine, parse_dates=["timestamp"])

    # add capacity for model features
    history = history.merge(lots[["lot_id","capacity"]], on="lot_id", how="left")
    return lots, buildings, history

def save_forecast(engine: Engine, forecast_rows: list[dict], model_version="rf_v1"):
    if not forecast_rows:
        return
    stmt = text("""
        INSERT INTO occupancy_forecasts
        (lot_id, forecast_for, predicted_occupied, predicted_available,
         predicted_occupancy_rate, model_version)
        VALUES
        (:lot_id, :forecast_for, :predicted_occupied, :predicted_available,
         :predicted_occupancy_rate, :model_version)
    """)
    payload = []
    for r in forecast_rows:
        payload.append({
            "lot_id": r["lot_id"],
            "forecast_for": r["forecast_for"],
            "predicted_occupied": int(r["predicted_occupied"]),
            "predicted_available": int(r["predicted_available"]),
            "predicted_occupancy_rate": float(r["predicted_occupancy_rate"]),
            "model_version": model_version,
        })
    with engine.begin() as conn:
        conn.execute(stmt, payload)
